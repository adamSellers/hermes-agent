import argparse
import json
import sqlite3
import time

from plugins.memory import discover_memory_cli_extension, load_memory_provider


def _memory_parser():
    setup_fn = discover_memory_cli_extension()
    assert setup_fn is not None
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="memory_command")
    setup_fn(subs)
    return parser


def test_nested_memory_cli_extension_registers_for_active_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )

    setup_fn = discover_memory_cli_extension()
    assert setup_fn is not None
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="memory_command")
    setup_fn(subs)

    args = parser.parse_args(["stats", "--json"])

    assert callable(args.func)


def test_cli_stats_uses_active_profile_database(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    provider.handle_tool_call(
        "remember",
        {
            "kind": "fact",
            "content": {
                "subject": "user",
                "predicate": "timezone",
                "object": "Australia/Sydney",
            },
        },
    )
    provider.shutdown()

    setup_fn = discover_memory_cli_extension()
    if setup_fn is None:
        (tmp_path / "config.yaml").write_text(
            "memory:\n  provider: cos-memory\n",
            encoding="utf-8",
        )
        setup_fn = discover_memory_cli_extension()
    assert setup_fn is not None
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="memory_command")
    setup_fn(subs)
    args = parser.parse_args(["stats", "--json"])
    args.func(args)

    out = capsys.readouterr().out
    stats = json.loads(out)
    assert stats["memory"]["facts"] == 1
    assert stats["db_path"].endswith("cos-memory.db")


def test_direct_setup_enables_provider_and_context_engine(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli.config import load_config
    from hermes_cli.memory_setup import cmd_setup_provider

    cmd_setup_provider("cos-memory")
    config = load_config()

    assert config["memory"]["provider"] == "cos-memory"
    assert config["context"]["engine"] == "cos-context"


def test_cli_queue_json_lists_failed_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    provider.shutdown()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    try:
        conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 7, '{}', 'failed', 3, ?, 'boom', ?, ?)
            """,
            (now, now, now),
        )
        conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('embed_turn', 's1', 8, '{}', 'pending', 0, ?, ?, ?)
            """,
            (now + 3600, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    parser = _memory_parser()
    args = parser.parse_args(["queue", "--failed", "--json"])
    args.func(args)

    result = json.loads(capsys.readouterr().out)
    assert result["counts"]["failed"] == 1
    assert [job["status"] for job in result["jobs"]] == ["failed"]
    assert result["jobs"][0]["last_error"] == "boom"


def test_cli_queue_status_does_not_start_workers_or_recover_stale_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    provider.shutdown()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 7, '{}', 'running', 1, ?, ?, ?)
            """,
            (now + 3600, now - 600, now - 600),
        )
        conn.commit()
    finally:
        conn.close()

    parser = _memory_parser()
    args = parser.parse_args(["queue", "--json"])
    args.func(args)

    result = json.loads(capsys.readouterr().out)
    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    try:
        status = conn.execute("SELECT status FROM memory_work_queue").fetchone()[0]
    finally:
        conn.close()

    assert result["counts"]["running"] == 1
    assert result["stale_running"] == 1
    assert result["recovered_stale_running"] == 0
    assert status == "running"


def test_cli_queue_retry_failed_reports_requeued_count(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    provider.shutdown()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    try:
        conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('consolidate', 's1', NULL, '{}', 'failed', 3, ?, 'boom', ?, ?)
            """,
            (now, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    parser = _memory_parser()
    args = parser.parse_args(["queue", "retry-failed", "--json"])
    args.func(args)

    result = json.loads(capsys.readouterr().out)
    assert result["retried"] == 1
    assert result["counts"]["pending"] == 1

    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    try:
        row = conn.execute(
            "SELECT status, attempts, last_error FROM memory_work_queue"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("pending", 0, None)


def test_cli_queue_recover_stale_is_explicit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    provider.shutdown()
    now = time.time()
    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    try:
        conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('embed_turn', 's1', 8, '{}', 'running', 1, ?, ?, ?)
            """,
            (now + 3600, now - 600, now - 600),
        )
        conn.commit()
    finally:
        conn.close()

    parser = _memory_parser()
    args = parser.parse_args(["queue", "recover-stale", "--json"])
    args.func(args)

    result = json.loads(capsys.readouterr().out)
    assert result["recovered_stale_running"] == 1
    assert result["counts"]["pending"] == 1


def test_cli_search_debug_prints_sources_and_scores(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": {
                    "subject": "user",
                    "predicate": "timezone",
                    "object": "Australia/Sydney",
                },
            },
        )
    )
    provider.shutdown()

    parser = _memory_parser()
    args = parser.parse_args(["search", "timezone", "--debug"])
    args.func(args)

    out = capsys.readouterr().out
    assert remembered["ref"] in out
    assert "source=fts" in out
    assert "lexical=0.75" in out
    assert "semantic=-" in out


def test_cli_remember_and_search_support_tags(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )

    parser = _memory_parser()
    content = json.dumps(
        {
            "subject": "shopping_list:groceries:dried_chick_peas",
            "subject_type": "thing",
            "predicate": "shopping_item_status",
            "object": "status=pending item=dried chick peas list=groceries",
        }
    )
    args = parser.parse_args(
        [
            "remember",
            "fact",
            content,
            "--tag",
            "shopping_list",
            "--tag",
            "shopping_list:groceries",
        ]
    )
    args.func(args)
    remembered = json.loads(capsys.readouterr().out)

    args = parser.parse_args(["search", "pending", "--tag", "shopping_list:groceries"])
    args.func(args)

    out = capsys.readouterr().out
    assert remembered["ref"] in out
    assert "dried chick peas" in out


def test_cli_doctor_reports_malformed_memory_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "memory:\n  provider: cos-memory\n",
        encoding="utf-8",
    )
    provider = load_memory_provider("cos-memory")
    provider.initialize(session_id="s1", hermes_home=str(tmp_path), platform="cli")
    assert provider._conn is not None
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO entities(
                type, canonical_name, aliases, attributes, first_seen,
                last_seen, salience
            ) VALUES ('person', 'user_friend_and_her_email_address', '[]', '{}', 1, 1, 0.1)
            """
        )
    provider.shutdown()

    parser = _memory_parser()
    args = parser.parse_args(["doctor", "--json"])
    args.func(args)

    result = json.loads(capsys.readouterr().out)
    assert result["ok"] is False
    assert any(
        item["code"] == "malformed_entity_name" and item["ref"].startswith("entity:")
        for item in result["findings"]
    )
