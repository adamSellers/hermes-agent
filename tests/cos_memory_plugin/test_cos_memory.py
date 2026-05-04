import json
import sqlite3
import time
from types import SimpleNamespace

from hermes_state import SessionDB
from plugins.memory import load_memory_provider


def _provider(tmp_path, session_id="s1"):
    provider = load_memory_provider("cos-memory")
    assert provider is not None
    provider.initialize(session_id=session_id, hermes_home=str(tmp_path), platform="cli")
    return provider


def _provider_without_workers(tmp_path, session_id="s1"):
    provider = load_memory_provider("cos-memory")
    assert provider is not None
    provider.initialize(
        session_id=session_id,
        hermes_home=str(tmp_path),
        platform="cli",
        start_workers=False,
    )
    return provider


def _read_turns(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM session_turns ORDER BY session_id, turn_id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def test_load_memory_provider_discovers_cos_memory():
    provider = load_memory_provider("cos-memory")

    assert provider is not None
    assert provider.name == "cos-memory"
    assert provider.is_available() is True

    schemas = provider.get_tool_schemas()
    assert [schema["name"] for schema in schemas] == [
        "recall_session",
        "recall_memory",
        "remember",
        "forget",
    ]
    params = schemas[0]["parameters"]
    assert params["required"] == ["query"]
    assert params["properties"]["max_results"]["maximum"] == 15


def test_initialize_creates_cos_memory_db_and_schema(tmp_path):
    provider = _provider(tmp_path)
    provider.shutdown()

    db_path = tmp_path / "cos-memory.db"
    assert db_path.exists()

    conn = sqlite3.connect(str(db_path))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {"schema_version", "session_turns", "memory_work_queue", "digest_cache"} <= tables
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        assert version == 1
    finally:
        conn.close()

    provider = _provider(tmp_path)
    provider.shutdown()


def test_sync_turn_records_stable_turn_ids_across_restart(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    provider.sync_turn("u1", "a1", session_id="s1")
    provider.sync_turn("u2", "a2", session_id="s1")
    provider.shutdown()

    provider = _provider(tmp_path, session_id="s1")
    provider.sync_turn("u3", "a3", session_id="s1")
    provider.shutdown()

    rows = _read_turns(tmp_path / "cos-memory.db")
    assert [(row["session_id"], row["turn_id"]) for row in rows] == [
        ("s1", 1),
        ("s1", 2),
        ("s1", 3),
    ]
    assert {row["compression_state"] for row in rows} == {"hot"}
    assert {row["session_embedding_status"] for row in rows} <= {"pending", "done"}
    assert {row["extraction_status"] for row in rows} <= {"pending", "done"}


def test_memory_stats_exposes_queue_counts_and_errors(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = time.time()
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('embed_turn', 's1', 1, '{}', 'pending', 1, ?, 'transient embed error', ?, ?)
            """,
            (now + 3600, now, now),
        )
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 2, '{}', 'failed', 3, ?, 'extract exploded', ?, ?)
            """,
            (now, now, now),
        )
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('consolidate', 's1', NULL, '{}', 'running', 1, ?, ?, ?)
            """,
            (now, now, now),
        )

    stats = provider.memory_stats()
    provider.shutdown()

    assert stats["queue"]["pending"] == 1
    assert stats["queue"]["running"] == 1
    assert stats["queue"]["failed"] == 1
    errors = {(row["status"], row["last_error"]) for row in stats["queue_errors"]}
    assert ("failed", "extract exploded") in errors
    assert ("pending", "transient embed error") in errors


def test_queue_status_reports_stale_running_jobs_without_recovery(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = time.time()
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('embed_turn', 's1', 1, '{}', 'running', 1, ?, ?, ?)
            """,
            (now + 3600, now - 600, now - 600),
        )

    result = provider.queue_status()
    row = provider._conn.execute(
        "SELECT status, last_error FROM memory_work_queue"
    ).fetchone()
    provider.shutdown()

    assert result["stale_running"] == 1
    assert result["recovered_stale_running"] == 0
    assert result["counts"]["running"] == 1
    assert row["status"] == "running"
    assert row["last_error"] is None


def test_recover_stale_queue_jobs_is_explicit(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = time.time()
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('embed_turn', 's1', 1, '{}', 'running', 1, ?, ?, ?)
            """,
            (now + 3600, now - 600, now - 600),
        )

    recovered = provider.recover_stale_queue_jobs(stale_after_seconds=300)
    result = provider.queue_status()
    row = provider._conn.execute(
        "SELECT status, last_error FROM memory_work_queue"
    ).fetchone()
    provider.shutdown()

    assert recovered == 1
    assert result["counts"]["pending"] == 1
    assert result["stale_running"] == 0
    assert row["status"] == "pending"
    assert row["last_error"] == "recovered stale running job"


def test_retry_failed_jobs_requeues_and_resets_attempts(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = time.time()
    with provider._lock:
        first = provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 1, '{}', 'failed', 3, ?, 'first failure', ?, ?)
            """,
            (now + 3600, now - 10, now - 10),
        ).lastrowid
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, last_error, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 2, '{}', 'failed', 3, ?, 'second failure', ?, ?)
            """,
            (now + 3600, now, now),
        )

    result = provider.retry_failed_jobs(limit=1)
    rows = provider._conn.execute(
        "SELECT id, status, attempts, last_error FROM memory_work_queue ORDER BY id"
    ).fetchall()
    provider.shutdown()

    assert result["retried"] == 1
    assert result["job_ids"] == [first]
    assert [(row["status"], row["attempts"], row["last_error"]) for row in rows] == [
        ("pending", 0, None),
        ("failed", 3, "second failure"),
    ]


def test_claim_job_is_single_consumer(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = time.time()
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO memory_work_queue(
                job_type, session_id, turn_id, payload, status, attempts,
                run_after, created_at, updated_at
            ) VALUES ('extract_turn', 's1', 1, '{}', 'pending', 0, ?, ?, ?)
            """,
            (now, now, now),
        )

    first = provider._claim_job()
    second = provider._claim_job()
    row = provider._conn.execute(
        "SELECT status, attempts FROM memory_work_queue"
    ).fetchone()
    provider.shutdown()

    assert first is not None
    assert first["status"] == "running"
    assert second is None
    assert row["status"] == "running"
    assert row["attempts"] == 1


def test_recall_session_searches_current_session_fts_only(tmp_path):
    state = SessionDB(tmp_path / "state.db")
    state.create_session("s1", source="cli")
    user_id = state.append_message("s1", role="user", content="remember blue comet token")
    state.append_message("s1", role="assistant", content="I noted the blue comet token.")
    state.close()

    provider = _provider(tmp_path, session_id="s1")
    provider.sync_turn("remember blue comet token", "I noted the blue comet token.", session_id="s1")
    result = json.loads(
        provider.handle_tool_call("recall_session", {"query": "blue comet", "max_results": 3})
    )
    provider.shutdown()

    assert "results" in result
    assert result["results"]
    first = result["results"][0]
    assert {"turn_id", "message_id", "session_id", "role", "timestamp", "snippet", "score"} <= set(first)
    assert first["session_id"] == "s1"
    assert any(item["message_id"] == user_id for item in result["results"])
    assert any(item["turn_id"] == 1 for item in result["results"])


def test_recall_session_includes_parents_but_not_siblings_or_children(tmp_path):
    state = SessionDB(tmp_path / "state.db")
    state.create_session("root", source="cli")
    state.create_session("child", source="cli", parent_session_id="root")
    state.create_session("sibling", source="cli", parent_session_id="root")
    state.create_session("grandchild", source="cli", parent_session_id="child")
    state.create_session("unrelated", source="cli")
    for sid in ("root", "child", "sibling", "grandchild", "unrelated"):
        state.append_message(sid, role="user", content=f"lineage-needle from {sid}")
    state.close()

    provider = _provider(tmp_path, session_id="child")
    child_result = json.loads(
        provider.handle_tool_call("recall_session", {"query": "lineage-needle", "max_results": 10})
    )
    child_sessions = {item["session_id"] for item in child_result["results"]}

    provider.on_session_switch("root")
    root_result = json.loads(
        provider.handle_tool_call("recall_session", {"query": "lineage-needle", "max_results": 10})
    )
    root_sessions = {item["session_id"] for item in root_result["results"]}
    provider.shutdown()

    assert child_sessions == {"root", "child"}
    assert root_sessions == {"root"}


def test_recall_session_clamps_limit_and_handles_bad_query(tmp_path):
    state = SessionDB(tmp_path / "state.db")
    state.create_session("s1", source="cli")
    for idx in range(20):
        state.append_message("s1", role="user", content=f"limit-token message {idx}")
    state.close()

    provider = _provider(tmp_path, session_id="s1")
    high = json.loads(
        provider.handle_tool_call("recall_session", {"query": "limit-token", "max_results": 99})
    )
    low = json.loads(
        provider.handle_tool_call("recall_session", {"query": "limit-token", "max_results": 0})
    )
    malformed = json.loads(
        provider.handle_tool_call("recall_session", {"query": "C++ (unterminated", "max_results": 5})
    )
    provider.shutdown()

    assert len(high["results"]) == 15
    assert len(low["results"]) == 1
    assert "results" in malformed


def test_embed_turn_stores_session_vectors(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    provider._embed_texts = lambda texts, purpose="memory": (
        [[1.0, 0.0] if "blue" in text else [0.0, 1.0] for text in texts],
        "test-embed",
    )
    provider._process_embed_turn(
        {"session_id": "s1", "turn_id": 1},
        {"user_content": "blue comet token", "assistant_content": "noted"},
    )
    provider.shutdown()

    conn = sqlite3.connect(str(tmp_path / "cos-memory.db"))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT role, embedding_model, embedding_vector FROM session_embedding_records ORDER BY role"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    assert {row["embedding_model"] for row in rows} == {"test-embed"}
    assert all(row["embedding_vector"] for row in rows)


def test_recall_session_uses_vectors_when_fts_misses(tmp_path):
    state = SessionDB(tmp_path / "state.db")
    state.create_session("s1", source="cli")
    state.close()

    provider = _provider(tmp_path, session_id="s1")

    def fake_embed(texts, purpose="memory"):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append([1.0, 0.0] if "space" in lowered or "comet" in lowered else [0.0, 1.0])
        return vectors, "test-embed"

    provider._embed_texts = fake_embed
    provider._process_embed_turn(
        {"session_id": "s1", "turn_id": 1},
        {"user_content": "blue comet token", "assistant_content": "noted"},
    )
    result = json.loads(
        provider.handle_tool_call("recall_session", {"query": "space marker", "max_results": 3})
    )
    provider.shutdown()

    assert result["results"]
    assert result["results"][0]["source"] == "vector"
    assert "blue comet token" in result["results"][0]["snippet"]


def test_remember_recall_and_forget_fact(tmp_path):
    provider = _provider(tmp_path, session_id="s1")

    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": {
                    "subject": "user",
                    "predicate": "default_flight_class",
                    "object": "economy",
                },
                "confidence": 0.9,
            },
        )
    )
    assert remembered["ok"] is True
    assert remembered["ref"].startswith("fact:")

    recalled = provider.handle_tool_call(
        "recall_memory",
        {"query": "economy", "max_results": 5},
    )
    assert remembered["ref"] in recalled
    assert "default_flight_class" in recalled

    forgotten = json.loads(
        provider.handle_tool_call(
            "forget",
            {"memory_ref": remembered["ref"], "reason": "test correction"},
        )
    )
    assert forgotten["ok"] is True
    recalled_after_forget = provider.handle_tool_call(
        "recall_memory",
        {"query": "economy", "max_results": 5},
    )
    assert recalled_after_forget == "No durable memory results."
    all_facts = provider.list_memory(kind="fact", include_superseded=True)
    assert all_facts and all_facts[0]["superseded"] is True
    provider.shutdown()


def test_remember_tags_support_skill_owned_state(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")

    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": {
                    "subject": "shopping_list:groceries:dried_chick_peas",
                    "subject_type": "thing",
                    "predicate": "shopping_item_status",
                    "object": "status=pending item=dried chick peas list=groceries",
                },
                "tags": ["shopping_list", "shopping_list:groceries", "status:pending"],
            },
        )
    )
    provider.handle_tool_call(
        "remember",
        {
            "kind": "fact",
            "content": {
                "subject": "shopping_list:hardware:screws",
                "subject_type": "thing",
                "predicate": "shopping_item_status",
                "object": "status=pending item=screws list=hardware",
            },
            "tags": ["shopping_list", "shopping_list:hardware", "status:pending"],
        },
    )

    recalled = provider.handle_tool_call(
        "recall_memory",
        {
            "query": "pending",
            "tags": ["shopping_list:groceries", "status:pending"],
            "kinds": ["fact"],
            "max_results": 100,
        },
    )
    tag_only = provider.search_memory(
        "",
        kinds={"fact"},
        tags=["shopping_list:groceries"],
        limit=100,
    )
    provider.shutdown()

    assert remembered["tags"] == ["shopping_list", "shopping_list:groceries", "status:pending"]
    assert "dried chick peas" in recalled
    assert "screws" not in recalled
    assert [row["ref"] for row in tag_only] == [remembered["ref"]]
    assert tag_only[0]["tags"] == ["shopping_list", "shopping_list:groceries", "status:pending"]


def test_tagged_fact_update_moves_item_between_status_tags(tmp_path):
    provider = _provider_without_workers(tmp_path, session_id="s1")
    content = {
        "subject": "shopping_list:groceries:dried_chick_peas",
        "subject_type": "thing",
        "predicate": "shopping_item_status",
        "object": "status=pending item=dried chick peas list=groceries",
    }
    pending = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": content,
                "tags": ["shopping_list", "shopping_list:groceries", "status:pending"],
            },
        )
    )
    pending_id = int(pending["ref"].split(":", 1)[1])
    assert provider._conn is not None
    with provider._lock:
        provider._conn.execute(
            """
            INSERT INTO memory_embedding_records(
                kind, row_id, text, embedding_model, embedding_vector,
                embedded_at, created_at, updated_at
            ) VALUES ('fact', ?, 'old pending shopping item', 'test-embed', '[1.0,0.0]', 1, 1, 1)
            """,
            (pending_id,),
        )
    bought_content = dict(content)
    bought_content["object"] = "status=bought item=dried chick peas list=groceries"
    bought = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": bought_content,
                "tags": ["shopping_list", "shopping_list:groceries", "status:bought"],
            },
        )
    )

    pending_rows = provider.search_memory(
        "",
        kinds={"fact"},
        tags=["shopping_list:groceries", "status:pending"],
        limit=100,
    )
    bought_rows = provider.search_memory(
        "",
        kinds={"fact"},
        tags=["shopping_list:groceries", "status:bought"],
        limit=100,
    )
    old_embedding = provider._conn.execute(
        "SELECT 1 FROM memory_embedding_records WHERE kind = 'fact' AND row_id = ?",
        (pending_id,),
    ).fetchone()
    provider.shutdown()

    assert pending["ref"] != bought["ref"]
    assert pending_rows == []
    assert [row["ref"] for row in bought_rows] == [bought["ref"]]
    assert "status=bought" in bought_rows[0]["summary"]
    assert old_embedding is None


def test_forget_entity_hides_it_from_normal_search(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "entity",
                "content": {
                    "type": "person",
                    "canonical_name": "Alena",
                },
            },
        )
    )

    assert "Alena" in provider.handle_tool_call("recall_memory", {"query": "Alena"})
    forgotten = json.loads(
        provider.handle_tool_call(
            "forget",
            {"memory_ref": remembered["ref"], "reason": "test correction"},
        )
    )
    assert forgotten["ok"] is True
    assert provider.handle_tool_call("recall_memory", {"query": "Alena"}) == "No durable memory results."
    all_entities = provider.list_memory(kind="entity", include_superseded=True)
    assert all_entities and all_entities[0]["superseded"] is True
    provider.shutdown()


def test_semantic_memory_search_uses_memory_vectors(tmp_path):
    provider = _provider(tmp_path, session_id="s1")

    def fake_embed(texts, purpose="memory"):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [1.0, 0.0]
                if "email" in lowered or "contact" in lowered or "reach" in lowered
                else [0.0, 1.0]
            )
        return vectors, "test-embed"

    provider._embed_texts = fake_embed
    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "fact",
                "content": {
                    "subject": "Alena",
                    "subject_type": "person",
                    "predicate": "email_address",
                    "object": "alena@example.com",
                },
            },
        )
    )
    _kind, raw_id = remembered["ref"].split(":", 1)
    provider._process_embed_memory_row(
        {"target_table": "fact", "target_row_id": int(raw_id)},
        {"kind": "fact", "row_id": int(raw_id)},
    )

    rows = provider.search_memory("how do I contact my friend?", limit=3)
    provider.shutdown()

    assert rows
    assert rows[0]["ref"] == remembered["ref"]
    assert rows[0]["source"] == "vector"
    assert "Alena email_address alena@example.com" in rows[0]["summary"]


def test_memory_search_filters_weak_vector_only_results_when_lexical_matches(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    provider._embed_texts = lambda texts, purpose="memory": ([[1.0, 0.0] for _text in texts], "test-embed")
    seller = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "entity",
                "content": {"type": "person", "canonical_name": "Owen Sellers"},
            },
        )
    )
    unrelated = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "entity",
                "content": {"type": "person", "canonical_name": "Alena"},
            },
        )
    )

    provider._memory_item = lambda kind, row_id: {
        "kind": "entity",
        "id": row_id,
        "ref": f"entity:{row_id}",
        "score": 0.0,
        "summary": "Alena (person)",
    } if f"entity:{row_id}" == unrelated["ref"] else None
    provider._search_memory_vector = lambda query, kinds=None, limit=20: [
        {
            "kind": "entity",
            "id": int(unrelated["ref"].split(":", 1)[1]),
            "ref": unrelated["ref"],
            "score": 0.38,
            "semantic_score": 0.38,
            "summary": "Alena (person)",
        }
    ]

    rows = provider.search_memory("Sellers", limit=20)
    provider.shutdown()

    refs = [row["ref"] for row in rows]
    assert seller["ref"] in refs
    assert unrelated["ref"] not in refs


def test_system_prompt_block_includes_open_commitment(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    remembered = json.loads(
        provider.handle_tool_call(
            "remember",
            {
                "kind": "commitment",
                "content": {
                    "description": "book flights to Tokyo",
                    "owner": "agent",
                },
            },
        )
    )
    assert remembered["ref"].startswith("commitment:")

    briefing = provider.system_prompt_block()
    provider.shutdown()

    assert "CHIEF OF STAFF BRIEFING" in briefing
    assert "book flights to Tokyo" in briefing


def test_heuristic_user_attribute_uses_user_subject(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    provider._llm_extract = lambda user_text, assistant_text="": None
    provider._process_extract_turn(
        {"session_id": "s1", "turn_id": 1},
        {"user_content": "Please remember that my timezone is Australia/Sydney."},
    )
    provider.consolidate_pending(session_id="s1")

    recalled = provider.handle_tool_call(
        "recall_memory",
        {"query": "Australia/Sydney", "max_results": 5},
    )
    provider.shutdown()

    assert "user timezone Australia/Sydney" in recalled
    assert "user_timezone" not in recalled


def test_llm_extraction_creates_person_entity_and_clean_fact(tmp_path, monkeypatch):
    provider = _provider(tmp_path, session_id="s1")
    payload = {
        "memories": [
            {
                "kind": "entity",
                "confidence": 0.85,
                "content": {
                    "type": "person",
                    "canonical_name": "Alena",
                    "aliases": [],
                    "attributes": {"relationship_to_user": "friend"},
                },
            },
            {
                "kind": "fact",
                "confidence": 0.85,
                "content": {
                    "subject": "Alena",
                    "subject_type": "person",
                    "predicate": "email address",
                    "object": "alena@example.com",
                },
            },
        ]
    }

    def fake_call_llm(**kwargs):
        assert kwargs["task"] == "memory_extraction"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(payload))
                )
            ]
        )

    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake_call_llm)
    provider._process_extract_turn(
        {"session_id": "s1", "turn_id": 1},
        {
            "user_content": (
                "Please remember Alena is my friend and her email address "
                "is alena@example.com."
            ),
            "assistant_content": "Noted.",
        },
    )
    provider.consolidate_pending(session_id="s1")

    recalled = provider.handle_tool_call(
        "recall_memory",
        {"query": "Alena", "max_results": 10},
    )
    provider.shutdown()

    assert "Alena (person)" in recalled
    assert "Alena email_address alena@example.com" in recalled
    assert "user_friend" not in recalled


def test_heuristic_skips_malformed_contact_attribute(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    provider._llm_extract = lambda user_text, assistant_text="": None
    provider._process_extract_turn(
        {"session_id": "s1", "turn_id": 1},
        {"user_content": "Please remember that my friend and her email address is Alena."},
    )
    provider.consolidate_pending(session_id="s1")

    recalled = provider.handle_tool_call(
        "recall_memory",
        {"query": "Alena", "max_results": 5},
    )
    provider.shutdown()

    assert recalled == "No durable memory results."


def test_memory_doctor_flags_malformed_names_and_embedding_hygiene(tmp_path):
    provider = _provider(tmp_path, session_id="s1")
    assert provider._conn is not None
    now = 123.0
    with provider._lock:
        entity_cur = provider._conn.execute(
            """
            INSERT INTO entities(
                type, canonical_name, aliases, attributes, first_seen,
                last_seen, salience
            ) VALUES ('person', 'user_friend_and_her_email_address', '[]', '{}', ?, ?, 0.1)
            """,
            (now, now),
        )
        entity_id = int(entity_cur.lastrowid)
        fact_cur = provider._conn.execute(
            """
            INSERT INTO facts(
                subject_entity_id, predicate, object_value, confidence,
                source_session_id, source_turn_id, created_at,
                last_confirmed_at, superseded_at
            ) VALUES (?, 'email_address', 'old@example.com', 0.8, 's1', 1, ?, ?, ?)
            """,
            (entity_id, now, now, now),
        )
        fact_id = int(fact_cur.lastrowid)
        provider._conn.execute(
            """
            INSERT INTO memory_embedding_records(
                kind, row_id, text, embedding_model, embedding_vector,
                embedded_at, created_at, updated_at
            ) VALUES ('fact', ?, 'stale fact text', 'test-embed', '[1.0,0.0]', ?, ?, ?)
            """,
            (fact_id, now, now, now),
        )
    provider._embedding_config = lambda: {"configured": True, "model": "test-embed"}

    result = provider.memory_doctor(limit=20)
    provider.shutdown()

    findings = result["findings"]
    assert result["ok"] is False
    assert any(
        item["ref"] == f"entity:{entity_id}" and item["code"] == "malformed_entity_name"
        for item in findings
    )
    assert any(
        item["ref"] == f"entity:{entity_id}" and item["code"] == "missing_embedding"
        for item in findings
    )
    assert any(
        item["ref"] == f"fact:{fact_id}" and item["code"] == "superseded_embedding"
        for item in findings
    )
