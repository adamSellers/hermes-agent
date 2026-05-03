import argparse
import json

from plugins.memory import discover_memory_cli_extension, load_memory_provider


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
