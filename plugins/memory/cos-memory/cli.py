"""CLI commands for the cos-memory provider."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

from hermes_constants import get_hermes_home
from plugins.memory import load_memory_provider


def _provider():
    provider = load_memory_provider("cos-memory")
    if provider is None:
        raise SystemExit("cos-memory provider could not be loaded")
    provider.initialize(
        session_id="cli-curation",
        hermes_home=str(get_hermes_home()),
        platform="cli",
        agent_context="primary",
    )
    return provider


def _parse_ref(memory_ref: str) -> tuple[Optional[str], Optional[int]]:
    text = (memory_ref or "").strip()
    if ":" not in text:
        return None, None
    kind, raw_id = text.split(":", 1)
    try:
        row_id = int(raw_id)
    except ValueError:
        return None, None
    kind = kind.strip().lower()
    if kind not in {"entity", "fact", "preference", "commitment", "relation"}:
        return None, None
    return kind, row_id


def _print_items(items: Iterable[dict[str, Any]]) -> None:
    found = False
    for item in items:
        found = True
        pin = " pinned" if item.get("pinned") else ""
        stale = " superseded" if item.get("superseded") else ""
        print(f"{item['ref']:<18} {item['kind']:<11}{pin}{stale}")
        print(f"  {item['summary']}")
    if not found:
        print("No records.")


def _confirm(prompt: str, *, assume_yes: bool = False) -> bool:
    if assume_yes:
        return True
    if not sys.stdin.isatty():
        print("Refusing without --yes in a non-interactive shell.")
        return False
    answer = input(f"{prompt} Type 'yes' to continue: ").strip().lower()
    return answer == "yes"


def cmd_enable(args) -> None:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    config.setdefault("memory", {})["provider"] = "cos-memory"
    if getattr(args, "context", True):
        config.setdefault("context", {})["engine"] = "cos-context"
    save_config(config)
    print("cos-memory enabled.")
    if getattr(args, "context", True):
        print("cos-context enabled.")
    print("Start a new Hermes session to activate.")


def cmd_disable(args) -> None:
    from hermes_cli.config import load_config, save_config

    config = load_config()
    if isinstance(config.get("memory"), dict):
        config["memory"]["provider"] = ""
    if isinstance(config.get("context"), dict) and config["context"].get("engine") == "cos-context":
        config["context"]["engine"] = "compressor"
    save_config(config)
    print("cos-memory disabled. Built-in memory remains active.")


def cmd_stats(args) -> None:
    provider = _provider()
    try:
        stats = provider.memory_stats()
        if getattr(args, "json", False):
            print(json.dumps(stats, indent=2, sort_keys=True))
            return
        print(f"Database: {stats.get('db_path', '')}")
        print(f"Sessions: {stats.get('sessions', 0)}")
        print(f"Turns:    {stats.get('turns', 0)}")
        print(f"Staging:  {stats.get('staging_pending', 0)} pending")
        print("Memory:")
        for table, count in stats.get("memory", {}).items():
            print(f"  {table:<13} {count}")
        print("Queue:")
        queue = stats.get("queue", {})
        if queue:
            for status, count in sorted(queue.items()):
                print(f"  {status:<13} {count}")
        else:
            print("  empty")
        embeddings = stats.get("embeddings", {})
        print("Embeddings:")
        print(f"  configured    {embeddings.get('configured', False)}")
        print(f"  model         {embeddings.get('model', '') or '-'}")
        print(
            "  sessions      "
            f"{embeddings.get('session_embedded', 0)}/{embeddings.get('session_records', 0)}"
        )
        print(
            "  memory        "
            f"{embeddings.get('memory_embedded', 0)}/{embeddings.get('memory_records', 0)}"
        )
    finally:
        provider.shutdown()


def cmd_list(args) -> None:
    provider = _provider()
    try:
        items = provider.list_memory(
            kind=getattr(args, "kind", "") or "",
            limit=getattr(args, "limit", 50),
            include_superseded=getattr(args, "all", False),
            status=getattr(args, "status", "") or "",
        )
        _print_items(items)
    finally:
        provider.shutdown()


def cmd_show(args) -> None:
    provider = _provider()
    try:
        kind, row_id = _parse_ref(args.memory_ref)
        if not kind or row_id is None:
            raise SystemExit("Use a typed memory ref like fact:123")
        item = provider.get_memory(kind, row_id)
        if not item:
            print(f"{args.memory_ref} not found.")
            return
        print(json.dumps(item, indent=2, sort_keys=True))
    finally:
        provider.shutdown()


def cmd_search(args) -> None:
    provider = _provider()
    try:
        kinds = set(getattr(args, "kind", []) or [])
        items = provider.search_memory(args.query, kinds=kinds, limit=args.limit)
        _print_items(items)
    finally:
        provider.shutdown()


def cmd_remember(args) -> None:
    provider = _provider()
    try:
        try:
            content = json.loads(args.content)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"content must be JSON: {exc}") from exc
        result = provider.handle_tool_call(
            "remember",
            {
                "kind": args.kind,
                "content": content,
                "confidence": args.confidence,
            },
        )
        print(result)
    finally:
        provider.shutdown()


def cmd_pin(args) -> None:
    provider = _provider()
    try:
        kind, row_id = _parse_ref(args.memory_ref)
        if not kind or row_id is None:
            raise SystemExit("Use a typed memory ref like fact:123")
        ok = provider.set_pin(kind, row_id, pinned=not args.unpin, actor="user")
        print(("Pinned" if not args.unpin else "Unpinned") if ok else "No matching memory.")
    finally:
        provider.shutdown()


def cmd_forget(args) -> None:
    provider = _provider()
    try:
        kind, row_id = _parse_ref(args.memory_ref)
        if not kind or row_id is None:
            raise SystemExit("Use a typed memory ref like fact:123")
        ok = provider.supersede_memory(kind, row_id, reason=args.reason or "", actor="user")
        print("Forgotten." if ok else "No matching memory.")
    finally:
        provider.shutdown()


def cmd_delete(args) -> None:
    provider = _provider()
    try:
        kind, row_id = _parse_ref(args.memory_ref)
        if not kind or row_id is None:
            raise SystemExit("Use a typed memory ref like fact:123")
        if not _confirm(f"This will permanently delete {args.memory_ref}.", assume_yes=args.yes):
            print("Cancelled.")
            return
        ok = provider.hard_delete_memory(kind, row_id, reason=args.reason or "manual delete", actor="user")
        print("Deleted." if ok else "No matching memory.")
    finally:
        provider.shutdown()


def cmd_commitments(args) -> None:
    provider = _provider()
    try:
        items = provider.list_memory(
            kind="commitment",
            limit=args.limit,
            include_superseded=args.all,
            status=args.status or "",
        )
        _print_items(items)
    finally:
        provider.shutdown()


def cmd_briefing(args) -> None:
    provider = _provider()
    try:
        print(provider.system_prompt_block())
    finally:
        provider.shutdown()


def cmd_review(args) -> None:
    provider = _provider()
    try:
        rows = provider.review_staging(limit=args.limit)
        if not rows:
            print("No pending staging rows.")
            return
        for row in rows:
            print(
                f"staging:{row['id']} {row['extraction_kind']} "
                f"session={row['session_id']} turn={row['turn_id']} "
                f"confidence={row['confidence']}"
            )
            print(f"  {row['payload']}")
    finally:
        provider.shutdown()


def cmd_consolidate(args) -> None:
    provider = _provider()
    try:
        count = provider.consolidate_pending(session_id=args.session or "")
        print(f"Consolidated {count} staging row(s).")
    finally:
        provider.shutdown()


def cmd_export(args) -> None:
    provider = _provider()
    try:
        payload = provider.export_memory()
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.out:
            Path(args.out).write_text(text + "\n", encoding="utf-8")
            print(f"Exported to {args.out}")
        else:
            print(text)
    finally:
        provider.shutdown()


def cmd_import(args) -> None:
    if not _confirm("This will import durable memory rows.", assume_yes=args.yes):
        print("Cancelled.")
        return
    provider = _provider()
    try:
        payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
        counts = provider.import_memory(payload)
        print(json.dumps(counts, indent=2, sort_keys=True))
    finally:
        provider.shutdown()


def cmd_rebuild_vectors(args) -> None:
    provider = _provider()
    try:
        include_sessions = not getattr(args, "memory_only", False)
        include_memory = not getattr(args, "sessions_only", False)
        result = provider.rebuild_embeddings(
            include_sessions=include_sessions,
            include_memory=include_memory,
            limit=getattr(args, "limit", 0) or 0,
        )
        if getattr(args, "json", False):
            print(json.dumps(result, indent=2, sort_keys=True))
            return
        if not result.get("configured"):
            print(result.get("error") or "memory_embedding endpoint is not configured")
            return
        print(f"Model:    {result.get('model', '')}")
        print(f"Sessions: {result.get('session_records', 0)}")
        print(f"Memory:   {result.get('memory_records', 0)}")
        errors = result.get("errors") or []
        if errors:
            print("Errors:")
            for error in errors:
                print(f"  {error}")
    finally:
        provider.shutdown()


def cos_memory_command(args) -> None:
    sub = getattr(args, "cos_memory_command", None) or getattr(args, "memory_command", None)
    handlers = {
        "enable": cmd_enable,
        "disable": cmd_disable,
        "stats": cmd_stats,
        "list": cmd_list,
        "show": cmd_show,
        "search": cmd_search,
        "remember": cmd_remember,
        "pin": cmd_pin,
        "unpin": cmd_pin,
        "forget": cmd_forget,
        "delete": cmd_delete,
        "commitments": cmd_commitments,
        "briefing": cmd_briefing,
        "review": cmd_review,
        "consolidate": cmd_consolidate,
        "export": cmd_export,
        "import": cmd_import,
        "rebuild-vectors": cmd_rebuild_vectors,
    }
    handler = handlers.get(sub)
    if handler is None:
        cmd_stats(args)
        return
    handler(args)


def _add_commands(subs, *, set_func: bool) -> None:
    enable = subs.add_parser("enable", help="Enable cos-memory and cos-context")
    enable.add_argument(
        "--no-context",
        action="store_false",
        dest="context",
        default=True,
        help="Enable only the memory provider, leaving context.engine unchanged",
    )

    subs.add_parser("disable", help="Disable cos-memory and return to built-in memory")

    stats = subs.add_parser("stats", help="Show cos-memory database and queue stats")
    stats.add_argument("--json", action="store_true", help="Print raw JSON")

    list_parser = subs.add_parser("list", help="List durable memories")
    list_parser.add_argument("--kind", choices=["entity", "fact", "preference", "commitment", "relation"])
    list_parser.add_argument("--limit", type=int, default=50)
    list_parser.add_argument("--all", action="store_true", help="Include superseded rows")
    list_parser.add_argument("--status", choices=["open", "done", "cancelled"], help="Filter commitments")

    show = subs.add_parser("show", help="Show one durable memory")
    show.add_argument("memory_ref", help="Typed ref like fact:123")

    search = subs.add_parser("search", help="Search durable memories")
    search.add_argument("query")
    search.add_argument("--kind", action="append", choices=["entity", "fact", "preference", "commitment", "relation"])
    search.add_argument("--limit", type=int, default=20)

    remember = subs.add_parser("remember", help="Manually add a durable memory from JSON")
    remember.add_argument("kind", choices=["entity", "fact", "preference", "commitment"])
    remember.add_argument("content", help='JSON content, e.g. {"subject":"user","predicate":"timezone","object":"AEST"}')
    remember.add_argument("--confidence", type=float, default=0.9)

    pin = subs.add_parser("pin", help="Pin a durable memory into the briefing")
    pin.add_argument("memory_ref")
    pin.set_defaults(unpin=False)

    unpin = subs.add_parser("unpin", help="Remove a durable memory pin")
    unpin.add_argument("memory_ref")
    unpin.set_defaults(unpin=True)

    forget = subs.add_parser("forget", help="Mark a durable memory superseded")
    forget.add_argument("memory_ref")
    forget.add_argument("--reason", default="manual correction")

    delete = subs.add_parser("delete", help="Permanently delete a durable memory")
    delete.add_argument("memory_ref")
    delete.add_argument("--reason", default="manual delete")
    delete.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    commitments = subs.add_parser("commitments", help="List commitments")
    commitments.add_argument("--status", choices=["open", "done", "cancelled"], default="open")
    commitments.add_argument("--limit", type=int, default=50)
    commitments.add_argument("--all", action="store_true", help="Include superseded rows")

    subs.add_parser("briefing", help="Print the chief-of-staff briefing block")

    review = subs.add_parser("review", help="List unconsolidated staging rows")
    review.add_argument("--limit", type=int, default=20)

    consolidate = subs.add_parser("consolidate", help="Consolidate staging rows")
    consolidate.add_argument("--session", default="", help="Limit to one session id")

    export = subs.add_parser("export", help="Export durable memory JSON")
    export.add_argument("--out", help="Write JSON to this path instead of stdout")

    import_parser = subs.add_parser("import", help="Import durable memory JSON")
    import_parser.add_argument("path")
    import_parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    rebuild = subs.add_parser("rebuild-vectors", help="Rebuild SQLite-backed memory embeddings")
    mode = rebuild.add_mutually_exclusive_group()
    mode.add_argument("--sessions-only", action="store_true", help="Only rebuild session-turn vectors")
    mode.add_argument("--memory-only", action="store_true", help="Only rebuild durable-memory vectors")
    rebuild.add_argument("--limit", type=int, default=0, help="Maximum rows to rebuild, 0 means all")
    rebuild.add_argument("--json", action="store_true", help="Print raw JSON")

    if set_func:
        for action in subs.choices.values():
            action.set_defaults(func=cos_memory_command)


def register_memory_subcommands(memory_subparsers) -> None:
    """Register provider commands under ``hermes memory``."""
    _add_commands(memory_subparsers, set_func=True)


def register_cli(subparser) -> None:
    """Register top-level ``hermes cos-memory`` commands."""
    subs = subparser.add_subparsers(dest="cos_memory_command")
    _add_commands(subs, set_func=False)
    subparser.set_defaults(func=cos_memory_command)
