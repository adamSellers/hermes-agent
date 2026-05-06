"""Local telemetry for tool_router discovery and execution.

The router telemetry database is intentionally compact: it records metadata
about searches, descriptions, executions, misses, and result sizes without
copying full tool outputs into another store.  This gives Hermes a local
dataset for tool aliases, skill candidates, and new-tool opportunities.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

DB_FILENAME = "router_telemetry.db"
UNSATISFIED_OUTCOMES = frozenset(
    {
        "search_no_matches",
        "describe_unknown",
        "unknown_target",
        "missing_target",
        "malformed_arguments",
        "skill_view_unavailable",
    }
)


def telemetry_db_path(db_path: Path | None = None) -> Path:
    return db_path or (get_hermes_home() / DB_FILENAME)


def _clip(value: Any, limit: int = 1000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _json_dumps(value: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = json.dumps(str(value), ensure_ascii=False)
    return _clip(text, limit)


def _connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = telemetry_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=0.25)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def ensure_schema(con: sqlite3.Connection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS tool_router_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            session_id TEXT,
            tool_call_id TEXT,
            action TEXT NOT NULL,
            query TEXT,
            requested_name TEXT,
            resolved_name TEXT,
            target_type TEXT,
            target_tool TEXT,
            success INTEGER,
            outcome TEXT,
            error TEXT,
            match_count INTEGER,
            matches_json TEXT,
            result_chars INTEGER,
            result_compacted INTEGER,
            catalog_hash TEXT,
            hidden_tool_count INTEGER,
            hidden_skill_count INTEGER,
            model TEXT,
            provider TEXT,
            platform TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_router_events_created "
        "ON tool_router_events(created_at)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_router_events_outcome "
        "ON tool_router_events(outcome)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_tool_router_events_target "
        "ON tool_router_events(target_type, resolved_name)"
    )


def record_router_event(event: Mapping[str, Any], *, db_path: Path | None = None) -> None:
    """Append one compact router event.

    This function is best-effort and fail-open.  Router execution should never
    block or fail because telemetry storage is temporarily locked/unavailable.
    """
    try:
        with _connect(db_path) as con:
            ensure_schema(con)
            con.execute(
                """
                INSERT INTO tool_router_events (
                    created_at, session_id, tool_call_id, action, query,
                    requested_name, resolved_name, target_type, target_tool,
                    success, outcome, error, match_count, matches_json,
                    result_chars, result_compacted, catalog_hash,
                    hidden_tool_count, hidden_skill_count, model, provider, platform
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    _clip(event.get("session_id"), 200),
                    _clip(event.get("tool_call_id"), 200),
                    _clip(event.get("action"), 50),
                    _clip(event.get("query"), 500),
                    _clip(event.get("requested_name"), 300),
                    _clip(event.get("resolved_name"), 300),
                    _clip(event.get("target_type"), 50),
                    _clip(event.get("target_tool"), 200),
                    None if event.get("success") is None else int(bool(event.get("success"))),
                    _clip(event.get("outcome"), 80),
                    _clip(event.get("error"), 1000),
                    event.get("match_count"),
                    _json_dumps(event.get("matches") or []),
                    event.get("result_chars"),
                    int(bool(event.get("result_compacted"))) if event.get("result_compacted") is not None else None,
                    _clip(event.get("catalog_hash"), 80),
                    event.get("hidden_tool_count"),
                    event.get("hidden_skill_count"),
                    _clip(event.get("model"), 200),
                    _clip(event.get("provider"), 100),
                    _clip(event.get("platform"), 100),
                ),
            )
    except Exception as exc:
        logger.debug("tool_router telemetry write failed: %s", exc)


def summarize_router_events(*, limit: int = 10, db_path: Path | None = None) -> Dict[str, Any]:
    """Return a compact local report for tool/skill routing improvement.

    The report is intentionally metadata-only.  It helps spot missing aliases,
    unknown requested tools, and skill-build candidates without duplicating full
    tool results into the telemetry store.
    """
    limit = max(1, min(int(limit or 10), 100))
    path = telemetry_db_path(db_path)
    if not path.exists():
        return {
            "success": True,
            "db_path": str(path),
            "event_count": 0,
            "unsatisfied": [],
            "top_targets": [],
            "recent": [],
        }

    try:
        with sqlite3.connect(path, timeout=0.25) as con:
            con.row_factory = sqlite3.Row
            ensure_schema(con)
            event_count = con.execute(
                "SELECT COUNT(*) AS count FROM tool_router_events"
            ).fetchone()["count"]
            unsatisfied_marks = ",".join("?" for _ in UNSATISFIED_OUTCOMES)
            unsatisfied_rows = con.execute(
                f"""
                SELECT
                    COALESCE(NULLIF(query, ''), NULLIF(requested_name, ''), '<unknown>') AS request,
                    outcome,
                    COUNT(*) AS count,
                    MAX(created_at) AS last_seen,
                    MAX(error) AS error
                FROM tool_router_events
                WHERE outcome IN ({unsatisfied_marks})
                GROUP BY request, outcome
                ORDER BY count DESC, last_seen DESC
                LIMIT ?
                """,
                (*sorted(UNSATISFIED_OUTCOMES), limit),
            ).fetchall()
            target_rows = con.execute(
                """
                SELECT
                    COALESCE(NULLIF(target_type, ''), 'tool') AS target_type,
                    COALESCE(NULLIF(resolved_name, ''), NULLIF(target_tool, ''), NULLIF(requested_name, ''), '<unknown>') AS target,
                    outcome,
                    COUNT(*) AS count,
                    MAX(created_at) AS last_seen
                FROM tool_router_events
                WHERE action = 'execute'
                GROUP BY target_type, target, outcome
                ORDER BY count DESC, last_seen DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            recent_rows = con.execute(
                """
                SELECT created_at, action, query, requested_name, resolved_name,
                       target_type, target_tool, success, outcome, error
                FROM tool_router_events
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return {
            "success": True,
            "db_path": str(path),
            "event_count": event_count,
            "unsatisfied": [dict(row) for row in unsatisfied_rows],
            "top_targets": [dict(row) for row in target_rows],
            "recent": [dict(row) for row in recent_rows],
        }
    except Exception as exc:
        logger.debug("tool_router telemetry summary failed: %s", exc)
        return {
            "success": False,
            "db_path": str(path),
            "error": str(exc),
        }
