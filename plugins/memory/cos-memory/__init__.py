"""Chief-of-staff local memory provider.

This provider is intentionally local-first and failure-tolerant. It owns:

- ``cos-memory.db`` under the active ``HERMES_HOME``
- a sidecar turn ledger
- a durable background work queue
- FTS-only session recall for v1 deployability
- structured durable memory tables and curation helpers

Optional semantic embeddings and richer auxiliary-LLM extraction can be added
behind the same queue without changing the public tools.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.memory_provider import MemoryProvider
from hermes_state import SessionDB
from tools.registry import tool_error

logger = logging.getLogger(__name__)

MEMORY_VECTOR_MIN_SCORE = 0.42
MEMORY_VECTOR_WITH_LEXICAL_MIN_SCORE = 0.50
RELATIONISH_ENTITY_NAME_TOKENS = {
    "and",
    "email",
    "address",
    "phone",
    "mobile",
    "friend",
    "partner",
    "spouse",
    "wife",
    "husband",
    "colleague",
    "manager",
    "mother",
    "father",
    "sister",
    "brother",
    "daughter",
    "son",
    "her",
    "his",
    "their",
    "my",
    "user",
}


RECALL_SESSION_SCHEMA = {
    "name": "recall_session",
    "description": (
        "Search this conversation's full session history. Use when relevant "
        "context may exist earlier but has been compressed or removed from "
        "live context."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language or keywords."},
            "max_results": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 15,
            },
        },
        "required": ["query"],
    },
}

RECALL_MEMORY_SCHEMA = {
    "name": "recall_memory",
    "description": (
        "Search durable user knowledge: people, projects, preferences, "
        "commitments, facts, and relations from past sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional durable-memory tags to filter by, such as "
                    "shopping_list or shopping_list:groceries."
                ),
            },
            "kinds": {
                "type": "array",
                "items": {
                    "enum": ["entity", "fact", "preference", "commitment", "relation"]
                },
            },
            "max_results": {
                "type": "integer",
                "default": 8,
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["query"],
    },
}

REMEMBER_SCHEMA = {
    "name": "remember",
    "description": (
        "Save a durable fact, preference, commitment, or entity about the "
        "user. Use when the user explicitly shares information that should "
        "persist across sessions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"enum": ["fact", "preference", "commitment", "entity"]},
            "content": {
                "type": "object",
                "description": (
                    "For fact: {subject, predicate, object}. For preference: "
                    "{domain, statement, strength}. For commitment: "
                    "{description, due_at?, owner?}. For entity: "
                    "{type, canonical_name, aliases?, attributes?}."
                ),
            },
            "confidence": {"type": "number", "default": 0.8},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Optional durable-memory tags for skill-owned state and "
                    "curation, such as shopping_list:groceries."
                ),
            },
        },
        "required": ["kind", "content"],
    },
}

FORGET_SCHEMA = {
    "name": "forget",
    "description": (
        "Mark a remembered item as superseded or cancelled. Use when the user "
        "corrects a memory or says it is no longer true."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "integer"},
            "memory_ref": {"type": "string", "description": "Typed ref like fact:123."},
            "kind": {"enum": ["fact", "preference", "commitment", "entity", "relation"]},
            "reason": {"type": "string"},
        },
        "required": ["reason"],
    },
}


MEMORY_EXTRACTION_PROMPT = """You extract durable memory for a local single-user assistant.

Return ONLY valid JSON, with this exact top-level shape:
{"memories":[{"kind":"entity|fact|preference|commitment","confidence":0.0,"content":{}}]}

Allowed content shapes:
- entity: {"type":"person|project|organization|place|thing","canonical_name":"...","aliases":[],"attributes":{}}
- fact: {"subject":"user or entity name","subject_type":"person|project|organization|place|thing","predicate":"snake_case_relation","object":"..."}
- preference: {"domain":"general|coding|communication|...","statement":"...","strength":"soft|strong|hard_rule"}
- commitment: {"description":"...","owner":"user|agent","due_at":null}

Rules:
- Extract only stable, durable information useful in later sessions.
- Ignore transient requests, chit-chat, and assistant claims.
- If the user asks to remember/save/note something, treat that as explicit consent.
- Do not store passwords, API keys, secrets, bank details, government IDs, or medical details unless the user explicitly asks to remember that exact item.
- Contact details such as email addresses may be stored only when explicitly provided for memory.
- Prefer clean entity+fact pairs over awkward synthesized subjects.
- Use snake_case predicates such as email_address, timezone, works_on, lives_in.
- If there is nothing worth storing, return {"memories":[]}.

Example input:
{"user_message":"remember Alena is my friend and her email address is alena@example.com","assistant_response":"Noted."}

Example output:
{"memories":[
  {"kind":"entity","confidence":0.85,"content":{"type":"person","canonical_name":"Alena","aliases":[],"attributes":{"relationship_to_user":"friend"}}},
  {"kind":"fact","confidence":0.85,"content":{"subject":"Alena","subject_type":"person","predicate":"email_address","object":"alena@example.com"}}
]}
"""


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session_turns (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    user_message_ids TEXT,
    assistant_message_ids TEXT,
    tool_message_ids TEXT,
    started_at REAL NOT NULL,
    completed_at REAL,
    compression_state TEXT NOT NULL DEFAULT 'hot',
    digest_cached INTEGER DEFAULT 0,
    session_embedding_status TEXT NOT NULL DEFAULT 'pending',
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    UNIQUE(session_id, turn_id)
);
CREATE INDEX IF NOT EXISTS idx_session_turns_session
    ON session_turns(session_id, turn_id);
CREATE INDEX IF NOT EXISTS idx_session_turns_state
    ON session_turns(session_id, compression_state);

CREATE TABLE IF NOT EXISTS memory_work_queue (
    id INTEGER PRIMARY KEY,
    job_type TEXT NOT NULL,
    session_id TEXT,
    turn_id INTEGER,
    target_table TEXT,
    target_row_id INTEGER,
    payload TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    run_after REAL NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_work_queue_ready
    ON memory_work_queue(status, run_after, created_at);

CREATE TABLE IF NOT EXISTS session_embedding_records (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding_model TEXT,
    embedding_vector TEXT,
    embedded_at REAL,
    embedding_error TEXT,
    created_at REAL NOT NULL,
    UNIQUE(session_id, turn_id, role)
);
CREATE INDEX IF NOT EXISTS idx_session_embedding_records_session
    ON session_embedding_records(session_id, turn_id);

CREATE TABLE IF NOT EXISTS memory_embedding_records (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding_model TEXT,
    embedding_vector TEXT,
    embedded_at REAL,
    embedding_error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(kind, row_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_embedding_records_kind
    ON memory_embedding_records(kind, row_id);

CREATE TABLE IF NOT EXISTS memory_tags (
    id INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    row_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    created_at REAL NOT NULL,
    UNIQUE(kind, row_id, tag)
);
CREATE INDEX IF NOT EXISTS idx_memory_tags_tag
    ON memory_tags(tag, kind, row_id);
CREATE INDEX IF NOT EXISTS idx_memory_tags_ref
    ON memory_tags(kind, row_id);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY,
    type TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    aliases TEXT,
    attributes TEXT,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    salience REAL DEFAULT 0.0,
    superseded_at REAL,
    pinned INTEGER DEFAULT 0,
    user_edited INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
CREATE INDEX IF NOT EXISTS idx_entities_salience ON entities(salience DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_entities_canonical
    ON entities(type, canonical_name);

CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    subject_entity_id INTEGER REFERENCES entities(id),
    predicate TEXT NOT NULL,
    object_value TEXT,
    object_entity_id INTEGER REFERENCES entities(id),
    confidence REAL NOT NULL DEFAULT 0.6,
    source_session_id TEXT,
    source_turn_id INTEGER,
    created_at REAL NOT NULL,
    last_confirmed_at REAL NOT NULL,
    superseded_by_id INTEGER REFERENCES facts(id),
    superseded_at REAL,
    pinned INTEGER DEFAULT 0,
    user_edited INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_facts_subject
    ON facts(subject_entity_id) WHERE superseded_by_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_facts_predicate ON facts(predicate);
CREATE INDEX IF NOT EXISTS idx_facts_active
    ON facts(subject_entity_id, predicate) WHERE superseded_by_id IS NULL;

CREATE TABLE IF NOT EXISTS preferences (
    id INTEGER PRIMARY KEY,
    domain TEXT NOT NULL,
    statement TEXT NOT NULL,
    strength TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.6,
    source_session_id TEXT,
    source_turn_id INTEGER,
    created_at REAL NOT NULL,
    last_confirmed_at REAL NOT NULL,
    superseded_by_id INTEGER REFERENCES preferences(id),
    superseded_at REAL,
    pinned INTEGER DEFAULT 0,
    user_edited INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_preferences_domain ON preferences(domain);
CREATE INDEX IF NOT EXISTS idx_preferences_active
    ON preferences(domain) WHERE superseded_by_id IS NULL;

CREATE TABLE IF NOT EXISTS commitments (
    id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    owner TEXT NOT NULL DEFAULT 'user',
    related_entity_id INTEGER REFERENCES entities(id),
    due_at REAL,
    status TEXT NOT NULL DEFAULT 'open',
    source_session_id TEXT,
    source_turn_id INTEGER,
    created_at REAL NOT NULL,
    completed_at REAL,
    last_confirmed_at REAL NOT NULL,
    superseded_by_id INTEGER REFERENCES commitments(id),
    superseded_at REAL,
    pinned INTEGER DEFAULT 0,
    user_edited INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_commitments_status
    ON commitments(status, due_at);
CREATE INDEX IF NOT EXISTS idx_commitments_owner
    ON commitments(owner, status);

CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY,
    from_entity_id INTEGER NOT NULL REFERENCES entities(id),
    to_entity_id INTEGER NOT NULL REFERENCES entities(id),
    relation TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.6,
    source_session_id TEXT,
    source_turn_id INTEGER,
    created_at REAL NOT NULL,
    last_confirmed_at REAL NOT NULL,
    superseded_by_id INTEGER REFERENCES relations(id),
    superseded_at REAL,
    pinned INTEGER DEFAULT 0,
    user_edited INTEGER DEFAULT 0,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_entity_id);
CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_entity_id);

CREATE TABLE IF NOT EXISTS staging_extractions (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    turn_id INTEGER,
    extraction_kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at REAL NOT NULL,
    consolidated_at REAL,
    consolidation_outcome TEXT
);
CREATE INDEX IF NOT EXISTS idx_staging_unconsolidated
    ON staging_extractions(session_id) WHERE consolidated_at IS NULL;

CREATE TABLE IF NOT EXISTS digest_cache (
    session_id TEXT NOT NULL,
    turn_id INTEGER NOT NULL,
    digest_json TEXT NOT NULL,
    digest_text TEXT NOT NULL,
    aux_model TEXT,
    tokens_original INTEGER,
    tokens_digest INTEGER,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS memory_audit (
    id INTEGER PRIMARY KEY,
    timestamp REAL NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    table_name TEXT NOT NULL,
    row_id INTEGER,
    diff TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON memory_audit(timestamp DESC);
"""


def _json_array(values: Iterable[int]) -> str:
    return json.dumps([int(v) for v in values], separators=(",", ":"))


def _json_obj(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, separators=(",", ":"))


def _json_vector(values: Iterable[float]) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _now() -> float:
    return time.time()


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def parse_memory_ref(memory_ref: str) -> Tuple[Optional[str], Optional[int]]:
    if not memory_ref:
        return None, None
    text = str(memory_ref).strip()
    if ":" not in text:
        return None, None
    kind, raw_id = text.split(":", 1)
    kind = kind.strip().lower()
    try:
        row_id = int(raw_id.strip())
    except ValueError:
        return None, None
    if kind not in {"entity", "fact", "preference", "commitment", "relation"}:
        return None, None
    return kind, row_id


def _normalize_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    values = [raw] if isinstance(raw, str) else raw
    if not isinstance(values, list):
        return []
    tags: List[str] = []
    seen = set()
    for value in values:
        tag = str(value or "").strip().lower()
        if not tag:
            continue
        tag = re.sub(r"\s+", "_", tag)
        tag = re.sub(r"[^a-z0-9:_-]+", "_", tag).strip("_")
        tag = re.sub(r"_+", "_", tag)
        tag = tag[:120]
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


class CosMemoryProvider(MemoryProvider):
    """Local chief-of-staff memory provider."""

    def __init__(self) -> None:
        self._session_id = ""
        self._hermes_home: Optional[Path] = None
        self._db_path: Optional[Path] = None
        self._state_db_path: Optional[Path] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._session_db: Optional[SessionDB] = None
        self._lock = threading.RLock()
        self._initialized = False
        self._shutdown = threading.Event()
        self._workers: List[threading.Thread] = []
        self._worker_count = 1
        self._max_job_attempts = 3
        self._job_retry_base_seconds = 30
        self._stale_job_seconds = 300
        self._last_stale_jobs_recovered = 0
        self._briefing_cache = ""
        self._briefing_cache_at = 0.0

    @property
    def name(self) -> str:
        return "cos-memory"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return []

    def post_setup(self, hermes_home: str, config: dict) -> None:
        """Activate the memory provider and matching context engine."""
        from hermes_cli.config import save_config

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        if not isinstance(config.get("context"), dict):
            config["context"] = {}
        config["memory"]["provider"] = self.name
        config["context"]["engine"] = "cos-context"
        save_config(config)
        print("\n  Memory provider: cos-memory")
        print("  Context engine:  cos-context")
        print("  Activation saved to config.yaml")
        print("  Start a new Hermes session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = Path(kwargs.get("hermes_home") or Path.home() / ".hermes")
        hermes_home.mkdir(parents=True, exist_ok=True)
        self._hermes_home = hermes_home
        self._db_path = hermes_home / "cos-memory.db"
        self._state_db_path = hermes_home / "state.db"
        self._session_id = session_id or ""

        self._conn = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=1.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._run_migrations()
        self._session_db = SessionDB(self._state_db_path)
        self._initialized = True
        self._shutdown.clear()
        if kwargs.get("start_workers", True):
            self._start_workers()

    def _run_migrations(self) -> None:
        assert self._conn is not None
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._ensure_column("entities", "superseded_at", "REAL")
            for column, declaration in (
                ("embedding_model", "TEXT"),
                ("embedding_vector", "TEXT"),
                ("embedded_at", "REAL"),
                ("embedding_error", "TEXT"),
            ):
                self._ensure_column("session_embedding_records", column, declaration)
            row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                self._conn.execute("INSERT INTO schema_version(version) VALUES (1)")

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        assert self._conn is not None
        existing = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        }
        if column not in existing:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _start_workers(self) -> None:
        if self._workers:
            return
        self._recover_stale_jobs()
        for idx in range(max(0, self._worker_count)):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"cos-memory-worker-{idx + 1}",
                daemon=True,
            )
            thread.start()
            self._workers.append(thread)

    def _recover_stale_jobs(self, stale_after_seconds: Optional[int] = None) -> int:
        if self._conn is None:
            return 0
        stale_after = self._stale_job_seconds if stale_after_seconds is None else stale_after_seconds
        cutoff = _now() - max(1, int(stale_after or self._stale_job_seconds))
        now = _now()
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE memory_work_queue
                SET status = 'pending',
                    last_error = COALESCE(NULLIF(TRIM(last_error), ''), 'recovered stale running job'),
                    updated_at = ?
                WHERE status = 'running' AND updated_at < ?
                """,
                (now, cutoff),
            )
            recovered = int(cur.rowcount or 0)
            if recovered:
                self._last_stale_jobs_recovered = recovered
            return recovered

    def _worker_loop(self) -> None:
        while not self._shutdown.is_set():
            job = self._claim_job()
            if not job:
                self._shutdown.wait(0.2)
                continue
            try:
                self._process_job(job)
                self._mark_job_done(job["id"])
            except Exception as exc:
                logger.debug("cos-memory job %s failed: %s", job.get("id"), exc)
                self._mark_job_failed(job, str(exc))

    def _claim_job(self) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            return None
        now = _now()
        with self._lock:
            committed = False
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    """
                    SELECT id FROM memory_work_queue
                    WHERE status = 'pending' AND run_after <= ?
                    ORDER BY created_at, id
                    LIMIT 1
                    """,
                    (now,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    committed = True
                    return None
                cur = self._conn.execute(
                    """
                    UPDATE memory_work_queue
                    SET status = 'running', attempts = attempts + 1, updated_at = ?
                    WHERE id = ? AND status = 'pending'
                    """,
                    (now, row["id"]),
                )
                if int(cur.rowcount or 0) != 1:
                    self._conn.execute("ROLLBACK")
                    committed = True
                    return None
                claimed = self._conn.execute(
                    "SELECT * FROM memory_work_queue WHERE id = ?",
                    (row["id"],),
                ).fetchone()
                self._conn.execute("COMMIT")
                committed = True
                return dict(claimed) if claimed is not None else None
            except sqlite3.OperationalError as exc:
                if not committed:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                logger.debug("cos-memory queue claim skipped: %s", exc)
                return None
            except sqlite3.Error:
                if not committed:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                raise

    def _mark_job_done(self, job_id: int) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE memory_work_queue SET status = 'done', updated_at = ? WHERE id = ?",
                (_now(), job_id),
            )

    def _mark_job_failed(self, job: Dict[str, Any], error: str) -> None:
        if self._conn is None:
            return
        attempts = int(job.get("attempts") or 0) + 1
        status = "failed" if attempts >= self._max_job_attempts else "pending"
        delay = self._job_retry_base_seconds * max(1, attempts)
        with self._lock:
            self._conn.execute(
                """
                UPDATE memory_work_queue
                SET status = ?, run_after = ?, last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, _now() + delay, error[:500], _now(), job["id"]),
            )

    def _queue_counts_locked(self) -> Dict[str, int]:
        assert self._conn is not None
        counts = {"pending": 0, "running": 0, "failed": 0, "done": 0}
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM memory_work_queue GROUP BY status"
        ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["n"])
        return counts

    def _queue_errors_locked(self, limit: int = 5) -> List[Dict[str, Any]]:
        assert self._conn is not None
        rows = self._conn.execute(
            """
            SELECT id, job_type, status, session_id, turn_id, target_table,
                   target_row_id, attempts, run_after, last_error, created_at,
                   updated_at
            FROM memory_work_queue
            WHERE status IN ('pending', 'running', 'failed')
              AND last_error IS NOT NULL
              AND TRIM(last_error) != ''
            ORDER BY
                CASE status
                    WHEN 'failed' THEN 0
                    WHEN 'running' THEN 1
                    WHEN 'pending' THEN 2
                    ELSE 3
                END,
                updated_at DESC,
                id DESC
            LIMIT ?
            """,
            (_clamp_int(limit, 5, 1, 50),),
        ).fetchall()
        return [dict(row) for row in rows]

    def queue_status(
        self,
        *,
        status: str = "",
        limit: int = 50,
        include_done: bool = False,
    ) -> Dict[str, Any]:
        """Return visible queue jobs and reliability counters for curation CLI."""
        if self._conn is None:
            return {}
        status = (status or "").strip().lower()
        if status and status not in {"pending", "running", "failed", "done"}:
            raise ValueError(f"unsupported queue status {status!r}")
        limit = _clamp_int(limit, 50, 1, 200)
        with self._lock:
            counts = self._queue_counts_locked()
            cutoff = _now() - self._stale_job_seconds
            stale_running = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM memory_work_queue
                    WHERE status = 'running' AND updated_at < ?
                    """,
                    (cutoff,),
                ).fetchone()[0]
            )
            params: List[Any] = []
            where = ""
            if status:
                where = "WHERE status = ?"
                params.append(status)
            elif not include_done:
                where = "WHERE status IN ('pending', 'running', 'failed')"
            rows = self._conn.execute(
                f"""
                SELECT id, job_type, status, session_id, turn_id, target_table,
                       target_row_id, attempts, run_after, last_error, created_at,
                       updated_at
                FROM memory_work_queue
                {where}
                ORDER BY
                    CASE status
                        WHEN 'failed' THEN 0
                        WHEN 'running' THEN 1
                        WHEN 'pending' THEN 2
                        WHEN 'done' THEN 3
                        ELSE 4
                    END,
                    updated_at DESC,
                    id DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
            errors = self._queue_errors_locked(limit=5)
        return {
            "counts": counts,
            "errors": errors,
            "jobs": [dict(row) for row in rows],
            "limit": limit,
            "stale_running": stale_running,
            "recovered_stale_running": 0,
            "last_recovered_stale_running": self._last_stale_jobs_recovered,
        }

    def retry_failed_jobs(self, *, limit: int = 0) -> Dict[str, Any]:
        """Requeue failed jobs when the user explicitly requests a retry."""
        if self._conn is None:
            return {}
        limit = max(0, _clamp_int(limit, 0, 0, 1000))
        with self._lock:
            sql = """
                SELECT id FROM memory_work_queue
                WHERE status = 'failed'
                ORDER BY updated_at, id
            """
            params: List[Any] = []
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            ids = [int(row["id"]) for row in self._conn.execute(sql, params).fetchall()]
            if ids:
                now = _now()
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"""
                    UPDATE memory_work_queue
                    SET status = 'pending',
                        attempts = 0,
                        run_after = ?,
                        last_error = NULL,
                        updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    (now, now, *ids),
                )
            counts = self._queue_counts_locked()
            errors = self._queue_errors_locked(limit=5)
        return {"retried": len(ids), "job_ids": ids, "counts": counts, "errors": errors}

    def _process_job(self, job: Dict[str, Any]) -> None:
        job_type = job.get("job_type")
        payload = self._loads(job.get("payload"), {})
        if job_type == "embed_turn":
            self._process_embed_turn(job, payload)
        elif job_type == "extract_turn":
            self._process_extract_turn(job, payload)
        elif job_type == "consolidate":
            self.consolidate_pending(session_id=job.get("session_id") or "")
        elif job_type == "embed_memory_row":
            self._process_embed_memory_row(job, payload)
        else:
            raise ValueError(f"unknown job_type {job_type!r}")

    @staticmethod
    def _loads(text: Any, default: Any) -> Any:
        try:
            return json.loads(text or "")
        except (TypeError, json.JSONDecodeError):
            return default

    def system_prompt_block(self) -> str:
        if not self._initialized or self._conn is None:
            return ""
        now = _now()
        if self._briefing_cache and now - self._briefing_cache_at < 30:
            return self._briefing_cache
        briefing = self._build_briefing()
        self._briefing_cache = briefing
        self._briefing_cache_at = now
        return briefing

    def _build_briefing(self) -> str:
        lines = [
            "# CHIEF OF STAFF BRIEFING",
            "Use recall_session(query) for earlier conversation details. "
            "Use recall_memory(query) for durable cross-session knowledge. "
            "Use remember(...) when the user explicitly shares durable info.",
        ]
        pinned = self._fetch_pinned_lines(limit=20)
        if pinned:
            lines.append("\n## PINNED")
            lines.extend(f"- {line}" for line in pinned)
        projects = self._fetch_entity_lines(types=("project", "person"), limit=12)
        if projects:
            lines.append("\n## KEY ENTITIES")
            lines.extend(f"- {line}" for line in projects)
        commitments = self._fetch_commitment_lines(limit=20)
        if commitments:
            lines.append("\n## OPEN COMMITMENTS")
            lines.extend(f"- {line}" for line in commitments)
        recent = self._fetch_recent_fact_lines(limit=20)
        if recent:
            lines.append("\n## RECENT FACTS")
            lines.extend(f"- {line}" for line in recent)
        text = "\n".join(lines)
        return text[:16000]

    def _fetch_pinned_lines(self, limit: int) -> List[str]:
        assert self._conn is not None
        lines: List[str] = []
        with self._lock:
            facts = self._conn.execute(
                """
                SELECT f.id, e.canonical_name AS subject, f.predicate, f.object_value
                FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                WHERE f.pinned = 1
                  AND f.superseded_by_id IS NULL
                  AND f.superseded_at IS NULL
                ORDER BY f.last_confirmed_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
            prefs = self._conn.execute(
                """
                SELECT id, domain, statement, strength FROM preferences
                WHERE pinned = 1
                  AND superseded_by_id IS NULL
                  AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        for row in facts:
            lines.append(
                f"fact:{row['id']} {row['subject'] or 'user'} {row['predicate']} {row['object_value']}"
            )
        for row in prefs:
            lines.append(
                f"preference:{row['id']} [{row['domain']}/{row['strength']}] {row['statement']}"
            )
        return lines[:limit]

    def _fetch_entity_lines(self, types: Tuple[str, ...], limit: int) -> List[str]:
        assert self._conn is not None
        placeholders = ",".join("?" for _ in types)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, type, canonical_name, notes, salience FROM entities
                WHERE type IN ({placeholders}) AND superseded_at IS NULL
                ORDER BY pinned DESC, salience DESC, last_seen DESC
                LIMIT ?
                """,
                (*types, limit),
            ).fetchall()
        return [
            f"entity:{r['id']} {r['canonical_name']} ({r['type']})"
            + (f" - {r['notes']}" if r["notes"] else "")
            for r in rows
        ]

    def _fetch_commitment_lines(self, limit: int) -> List[str]:
        assert self._conn is not None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, description, owner, due_at FROM commitments
                WHERE status = 'open'
                  AND superseded_by_id IS NULL
                  AND superseded_at IS NULL
                ORDER BY due_at IS NULL, due_at ASC, last_confirmed_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        lines = []
        for row in rows:
            due = f" due={row['due_at']}" if row["due_at"] else ""
            lines.append(f"commitment:{row['id']} [{row['owner']}]{due} {row['description']}")
        return lines

    def _fetch_recent_fact_lines(self, limit: int) -> List[str]:
        assert self._conn is not None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT f.id, e.canonical_name AS subject, f.predicate, f.object_value
                FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                WHERE f.superseded_by_id IS NULL
                  AND f.superseded_at IS NULL
                ORDER BY f.last_confirmed_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            f"fact:{r['id']} {r['subject'] or 'user'} {r['predicate']} {r['object_value']}"
            for r in rows
        ]

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        if not self._initialized or self._conn is None:
            return
        sid = session_id or self._session_id
        if not sid:
            return
        self._session_id = sid
        now = _now()
        user_ids = self._resolve_message_ids(sid, "user", user_content)
        assistant_ids = self._resolve_message_ids(sid, "assistant", assistant_content)
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(turn_id), 0) + 1 AS next_turn "
                "FROM session_turns WHERE session_id = ?",
                (sid,),
            ).fetchone()
            turn_id = int(row["next_turn"] if row else 1)
            self._conn.execute(
                """
                INSERT INTO session_turns (
                    session_id, turn_id, user_message_ids, assistant_message_ids,
                    tool_message_ids, started_at, completed_at,
                    session_embedding_status, extraction_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', 'pending')
                """,
                (
                    sid,
                    turn_id,
                    _json_array(user_ids),
                    _json_array(assistant_ids),
                    _json_array([]),
                    now,
                    now,
                ),
            )
            payload = {
                "user_content": user_content or "",
                "assistant_content": assistant_content or "",
            }
            self._enqueue_job_locked("embed_turn", sid, turn_id, now, payload)
            self._enqueue_job_locked("extract_turn", sid, turn_id, now, payload)

    def _enqueue_job_locked(
        self,
        job_type: str,
        session_id: str,
        turn_id: Optional[int],
        now: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO memory_work_queue (
                job_type, session_id, turn_id, payload, status,
                run_after, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                job_type,
                session_id,
                turn_id,
                _json_obj(payload or {}),
                now,
                now,
                now,
            ),
        )

    def _resolve_message_ids(self, session_id: str, role: str, content: str) -> List[int]:
        if self._session_db is None or not content:
            return []
        try:
            stored = SessionDB._encode_content(content)
            with self._session_db._lock:
                rows = self._session_db._conn.execute(
                    """
                    SELECT id FROM messages
                    WHERE session_id = ? AND role = ? AND content = ?
                    ORDER BY timestamp DESC, id DESC
                    LIMIT 3
                    """,
                    (session_id, role, stored),
                ).fetchall()
        except Exception:
            return []
        return [int(row["id"]) for row in rows]

    def _process_embed_turn(self, job: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if self._conn is None:
            return
        sid = job.get("session_id") or ""
        turn_id = int(job.get("turn_id") or 0)
        now = _now()
        rows = [
            ("user", payload.get("user_content") or ""),
            ("assistant", payload.get("assistant_content") or ""),
        ]
        rows = [(role, text) for role, text in rows if text]
        embeddings = self._embed_texts([text for _role, text in rows], purpose="session_turn")
        vectors: List[Optional[List[float]]] = [None] * len(rows)
        embedding_model = ""
        if embeddings is not None:
            vectors, embedding_model = embeddings
        with self._lock:
            for idx, (role, text) in enumerate(rows):
                vector = vectors[idx] if idx < len(vectors) else None
                self._conn.execute(
                    """
                    INSERT INTO session_embedding_records(
                        session_id, turn_id, role, text, embedding_model,
                        embedding_vector, embedded_at, embedding_error, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
                    ON CONFLICT(session_id, turn_id, role) DO UPDATE SET
                        text = excluded.text,
                        embedding_model = excluded.embedding_model,
                        embedding_vector = excluded.embedding_vector,
                        embedded_at = excluded.embedded_at,
                        embedding_error = NULL,
                        created_at = excluded.created_at
                    """,
                    (
                        sid,
                        turn_id,
                        role,
                        text,
                        embedding_model if vector is not None else None,
                        _json_vector(vector) if vector is not None else None,
                        now if vector is not None else None,
                        now,
                    ),
                )
            self._conn.execute(
                """
                UPDATE session_turns
                SET session_embedding_status = 'done'
                WHERE session_id = ? AND turn_id = ?
                """,
                (sid, turn_id),
            )

    def _process_embed_memory_row(self, job: Dict[str, Any], payload: Dict[str, Any]) -> None:
        if self._conn is None:
            return
        kind = str(payload.get("kind") or job.get("target_table") or "").strip().lower()
        row_id = int(payload.get("row_id") or job.get("target_row_id") or 0)
        if not kind or not row_id:
            return
        text = self._memory_text_for_embedding(kind, row_id)
        if not text:
            with self._lock:
                self._conn.execute(
                    "DELETE FROM memory_embedding_records WHERE kind = ? AND row_id = ?",
                    (kind, row_id),
                )
            return
        embeddings = self._embed_texts([text], purpose=f"memory_{kind}")
        now = _now()
        vector: Optional[List[float]] = None
        embedding_model = ""
        if embeddings is not None:
            vectors, embedding_model = embeddings
            vector = vectors[0] if vectors else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO memory_embedding_records(
                    kind, row_id, text, embedding_model, embedding_vector,
                    embedded_at, embedding_error, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, ?)
                ON CONFLICT(kind, row_id) DO UPDATE SET
                    text = excluded.text,
                    embedding_model = excluded.embedding_model,
                    embedding_vector = excluded.embedding_vector,
                    embedded_at = excluded.embedded_at,
                    embedding_error = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    kind,
                    row_id,
                    text,
                    embedding_model if vector is not None else None,
                    _json_vector(vector) if vector is not None else None,
                    now if vector is not None else None,
                    now,
                    now,
                ),
            )

    def _embedding_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception:
            config = {}
        aux = config.get("auxiliary", {}) if isinstance(config, dict) else {}
        task = aux.get("memory_embedding", {}) if isinstance(aux, dict) else {}
        fallback = aux.get("memory_extraction", {}) if isinstance(aux, dict) else {}
        task = task if isinstance(task, dict) else {}
        fallback = fallback if isinstance(fallback, dict) else {}
        if task.get("enabled") is False:
            return {"configured": False}

        model = (
            str(task.get("model") or "").strip()
            or os.environ.get("HERMES_MEMORY_EMBEDDING_MODEL", "").strip()
            or "nomicai-modernbert-embed-base-bf16"
        )
        base_url = (
            str(task.get("base_url") or "").strip()
            or os.environ.get("HERMES_MEMORY_EMBEDDING_BASE_URL", "").strip()
            or str(fallback.get("base_url") or "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
        )
        api_key = (
            str(task.get("api_key") or "").strip()
            or os.environ.get("HERMES_MEMORY_EMBEDDING_API_KEY", "").strip()
            or str(fallback.get("api_key") or "").strip()
            or os.environ.get("OPENAI_API_KEY", "").strip()
        )
        if not base_url:
            try:
                from agent.auxiliary_client import _resolve_custom_runtime

                runtime = _resolve_custom_runtime()
                custom_base = runtime[0] if runtime else ""
                custom_key = runtime[1] if len(runtime) > 1 else ""
                if custom_base:
                    base_url = str(custom_base).strip()
                if custom_key and not api_key:
                    api_key = str(custom_key).strip()
            except Exception:
                pass
        try:
            timeout = float(task.get("timeout", 30))
        except (TypeError, ValueError):
            timeout = 30.0
        extra_body = task.get("extra_body") if isinstance(task.get("extra_body"), dict) else {}
        return {
            "configured": bool(model and base_url),
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "extra_body": dict(extra_body),
        }

    def _embed_texts(
        self,
        texts: List[str],
        *,
        purpose: str = "memory",
    ) -> Optional[Tuple[List[Optional[List[float]]], str]]:
        clean_texts = [str(text or "").strip() for text in texts]
        if not clean_texts:
            return None
        cfg = self._embedding_config()
        if not cfg.get("configured"):
            return None
        model = str(cfg.get("model") or "").strip()
        base_url = str(cfg.get("base_url") or "").strip().rstrip("/")
        endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
        body = {
            "model": model,
            "input": clean_texts,
        }
        body.update(cfg.get("extra_body") or {})
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = str(cfg.get("api_key") or "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(cfg.get("timeout") or 30)) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"embedding request failed for {purpose}: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"embedding endpoint unavailable for {purpose}: {exc}") from exc
        payload = self._loads(raw, {})
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            raise RuntimeError("embedding response did not include a data list")
        vectors: List[Optional[List[float]]] = []
        by_index: Dict[int, List[float]] = {}
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            raw_vector = item.get("embedding")
            vector = self._normalize_embedding_vector(raw_vector)
            if vector is None:
                continue
            try:
                item_idx = int(item.get("index", idx))
            except (TypeError, ValueError):
                item_idx = idx
            by_index[item_idx] = vector
        for idx in range(len(clean_texts)):
            vectors.append(by_index.get(idx))
        if any(vector is None for vector in vectors):
            raise RuntimeError("embedding response was missing one or more vectors")
        return vectors, model

    def _normalize_embedding_vector(self, raw_vector: Any) -> Optional[List[float]]:
        if not isinstance(raw_vector, list) or not raw_vector:
            return None
        out: List[float] = []
        for value in raw_vector:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(parsed):
                return None
            out.append(parsed)
        return out

    def _decode_embedding_vector(self, raw_vector: Any) -> Optional[List[float]]:
        try:
            data = json.loads(raw_vector or "[]")
        except (TypeError, json.JSONDecodeError):
            return None
        return self._normalize_embedding_vector(data)

    def _cosine_similarity(self, left: List[float], right: List[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm <= 0 or right_norm <= 0:
            return 0.0
        return dot / (left_norm * right_norm)

    def _process_extract_turn(self, job: Dict[str, Any], payload: Dict[str, Any]) -> None:
        sid = job.get("session_id") or ""
        turn_id = int(job.get("turn_id") or 0)
        user_text = payload.get("user_content") or ""
        assistant_text = payload.get("assistant_content") or ""
        candidates = self._llm_extract(user_text, assistant_text)
        if candidates is None:
            candidates = self._heuristic_extract(user_text)
        now = _now()
        if self._conn is None:
            return
        with self._lock:
            for kind, confidence, item in candidates:
                self._conn.execute(
                    """
                    INSERT INTO staging_extractions(
                        session_id, turn_id, extraction_kind, payload,
                        confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (sid, turn_id, kind, _json_obj(item), confidence, now),
                )
            self._conn.execute(
                """
                UPDATE session_turns
                SET extraction_status = 'done'
                WHERE session_id = ? AND turn_id = ?
                """,
                (sid, turn_id),
            )
            if candidates:
                self._enqueue_job_locked("consolidate", sid, None, now, {})

    def _llm_extract(
        self,
        user_text: str,
        assistant_text: str = "",
    ) -> Optional[List[Tuple[str, float, Dict[str, Any]]]]:
        if not user_text or not user_text.strip():
            return []
        try:
            from agent.auxiliary_client import call_llm, extract_content_or_reasoning

            response = call_llm(
                task="memory_extraction",
                messages=[
                    {"role": "system", "content": MEMORY_EXTRACTION_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_message": user_text,
                                "assistant_response": assistant_text or "",
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0,
                max_tokens=1200,
                timeout=45,
            )
            text = extract_content_or_reasoning(response)
            return self._parse_llm_extraction(text, user_text=user_text)
        except Exception as exc:
            logger.debug("cos-memory LLM extraction unavailable; using fallback: %s", exc)
            return None

    def _parse_llm_extraction(
        self,
        text: str,
        *,
        user_text: str,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        data = self._loads(self._extract_json_object(text), None)
        if not isinstance(data, dict):
            raise ValueError("memory extraction returned non-object JSON")
        memories = data.get("memories")
        if memories is None:
            memories = data.get("items", [])
        if not isinstance(memories, list):
            raise ValueError("memory extraction 'memories' must be a list")

        lowered = user_text.lower()
        sensitive = self._contains_sensitive_memory(user_text)
        explicit = bool(re.search(r"\b(remember|save|note|store|keep track of)\b", lowered))
        out: List[Tuple[str, float, Dict[str, Any]]] = []
        for item in memories:
            parsed = self._coerce_extraction_item(item)
            if parsed is None:
                continue
            kind, confidence, content = parsed
            if sensitive and not explicit:
                continue
            if kind in {"entity", "fact", "preference", "commitment"}:
                out.append((kind, confidence, content))
        return out

    def _extract_json_object(self, text: str) -> str:
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
            text = re.sub(r"\s*```$", "", text)
        first = text.find("{")
        last = text.rfind("}")
        if first >= 0 and last >= first:
            return text[first:last + 1]
        return text

    def _coerce_extraction_item(
        self,
        item: Any,
    ) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        if not isinstance(item, dict):
            return None
        kind = str(item.get("kind") or item.get("type") or "").strip().lower()
        if kind not in {"entity", "fact", "preference", "commitment"}:
            return None
        content = item.get("content")
        if not isinstance(content, dict):
            content = {
                key: value
                for key, value in item.items()
                if key not in {"kind", "type", "confidence"}
            }
        try:
            confidence = float(item.get("confidence", 0.6))
        except (TypeError, ValueError):
            confidence = 0.6
        confidence = max(0.0, min(1.0, confidence))
        normalized = self._normalize_extraction_content(kind, content)
        if normalized is None:
            return None
        return kind, confidence, normalized

    def _normalize_extraction_content(
        self,
        kind: str,
        content: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if kind == "entity":
            name = str(content.get("canonical_name") or content.get("name") or "").strip()
            if not name:
                return None
            return {
                "type": str(content.get("type") or "thing").strip().lower(),
                "canonical_name": name,
                "aliases": content.get("aliases") if isinstance(content.get("aliases"), list) else [],
                "attributes": content.get("attributes") if isinstance(content.get("attributes"), dict) else {},
            }
        if kind == "fact":
            subject = str(content.get("subject") or "user").strip()
            predicate = self._normalize_predicate(str(content.get("predicate") or "has_status"))
            obj = str(content.get("object") or content.get("object_value") or "").strip()
            if not subject or not predicate or not obj:
                return None
            normalized = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }
            subject_type = str(content.get("subject_type") or "").strip().lower()
            if subject_type:
                normalized["subject_type"] = subject_type
            return normalized
        if kind == "preference":
            statement = str(content.get("statement") or "").strip()
            if not statement:
                return None
            return {
                "domain": str(content.get("domain") or "general").strip().lower(),
                "statement": statement,
                "strength": str(content.get("strength") or "soft").strip().lower(),
            }
        if kind == "commitment":
            description = str(content.get("description") or "").strip()
            if not description:
                return None
            return {
                "description": description,
                "owner": str(content.get("owner") or "user").strip().lower(),
                "due_at": content.get("due_at"),
            }
        return None

    def _contains_sensitive_memory(self, text: str) -> bool:
        lowered = (text or "").lower()
        sensitive = (
            "password",
            "api key",
            "secret key",
            "bank account",
            "medicare",
            "medical",
            "passport",
            "social security",
        )
        return any(word in lowered for word in sensitive)

    def _heuristic_extract(self, text: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Conservative deterministic extraction fallback.

        LLM extraction is the primary path. This fallback only catches
        explicit, low-risk statements when the auxiliary model is unavailable.
        """
        if not text:
            return []
        lowered = text.lower()
        if self._contains_sensitive_memory(text) and "remember" not in lowered:
            return []

        out: List[Tuple[str, float, Dict[str, Any]]] = []
        pref_patterns = [
            (r"\bi prefer ([^.!\n]+)", "general", "strong"),
            (r"\bi like ([^.!\n]+)", "general", "soft"),
            (r"\bi don't like ([^.!\n]+)", "general", "soft"),
            (r"\bnever ([^.!\n]+)", "general", "hard_rule"),
            (r"\balways ([^.!\n]+)", "general", "hard_rule"),
        ]
        for pattern, domain, strength in pref_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                statement = match.group(0).strip()
                out.append(("preference", 0.65, {
                    "domain": domain,
                    "statement": statement,
                    "strength": strength,
                }))
                break

        fact_patterns = [
            (r"\bmy ([A-Za-z0-9 _'-]{2,60}) is ([^.!\n]{1,160})", "user_attribute"),
            (r"\bi live in ([^.!\n]{2,120})", "lives_in"),
            (r"\bi work on ([^.!\n]{2,120})", "works_on"),
        ]
        for pattern, predicate in fact_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            if predicate == "user_attribute":
                attribute = match.group(1).strip()
                obj = match.group(2).strip()
                normalized_predicate = self._normalize_predicate(attribute)
                if not normalized_predicate:
                    continue
                if self._looks_like_contact_attribute(normalized_predicate):
                    # Avoid storing malformed PII/contact facts from broad
                    # "my X is Y" matches. The LLM extraction path handles
                    # explicit contact memories with concrete values.
                    continue
                subject = "user"
                predicate = normalized_predicate
            else:
                subject = "user"
                obj = match.group(1).strip()
            out.append(("fact", 0.6, {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
            }))
            break

        if re.search(r"\b(remind me|i need to|todo|to do|please book|please schedule)\b", text, re.I):
            out.append(("commitment", 0.55, {
                "description": text.strip()[:240],
                "owner": "user",
            }))
        return out

    def _normalize_predicate(self, text: str) -> str:
        text = re.sub(r"\b(and|the|a|an)\b", " ", text.lower())
        text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
        text = re.sub(r"_+", "_", text)
        return text[:80]

    def _looks_like_contact_attribute(self, predicate: str) -> bool:
        contact_terms = ("email", "e_mail", "phone", "mobile", "address")
        return any(term in predicate for term in contact_terms)

    def _looks_like_relationish_entity_name(self, name: str) -> bool:
        lowered = (name or "").strip().lower()
        if not lowered:
            return False
        tokens = [token for token in re.split(r"[^a-z0-9]+", lowered) if token]
        if len(tokens) < 2:
            return False
        token_set = set(tokens)
        relationish_count = len(token_set & RELATIONISH_ENTITY_NAME_TOKENS)
        has_relation_phrase = (
            "and" in token_set
            or {"email", "address"} <= token_set
            or "phone" in token_set
            or "mobile" in token_set
        )
        has_synthetic_subject = token_set & {"user", "my", "her", "his", "their"}
        return relationish_count >= 2 and (has_relation_phrase or bool(has_synthetic_subject))

    def consolidate_pending(self, session_id: str = "") -> int:
        if self._conn is None:
            return 0
        params: Tuple[Any, ...]
        if session_id:
            where = "WHERE consolidated_at IS NULL AND session_id = ?"
            params = (session_id,)
        else:
            where = "WHERE consolidated_at IS NULL"
            params = ()
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM staging_extractions {where} ORDER BY created_at, id",
                params,
            ).fetchall()
        count = 0
        for row in rows:
            payload = self._loads(row["payload"], {})
            outcome = "created"
            try:
                self._create_memory_from_payload(
                    row["extraction_kind"],
                    payload,
                    confidence=float(row["confidence"] or 0.6),
                    source_session_id=row["session_id"],
                    source_turn_id=row["turn_id"],
                    actor="consolidation",
                    user_edited=False,
                )
            except Exception as exc:
                logger.debug("cos-memory consolidation skipped row %s: %s", row["id"], exc)
                outcome = "rejected"
            with self._lock:
                self._conn.execute(
                    """
                    UPDATE staging_extractions
                    SET consolidated_at = ?, consolidation_outcome = ?
                    WHERE id = ?
                    """,
                    (_now(), outcome, row["id"]),
                )
            count += 1
        if count:
            self._briefing_cache = ""
        return count

    def _create_memory_from_payload(
        self,
        kind: str,
        payload: Dict[str, Any],
        *,
        confidence: float,
        source_session_id: str,
        source_turn_id: Optional[int],
        actor: str,
        user_edited: bool,
        tags: Optional[List[str]] = None,
    ) -> Tuple[str, int]:
        kind = (kind or "").strip().lower()
        record_tags = _normalize_tags(tags if tags is not None else payload.get("tags"))
        if kind == "entity":
            row_id = self._upsert_entity(
                payload.get("type") or "thing",
                payload.get("canonical_name") or payload.get("name") or "",
                aliases=payload.get("aliases") or [],
                attributes=payload.get("attributes") or {},
                actor=actor,
                user_edited=user_edited,
            )
            self._set_memory_tags("entity", row_id, record_tags)
            self._enqueue_memory_embedding("entity", row_id)
            return "entity", row_id
        if kind == "fact":
            row_id = self._create_fact(payload, confidence, source_session_id, source_turn_id, actor, user_edited)
            self._set_memory_tags("fact", row_id, record_tags)
            self._enqueue_memory_embedding("fact", row_id)
            return "fact", row_id
        if kind == "preference":
            row_id = self._create_preference(payload, confidence, source_session_id, source_turn_id, actor, user_edited)
            self._set_memory_tags("preference", row_id, record_tags)
            self._enqueue_memory_embedding("preference", row_id)
            return "preference", row_id
        if kind == "commitment":
            row_id = self._create_commitment(payload, confidence, source_session_id, source_turn_id, actor, user_edited)
            self._set_memory_tags("commitment", row_id, record_tags)
            self._enqueue_memory_embedding("commitment", row_id)
            return "commitment", row_id
        raise ValueError(f"unsupported memory kind {kind!r}")

    def _set_memory_tags(self, kind: str, row_id: int, tags: List[str]) -> None:
        if self._conn is None or not tags:
            return
        now = _now()
        with self._lock:
            self._conn.execute(
                "DELETE FROM memory_tags WHERE kind = ? AND row_id = ?",
                (kind, row_id),
            )
            self._conn.executemany(
                """
                INSERT OR IGNORE INTO memory_tags(kind, row_id, tag, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [(kind, row_id, tag, now) for tag in tags],
            )

    def _memory_tags(self, kind: str, row_id: int) -> List[str]:
        if self._conn is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT tag FROM memory_tags
                WHERE kind = ? AND row_id = ?
                ORDER BY tag
                """,
                (kind, row_id),
            ).fetchall()
        return [str(row["tag"]) for row in rows]

    def _enqueue_memory_embedding(self, kind: str, row_id: int) -> None:
        if self._conn is None:
            return
        now = _now()
        with self._lock:
            self._enqueue_job_locked(
                "embed_memory_row",
                self._session_id,
                None,
                now,
                {"kind": kind, "row_id": int(row_id)},
            )

    def _upsert_entity(
        self,
        entity_type: str,
        canonical_name: str,
        *,
        aliases: List[str],
        attributes: Dict[str, Any],
        actor: str,
        user_edited: bool,
    ) -> int:
        if self._conn is None:
            raise RuntimeError("not initialized")
        name = str(canonical_name or "").strip()
        if not name:
            raise ValueError("entity canonical_name is required")
        entity_type = str(entity_type or "thing").strip().lower()
        now = _now()
        with self._lock:
            existing = self._conn.execute(
                "SELECT id, salience FROM entities WHERE type = ? AND canonical_name = ?",
                (entity_type, name),
            ).fetchone()
            if existing:
                row_id = int(existing["id"])
                self._conn.execute(
                    """
                    UPDATE entities
                    SET last_seen = ?, salience = MIN(1.0, salience + 0.05),
                        aliases = COALESCE(NULLIF(?, '[]'), aliases),
                        attributes = CASE WHEN ? != '{}' THEN ? ELSE attributes END,
                        user_edited = MAX(user_edited, ?)
                    WHERE id = ?
                    """,
                    (
                        now,
                        _json_obj(aliases or []),
                        _json_obj(attributes or {}),
                        _json_obj(attributes or {}),
                        1 if user_edited else 0,
                        row_id,
                    ),
                )
                self._audit(actor, "update", "entities", row_id, {"canonical_name": name})
                return row_id
            cur = self._conn.execute(
                """
                INSERT INTO entities(
                    type, canonical_name, aliases, attributes, first_seen,
                    last_seen, salience, user_edited
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entity_type,
                    name,
                    _json_obj(aliases or []),
                    _json_obj(attributes or {}),
                    now,
                    now,
                    0.1,
                    1 if user_edited else 0,
                ),
            )
            row_id = int(cur.lastrowid)
            self._audit(actor, "create", "entities", row_id, {"canonical_name": name})
            return row_id

    def _create_fact(
        self,
        payload: Dict[str, Any],
        confidence: float,
        source_session_id: str,
        source_turn_id: Optional[int],
        actor: str,
        user_edited: bool,
    ) -> int:
        subject = str(payload.get("subject") or "user").strip()
        predicate = str(payload.get("predicate") or "has_status").strip()
        obj = str(payload.get("object") or payload.get("object_value") or "").strip()
        if not obj:
            raise ValueError("fact object is required")
        subject_type = str(payload.get("subject_type") or "thing").strip().lower()
        subject_id = self._upsert_entity(subject_type, subject, aliases=[], attributes={}, actor=actor, user_edited=False)
        now = _now()
        with self._lock:
            old = self._conn.execute(
                """
                SELECT id, object_value, pinned, user_edited FROM facts
                WHERE subject_entity_id = ? AND predicate = ?
                  AND superseded_by_id IS NULL
                  AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC LIMIT 1
                """,
                (subject_id, predicate),
            ).fetchone()
            if old and old["object_value"] == obj:
                row_id = int(old["id"])
                self._conn.execute(
                    """
                    UPDATE facts
                    SET confidence = MAX(confidence, ?), last_confirmed_at = ?
                    WHERE id = ?
                    """,
                    (confidence, now, row_id),
                )
                self._audit(actor, "update", "facts", row_id, {"confirmed": True})
                return row_id
            cur = self._conn.execute(
                """
                INSERT INTO facts(
                    subject_entity_id, predicate, object_value, confidence,
                    source_session_id, source_turn_id, created_at,
                    last_confirmed_at, user_edited
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    subject_id,
                    predicate,
                    obj,
                    confidence,
                    source_session_id,
                    source_turn_id,
                    now,
                    now,
                    1 if user_edited else 0,
                ),
            )
            row_id = int(cur.lastrowid)
            if old and not old["pinned"] and not old["user_edited"]:
                self._conn.execute(
                    "UPDATE facts SET superseded_by_id = ?, superseded_at = ? WHERE id = ?",
                    (row_id, now, old["id"]),
                )
                self._conn.execute(
                    "DELETE FROM memory_embedding_records WHERE kind = 'fact' AND row_id = ?",
                    (old["id"],),
                )
            self._audit(actor, "create", "facts", row_id, payload)
            return row_id

    def _create_preference(
        self,
        payload: Dict[str, Any],
        confidence: float,
        source_session_id: str,
        source_turn_id: Optional[int],
        actor: str,
        user_edited: bool,
    ) -> int:
        domain = str(payload.get("domain") or "general").strip().lower()
        statement = str(payload.get("statement") or "").strip()
        strength = str(payload.get("strength") or "soft").strip()
        if not statement:
            raise ValueError("preference statement is required")
        now = _now()
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT id FROM preferences
                WHERE domain = ? AND statement = ?
                  AND superseded_by_id IS NULL
                  AND superseded_at IS NULL
                LIMIT 1
                """,
                (domain, statement),
            ).fetchone()
            if existing:
                row_id = int(existing["id"])
                self._conn.execute(
                    """
                    UPDATE preferences
                    SET confidence = MAX(confidence, ?), last_confirmed_at = ?,
                        user_edited = MAX(user_edited, ?)
                    WHERE id = ?
                    """,
                    (confidence, now, 1 if user_edited else 0, row_id),
                )
                self._audit(actor, "update", "preferences", row_id, payload)
                return row_id
            cur = self._conn.execute(
                """
                INSERT INTO preferences(
                    domain, statement, strength, confidence, source_session_id,
                    source_turn_id, created_at, last_confirmed_at, user_edited
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (domain, statement, strength, confidence, source_session_id, source_turn_id, now, now, 1 if user_edited else 0),
            )
            row_id = int(cur.lastrowid)
            self._audit(actor, "create", "preferences", row_id, payload)
            return row_id

    def _create_commitment(
        self,
        payload: Dict[str, Any],
        confidence: float,
        source_session_id: str,
        source_turn_id: Optional[int],
        actor: str,
        user_edited: bool,
    ) -> int:
        description = str(payload.get("description") or "").strip()
        if not description:
            raise ValueError("commitment description is required")
        owner = str(payload.get("owner") or "user").strip()
        due_at = payload.get("due_at")
        now = _now()
        with self._lock:
            existing = self._conn.execute(
                """
                SELECT id FROM commitments
                WHERE description = ? AND COALESCE(due_at, 0) = COALESCE(?, 0)
                  AND superseded_by_id IS NULL
                  AND superseded_at IS NULL
                LIMIT 1
                """,
                (description, due_at),
            ).fetchone()
            if existing:
                row_id = int(existing["id"])
                self._conn.execute(
                    "UPDATE commitments SET last_confirmed_at = ? WHERE id = ?",
                    (now, row_id),
                )
                self._audit(actor, "update", "commitments", row_id, payload)
                return row_id
            cur = self._conn.execute(
                """
                INSERT INTO commitments(
                    description, owner, due_at, status, source_session_id,
                    source_turn_id, created_at, last_confirmed_at, user_edited
                ) VALUES (?, ?, ?, 'open', ?, ?, ?, ?, ?)
                """,
                (description, owner, due_at, source_session_id, source_turn_id, now, now, 1 if user_edited else 0),
            )
            row_id = int(cur.lastrowid)
            self._audit(actor, "create", "commitments", row_id, payload)
            return row_id

    def _audit(self, actor: str, action: str, table_name: str, row_id: int, diff: Any) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            INSERT INTO memory_audit(timestamp, actor, action, table_name, row_id, diff)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (_now(), actor, action, table_name, row_id, _json_obj(diff)),
        )

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        if new_session_id:
            self._session_id = new_session_id

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        return ""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        self.consolidate_pending(session_id=self._session_id)

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [RECALL_SESSION_SCHEMA, RECALL_MEMORY_SCHEMA, REMEMBER_SCHEMA, FORGET_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "recall_session":
            return self._handle_recall_session(args)
        if tool_name == "recall_memory":
            return self._handle_recall_memory(args)
        if tool_name == "remember":
            return self._handle_remember(args)
        if tool_name == "forget":
            return self._handle_forget(args)
        return tool_error(f"Unknown cos-memory tool: {tool_name}")

    def _handle_recall_session(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        max_results = _clamp_int(args.get("max_results"), 5, 1, 15)
        if not query:
            return json.dumps({"results": [], "error": "query is required"})
        if self._session_db is None or not self._session_id:
            return json.dumps({"results": [], "error": "session database is not available"})
        results = self._recall_session_hybrid(query, max_results)
        return json.dumps({"results": results}, ensure_ascii=False)

    def _recall_session_hybrid(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        lexical = self._recall_session_fts(query, max_results)
        vector = self._recall_session_vector(query, max_results)
        if not vector:
            return lexical
        merged: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        order: List[Tuple[Any, ...]] = []
        for item in lexical:
            key = (
                item.get("session_id"),
                item.get("message_id"),
                item.get("turn_id"),
                item.get("role"),
                item.get("snippet"),
            )
            merged[key] = dict(item)
            merged[key]["source"] = "fts"
            order.append(key)
        for item in vector:
            key = (
                item.get("session_id"),
                item.get("message_id"),
                item.get("turn_id"),
                item.get("role"),
                item.get("snippet"),
            )
            if key in merged:
                merged[key]["semantic_score"] = item.get("semantic_score", item.get("score", 0.0))
                merged[key]["source"] = "hybrid"
            else:
                merged[key] = dict(item)
                order.append(key)
        items = [merged[key] for key in order]
        items.sort(
            key=lambda item: (
                item.get("source") != "hybrid",
                item.get("source") != "vector",
                -(float(item.get("semantic_score") or 0.0)),
            )
        )
        return items[:max_results]

    def _recall_session_fts(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        assert self._session_db is not None
        fts_query = SessionDB._sanitize_fts5_query(query)
        if not fts_query:
            return []
        session_ids = self._session_lineage_ids(self._session_id)
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        sql = f"""
            SELECT m.id, m.session_id, m.role, m.timestamp, m.tool_name,
                   snippet(messages_fts, 0, '>>>', '<<<', '...', 40) AS snippet,
                   m.content, rank
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            WHERE messages_fts MATCH ?
              AND m.session_id IN ({placeholders})
            ORDER BY rank
            LIMIT ?
        """
        try:
            with self._session_db._lock:
                rows = self._session_db._conn.execute(
                    sql, [fts_query, *session_ids, max_results]
                ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for row in rows:
            content = SessionDB._decode_content(row["content"])
            snippet = row["snippet"] or (content[:240] if isinstance(content, str) else "")
            out.append({
                "turn_id": self._turn_id_for_message(row["session_id"], row["id"]),
                "message_id": int(row["id"]),
                "session_id": row["session_id"],
                "role": row["role"],
                "timestamp": row["timestamp"],
                "snippet": snippet,
                "score": float(row["rank"] or 0.0),
            })
        return out

    def _session_lineage_ids(self, session_id: str) -> List[str]:
        if self._session_db is None:
            return [session_id] if session_id else []
        try:
            return self._session_db._session_lineage_root_to_tip(session_id)
        except Exception:
            return [session_id] if session_id else []

    def _turn_id_for_message(self, session_id: str, message_id: int) -> Optional[int]:
        if self._conn is None:
            return None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT turn_id, user_message_ids, assistant_message_ids, tool_message_ids
                FROM session_turns WHERE session_id = ? ORDER BY turn_id DESC
                """,
                (session_id,),
            ).fetchall()
        for row in rows:
            for column in ("user_message_ids", "assistant_message_ids", "tool_message_ids"):
                ids = self._loads(row[column], [])
                if int(message_id) in {int(v) for v in ids}:
                    return int(row["turn_id"])
        return None

    def _recall_session_vector(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        session_ids = self._session_lineage_ids(self._session_id)
        if not session_ids:
            return []
        placeholders = ",".join("?" for _ in session_ids)
        with self._lock:
            has_vectors = self._conn.execute(
                f"""
                SELECT 1 FROM session_embedding_records
                WHERE session_id IN ({placeholders}) AND embedding_vector IS NOT NULL
                LIMIT 1
                """,
                tuple(session_ids),
            ).fetchone()
        if not has_vectors:
            return []
        embeddings = self._embed_texts([query], purpose="session_query")
        if embeddings is None:
            return []
        vectors, _model = embeddings
        query_vector = vectors[0] if vectors else None
        if query_vector is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT session_id, turn_id, role, text, embedding_vector, embedded_at
                FROM session_embedding_records
                WHERE session_id IN ({placeholders}) AND embedding_vector IS NOT NULL
                """,
                tuple(session_ids),
            ).fetchall()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            vector = self._decode_embedding_vector(row["embedding_vector"])
            if vector is None:
                continue
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity <= 0:
                continue
            text = row["text"] or ""
            scored.append({
                "turn_id": int(row["turn_id"]),
                "message_id": None,
                "session_id": row["session_id"],
                "role": row["role"],
                "timestamp": row["embedded_at"],
                "snippet": text[:240],
                "score": float(similarity),
                "semantic_score": float(similarity),
                "source": "vector",
            })
        scored.sort(key=lambda item: item["semantic_score"], reverse=True)
        return scored[:max_results]

    def _handle_recall_memory(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query") or "").strip()
        tags = _normalize_tags(args.get("tags"))
        max_results = _clamp_int(args.get("max_results"), 8, 1, 100)
        kinds = args.get("kinds") or []
        if isinstance(kinds, str):
            kinds = [kinds]
        allowed = {str(k).strip().lower() for k in kinds if k}
        if not query and not tags:
            return "No query provided."
        rows = self.search_memory(query, kinds=allowed, tags=tags, limit=max_results)
        if not rows:
            return "No durable memory results."
        lines = []
        for item in rows:
            tag_line = ""
            if item.get("tags"):
                tag_line = f"  tags: {', '.join(item['tags'])}\n"
            lines.append(
                f"- ref: {item['ref']}\n"
                f"  kind: {item['kind']}\n"
                f"  summary: {item['summary']}\n"
                f"{tag_line}"
                f"  score: {item['score']:.2f}"
            )
        return "\n".join(lines)

    def search_memory(
        self,
        query: str,
        *,
        kinds: Optional[set] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        tags = _normalize_tags(tags)
        search_limit = max(limit * 2, limit)
        if tags:
            lexical = self._search_memory_lexical(query, kinds=kinds, tags=tags, limit=search_limit)
        else:
            lexical = self._search_memory_lexical(query, kinds=kinds, limit=search_limit)
        for item in lexical:
            item.setdefault("source", "fts")
            item.setdefault("lexical_score", float(item.get("score") or 0.0))
        if tags:
            vector = self._search_memory_vector(query, kinds=kinds, tags=tags, limit=search_limit)
        else:
            vector = self._search_memory_vector(query, kinds=kinds, limit=search_limit)
        if not vector:
            return lexical[:limit]
        merged: Dict[str, Dict[str, Any]] = {}
        for item in lexical:
            merged[item["ref"]] = dict(item)
            merged[item["ref"]]["lexical_score"] = float(item.get("score") or 0.0)
            merged[item["ref"]]["source"] = "fts"
        for item in vector:
            ref = item["ref"]
            semantic_score = float(item.get("semantic_score") or item.get("score") or 0.0)
            if ref in merged:
                lexical_score = float(merged[ref].get("lexical_score") or merged[ref].get("score") or 0.0)
                merged[ref]["semantic_score"] = semantic_score
                merged[ref]["score"] = min(1.0, max(lexical_score, semantic_score) + 0.12)
                merged[ref]["source"] = "hybrid"
            else:
                min_score = (
                    MEMORY_VECTOR_WITH_LEXICAL_MIN_SCORE
                    if lexical
                    else MEMORY_VECTOR_MIN_SCORE
                )
                if semantic_score < min_score:
                    continue
                merged[ref] = dict(item)
                merged[ref]["source"] = "vector"
        results = list(merged.values())
        results.sort(
            key=lambda item: (
                -float(item.get("score") or 0.0),
                item.get("source") != "hybrid",
                item.get("source") != "vector",
            )
        )
        return results[:limit]

    def _tag_filter_clause(self, kind: str, row_expr: str, tags: List[str]) -> Tuple[str, List[Any]]:
        if not tags:
            return "", []
        placeholders = ",".join("?" for _ in tags)
        return (
            f"""
              AND (
                SELECT COUNT(DISTINCT mt.tag)
                FROM memory_tags mt
                WHERE mt.kind = ?
                  AND mt.row_id = {row_expr}
                  AND mt.tag IN ({placeholders})
              ) = ?
            """,
            [kind, *tags, len(tags)],
        )

    def _search_memory_lexical(
        self,
        query: str,
        *,
        kinds: Optional[set] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        tags = _normalize_tags(tags)
        needle = f"%{query.lower()}%" if query else "%%"
        kinds = kinds or set()
        results: List[Dict[str, Any]] = []
        with self._lock:
            if not kinds or "entity" in kinds:
                tag_sql, tag_params = self._tag_filter_clause("entity", "entities.id", tags)
                rows = self._conn.execute(
                    f"""
                    SELECT id, type, canonical_name, notes FROM entities
                    WHERE superseded_at IS NULL
                      AND (lower(canonical_name) LIKE ? OR lower(COALESCE(notes, '')) LIKE ?)
                      {tag_sql}
                    ORDER BY pinned DESC, salience DESC, last_seen DESC LIMIT ?
                    """,
                    (needle, needle, *tag_params, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "kind": "entity",
                        "id": r["id"],
                        "ref": f"entity:{r['id']}",
                        "score": 0.8,
                        "summary": f"{r['canonical_name']} ({r['type']})",
                        "tags": self._memory_tags("entity", int(r["id"])),
                    })
            if not kinds or "fact" in kinds:
                tag_sql, tag_params = self._tag_filter_clause("fact", "f.id", tags)
                rows = self._conn.execute(
                    f"""
                    SELECT f.id, e.canonical_name AS subject, f.predicate, f.object_value
                    FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                    WHERE f.superseded_by_id IS NULL
                      AND f.superseded_at IS NULL
                      AND lower(COALESCE(e.canonical_name, '') || ' ' || f.predicate || ' ' || COALESCE(f.object_value, '')) LIKE ?
                      {tag_sql}
                    ORDER BY f.pinned DESC, f.last_confirmed_at DESC LIMIT ?
                    """,
                    (needle, *tag_params, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "kind": "fact",
                        "id": r["id"],
                        "ref": f"fact:{r['id']}",
                        "score": 0.75,
                        "summary": f"{r['subject'] or 'user'} {r['predicate']} {r['object_value']}",
                        "tags": self._memory_tags("fact", int(r["id"])),
                    })
            if not kinds or "preference" in kinds:
                tag_sql, tag_params = self._tag_filter_clause("preference", "preferences.id", tags)
                rows = self._conn.execute(
                    f"""
                    SELECT id, domain, statement, strength FROM preferences
                    WHERE superseded_by_id IS NULL
                      AND superseded_at IS NULL
                      AND lower(domain || ' ' || statement || ' ' || strength) LIKE ?
                      {tag_sql}
                    ORDER BY pinned DESC, last_confirmed_at DESC LIMIT ?
                    """,
                    (needle, *tag_params, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "kind": "preference",
                        "id": r["id"],
                        "ref": f"preference:{r['id']}",
                        "score": 0.72,
                        "summary": f"[{r['domain']}/{r['strength']}] {r['statement']}",
                        "tags": self._memory_tags("preference", int(r["id"])),
                    })
            if not kinds or "commitment" in kinds:
                tag_sql, tag_params = self._tag_filter_clause("commitment", "commitments.id", tags)
                rows = self._conn.execute(
                    f"""
                    SELECT id, description, owner, status, due_at FROM commitments
                    WHERE superseded_by_id IS NULL
                      AND superseded_at IS NULL
                      AND lower(description || ' ' || owner || ' ' || status) LIKE ?
                      {tag_sql}
                    ORDER BY status = 'open' DESC, due_at IS NULL, due_at ASC, last_confirmed_at DESC LIMIT ?
                    """,
                    (needle, *tag_params, limit),
                ).fetchall()
                for r in rows:
                    results.append({
                        "kind": "commitment",
                        "id": r["id"],
                        "ref": f"commitment:{r['id']}",
                        "score": 0.7,
                        "summary": f"[{r['status']}/{r['owner']}] {r['description']}",
                        "tags": self._memory_tags("commitment", int(r["id"])),
                    })
        return results[:limit]

    def _search_memory_vector(
        self,
        query: str,
        *,
        kinds: Optional[set] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        if not query:
            return []
        tags = _normalize_tags(tags)
        kinds = kinds or set()
        filters = ["m.embedding_vector IS NOT NULL"]
        params: List[Any] = []
        if kinds:
            placeholders = ",".join("?" for _ in kinds)
            filters.append(f"m.kind IN ({placeholders})")
            params.extend(sorted(kinds))
        if tags:
            placeholders = ",".join("?" for _ in tags)
            filters.append(
                f"""
                (
                    SELECT COUNT(DISTINCT mt.tag)
                    FROM memory_tags mt
                    WHERE mt.kind = m.kind
                      AND mt.row_id = m.row_id
                      AND mt.tag IN ({placeholders})
                ) = ?
                """
            )
            params.extend([*tags, len(tags)])
        where = " AND ".join(filters)
        with self._lock:
            has_vectors = self._conn.execute(
                f"SELECT 1 FROM memory_embedding_records m WHERE {where} LIMIT 1",
                tuple(params),
            ).fetchone()
        if not has_vectors:
            return []
        embeddings = self._embed_texts([query], purpose="memory_query")
        if embeddings is None:
            return []
        vectors, _model = embeddings
        query_vector = vectors[0] if vectors else None
        if query_vector is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT m.kind, m.row_id, m.embedding_vector
                FROM memory_embedding_records m
                WHERE {where}
                """,
                tuple(params),
            ).fetchall()
        scored: List[Dict[str, Any]] = []
        for row in rows:
            vector = self._decode_embedding_vector(row["embedding_vector"])
            if vector is None:
                continue
            similarity = self._cosine_similarity(query_vector, vector)
            if similarity <= 0:
                continue
            item = self._memory_item(row["kind"], int(row["row_id"]))
            if item is None:
                continue
            item["score"] = float(similarity)
            item["semantic_score"] = float(similarity)
            scored.append(item)
        scored.sort(key=lambda item: item["semantic_score"], reverse=True)
        return scored[:limit]

    def _memory_item(self, kind: str, row_id: int) -> Optional[Dict[str, Any]]:
        if self._conn is None:
            return None
        kind = (kind or "").strip().lower()
        tags_text = " ".join(self._memory_tags(kind, row_id))
        with self._lock:
            if kind == "entity":
                row = self._conn.execute(
                    """
                    SELECT id, type, canonical_name, notes, pinned, superseded_at
                    FROM entities WHERE id = ? AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return {
                        "kind": "entity",
                        "id": int(row["id"]),
                        "ref": f"entity:{row['id']}",
                        "score": 0.0,
                        "summary": f"{row['canonical_name']} ({row['type']})"
                        + (f" - {row['notes']}" if row["notes"] else ""),
                        "tags": self._memory_tags("entity", int(row["id"])),
                    }
            if kind == "fact":
                row = self._conn.execute(
                    """
                    SELECT f.id, e.canonical_name AS subject, f.predicate, f.object_value
                    FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                    WHERE f.id = ? AND f.superseded_by_id IS NULL AND f.superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return {
                        "kind": "fact",
                        "id": int(row["id"]),
                        "ref": f"fact:{row['id']}",
                        "score": 0.0,
                        "summary": f"{row['subject'] or 'user'} {row['predicate']} {row['object_value']}",
                        "tags": self._memory_tags("fact", int(row["id"])),
                    }
            if kind == "preference":
                row = self._conn.execute(
                    """
                    SELECT id, domain, statement, strength
                    FROM preferences
                    WHERE id = ? AND superseded_by_id IS NULL AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return {
                        "kind": "preference",
                        "id": int(row["id"]),
                        "ref": f"preference:{row['id']}",
                        "score": 0.0,
                        "summary": f"[{row['domain']}/{row['strength']}] {row['statement']}",
                        "tags": self._memory_tags("preference", int(row["id"])),
                    }
            if kind == "commitment":
                row = self._conn.execute(
                    """
                    SELECT id, description, owner, status, due_at
                    FROM commitments
                    WHERE id = ? AND superseded_by_id IS NULL AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    due = f" due={row['due_at']}" if row["due_at"] else ""
                    return {
                        "kind": "commitment",
                        "id": int(row["id"]),
                        "ref": f"commitment:{row['id']}",
                        "score": 0.0,
                        "summary": f"[{row['status']}/{row['owner']}]{due} {row['description']}",
                        "tags": self._memory_tags("commitment", int(row["id"])),
                    }
            if kind == "relation":
                row = self._conn.execute(
                    """
                    SELECT r.id, r.relation, f.canonical_name AS from_name,
                           t.canonical_name AS to_name
                    FROM relations r
                    JOIN entities f ON f.id = r.from_entity_id
                    JOIN entities t ON t.id = r.to_entity_id
                    WHERE r.id = ? AND r.superseded_by_id IS NULL AND r.superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return {
                        "kind": "relation",
                        "id": int(row["id"]),
                        "ref": f"relation:{row['id']}",
                        "score": 0.0,
                        "summary": f"{row['from_name']} {row['relation']} {row['to_name']}",
                        "tags": self._memory_tags("relation", int(row["id"])),
                    }
        return None

    def _memory_text_for_embedding(self, kind: str, row_id: int) -> Optional[str]:
        if self._conn is None:
            return None
        kind = (kind or "").strip().lower()
        tags_text = " ".join(self._memory_tags(kind, row_id))
        with self._lock:
            if kind == "entity":
                row = self._conn.execute(
                    """
                    SELECT type, canonical_name, aliases, attributes, notes
                    FROM entities WHERE id = ? AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return " ".join(
                        part for part in (
                            f"entity {row['type']}",
                            row["canonical_name"],
                            row["aliases"] or "",
                            row["attributes"] or "",
                            row["notes"] or "",
                            tags_text,
                        )
                        if part
                    )
            if kind == "fact":
                row = self._conn.execute(
                    """
                    SELECT e.canonical_name AS subject, f.predicate, f.object_value
                    FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                    WHERE f.id = ? AND f.superseded_by_id IS NULL AND f.superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return " ".join(
                        part for part in (
                            "fact",
                            row["subject"] or "user",
                            row["predicate"],
                            row["object_value"],
                            tags_text,
                        )
                        if part
                    )
            if kind == "preference":
                row = self._conn.execute(
                    """
                    SELECT domain, statement, strength FROM preferences
                    WHERE id = ? AND superseded_by_id IS NULL AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return " ".join(
                        part for part in (
                            "preference",
                            row["domain"],
                            row["strength"],
                            row["statement"],
                            tags_text,
                        )
                        if part
                    )
            if kind == "commitment":
                row = self._conn.execute(
                    """
                    SELECT description, owner, status, due_at FROM commitments
                    WHERE id = ? AND superseded_by_id IS NULL AND superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    due = f" due {row['due_at']}" if row["due_at"] else ""
                    return " ".join(
                        part for part in (
                            f"commitment {row['status']}",
                            f"owner {row['owner']}{due}",
                            row["description"],
                            tags_text,
                        )
                        if part
                    )
            if kind == "relation":
                row = self._conn.execute(
                    """
                    SELECT r.relation, f.canonical_name AS from_name,
                           t.canonical_name AS to_name
                    FROM relations r
                    JOIN entities f ON f.id = r.from_entity_id
                    JOIN entities t ON t.id = r.to_entity_id
                    WHERE r.id = ? AND r.superseded_by_id IS NULL AND r.superseded_at IS NULL
                    """,
                    (row_id,),
                ).fetchone()
                if row:
                    return " ".join(
                        part for part in (
                            "relation",
                            row["from_name"],
                            row["relation"],
                            row["to_name"],
                            tags_text,
                        )
                        if part
                    )
        return None

    def list_memory(
        self,
        *,
        kind: str = "",
        limit: int = 50,
        include_superseded: bool = False,
        status: str = "",
    ) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        limit = _clamp_int(limit, 50, 1, 500)
        kind = (kind or "").strip().lower()
        kinds = [kind] if kind else ["entity", "fact", "preference", "commitment", "relation"]
        out: List[Dict[str, Any]] = []
        with self._lock:
            if "entity" in kinds:
                where = "" if include_superseded else "WHERE superseded_at IS NULL"
                rows = self._conn.execute(
                    f"""
                    SELECT id, type, canonical_name, notes, salience, pinned,
                           superseded_at
                    FROM entities
                    {where}
                    ORDER BY pinned DESC, salience DESC, last_seen DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                for r in rows:
                    out.append({
                        "ref": f"entity:{r['id']}",
                        "kind": "entity",
                        "id": int(r["id"]),
                        "pinned": bool(r["pinned"]),
                        "superseded": bool(r["superseded_at"]),
                        "summary": f"{r['canonical_name']} ({r['type']})"
                        + (f" - {r['notes']}" if r["notes"] else ""),
                    })
            if "fact" in kinds:
                where = "" if include_superseded else "WHERE f.superseded_by_id IS NULL AND f.superseded_at IS NULL"
                rows = self._conn.execute(
                    f"""
                    SELECT f.id, e.canonical_name AS subject, f.predicate,
                           f.object_value, f.pinned, f.superseded_at
                    FROM facts f LEFT JOIN entities e ON e.id = f.subject_entity_id
                    {where}
                    ORDER BY f.pinned DESC, f.last_confirmed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                for r in rows:
                    out.append({
                        "ref": f"fact:{r['id']}",
                        "kind": "fact",
                        "id": int(r["id"]),
                        "pinned": bool(r["pinned"]),
                        "superseded": bool(r["superseded_at"]),
                        "summary": f"{r['subject'] or 'user'} {r['predicate']} {r['object_value']}",
                    })
            if "preference" in kinds:
                where = "" if include_superseded else "WHERE superseded_by_id IS NULL AND superseded_at IS NULL"
                rows = self._conn.execute(
                    f"""
                    SELECT id, domain, statement, strength, pinned, superseded_at
                    FROM preferences
                    {where}
                    ORDER BY pinned DESC, last_confirmed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                for r in rows:
                    out.append({
                        "ref": f"preference:{r['id']}",
                        "kind": "preference",
                        "id": int(r["id"]),
                        "pinned": bool(r["pinned"]),
                        "superseded": bool(r["superseded_at"]),
                        "summary": f"[{r['domain']}/{r['strength']}] {r['statement']}",
                    })
            if "commitment" in kinds:
                filters = []
                params: List[Any] = []
                if not include_superseded:
                    filters.append("superseded_by_id IS NULL")
                    filters.append("superseded_at IS NULL")
                if status:
                    filters.append("status = ?")
                    params.append(status)
                where = f"WHERE {' AND '.join(filters)}" if filters else ""
                rows = self._conn.execute(
                    f"""
                    SELECT id, description, owner, status, due_at, pinned,
                           superseded_at
                    FROM commitments
                    {where}
                    ORDER BY status = 'open' DESC, pinned DESC,
                             due_at IS NULL, due_at ASC, last_confirmed_at DESC
                    LIMIT ?
                    """,
                    (*params, limit),
                ).fetchall()
                for r in rows:
                    due = f" due={r['due_at']}" if r["due_at"] else ""
                    out.append({
                        "ref": f"commitment:{r['id']}",
                        "kind": "commitment",
                        "id": int(r["id"]),
                        "pinned": bool(r["pinned"]),
                        "superseded": bool(r["superseded_at"]),
                        "summary": f"[{r['status']}/{r['owner']}]{due} {r['description']}",
                    })
            if "relation" in kinds:
                where = "" if include_superseded else "WHERE r.superseded_by_id IS NULL AND r.superseded_at IS NULL"
                rows = self._conn.execute(
                    f"""
                    SELECT r.id, r.relation, r.pinned, r.superseded_at,
                           f.canonical_name AS from_name,
                           t.canonical_name AS to_name
                    FROM relations r
                    JOIN entities f ON f.id = r.from_entity_id
                    JOIN entities t ON t.id = r.to_entity_id
                    {where}
                    ORDER BY r.pinned DESC, r.last_confirmed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                for r in rows:
                    out.append({
                        "ref": f"relation:{r['id']}",
                        "kind": "relation",
                        "id": int(r["id"]),
                        "pinned": bool(r["pinned"]),
                        "superseded": bool(r["superseded_at"]),
                        "summary": f"{r['from_name']} {r['relation']} {r['to_name']}",
                    })
        return out[:limit]

    def get_memory(self, kind: str, row_id: int) -> Optional[Dict[str, Any]]:
        items = self.list_memory(kind=kind, limit=500, include_superseded=True)
        for item in items:
            if int(item["id"]) == int(row_id):
                return item
        return None

    def review_staging(self, *, limit: int = 20) -> List[Dict[str, Any]]:
        if self._conn is None:
            return []
        limit = _clamp_int(limit, 20, 1, 200)
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_id, turn_id, extraction_kind, payload,
                       confidence, created_at
                FROM staging_extractions
                WHERE consolidated_at IS NULL
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def recover_stale_queue_jobs(self, *, stale_after_seconds: int = 300) -> int:
        return self._recover_stale_jobs(stale_after_seconds)

    def memory_stats(self) -> Dict[str, Any]:
        if self._conn is None:
            return {}
        stats: Dict[str, Any] = {"db_path": str(self._db_path or "")}
        embedding_cfg = self._embedding_config()
        with self._lock:
            stats["turns"] = self._conn.execute("SELECT COUNT(*) FROM session_turns").fetchone()[0]
            stats["sessions"] = self._conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM session_turns"
            ).fetchone()[0]
            stats["staging_pending"] = self._conn.execute(
                "SELECT COUNT(*) FROM staging_extractions WHERE consolidated_at IS NULL"
            ).fetchone()[0]
            stats["queue"] = self._queue_counts_locked()
            stats["queue_errors"] = self._queue_errors_locked(limit=5)
            stale_running = int(
                self._conn.execute(
                    """
                    SELECT COUNT(*) FROM memory_work_queue
                    WHERE status = 'running' AND updated_at < ?
                    """,
                    (_now() - self._stale_job_seconds,),
                ).fetchone()[0]
            )
            stats["queue_stale_running"] = stale_running
            stats["queue_last_recovered_stale_running"] = self._last_stale_jobs_recovered
            stats["queue_detail"] = {
                "failed_recent": [
                    {
                        "id": int(error["id"]),
                        "job_type": error["job_type"],
                        "attempts": int(error["attempts"] or 0),
                        "last_error": error["last_error"] or "",
                    }
                    for error in stats["queue_errors"]
                    if error.get("status") == "failed"
                ],
                "stale_running": stale_running,
                "last_recovered_stale_running": self._last_stale_jobs_recovered,
                "errors": stats["queue_errors"],
            }
            stats["memory"] = {}
            for table in ("entities", "facts", "preferences", "commitments", "relations"):
                stats["memory"][table] = self._conn.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            stats["embeddings"] = {
                "configured": bool(embedding_cfg.get("configured")),
                "model": embedding_cfg.get("model") or "",
                "session_records": self._conn.execute(
                    "SELECT COUNT(*) FROM session_embedding_records"
                ).fetchone()[0],
                "session_embedded": self._conn.execute(
                    "SELECT COUNT(*) FROM session_embedding_records WHERE embedding_vector IS NOT NULL"
                ).fetchone()[0],
                "memory_records": self._conn.execute(
                    "SELECT COUNT(*) FROM memory_embedding_records"
                ).fetchone()[0],
                "memory_embedded": self._conn.execute(
                    "SELECT COUNT(*) FROM memory_embedding_records WHERE embedding_vector IS NOT NULL"
                ).fetchone()[0],
            }
        return stats

    def memory_doctor(self, *, limit: int = 100) -> Dict[str, Any]:
        if self._conn is None:
            return {"ok": True, "findings": [], "counts": {"findings": 0}}
        limit = _clamp_int(limit, 100, 1, 500)
        findings: List[Dict[str, Any]] = []

        def add(severity: str, code: str, kind: str, row_id: Any, detail: str) -> None:
            if len(findings) >= limit:
                return
            try:
                parsed_id = int(row_id)
            except (TypeError, ValueError):
                return
            findings.append({
                "severity": severity,
                "code": code,
                "ref": f"{kind}:{parsed_id}",
                "kind": kind,
                "id": parsed_id,
                "detail": detail,
            })

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, canonical_name FROM entities
                WHERE superseded_at IS NULL
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for row in rows:
                if self._looks_like_relationish_entity_name(row["canonical_name"] or ""):
                    add(
                        "warn",
                        "malformed_entity_name",
                        "entity",
                        row["id"],
                        "canonical_name looks relation-like",
                    )

            blank_checks = [
                (
                    "entity",
                    "entities",
                    "TRIM(COALESCE(canonical_name, '')) = '' OR TRIM(COALESCE(type, '')) = ''",
                ),
                (
                    "fact",
                    "facts",
                    "TRIM(COALESCE(predicate, '')) = '' OR TRIM(COALESCE(object_value, '')) = ''",
                ),
                (
                    "preference",
                    "preferences",
                    "TRIM(COALESCE(domain, '')) = '' OR TRIM(COALESCE(statement, '')) = '' "
                    "OR TRIM(COALESCE(strength, '')) = ''",
                ),
                (
                    "commitment",
                    "commitments",
                    "TRIM(COALESCE(description, '')) = '' OR TRIM(COALESCE(owner, '')) = '' "
                    "OR TRIM(COALESCE(status, '')) = ''",
                ),
            ]
            for kind, table, where in blank_checks:
                if len(findings) >= limit:
                    break
                rows = self._conn.execute(
                    f"SELECT id FROM {table} WHERE {where} ORDER BY id LIMIT ?",
                    (limit - len(findings),),
                ).fetchall()
                for row in rows:
                    add("error", "blank_summary", kind, row["id"], "required summary field is blank")

            if len(findings) < limit:
                rows = self._conn.execute(
                    """
                    SELECT r.id
                    FROM relations r
                    LEFT JOIN entities f ON f.id = r.from_entity_id
                    LEFT JOIN entities t ON t.id = r.to_entity_id
                    WHERE TRIM(COALESCE(r.relation, '')) = ''
                       OR f.id IS NULL
                       OR t.id IS NULL
                    ORDER BY r.id
                    LIMIT ?
                    """,
                    (limit - len(findings),),
                ).fetchall()
                for row in rows:
                    add("error", "blank_summary", "relation", row["id"], "relation summary cannot be rendered")

            superseded_checks = [
                ("entity", "entities", "t.superseded_at IS NOT NULL"),
                (
                    "fact",
                    "facts",
                    "t.superseded_by_id IS NOT NULL OR t.superseded_at IS NOT NULL",
                ),
                (
                    "preference",
                    "preferences",
                    "t.superseded_by_id IS NOT NULL OR t.superseded_at IS NOT NULL",
                ),
                (
                    "commitment",
                    "commitments",
                    "t.superseded_by_id IS NOT NULL OR t.superseded_at IS NOT NULL",
                ),
                (
                    "relation",
                    "relations",
                    "t.superseded_by_id IS NOT NULL OR t.superseded_at IS NOT NULL",
                ),
            ]
            for kind, table, where in superseded_checks:
                if len(findings) >= limit:
                    break
                rows = self._conn.execute(
                    f"""
                    SELECT m.row_id
                    FROM memory_embedding_records m
                    JOIN {table} t ON t.id = m.row_id
                    WHERE m.kind = ?
                      AND ({where})
                      AND (
                        m.embedding_vector IS NOT NULL
                        OR TRIM(COALESCE(m.text, '')) != ''
                      )
                    ORDER BY m.row_id
                    LIMIT ?
                    """,
                    (kind, limit - len(findings)),
                ).fetchall()
                for row in rows:
                    add(
                        "warn",
                        "superseded_embedding",
                        kind,
                        row["row_id"],
                        "superseded row still has embedding text/vector",
                    )

            tables = {
                "entity": "entities",
                "fact": "facts",
                "preference": "preferences",
                "commitment": "commitments",
                "relation": "relations",
            }
            for kind, table in tables.items():
                if len(findings) >= limit:
                    break
                rows = self._conn.execute(
                    f"""
                    SELECT m.row_id
                    FROM memory_embedding_records m
                    LEFT JOIN {table} t ON t.id = m.row_id
                    WHERE m.kind = ? AND t.id IS NULL
                    ORDER BY m.row_id
                    LIMIT ?
                    """,
                    (kind, limit - len(findings)),
                ).fetchall()
                for row in rows:
                    add("warn", "orphan_embedding", kind, row["row_id"], "embedding row has no memory record")

            if len(findings) < limit:
                rows = self._conn.execute(
                    """
                    SELECT kind, row_id, embedding_vector
                    FROM memory_embedding_records
                    WHERE embedding_vector IS NOT NULL
                    ORDER BY kind, row_id
                    LIMIT ?
                    """,
                    (limit - len(findings),),
                ).fetchall()
                for row in rows:
                    if self._decode_embedding_vector(row["embedding_vector"]) is None:
                        add(
                            "error",
                            "invalid_embedding",
                            row["kind"],
                            row["row_id"],
                            "embedding vector is malformed",
                        )

        embedding_cfg = self._embedding_config()
        if embedding_cfg.get("configured"):
            for kind, row_id in self._active_memory_refs():
                if len(findings) >= limit:
                    break
                with self._lock:
                    row = self._conn.execute(
                        """
                        SELECT embedding_vector
                        FROM memory_embedding_records
                        WHERE kind = ? AND row_id = ?
                        """,
                        (kind, row_id),
                    ).fetchone()
                if row is None or not row["embedding_vector"]:
                    add(
                        "warn",
                        "missing_embedding",
                        kind,
                        row_id,
                        "active row has no embedding vector",
                    )

        return {
            "ok": not findings,
            "findings": findings,
            "counts": {"findings": len(findings)},
            "embeddings_configured": bool(embedding_cfg.get("configured")),
            "truncated": len(findings) >= limit,
        }

    def rebuild_embeddings(
        self,
        *,
        include_sessions: bool = True,
        include_memory: bool = True,
        limit: int = 0,
    ) -> Dict[str, Any]:
        if self._conn is None:
            return {}
        cfg = self._embedding_config()
        result: Dict[str, Any] = {
            "configured": bool(cfg.get("configured")),
            "model": cfg.get("model") or "",
            "session_records": 0,
            "memory_records": 0,
            "errors": [],
        }
        if not cfg.get("configured"):
            result["error"] = "memory_embedding endpoint is not configured"
            return result
        if include_memory:
            result["pruned_inactive_memory_records"] = self._prune_inactive_memory_embeddings()
        remaining = max(0, int(limit or 0))
        if include_sessions:
            with self._lock:
                session_rows = [
                    dict(row)
                    for row in self._conn.execute(
                        """
                        SELECT id, session_id, turn_id, role, text
                        FROM session_embedding_records
                        WHERE text != ''
                        ORDER BY session_id, turn_id, role
                        """
                    ).fetchall()
                ]
            if remaining:
                session_rows = session_rows[:remaining]
            batch_size = 16
            for start in range(0, len(session_rows), batch_size):
                batch = session_rows[start:start + batch_size]
                try:
                    embedded = self._embed_texts([row["text"] for row in batch], purpose="session_rebuild")
                    if embedded is None:
                        continue
                    vectors, model = embedded
                    now = _now()
                    with self._lock:
                        for row, vector in zip(batch, vectors):
                            self._conn.execute(
                                """
                                UPDATE session_embedding_records
                                SET embedding_model = ?, embedding_vector = ?,
                                    embedded_at = ?, embedding_error = NULL
                                WHERE id = ?
                                """,
                                (model, _json_vector(vector or []), now, row["id"]),
                            )
                            result["session_records"] += 1
                except Exception as exc:
                    result["errors"].append(str(exc)[:300])
            if remaining:
                remaining = max(0, remaining - len(session_rows))
        if include_memory and (not limit or remaining > 0):
            refs = self._active_memory_refs(limit=remaining)
            for kind, row_id in refs:
                try:
                    self._process_embed_memory_row(
                        {"target_table": kind, "target_row_id": row_id},
                        {"kind": kind, "row_id": row_id},
                    )
                    result["memory_records"] += 1
                except Exception as exc:
                    result["errors"].append(f"{kind}:{row_id} {str(exc)[:240]}")
        return result

    def _prune_inactive_memory_embeddings(self) -> int:
        if self._conn is None:
            return 0
        deletes = [
            (
                "entity",
                """
                DELETE FROM memory_embedding_records
                WHERE kind = 'entity'
                  AND NOT EXISTS (
                    SELECT 1 FROM entities e
                    WHERE e.id = memory_embedding_records.row_id
                      AND e.superseded_at IS NULL
                  )
                """,
            ),
            (
                "fact",
                """
                DELETE FROM memory_embedding_records
                WHERE kind = 'fact'
                  AND NOT EXISTS (
                    SELECT 1 FROM facts f
                    WHERE f.id = memory_embedding_records.row_id
                      AND f.superseded_by_id IS NULL
                      AND f.superseded_at IS NULL
                  )
                """,
            ),
            (
                "preference",
                """
                DELETE FROM memory_embedding_records
                WHERE kind = 'preference'
                  AND NOT EXISTS (
                    SELECT 1 FROM preferences p
                    WHERE p.id = memory_embedding_records.row_id
                      AND p.superseded_by_id IS NULL
                      AND p.superseded_at IS NULL
                  )
                """,
            ),
            (
                "commitment",
                """
                DELETE FROM memory_embedding_records
                WHERE kind = 'commitment'
                  AND NOT EXISTS (
                    SELECT 1 FROM commitments c
                    WHERE c.id = memory_embedding_records.row_id
                      AND c.superseded_by_id IS NULL
                      AND c.superseded_at IS NULL
                  )
                """,
            ),
            (
                "relation",
                """
                DELETE FROM memory_embedding_records
                WHERE kind = 'relation'
                  AND NOT EXISTS (
                    SELECT 1 FROM relations r
                    WHERE r.id = memory_embedding_records.row_id
                      AND r.superseded_by_id IS NULL
                      AND r.superseded_at IS NULL
                  )
                """,
            ),
        ]
        pruned = 0
        with self._lock:
            for _kind, sql in deletes:
                cur = self._conn.execute(sql)
                pruned += int(cur.rowcount or 0)
        return pruned

    def _active_memory_refs(self, *, limit: int = 0) -> List[Tuple[str, int]]:
        if self._conn is None:
            return []
        refs: List[Tuple[str, int]] = []
        queries = [
            ("entity", "SELECT id FROM entities WHERE superseded_at IS NULL ORDER BY last_seen DESC"),
            (
                "fact",
                """
                SELECT id FROM facts
                WHERE superseded_by_id IS NULL AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC
                """,
            ),
            (
                "preference",
                """
                SELECT id FROM preferences
                WHERE superseded_by_id IS NULL AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC
                """,
            ),
            (
                "commitment",
                """
                SELECT id FROM commitments
                WHERE superseded_by_id IS NULL AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC
                """,
            ),
            (
                "relation",
                """
                SELECT id FROM relations
                WHERE superseded_by_id IS NULL AND superseded_at IS NULL
                ORDER BY last_confirmed_at DESC
                """,
            ),
        ]
        with self._lock:
            for kind, sql in queries:
                for row in self._conn.execute(sql).fetchall():
                    refs.append((kind, int(row["id"])))
                    if limit and len(refs) >= limit:
                        return refs
        return refs

    def export_memory(self) -> Dict[str, Any]:
        if self._conn is None:
            return {}
        tables = ("entities", "facts", "preferences", "commitments", "relations", "memory_tags")
        payload: Dict[str, Any] = {
            "format": "cos-memory-export-v1",
            "exported_at": _now(),
            "tables": {},
        }
        with self._lock:
            for table in tables:
                payload["tables"][table] = [
                    dict(row) for row in self._conn.execute(f"SELECT * FROM {table}").fetchall()
                ]
        return payload

    def import_memory(self, payload: Dict[str, Any]) -> Dict[str, int]:
        if self._conn is None:
            return {}
        if not isinstance(payload, dict) or payload.get("format") != "cos-memory-export-v1":
            raise ValueError("unsupported cos-memory export format")
        tables = ("entities", "facts", "preferences", "commitments", "relations", "memory_tags")
        counts: Dict[str, int] = {}
        with self._lock:
            for table in tables:
                rows = payload.get("tables", {}).get(table, [])
                if not isinstance(rows, list):
                    continue
                allowed = {
                    row["name"]
                    for row in self._conn.execute(f"PRAGMA table_info({table})").fetchall()
                }
                inserted = 0
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    cols = [col for col in row.keys() if col in allowed]
                    if not cols:
                        continue
                    placeholders = ",".join("?" for _ in cols)
                    col_sql = ",".join(cols)
                    cur = self._conn.execute(
                        f"INSERT OR IGNORE INTO {table} ({col_sql}) VALUES ({placeholders})",
                        tuple(row[col] for col in cols),
                    )
                    inserted += int(cur.rowcount or 0)
                counts[table] = inserted
        if any(counts.values()):
            self._briefing_cache = ""
        return counts

    def _handle_remember(self, args: Dict[str, Any]) -> str:
        kind = str(args.get("kind") or "").strip().lower()
        content = args.get("content") or {}
        if not isinstance(content, dict):
            return tool_error("remember content must be an object")
        tags = _normalize_tags(args.get("tags") or content.get("tags"))
        try:
            confidence = float(args.get("confidence", 0.8))
        except (TypeError, ValueError):
            confidence = 0.8
        try:
            item_kind, row_id = self._create_memory_from_payload(
                kind,
                content,
                confidence=confidence,
                source_session_id=self._session_id,
                source_turn_id=None,
                actor="agent",
                user_edited=False,
                tags=tags,
            )
        except Exception as exc:
            return tool_error(f"remember failed: {exc}")
        self._briefing_cache = ""
        return json.dumps({
            "ok": True,
            "ref": f"{item_kind}:{row_id}",
            "kind": item_kind,
            "id": row_id,
            "tags": tags,
        })

    def _handle_forget(self, args: Dict[str, Any]) -> str:
        kind, row_id = parse_memory_ref(str(args.get("memory_ref") or ""))
        if kind is None:
            kind = str(args.get("kind") or "").strip().lower() or None
            try:
                row_id = int(args.get("memory_id"))
            except (TypeError, ValueError):
                row_id = None
        reason = str(args.get("reason") or "").strip()
        if not kind or not row_id:
            return tool_error("forget requires memory_ref or memory_id + kind")
        ok = self.supersede_memory(kind, row_id, reason=reason, actor="agent")
        self._briefing_cache = ""
        return json.dumps({"ok": ok, "ref": f"{kind}:{row_id}"})

    def supersede_memory(self, kind: str, row_id: int, *, reason: str, actor: str = "user") -> bool:
        if self._conn is None:
            return False
        table = {
            "fact": "facts",
            "preference": "preferences",
            "commitment": "commitments",
            "relation": "relations",
            "entity": "entities",
        }.get(kind)
        if table is None:
            return False
        with self._lock:
            if kind == "commitment":
                cur = self._conn.execute(
                    "UPDATE commitments SET status = 'cancelled', superseded_at = ?, notes = ? WHERE id = ?",
                    (_now(), reason, row_id),
                )
            elif kind == "entity":
                cur = self._conn.execute(
                    "UPDATE entities SET salience = 0, superseded_at = ?, notes = ? WHERE id = ?",
                    (_now(), reason, row_id),
                )
            else:
                cur = self._conn.execute(
                    f"UPDATE {table} SET superseded_at = ?, notes = ? WHERE id = ?",
                    (_now(), reason, row_id),
                )
            if cur.rowcount:
                self._conn.execute(
                    "DELETE FROM memory_embedding_records WHERE kind = ? AND row_id = ?",
                    (kind, row_id),
                )
                self._audit(actor, "supersede", table, row_id, {"reason": reason})
                return True
        return False

    def hard_delete_memory(self, kind: str, row_id: int, *, reason: str, actor: str = "user") -> bool:
        if self._conn is None:
            return False
        table = {
            "entity": "entities",
            "fact": "facts",
            "preference": "preferences",
            "commitment": "commitments",
            "relation": "relations",
        }.get(kind)
        if table is None:
            return False
        with self._lock:
            cur = self._conn.execute(f"DELETE FROM {table} WHERE id = ?", (row_id,))
            if cur.rowcount:
                self._conn.execute(
                    "DELETE FROM memory_embedding_records WHERE kind = ? AND row_id = ?",
                    (kind, row_id),
                )
                self._conn.execute(
                    "DELETE FROM memory_tags WHERE kind = ? AND row_id = ?",
                    (kind, row_id),
                )
                self._audit(actor, "delete", table, row_id, {"redacted": True, "reason": reason})
                self._briefing_cache = ""
                return True
        return False

    def set_pin(self, kind: str, row_id: int, pinned: bool, *, actor: str = "user") -> bool:
        if self._conn is None:
            return False
        table = {
            "entity": "entities",
            "fact": "facts",
            "preference": "preferences",
            "commitment": "commitments",
            "relation": "relations",
        }.get(kind)
        if table is None:
            return False
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE {table} SET pinned = ? WHERE id = ?",
                (1 if pinned else 0, row_id),
            )
            if cur.rowcount:
                self._audit(actor, "pin" if pinned else "unpin", table, row_id, {})
                self._briefing_cache = ""
                return True
        return False

    def get_cached_digest(self, session_id: str, turn_id: int) -> Optional[dict]:
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT digest_json, digest_text FROM digest_cache WHERE session_id = ? AND turn_id = ?",
                (session_id, turn_id),
            ).fetchone()
        if row is None:
            return None
        data = self._loads(row["digest_json"], {})
        data["digest_text"] = row["digest_text"]
        return data

    def put_cached_digest(self, session_id: str, turn_id: int, digest: dict, rendered: str) -> None:
        if self._conn is None:
            return
        now = _now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO digest_cache(
                    session_id, turn_id, digest_json, digest_text, aux_model,
                    tokens_original, tokens_digest, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, turn_id) DO UPDATE SET
                    digest_json = excluded.digest_json,
                    digest_text = excluded.digest_text,
                    aux_model = excluded.aux_model,
                    tokens_original = excluded.tokens_original,
                    tokens_digest = excluded.tokens_digest,
                    created_at = excluded.created_at
                """,
                (
                    session_id,
                    turn_id,
                    json.dumps(digest, ensure_ascii=False),
                    rendered,
                    digest.get("aux_model"),
                    digest.get("tokens_original"),
                    digest.get("tokens_digest"),
                    now,
                ),
            )

    def shutdown(self) -> None:
        self._shutdown.set()
        for worker in list(self._workers):
            worker.join(timeout=1.0)
        self._workers.clear()
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            if self._session_db is not None:
                try:
                    self._session_db.close()
                except Exception:
                    pass
                self._session_db = None
            self._initialized = False


def register(ctx) -> None:
    ctx.register_memory_provider(CosMemoryProvider())
