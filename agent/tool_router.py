"""Single-tool router for Hermes' full tool catalog.

The router keeps the model-visible tool surface tiny while preserving the
existing internal tool registry and dispatch paths.  Search and describe are
handled here; execute is completed by ``AIAgent`` so agent-level tools,
memory/context-provider tools, hooks, approvals, and guardrails stay in the
normal execution path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

logger = logging.getLogger(__name__)

TOOL_ROUTER_NAME = "tool_router"

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_EMBEDDING_CACHE: Dict[str, Dict[str, Any]] = {}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def _coerce_int(value: Any, default: int, *, minimum: int = 1, maximum: int = 50) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def normalize_tool_routing_config(agent_config: Mapping[str, Any] | None) -> Dict[str, Any]:
    """Return normalized ``agent.tool_routing`` settings.

    Defaults are intentionally router-first for local Hermes deployments.  Users
    can set ``agent.tool_routing.mode: direct`` to restore legacy schema exposure.
    """
    cfg = {}
    if isinstance(agent_config, Mapping):
        raw = agent_config.get("tool_routing", {})
        if isinstance(raw, Mapping):
            cfg = dict(raw)

    mode = str(cfg.get("mode", "router") or "router").strip().lower()
    if mode not in {"router", "direct"}:
        mode = "router"

    return {
        "mode": mode,
        "search_limit": _coerce_int(cfg.get("search_limit"), 8, minimum=1, maximum=25),
        "embedding_index": _coerce_bool(cfg.get("embedding_index"), True),
        "compact_history": _coerce_bool(cfg.get("compact_history"), True),
        "execute_result_inline_chars": _coerce_int(
            cfg.get("execute_result_inline_chars"),
            3000,
            minimum=500,
            maximum=100_000,
        ),
        "skill_result_inline_chars": _coerce_int(
            cfg.get("skill_result_inline_chars"),
            20000,
            minimum=1000,
            maximum=200_000,
        ),
        "history_execute_inline_chars": _coerce_int(
            cfg.get("history_execute_inline_chars"),
            2000,
            minimum=500,
            maximum=100_000,
        ),
        "history_preview_chars": _coerce_int(
            cfg.get("history_preview_chars"),
            600,
            minimum=120,
            maximum=5000,
        ),
        "telemetry": _coerce_bool(cfg.get("telemetry"), True),
    }


def build_tool_router_schema(*, catalog_count: int = 0, catalog_hash: str = "") -> Dict[str, Any]:
    """Return the OpenAI-format function schema for the single exposed router."""
    description = (
        "Discover and use Hermes tools and skills without exposing every schema in the "
        "prompt. Use action='search' to find relevant tools, action='describe' "
        "to inspect one exact tool schema or skill, and action='execute' to run "
        "that tool or load that skill. Skill results use names like "
        "'skill:weather-lookup'. Hidden tools are not directly callable; call "
        "this router instead."
    )
    if catalog_count:
        description += f" The hidden catalog currently contains {catalog_count} tools and skills."
    if catalog_hash:
        description += f" Catalog hash: {catalog_hash}."

    return {
        "name": TOOL_ROUTER_NAME,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["search", "describe", "execute"],
                    "description": "Router action to perform.",
                },
                "query": {
                    "type": "string",
                    "description": "Natural-language capability search query for action='search'.",
                },
                "tool_name": {
                    "type": "string",
                    "description": "Exact hidden tool name for action='describe' or action='execute'.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments object to pass to tool_name for action='execute'.",
                    "additionalProperties": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum search matches to return.",
                    "minimum": 1,
                    "maximum": 25,
                },
            },
            "required": ["action"],
        },
    }


@dataclass(frozen=True)
class ToolCatalogEntry:
    name: str
    description: str
    parameters: Dict[str, Any]
    required: List[str]
    toolset: str
    schema: Dict[str, Any]
    search_text: str
    parameter_names: List[str]


@dataclass(frozen=True)
class SkillCatalogEntry:
    name: str
    description: str
    category: str
    search_text: str


class ToolRouter:
    """Search and describe hidden Hermes tools and skills."""

    def __init__(
        self,
        tool_schemas: Iterable[Dict[str, Any]],
        *,
        skill_entries: Iterable[Mapping[str, Any]] | None = None,
        search_limit: int = 8,
        embedding_index: bool = True,
        toolset_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> None:
        self.search_limit = _coerce_int(search_limit, 8, minimum=1, maximum=25)
        self.embedding_index_enabled = bool(embedding_index)
        self.entries = self._build_entries(tool_schemas, toolset_lookup=toolset_lookup)
        self.entries_by_name = {entry.name: entry for entry in self.entries}
        self.skill_entries = self._build_skill_entries(skill_entries or [])
        self.skills_by_ref = {
            self.skill_ref(entry.name): entry
            for entry in self.skill_entries
        }
        self.catalog_hash = self._catalog_hash(self.entries, self.skill_entries)
        self.embedding_status = "disabled"
        if self.embedding_index_enabled:
            self.embedding_status = "building"
            self._start_embedding_index_build()

    def _build_entries(
        self,
        tool_schemas: Iterable[Dict[str, Any]],
        *,
        toolset_lookup: Optional[Callable[[str], Optional[str]]],
    ) -> List[ToolCatalogEntry]:
        entries: List[ToolCatalogEntry] = []
        for wrapped in tool_schemas or []:
            if not isinstance(wrapped, Mapping):
                continue
            func = wrapped.get("function") if wrapped.get("type") == "function" else wrapped
            if not isinstance(func, Mapping):
                continue
            name = str(func.get("name") or "").strip()
            if not name or name == TOOL_ROUTER_NAME:
                continue
            description = str(func.get("description") or "")
            parameters = func.get("parameters") if isinstance(func.get("parameters"), dict) else {}
            required = parameters.get("required") if isinstance(parameters, dict) else []
            if not isinstance(required, list):
                required = []
            properties = parameters.get("properties") if isinstance(parameters, dict) else {}
            parameter_names = sorted(str(k) for k in properties.keys()) if isinstance(properties, dict) else []
            prop_text = ""
            if isinstance(properties, dict):
                prop_text = " ".join(
                    f"{key} {value.get('description', '') if isinstance(value, dict) else ''}"
                    for key, value in properties.items()
                )
            toolset = ""
            if toolset_lookup is not None:
                try:
                    toolset = toolset_lookup(name) or ""
                except Exception:
                    toolset = ""
            schema = dict(func)
            search_text = " ".join([name, description, " ".join(parameter_names), prop_text]).lower()
            entries.append(
                ToolCatalogEntry(
                    name=name,
                    description=description,
                    parameters=parameters,
                    required=[str(item) for item in required],
                    toolset=toolset,
                    schema=schema,
                    search_text=search_text,
                    parameter_names=parameter_names,
                )
            )
        return sorted(entries, key=lambda entry: entry.name)

    def _build_skill_entries(
        self,
        skill_entries: Iterable[Mapping[str, Any]],
    ) -> List[SkillCatalogEntry]:
        entries: List[SkillCatalogEntry] = []
        seen: set[str] = set()
        for raw in skill_entries or []:
            if not isinstance(raw, Mapping):
                continue
            name = str(raw.get("name") or "").strip()
            if not name or name in seen:
                continue
            description = " ".join(str(raw.get("description") or "").split())
            category = str(raw.get("category") or "").strip()
            search_text = " ".join([name, description, category, "skill workflow procedure"]).lower()
            seen.add(name)
            entries.append(
                SkillCatalogEntry(
                    name=name,
                    description=description,
                    category=category,
                    search_text=search_text,
                )
            )
        return sorted(entries, key=lambda entry: (entry.category, entry.name))

    def _catalog_hash(
        self,
        entries: List[ToolCatalogEntry],
        skill_entries: List[SkillCatalogEntry],
    ) -> str:
        payload = {
            "tools": [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "parameters": entry.parameters,
                    "toolset": entry.toolset,
                }
                for entry in entries
            ],
            "skills": [
                {
                    "name": entry.name,
                    "description": entry.description,
                    "category": entry.category,
                }
                for entry in skill_entries
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _start_embedding_index_build(self) -> None:
        cached = _EMBEDDING_CACHE.get(self.catalog_hash)
        if cached is not None:
            self.embedding_status = cached.get("status", "ready")
            return

        _EMBEDDING_CACHE[self.catalog_hash] = {"status": "building"}
        thread = threading.Thread(
            target=self._build_embedding_index_best_effort,
            name=f"tool-router-embed-{self.catalog_hash[:8]}",
            daemon=True,
        )
        thread.start()

    def _embedding_config(self) -> Dict[str, Any]:
        try:
            from hermes_cli.config import load_config
            cfg = load_config()
        except Exception:
            return {}
        aux = cfg.get("auxiliary") if isinstance(cfg, dict) else {}
        task = aux.get("memory_embedding") if isinstance(aux, dict) else {}
        return task if isinstance(task, dict) else {}

    def _build_embedding_index_best_effort(self) -> None:
        """Best-effort background embedding warmup.

        Search never waits for this.  A missing endpoint or request failure only
        updates status so diagnostics can explain why lexical search is in use.
        """
        try:
            cfg = self._embedding_config()
            base_url = str(cfg.get("base_url") or "").strip().rstrip("/")
            if not base_url:
                self.embedding_status = "unavailable"
                _EMBEDDING_CACHE[self.catalog_hash] = {"status": self.embedding_status}
                return
            endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
            model = str(cfg.get("model") or "nomicai-modernbert-embed-base-bf16")
            timeout = float(cfg.get("timeout") or 10)
            texts = [entry.search_text[:2000] for entry in self.entries]
            texts.extend(entry.search_text[:2000] for entry in self.skill_entries)
            if not texts:
                self.embedding_status = "ready"
                _EMBEDDING_CACHE[self.catalog_hash] = {"status": self.embedding_status, "vectors": []}
                return
            body = json.dumps({"model": model, "input": texts}).encode("utf-8")
            request = urllib.request.Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            api_key = str(cfg.get("api_key") or "").strip()
            if api_key:
                request.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = payload.get("data")
            vectors = [item.get("embedding") for item in data] if isinstance(data, list) else []
            self.embedding_status = "ready" if len(vectors) == len(texts) else "failed"
            _EMBEDDING_CACHE[self.catalog_hash] = {
                "status": self.embedding_status,
                "vectors": vectors if self.embedding_status == "ready" else [],
                "model": model,
                "indexed_at": time.time(),
            }
        except (OSError, urllib.error.URLError, ValueError, TypeError, KeyError) as exc:
            self.embedding_status = "failed"
            _EMBEDDING_CACHE[self.catalog_hash] = {
                "status": self.embedding_status,
                "error": str(exc)[:200],
            }
            logger.debug("tool router embedding index unavailable: %s", exc)

    def search(self, query: str = "", *, limit: Optional[int] = None) -> Dict[str, Any]:
        limit = _coerce_int(limit, self.search_limit, minimum=1, maximum=25)
        query = str(query or "").strip()
        tokens = [token.lower() for token in _WORD_RE.findall(query)]
        scored = []
        for entry in self.entries:
            score = self._score_entry(entry, query.lower(), tokens)
            if score > 0 or not tokens:
                scored.append((score, "tool", entry))
        for entry in self.skill_entries:
            score = self._score_skill_entry(entry, query.lower(), tokens)
            if score > 0 or not tokens:
                scored.append((score, "skill", entry))
        scored.sort(key=lambda pair: (-pair[0], pair[1], pair[2].name))
        matches = [
            self._compact_entry(entry, score)
            if kind == "tool"
            else self._compact_skill_entry(entry, score)
            for score, kind, entry in scored[:limit]
        ]
        return {
            "success": True,
            "action": "search",
            "query": query,
            "matches": matches,
            "catalog_hash": self.catalog_hash,
            "catalog_count": len(self.entries) + len(self.skill_entries),
            "tool_count": len(self.entries),
            "skill_count": len(self.skill_entries),
            "embedding_status": self.embedding_status,
        }

    def describe(self, tool_name: str) -> Dict[str, Any]:
        name = str(tool_name or "").strip()
        entry = self.entries_by_name.get(name)
        if entry is None:
            skill = self.skill_for_ref(name)
            if skill is not None:
                return self.describe_skill(skill)
            return self.unknown_tool(name)
        return {
            "success": True,
            "action": "describe",
            "type": "tool",
            "tool_name": entry.name,
            "toolset": entry.toolset,
            "schema": entry.schema,
            "catalog_hash": self.catalog_hash,
        }

    def describe_skill(self, entry: SkillCatalogEntry) -> Dict[str, Any]:
        return {
            "success": True,
            "action": "describe",
            "type": "skill",
            "name": self.skill_ref(entry.name),
            "skill_name": entry.name,
            "category": entry.category,
            "description": entry.description,
            "load_with": {
                "tool_name": self.skill_ref(entry.name),
                "arguments": {},
            },
            "equivalent_tool": {
                "tool_name": "skill_view",
                "arguments": {"name": entry.name},
            },
            "catalog_hash": self.catalog_hash,
        }

    def unknown_tool(self, tool_name: str) -> Dict[str, Any]:
        suggestions = self.search(tool_name, limit=5).get("matches", [])
        return {
            "success": False,
            "error": f"Unknown routed tool: {tool_name}",
            "tool_name": tool_name,
            "suggestions": suggestions,
            "catalog_hash": self.catalog_hash,
        }

    def _score_entry(self, entry: ToolCatalogEntry, query: str, tokens: List[str]) -> int:
        if not tokens:
            return 1
        score = 0
        name = entry.name.lower()
        description = entry.description.lower()
        params = " ".join(entry.parameter_names).lower()
        if query and query == name:
            score += 100
        elif query and query in name:
            score += 30
        for token in tokens:
            if token == name:
                score += 80
            elif token in name:
                score += 12
            if token in description:
                score += 5
            if token in params:
                score += 4
            if token in entry.search_text:
                score += 1
        return score

    def _score_skill_entry(self, entry: SkillCatalogEntry, query: str, tokens: List[str]) -> int:
        if not tokens:
            return 1
        score = 0
        name = entry.name.lower()
        description = entry.description.lower()
        category = entry.category.lower()
        ref = self.skill_ref(entry.name).lower()
        if query and query in {name, ref}:
            score += 95
        elif query and (query in name or query in ref):
            score += 28
        for token in tokens:
            if token == name or token == ref:
                score += 80
            elif token in name or token in ref:
                score += 12
            if token in description:
                score += 5
            if token in category:
                score += 4
            if token in entry.search_text:
                score += 1
        return score

    def _compact_entry(self, entry: ToolCatalogEntry, score: int) -> Dict[str, Any]:
        description = " ".join(entry.description.split())
        if len(description) > 120:
            description = description[:117].rstrip() + "..."
        return {
            "type": "tool",
            "name": entry.name,
            "description": description,
            "toolset": entry.toolset,
            "required": entry.required[:8],
            "parameters": entry.parameter_names[:12],
            "score": score,
        }

    def _compact_skill_entry(self, entry: SkillCatalogEntry, score: int) -> Dict[str, Any]:
        description = " ".join(entry.description.split())
        if len(description) > 120:
            description = description[:117].rstrip() + "..."
        return {
            "type": "skill",
            "name": self.skill_ref(entry.name),
            "skill_name": entry.name,
            "description": description,
            "category": entry.category,
            "load_with": "tool_router.execute",
            "score": score,
        }

    @staticmethod
    def skill_ref(skill_name: str) -> str:
        return f"skill:{skill_name}"

    def skill_for_ref(self, value: str) -> Optional[SkillCatalogEntry]:
        name = str(value or "").strip()
        if name.startswith("skill:"):
            return self.skills_by_ref.get(name)
        # Convenience for callers that pass a bare skill name and there is no
        # real tool with the same name.
        if name not in self.entries_by_name:
            return self.skills_by_ref.get(self.skill_ref(name))
        return None

    def is_skill_ref(self, value: str) -> bool:
        return self.skill_for_ref(value) is not None

    def status(self, *, exposed_tool_count: int = 1) -> Dict[str, Any]:
        return {
            "mode": "router",
            "exposed_tool_count": exposed_tool_count,
            "hidden_tool_count": len(self.entries),
            "hidden_skill_count": len(self.skill_entries),
            "catalog_hash": self.catalog_hash,
            "embedding_status": self.embedding_status,
        }


def parse_router_arguments_checked(value: Any) -> tuple[Dict[str, Any], Optional[str]]:
    """Normalize ``arguments`` and return a validation error when malformed."""
    if value is None:
        return {}, None
    if isinstance(value, dict):
        return value, None
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}, "arguments must be an object or a JSON object string."
        if isinstance(parsed, dict):
            return parsed, None
        return {}, "arguments JSON must decode to an object."
    return {}, "arguments must be an object or a JSON object string."


def parse_router_arguments(value: Any) -> Dict[str, Any]:
    """Normalize the router's nested ``arguments`` field."""
    parsed, _ = parse_router_arguments_checked(value)
    return parsed


def router_json(data: Mapping[str, Any]) -> str:
    return json.dumps(dict(data), ensure_ascii=False, separators=(",", ":"))
