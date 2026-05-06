"""Chief-of-staff context engine.

Deployable MVP:
- early threshold compression
- hot-zone verbatim turns
- warm-zone deterministic per-turn digests
- cold-zone drop from live context

The engine deliberately avoids synchronous auxiliary LLM calls during
compression. Richer digest generation can populate the same in-process/cache
path later without blocking a user-facing turn.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple

from agent.context_engine import ContextEngine


MARKER = "[COMPRESSED CONVERSATION HISTORY"
END_MARKER = "[END COMPRESSED CONVERSATION HISTORY]"
DEFAULT_THRESHOLD_TOKENS = 24000
DEFAULT_THRESHOLD_PERCENT = 0.10
DEFAULT_HOT_TOOL_MAX_CHARS = 6000
DEFAULT_HOT_TOOL_HEAD_CHARS = 3500
DEFAULT_HOT_TOOL_TAIL_CHARS = 1000


def _coerce_int(value: Any, default: int, *, minimum: int = 1) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def _coerce_float(value: Any, default: float, *, minimum: float = 0.01, maximum: float = 1.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _load_cos_context_settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
    except Exception:
        cfg = {}
    context_cfg = cfg.get("context") if isinstance(cfg, dict) else {}
    if not isinstance(context_cfg, dict):
        context_cfg = {}
    settings = context_cfg.get("cos_context") or context_cfg.get("cos-context") or {}
    return settings if isinstance(settings, dict) else {}


class CosContextEngine(ContextEngine):
    def __init__(self) -> None:
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        settings = _load_cos_context_settings()
        self.threshold_ceiling_tokens = _coerce_int(
            os.getenv("HERMES_COS_CONTEXT_THRESHOLD_TOKENS")
            or settings.get("threshold_tokens"),
            DEFAULT_THRESHOLD_TOKENS,
            minimum=4000,
        )
        self.threshold_tokens = self.threshold_ceiling_tokens
        self.context_length = 256000
        self.compression_count = 0
        self.threshold_percent = _coerce_float(
            os.getenv("HERMES_COS_CONTEXT_THRESHOLD")
            or settings.get("threshold"),
            DEFAULT_THRESHOLD_PERCENT,
            minimum=0.02,
            maximum=0.50,
        )
        self.protect_first_n = 1
        self.protect_last_n = 20
        self.hot_zone_turns = _coerce_int(settings.get("hot_zone_turns"), 10)
        self.warm_zone_turns = _coerce_int(settings.get("warm_zone_turns"), 20)
        self.digest_max_tokens = _coerce_int(settings.get("digest_max_tokens"), 250)
        self.hot_tool_max_chars = _coerce_int(
            os.getenv("HERMES_COS_CONTEXT_HOT_TOOL_MAX_CHARS")
            or settings.get("hot_tool_max_chars"),
            DEFAULT_HOT_TOOL_MAX_CHARS,
            minimum=1000,
        )
        self.hot_tool_head_chars = _coerce_int(
            settings.get("hot_tool_head_chars"),
            DEFAULT_HOT_TOOL_HEAD_CHARS,
            minimum=500,
        )
        self.hot_tool_tail_chars = _coerce_int(
            settings.get("hot_tool_tail_chars"),
            DEFAULT_HOT_TOOL_TAIL_CHARS,
            minimum=0,
        )
        self.hot_zone_turns_current = 0
        self.warm_zone_turns_current = 0
        self.cold_zone_turns_total = 0
        self.hot_tool_compactions_count = 0
        self.last_digest_latency_ms = 0
        self.digest_failures_count = 0
        self.sync_digest_fallbacks_count = 0
        self.recall_session_available = False

    @property
    def name(self) -> str:
        return "cos-context"

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        self.context_length = context_length or self.context_length
        self.threshold_tokens = min(
            self.threshold_ceiling_tokens,
            int(self.context_length * self.threshold_percent),
        )

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(usage.get("total_tokens") or 0)

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens is not None else self.last_prompt_tokens
        if tokens and tokens >= self.threshold_tokens:
            return True
        return self._count_live_user_turns(getattr(self, "_last_messages", [])) > self.hot_zone_turns

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        self._last_messages = messages
        if self._estimate_tokens(messages) >= self.threshold_tokens:
            return True
        return self._count_live_user_turns(messages) > self.hot_zone_turns

    def has_content_to_compress(self, messages: List[Dict[str, Any]]) -> bool:
        return self._count_live_user_turns(messages) > self.hot_zone_turns

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
        focus_topic: str = None,
    ) -> List[Dict[str, Any]]:
        self._last_messages = messages
        start = time.time()
        messages, hot_tool_compactions = self._compact_hot_tool_outputs_if_needed(
            messages,
            current_tokens=current_tokens,
        )
        system_messages, existing_digest_entries, body = self._split_messages(messages)
        turns = self._group_turns(body)
        if len(turns) <= self.hot_zone_turns:
            if hot_tool_compactions:
                self.hot_zone_turns_current = len(turns)
                self.warm_zone_turns_current = len(existing_digest_entries)
                self.hot_tool_compactions_count += hot_tool_compactions
                self.last_digest_latency_ms = int((time.time() - start) * 1000)
                self.compression_count += 1
                return messages
            return messages

        warm_candidates = turns[:-self.hot_zone_turns]
        hot_turns = turns[-self.hot_zone_turns:]
        new_entries = [self._digest_turn(turn_id, turn) for turn_id, turn in warm_candidates]
        all_entries = [*existing_digest_entries, *new_entries]
        if len(all_entries) > self.warm_zone_turns:
            self.cold_zone_turns_total += len(all_entries) - self.warm_zone_turns
            all_entries = all_entries[-self.warm_zone_turns:]

        compressed_block = self._render_digest_block(all_entries) if all_entries else None
        new_messages: List[Dict[str, Any]] = list(system_messages)
        if compressed_block:
            new_messages.append({"role": "system", "content": compressed_block})
        for _turn_id, turn in hot_turns:
            new_messages.extend(turn)

        self.hot_zone_turns_current = len(hot_turns)
        self.warm_zone_turns_current = len(all_entries)
        self.hot_tool_compactions_count += hot_tool_compactions
        self.last_digest_latency_ms = int((time.time() - start) * 1000)
        self.compression_count += 1
        return new_messages

    def _compact_hot_tool_outputs_if_needed(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: int = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        estimated = int(current_tokens or self._estimate_tokens(messages))
        if estimated < self.threshold_tokens:
            return messages, 0

        compacted: List[Dict[str, Any]] = []
        count = 0
        for msg in messages:
            if msg.get("role") != "tool":
                compacted.append(msg)
                continue
            content = msg.get("content")
            if not isinstance(content, str) or len(content) <= self.hot_tool_max_chars:
                compacted.append(msg)
                continue
            updated = dict(msg)
            updated["content"] = self._compact_tool_text(content)
            compacted.append(updated)
            count += 1
        return compacted, count

    def _compact_tool_text(self, text: str) -> str:
        head = text[: self.hot_tool_head_chars].rstrip()
        tail = ""
        if self.hot_tool_tail_chars:
            tail = text[-self.hot_tool_tail_chars :].lstrip()
        omitted = max(0, len(text) - len(head) - len(tail))
        parts = [
            "[Large tool output compacted by cos-context]",
            f"Original size: {len(text):,} characters.",
            f"Omitted middle: {omitted:,} characters.",
            "",
            "BEGIN PRESERVED HEAD",
            head,
        ]
        if tail:
            parts.extend(["", "BEGIN PRESERVED TAIL", tail])
        return "\n".join(parts)

    def _estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            total += max(1, len(content) // 4)
            if msg.get("tool_calls"):
                total += len(json.dumps(msg.get("tool_calls"), ensure_ascii=False)) // 4
        return total

    def _count_live_user_turns(self, messages: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for msg in messages
            if msg.get("role") == "user" and not self._is_compressed_block(msg)
        )

    def _split_messages(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[str], List[Dict[str, Any]]]:
        system_messages: List[Dict[str, Any]] = []
        digest_entries: List[str] = []
        body: List[Dict[str, Any]] = []
        for msg in messages:
            if self._is_compressed_block(msg):
                digest_entries.extend(self._parse_digest_entries(str(msg.get("content") or "")))
                continue
            if msg.get("role") == "system" and not body:
                system_messages.append(msg)
                continue
            body.append(msg)
        return system_messages, digest_entries, body

    def _is_compressed_block(self, msg: Dict[str, Any]) -> bool:
        content = msg.get("content")
        return isinstance(content, str) and content.startswith(MARKER)

    def _parse_digest_entries(self, text: str) -> List[str]:
        entries: List[str] = []
        current: List[str] = []
        for line in text.splitlines():
            if line.startswith(MARKER) or line.startswith(END_MARKER) or not line.strip():
                continue
            if line.startswith("Turn ") and current:
                entries.append("\n".join(current).strip())
                current = [line]
            else:
                current.append(line)
        if current:
            entries.append("\n".join(current).strip())
        return [e for e in entries if e]

    def _group_turns(self, messages: List[Dict[str, Any]]) -> List[Tuple[int, List[Dict[str, Any]]]]:
        turns: List[Tuple[int, List[Dict[str, Any]]]] = []
        current: List[Dict[str, Any]] = []
        turn_id = 0
        for msg in messages:
            if msg.get("role") == "user":
                if current:
                    turns.append((turn_id, current))
                turn_id += 1
                current = [msg]
            else:
                if not current:
                    turn_id += 1
                current.append(msg)
        if current:
            turns.append((turn_id, current))
        return turns

    def _digest_turn(self, turn_id: int, turn: List[Dict[str, Any]]) -> str:
        self.sync_digest_fallbacks_count += 1
        user_text = ""
        actions: List[str] = []
        outcome = ""
        for msg in turn:
            role = msg.get("role")
            content = self._string_content(msg.get("content"))
            if role == "user" and not user_text:
                user_text = content
            elif role == "assistant" and content:
                outcome = content
            elif role == "tool":
                actions.append(msg.get("tool_name") or "tool")

        intent = self._shorten(user_text, 180) or "(no user text)"
        outcome_text = self._shorten(outcome, 180)
        action_text = ", ".join(sorted(set(actions))) if actions else "none recorded"
        lines = [
            f"Turn {turn_id} - {self._tag(intent)}",
            f"- Intent: {intent}",
            f"- Actions: {action_text}",
        ]
        if outcome_text:
            lines.append(f"- Outcome: {outcome_text}")
        return "\n".join(lines)

    def _render_digest_block(self, entries: List[str]) -> str:
        return "\n\n".join([
            f"{MARKER} - latest {len(entries)} warm turns]",
            *entries,
            END_MARKER,
        ])

    def _string_content(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)

    def _shorten(self, text: str, max_chars: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 1].rstrip() + "..."

    def _tag(self, text: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", text.lower())[:3]
        return "-".join(words) if words else "conversation"

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "hot_zone_turns_current": self.hot_zone_turns_current,
            "warm_zone_turns_current": self.warm_zone_turns_current,
            "cold_zone_turns_total": self.cold_zone_turns_total,
            "hot_tool_compactions_count": self.hot_tool_compactions_count,
            "last_digest_latency_ms": self.last_digest_latency_ms,
            "digest_failures_count": self.digest_failures_count,
            "sync_digest_fallbacks_count": self.sync_digest_fallbacks_count,
            "recall_session_available": self.recall_session_available,
        })
        return status

    def on_session_start(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id

    def on_session_reset(self) -> None:
        super().on_session_reset()
        self.hot_zone_turns_current = 0
        self.warm_zone_turns_current = 0
        self.hot_tool_compactions_count = 0


def register(ctx) -> None:
    ctx.register_context_engine(CosContextEngine())
