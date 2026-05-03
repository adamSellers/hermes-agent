# Cos Memory Slice 0 Compatibility Note

Date: 2026-05-03

This note records the Slice 0 integration spike for the chief-of-staff
memory/context build.

## Confirmed Hermes Surfaces

- Context engine ABC: `agent/context_engine.py`
  - Required: `name`, `update_from_response()`, `should_compress()`, `compress()`.
  - Optional: `should_compress_preflight()`, `has_content_to_compress()`,
    session lifecycle hooks, tool schemas, tool dispatch, status, model update.
- Context engine discovery: `plugins/context_engine/<name>/`
  - Selected by `context.engine` in `config.yaml`.
  - Loader expects `__init__.py`; it can collect an engine via `register(ctx)`
    or instantiate a `ContextEngine` subclass.
- Memory provider ABC: `agent/memory_provider.py`
  - Required: `name`, `is_available()`, `initialize()`, `get_tool_schemas()`.
  - Useful hooks: `system_prompt_block()`, `prefetch()`, `queue_prefetch()`,
    `sync_turn()`, `on_session_switch()`, `on_pre_compress()`,
    `on_session_end()`, `on_memory_write()`.
- Memory provider discovery: `plugins/memory/<name>/`
  - Selected by `memory.provider` in `config.yaml`.
  - Loader expects `__init__.py`; it can collect a provider via `register(ctx)`
    or instantiate a `MemoryProvider` subclass.
- Memory tool routing: `agent/memory_manager.py` and `run_agent.py`
  - Provider tools are added to the model tool surface.
  - Tool calls are routed through `MemoryManager.handle_tool_call()`.
- Session storage: `hermes_state.py`
  - `state.db` has `sessions`, `messages`, `messages_fts`, and
    `messages_fts_trigram`.
  - `sessions.parent_session_id` stores compression/fork lineage.
  - `SessionDB._session_lineage_root_to_tip(session_id)` gives the current
    root-to-current chain.

## Important Constraints

- `MemoryProvider.sync_turn()` receives only `user_content`,
  `assistant_content`, and optional `session_id`; it does not receive full
  message objects or message IDs. `cos-memory` must derive message IDs from
  `state.db` best-effort or leave arrays empty.
- `MemoryProvider.handle_tool_call()` is called without `session_id` in the
  current run loop, so providers must track `_session_id` from `initialize()`,
  `sync_turn()`, and `on_session_switch()`.
- `MemoryProvider.on_session_end()` receives `messages` only, not a
  `session_id`.
- `on_pre_compress(messages)` returns text intended for the built-in
  compressor prompt. For `cos-memory`, it should remain cheap and should not
  perform unbounded extraction inline.
- Memory provider CLI registration currently exposes active provider CLIs as
  top-level plugin commands, not as `hermes memory <command>`. The v1 CLI
  requirement will need either a generic framework extension or a spec change.
- Context engines are selected by `context.engine`; they are not auto-enabled.
- Memory providers are selected by `memory.provider`; only one external
  provider can be active.

## Slice 1 Decision

Slice 1 proceeds as a bundled memory provider at:

`plugins/memory/cos-memory/`

It implements:

- `cos-memory.db` creation under `HERMES_HOME`.
- `session_turns` turn ledger.
- `memory_work_queue` schema placeholder.
- `digest_cache` schema for later cooperation with `cos-context`.
- FTS-only `recall_session` scoped to the current session lineage.

Embeddings, durable facts/preferences/commitments, extraction, consolidation,
and CLI curation are deferred to later slices.

