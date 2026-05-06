# cos-memory Deployment

`cos-memory` is a local-first durable memory provider paired with the
`cos-context` context engine.

## Activate

Use the setup flow:

```bash
hermes memory setup cos-memory
```

Or run `hermes memory setup` and select `cos-memory` interactively. The
provider setup enables:

```yaml
memory:
  provider: cos-memory
context:
  engine: cos-context
```

Start a new Hermes session after activation so the agent loads the provider
and context engine at startup.

## Verify

```bash
hermes memory status
hermes memory stats --json
hermes memory queue --json
hermes memory doctor --json
```

When `memory.provider` is `cos-memory`, these provider commands are available:

```bash
hermes memory list
hermes memory show fact:123
hermes memory search "project name"
hermes memory commitments
hermes memory briefing
hermes memory review
hermes memory consolidate
hermes memory queue --json
hermes memory doctor --json
hermes memory rebuild-vectors
hermes memory export --out cos-memory-export.json
hermes memory import --in cos-memory-export.json
```

The active provider also exposes a top-level convenience command:

```bash
hermes cos-memory stats
```

## Runtime Behavior

- `sync_turn` writes a profile-scoped turn ledger to `cos-memory.db` and
  queues background extraction work without blocking the user turn.
- Background extraction first uses the auxiliary `memory_extraction` LLM task
  to produce strict JSON entity/fact/preference/commitment candidates, then
  falls back to conservative deterministic extraction if that auxiliary call
  is unavailable.
- `recall_session(query)` searches the current session lineage through
  Hermes `state.db` FTS, then blends in SQLite-backed semantic turn vectors
  when the local embedding endpoint is configured.
- `recall_memory(query)` searches curated durable entities, facts,
  preferences, commitments, and relations with hybrid lexical/semantic
  ranking when vectors are available.
- `remember(...)` and `forget(...)` give the running agent typed memory
  curation tools.
- `cos-context` keeps recent hot turns verbatim, digests warm turns, and drops
  cold turns from live context once they are represented in compressed history.
- Tool-router mode can keep the model-visible tool surface small while still
  allowing the hidden memory tools and skills to be discovered and executed.
- Private memory-backed skills currently include `shopping-list` and
  `x-account`; they are deployed by the wrapper repo sync script.

## Current MVP Boundaries

- The v1 provider is SQLite-only. There is no external vector service.
  Embeddings are stored as JSON vectors inside `cos-memory.db` and can be
  rebuilt with `hermes memory rebuild-vectors`.
- Automatic extraction uses the configured local auxiliary LLM for structured
  extraction. Deterministic rules are retained only as a fallback.
- Sensitive-looking values are not stored by heuristic extraction unless the
  user explicitly asks Hermes to remember them.

## Wrapper Repo Deploy

From the Mac Studio wrapper repo:

```bash
./scripts/deploy_cos_memory_to_bot.sh --test
./scripts/smoke_cos_memory_on_bot.sh
```

The deploy script syncs the selected Hermes memory/context/router paths and
private skills to `~/.hermes/hermes-agent` on `oakley@bot.oakroad`. Restart the
gateway only after code sync, activation, and smoke checks are complete.

## Local LLM Extraction

On the NUC, point the `memory_extraction` auxiliary task at the Mac Studio
OpenAI-compatible inference endpoint, or leave it as `auto` when the main
Hermes model already resolves to that endpoint:

```yaml
auxiliary:
  memory_extraction:
    provider: custom
    model: gemma-4-31b-it-4bit
    base_url: http://<mac-studio-host>:<port>/v1
    api_key: ""
    timeout: 45
    extra_body: {}
  memory_embedding:
    provider: custom
    model: nomicai-modernbert-embed-base-bf16
    base_url: http://<mac-studio-host>:<port>/v1
    api_key: ""
    timeout: 30
    extra_body: {}
```

If `memory_embedding.base_url` is blank, `cos-memory` will try to reuse the
`memory_extraction` endpoint and then the active custom Hermes model endpoint.
Set it explicitly when the embedding server or port differs from chat
inference.

To backfill vectors for rows captured before embeddings were enabled:

```bash
hermes memory rebuild-vectors
```

After changing config on the NUC, restart the gateway so the running agent
loads the updated task routing:

```bash
sudo systemctl restart hermes-gateway
```
