---
name: x-account
description: Manage the authenticated X account. Search posts, inspect timelines, draft and publish posts, edit/delete own posts, reply, quote, like, repost, bookmark, follow, and unfollow.
version: 1.0.0
author: Adam
license: MIT
metadata:
  hermes:
    tags: [Social Media, Personal, X, Twitter, Publishing]
    related_skills: []
    requires_tools: [recall_memory, remember]
---

# X Account

Operate the authenticated X account through the bundled CLI. Use this skill
for account management, publishing, and social listening on X/Twitter.

The CLI is:

```bash
python ${HERMES_SKILL_DIR}/scripts/x_account.py <command> ...
```

## When to Use

Load this skill when the user asks to:

- Search X/Twitter, inspect posts, read mentions, or check the home timeline
- Post an update, reply, quote, edit, or delete the account's own post
- Like, unlike, repost, bookmark, follow, unfollow, or otherwise manage the
  authenticated account's normal public interactions
- Draft posts for X, review engagement, or monitor topics/accounts

Do not load for private messaging, paid ads, media uploads, analytics exports,
or non-X social platforms.

## Credentials

The script reads credentials from the environment. It supports either:

- OAuth 2 user token: `X_USER_ACCESS_TOKEN` or `X_API_USER_ACCESS_TOKEN`
- App bearer token for read-only search: `X_BEARER_TOKEN` or `X_API_BEARER_TOKEN`
- OAuth 1.0a user context: `X_API_KEY`, `X_API_SECRET`,
  `X_ACCESS_TOKEN`, `X_ACCESS_TOKEN_SECRET`

Never print, remember, or summarize credential values.

## Autonomy Rules

- Read/search commands are safe to run directly.
- Public side effects require clear user intent or an approved operating brief
  already stored in memory. Side effects include posting, editing, deleting,
  liking, reposting, bookmarking, following, and unfollowing.
- The CLI does not execute side effects unless `--yes` is passed. Without
  `--yes`, it returns a dry-run JSON plan.
- If intent is ambiguous, show the dry-run plan or draft and ask before using
  `--yes`.
- Do not automate spam, harassment, deceptive engagement, impersonation,
  credential disclosure, or bulk follow/like/repost loops.

## Quick Reference

| Intent | Command |
| --- | --- |
| Authenticated account | `python ${HERMES_SKILL_DIR}/scripts/x_account.py me` |
| Search recent posts | `python ${HERMES_SKILL_DIR}/scripts/x_account.py search "query lang:en -is:retweet" --limit 10` |
| Fetch one post | `python ${HERMES_SKILL_DIR}/scripts/x_account.py tweet <post_id>` |
| Own timeline | `python ${HERMES_SKILL_DIR}/scripts/x_account.py timeline --limit 10` |
| Mentions | `python ${HERMES_SKILL_DIR}/scripts/x_account.py mentions --limit 20` |
| Home timeline | `python ${HERMES_SKILL_DIR}/scripts/x_account.py home --limit 20` |
| Draft a post/update request | `python ${HERMES_SKILL_DIR}/scripts/x_account.py post --text "..."` |
| Publish a post/update | `python ${HERMES_SKILL_DIR}/scripts/x_account.py update --text "..." --yes` |
| Reply | `python ${HERMES_SKILL_DIR}/scripts/x_account.py post --reply-to <post_id> --text "..." --yes` |
| Quote | `python ${HERMES_SKILL_DIR}/scripts/x_account.py post --quote <post_id> --text "..." --yes` |
| Edit own post | `python ${HERMES_SKILL_DIR}/scripts/x_account.py edit <post_id> --text "..." --yes` |
| Delete own post | `python ${HERMES_SKILL_DIR}/scripts/x_account.py delete <post_id> --yes` |
| Like / unlike | `python ${HERMES_SKILL_DIR}/scripts/x_account.py like <post_id> --yes` |
| Repost / unrepost | `python ${HERMES_SKILL_DIR}/scripts/x_account.py repost <post_id> --yes` |
| Bookmark | `python ${HERMES_SKILL_DIR}/scripts/x_account.py bookmark <post_id> --yes` |
| Follow user | `python ${HERMES_SKILL_DIR}/scripts/x_account.py follow @username --yes` |

## Procedure

### 1. Read or Research

Use search/timeline/mentions/home commands. Summarize findings with post IDs,
authors, dates, and direct X URLs when possible:

```text
https://x.com/i/web/status/<post_id>
```

Prefer focused queries with X operators such as `lang:en`, `from:username`,
`to:username`, `@username`, `#tag`, `-is:retweet`, and `has:links`.

### 2. Draft Before Publishing

For new posts, replies, and quotes:

1. Draft the text in the user's requested voice.
2. Check for ambiguity, sensitive claims, accidental private information, and
   whether the user intended immediate publication.
3. If intent is clear, call the CLI with `--yes`; otherwise call without
   `--yes` and present the dry-run plan.

### 3. Execute Side Effects

After a side-effect command:

1. Inspect the returned JSON. Success requires `"ok": true`.
2. If a post was created or edited, fetch the returned post ID with `tweet`.
3. If the action failed, report the X error and do not claim success.
4. Record an audit memory with `remember`.

Use this memory shape:

```json
remember({
  "kind": "fact",
  "content": {
    "subject": "x_account:activity:<action>:<id_or_timestamp>",
    "subject_type": "thing",
    "predicate": "x_account_action",
    "object": "action=<action> target=<post_or_user_id> result=<summary>"
  },
  "tags": ["x_account", "x_account:activity", "x_action:<action>"],
  "confidence": 0.95
})
```

Do not store credentials in memory. For published posts, storing the post ID,
action, public text summary, and timestamp is okay.

### 4. Edit or Delete

Before editing or deleting, fetch the target post first and confirm it is the
intended account-owned post. X edits are time-limited and each edit creates a
new post ID; if editing fails because the edit window has closed, offer to
delete and repost only if the user explicitly wants that.

### 5. Follows and Engagement

Before following/unfollowing or engaging with a post, look up the user or post
and make sure it matches the user's intent. Avoid batches. For multi-account
curation, handle one account at a time and summarize before executing.

## Pitfalls

- Do not use `--yes` just because a command supports it. The user's intent or
  stored operating brief must justify public action.
- Do not infer that a post succeeded from a 2xx-looking transcript; inspect
  the CLI JSON.
- Do not treat read-only bearer tokens as sufficient for user-context actions.
  If X returns auth/scope errors, explain which credential scope appears to be
  missing.
- Do not promise true full-archive search unless the account has API access for
  it. The default `search` command is recent search.
- Do not bulk-like, bulk-follow, or mass-repost.

## Verification

Smoke test the CLI without touching X:

```bash
python ${HERMES_SKILL_DIR}/scripts/x_account.py post --text "dry run"
python ${HERMES_SKILL_DIR}/scripts/x_account.py like 1234567890
```

Both should return dry-run JSON with `confirmation_required: true`.

Credential smoke:

```bash
python ${HERMES_SKILL_DIR}/scripts/x_account.py me
python ${HERMES_SKILL_DIR}/scripts/x_account.py search "x api lang:en -is:retweet" --limit 10
```

Deploy via the standard NUC sync script, restart/reload the gateway, then run
the smoke tests on the NUC under the Hermes environment.
