#!/usr/bin/env python3
"""Small dependency-free CLI for operating an authenticated X account.

The script supports OAuth 2.0 bearer/user tokens and OAuth 1.0a user context.
It intentionally returns JSON for every command so Hermes can inspect results
and decide what to do next.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable


API_BASE = os.environ.get("X_API_BASE_URL", "https://api.x.com/2").rstrip("/")

DEFAULT_TWEET_FIELDS = (
    "author_id,created_at,conversation_id,edit_controls,"
    "edit_history_tweet_ids,public_metrics,referenced_tweets,reply_settings,lang"
)
DEFAULT_USER_FIELDS = (
    "created_at,description,location,name,profile_image_url,protected,"
    "public_metrics,url,username,verified,verified_type"
)
DEFAULT_EXPANSIONS = "author_id,referenced_tweets.id,referenced_tweets.id.author_id"

USER_TOKEN_ENV = (
    "X_USER_ACCESS_TOKEN",
    "X_API_USER_ACCESS_TOKEN",
    "TWITTER_USER_ACCESS_TOKEN",
)
BEARER_ENV = USER_TOKEN_ENV + (
    "X_BEARER_TOKEN",
    "X_API_BEARER_TOKEN",
    "TWITTER_BEARER_TOKEN",
    "BEARER_TOKEN",
)
OAUTH1_ENV = {
    "consumer_key": ("X_API_KEY", "X_CONSUMER_KEY", "TWITTER_API_KEY", "TWITTER_CONSUMER_KEY"),
    "consumer_secret": (
        "X_API_SECRET",
        "X_CONSUMER_SECRET",
        "TWITTER_API_SECRET",
        "TWITTER_CONSUMER_SECRET",
    ),
    "access_token": ("X_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN"),
    "access_secret": ("X_ACCESS_TOKEN_SECRET", "TWITTER_ACCESS_TOKEN_SECRET"),
}


class CliError(Exception):
    pass


def env_first(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value.strip()
    return None


def oauth1_credentials() -> dict[str, str] | None:
    values: dict[str, str] = {}
    for key, names in OAUTH1_ENV.items():
        value = env_first(names)
        if not value:
            return None
        values[key] = value
    return values


def flatten_query(query: dict[str, Any] | None) -> list[tuple[str, str]]:
    if not query:
        return []
    pairs: list[tuple[str, str]] = []
    for key, value in query.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                pairs.append((str(key), str(item)))
        else:
            pairs.append((str(key), str(value)))
    return pairs


def pct(value: str) -> str:
    return urllib.parse.quote(value, safe="~")


def build_url(path: str, query: dict[str, Any] | None = None) -> str:
    if path.startswith("http://") or path.startswith("https://"):
        base = path
    else:
        base = f"{API_BASE}/{path.lstrip('/')}"
    pairs = flatten_query(query)
    if pairs:
        return f"{base}?{urllib.parse.urlencode(pairs)}"
    return base


def oauth1_header(method: str, url: str, query: dict[str, Any] | None) -> str:
    creds = oauth1_credentials()
    if not creds:
        raise CliError(
            "OAuth 1.0a requested but credentials are incomplete. Set "
            "X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, and X_ACCESS_TOKEN_SECRET."
        )

    oauth_params = {
        "oauth_consumer_key": creds["consumer_key"],
        "oauth_nonce": secrets.token_hex(16),
        "oauth_signature_method": "HMAC-SHA1",
        "oauth_timestamp": str(int(time.time())),
        "oauth_token": creds["access_token"],
        "oauth_version": "1.0",
    }
    split = urllib.parse.urlsplit(url)
    base_url = urllib.parse.urlunsplit((split.scheme, split.netloc, split.path, "", ""))
    signature_params = flatten_query(query) + list(oauth_params.items())
    encoded_params = [(pct(k), pct(v)) for k, v in signature_params]
    encoded_params.sort()
    param_string = "&".join(f"{k}={v}" for k, v in encoded_params)
    base_string = "&".join([pct(method.upper()), pct(base_url), pct(param_string)])
    signing_key = f"{pct(creds['consumer_secret'])}&{pct(creds['access_secret'])}"
    signature = hmac.new(signing_key.encode(), base_string.encode(), hashlib.sha1).digest()
    oauth_params["oauth_signature"] = base64.b64encode(signature).decode("ascii")
    return "OAuth " + ", ".join(
        f'{pct(k)}="{pct(v)}"' for k, v in sorted(oauth_params.items())
    )


def choose_auth(method: str, auth_mode: str, require_user: bool) -> tuple[str, str]:
    if auth_mode == "oauth1":
        return ("oauth1", "")
    if auth_mode == "bearer":
        token = env_first(BEARER_ENV)
        if not token:
            raise CliError("Bearer auth requested but no X bearer/user token env var is set.")
        return ("bearer", token)

    user_token = env_first(USER_TOKEN_ENV)
    any_bearer = env_first(BEARER_ENV)
    has_oauth1 = oauth1_credentials() is not None

    if require_user:
        if user_token:
            return ("bearer", user_token)
        if has_oauth1:
            return ("oauth1", "")
        if any_bearer:
            return ("bearer", any_bearer)
    else:
        if any_bearer:
            return ("bearer", any_bearer)
        if has_oauth1:
            return ("oauth1", "")

    raise CliError(
        "No usable X credentials found. Set X_USER_ACCESS_TOKEN or X_BEARER_TOKEN, "
        "or set OAuth1 env vars: X_API_KEY, X_API_SECRET, X_ACCESS_TOKEN, "
        "X_ACCESS_TOKEN_SECRET."
    )


def api_request(
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    auth_mode: str = "auto",
    require_user: bool = False,
) -> dict[str, Any]:
    url = build_url(path, query)
    headers = {
        "Accept": "application/json",
        "User-Agent": "Hermes-X-Account-Skill/1.0",
    }
    body_bytes = None
    if body is not None:
        body_bytes = json.dumps(body, separators=(",", ":")).encode()
        headers["Content-Type"] = "application/json"

    kind, token = choose_auth(method, auth_mode, require_user)
    if kind == "oauth1":
        headers["Authorization"] = oauth1_header(method, url, query)
    else:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw else {}
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "rate_limit": rate_headers(response.headers),
                "body": parsed,
            }
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed: Any = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return {
            "ok": False,
            "status": exc.code,
            "rate_limit": rate_headers(exc.headers),
            "body": parsed,
        }
    except urllib.error.URLError as exc:
        raise CliError(f"Network error calling X API: {exc.reason}") from exc


def rate_headers(headers: Any) -> dict[str, str]:
    keys = {
        "x-rate-limit-limit": "limit",
        "x-rate-limit-remaining": "remaining",
        "x-rate-limit-reset": "reset",
    }
    return {label: headers[name] for name, label in keys.items() if name in headers}


def emit(result: dict[str, Any], compact: bool = False) -> int:
    print(json.dumps(result, indent=None if compact else 2, sort_keys=True))
    return 0 if result.get("ok", False) else 1


def plan_or_execute(
    args: argparse.Namespace,
    method: str,
    path: str,
    *,
    query: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
    require_user: bool = True,
) -> dict[str, Any]:
    if args.dry_run or not args.yes:
        return {
            "ok": True,
            "confirmation_required": not args.yes,
            "dry_run": True,
            "method": method,
            "path": path,
            "query": query or {},
            "body": body or {},
            "note": "Re-run with --yes to execute this X account side effect.",
        }
    return api_request(method, path, query=query, body=body, auth_mode=args.auth, require_user=require_user)


def common_tweet_query(max_results: int | None = None) -> dict[str, Any]:
    query: dict[str, Any] = {
        "tweet.fields": DEFAULT_TWEET_FIELDS,
        "user.fields": DEFAULT_USER_FIELDS,
        "expansions": DEFAULT_EXPANSIONS,
    }
    if max_results is not None:
        query["max_results"] = str(max(10, min(100, max_results)))
    return query


def me_id(args: argparse.Namespace) -> str:
    response = api_request(
        "GET",
        "/users/me",
        query={"user.fields": DEFAULT_USER_FIELDS},
        auth_mode=args.auth,
        require_user=True,
    )
    if not response.get("ok"):
        raise CliError(f"Could not resolve authenticated user id: {json.dumps(response)}")
    return str(response["body"]["data"]["id"])


def resolve_user(ref: str, args: argparse.Namespace, *, execute: bool = True) -> str:
    cleaned = ref.strip().lstrip("@")
    if cleaned.isdigit():
        return cleaned
    if not execute:
        return f"{{user_id_for_{cleaned}}}"
    response = api_request(
        "GET",
        f"/users/by/username/{urllib.parse.quote(cleaned)}",
        query={"user.fields": DEFAULT_USER_FIELDS},
        auth_mode=args.auth,
    )
    if not response.get("ok"):
        raise CliError(f"Could not resolve user @{cleaned}: {json.dumps(response)}")
    return str(response["body"]["data"]["id"])


def cmd_me(args: argparse.Namespace) -> dict[str, Any]:
    return api_request(
        "GET",
        "/users/me",
        query={"user.fields": DEFAULT_USER_FIELDS},
        auth_mode=args.auth,
        require_user=True,
    )


def cmd_lookup(args: argparse.Namespace) -> dict[str, Any]:
    ref = args.user.strip().lstrip("@")
    if ref.isdigit():
        path = f"/users/{ref}"
    else:
        path = f"/users/by/username/{urllib.parse.quote(ref)}"
    return api_request("GET", path, query={"user.fields": DEFAULT_USER_FIELDS}, auth_mode=args.auth)


def cmd_tweet(args: argparse.Namespace) -> dict[str, Any]:
    query = common_tweet_query()
    return api_request("GET", f"/tweets/{args.tweet_id}", query=query, auth_mode=args.auth)


def cmd_search(args: argparse.Namespace) -> dict[str, Any]:
    query = common_tweet_query(args.limit)
    query["query"] = args.query
    if args.sort_order:
        query["sort_order"] = args.sort_order
    if args.next_token:
        query["next_token"] = args.next_token
    return api_request("GET", "/tweets/search/recent", query=query, auth_mode=args.auth)


def cmd_timeline(args: argparse.Namespace) -> dict[str, Any]:
    user_id = args.user_id or (resolve_user(args.username, args) if args.username else me_id(args))
    query = common_tweet_query(args.limit)
    if args.exclude:
        query["exclude"] = ",".join(args.exclude)
    if args.pagination_token:
        query["pagination_token"] = args.pagination_token
    return api_request("GET", f"/users/{user_id}/tweets", query=query, auth_mode=args.auth)


def cmd_mentions(args: argparse.Namespace) -> dict[str, Any]:
    user_id = args.user_id or me_id(args)
    query = common_tweet_query(args.limit)
    if args.pagination_token:
        query["pagination_token"] = args.pagination_token
    return api_request("GET", f"/users/{user_id}/mentions", query=query, auth_mode=args.auth)


def cmd_home(args: argparse.Namespace) -> dict[str, Any]:
    user_id = args.user_id or me_id(args)
    query = common_tweet_query(args.limit)
    if args.pagination_token:
        query["pagination_token"] = args.pagination_token
    return api_request(
        "GET",
        f"/users/{user_id}/timelines/reverse_chronological",
        query=query,
        auth_mode=args.auth,
        require_user=True,
    )


def cmd_post(args: argparse.Namespace) -> dict[str, Any]:
    body: dict[str, Any] = {"text": args.text}
    if args.reply_to:
        body["reply"] = {"in_reply_to_tweet_id": args.reply_to}
    if args.quote:
        body["quote_tweet_id"] = args.quote
    if args.reply_settings:
        body["reply_settings"] = args.reply_settings
    if args.made_with_ai:
        body["made_with_ai"] = True
    return plan_or_execute(args, "POST", "/tweets", body=body)


def cmd_edit(args: argparse.Namespace) -> dict[str, Any]:
    body = {
        "text": args.text,
        "edit_options": {"previous_post_id": args.tweet_id},
    }
    return plan_or_execute(args, "POST", "/tweets", body=body)


def cmd_delete(args: argparse.Namespace) -> dict[str, Any]:
    return plan_or_execute(args, "DELETE", f"/tweets/{args.tweet_id}")


def cmd_like(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "POST", f"/users/{user_id}/likes", body={"tweet_id": args.tweet_id})


def cmd_unlike(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "DELETE", f"/users/{user_id}/likes/{args.tweet_id}")


def cmd_repost(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "POST", f"/users/{user_id}/retweets", body={"tweet_id": args.tweet_id})


def cmd_unrepost(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "DELETE", f"/users/{user_id}/retweets/{args.tweet_id}")


def cmd_bookmarks(args: argparse.Namespace) -> dict[str, Any]:
    user_id = args.user_id or me_id(args)
    query = common_tweet_query(args.limit)
    if args.pagination_token:
        query["pagination_token"] = args.pagination_token
    return api_request("GET", f"/users/{user_id}/bookmarks", query=query, auth_mode=args.auth, require_user=True)


def cmd_bookmark(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "POST", f"/users/{user_id}/bookmarks", body={"tweet_id": args.tweet_id})


def cmd_unbookmark(args: argparse.Namespace) -> dict[str, Any]:
    user_id = me_id(args) if args.yes and not args.dry_run else "{authenticated_user_id}"
    return plan_or_execute(args, "DELETE", f"/users/{user_id}/bookmarks/{args.tweet_id}")


def cmd_follow(args: argparse.Namespace) -> dict[str, Any]:
    execute = args.yes and not args.dry_run
    user_id = me_id(args) if execute else "{authenticated_user_id}"
    target_id = resolve_user(args.user, args, execute=execute)
    return plan_or_execute(args, "POST", f"/users/{user_id}/following", body={"target_user_id": target_id})


def cmd_unfollow(args: argparse.Namespace) -> dict[str, Any]:
    execute = args.yes and not args.dry_run
    user_id = me_id(args) if execute else "{authenticated_user_id}"
    target_id = resolve_user(args.user, args, execute=execute)
    return plan_or_execute(args, "DELETE", f"/users/{user_id}/following/{target_id}")


def add_write_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yes", action="store_true", help="Execute the side effect.")
    parser.add_argument("--dry-run", action="store_true", help="Show request without executing.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Operate an authenticated X account.")
    parser.add_argument("--auth", choices=["auto", "bearer", "oauth1"], default="auto")
    parser.add_argument("--compact", action="store_true", help="Print compact JSON.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("me", help="Show authenticated user.")
    p.set_defaults(func=cmd_me)

    p = sub.add_parser("lookup", help="Look up a user by @username or ID.")
    p.add_argument("user")
    p.set_defaults(func=cmd_lookup)

    p = sub.add_parser("tweet", help="Fetch one Post by ID.")
    p.add_argument("tweet_id")
    p.set_defaults(func=cmd_tweet)

    p = sub.add_parser("search", help="Search recent public Posts.")
    p.add_argument("query")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--sort-order", choices=["recency", "relevancy"])
    p.add_argument("--next-token")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("timeline", help="Fetch a user timeline.")
    p.add_argument("--user-id")
    p.add_argument("--username")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--exclude", action="append", choices=["retweets", "replies"])
    p.add_argument("--pagination-token")
    p.set_defaults(func=cmd_timeline)

    p = sub.add_parser("mentions", help="Fetch mentions for authenticated user or --user-id.")
    p.add_argument("--user-id")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--pagination-token")
    p.set_defaults(func=cmd_mentions)

    p = sub.add_parser("home", help="Fetch authenticated user's reverse chronological home timeline.")
    p.add_argument("--user-id")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--pagination-token")
    p.set_defaults(func=cmd_home)

    p = sub.add_parser("post", help="Create a new Post, reply, or quote.")
    p.add_argument("--text", required=True)
    p.add_argument("--reply-to")
    p.add_argument("--quote")
    p.add_argument("--reply-settings", choices=["everyone", "mentionedUsers", "following", "verified"])
    p.add_argument("--made-with-ai", action="store_true")
    add_write_flags(p)
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("update", help="Alias for post; publish an account update.")
    p.add_argument("--text", required=True)
    p.add_argument("--reply-to")
    p.add_argument("--quote")
    p.add_argument("--reply-settings", choices=["everyone", "mentionedUsers", "following", "verified"])
    p.add_argument("--made-with-ai", action="store_true")
    add_write_flags(p)
    p.set_defaults(func=cmd_post)

    p = sub.add_parser("edit", help="Edit a recent editable Post.")
    p.add_argument("tweet_id")
    p.add_argument("--text", required=True)
    add_write_flags(p)
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("delete", help="Delete a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_delete)

    p = sub.add_parser("like", help="Like a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_like)

    p = sub.add_parser("unlike", help="Unlike a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_unlike)

    p = sub.add_parser("repost", help="Repost a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_repost)

    p = sub.add_parser("unrepost", help="Remove your repost of a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_unrepost)

    p = sub.add_parser("bookmarks", help="List authenticated user's bookmarks.")
    p.add_argument("--user-id")
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--pagination-token")
    p.set_defaults(func=cmd_bookmarks)

    p = sub.add_parser("bookmark", help="Bookmark a Post.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_bookmark)

    p = sub.add_parser("unbookmark", help="Remove a bookmark.")
    p.add_argument("tweet_id")
    add_write_flags(p)
    p.set_defaults(func=cmd_unbookmark)

    p = sub.add_parser("follow", help="Follow a user by @username or ID.")
    p.add_argument("user")
    add_write_flags(p)
    p.set_defaults(func=cmd_follow)

    p = sub.add_parser("unfollow", help="Unfollow a user by @username or ID.")
    p.add_argument("user")
    add_write_flags(p)
    p.set_defaults(func=cmd_unfollow)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return emit(args.func(args), compact=args.compact)
    except CliError as exc:
        return emit({"ok": False, "error": str(exc)}, compact=args.compact)


if __name__ == "__main__":
    sys.exit(main())
