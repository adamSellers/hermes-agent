import sqlite3

from agent.tool_router_telemetry import record_router_event, summarize_router_events


def test_record_router_event_writes_compact_row(tmp_path):
    db_path = tmp_path / "router.db"

    record_router_event(
        {
            "session_id": "session-1",
            "tool_call_id": "call-1",
            "action": "search",
            "query": "weather forecast near me",
            "success": True,
            "outcome": "search_results",
            "match_count": 1,
            "matches": [{"type": "skill", "name": "skill:weather-lookup", "score": 42}],
            "catalog_hash": "abc123",
            "hidden_tool_count": 4,
            "hidden_skill_count": 2,
            "model": "gemma",
            "provider": "custom",
            "platform": "cli",
        },
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as con:
        row = con.execute(
            """
            SELECT action, query, outcome, match_count, matches_json,
                   hidden_tool_count, hidden_skill_count
            FROM tool_router_events
            """
        ).fetchone()

    assert row[0] == "search"
    assert row[1] == "weather forecast near me"
    assert row[2] == "search_results"
    assert row[3] == 1
    assert "weather-lookup" in row[4]
    assert row[5] == 4
    assert row[6] == 2


def test_summarize_router_events_surfaces_unsatisfied_requests(tmp_path):
    db_path = tmp_path / "router.db"
    record_router_event(
        {
            "action": "execute",
            "requested_name": "missing_calendar_tool",
            "success": False,
            "outcome": "unknown_target",
            "error": "Unknown routed tool",
        },
        db_path=db_path,
    )
    record_router_event(
        {
            "action": "execute",
            "requested_name": "skill:weather-lookup",
            "resolved_name": "skill:weather-lookup",
            "target_type": "skill",
            "target_tool": "skill_view",
            "success": True,
            "outcome": "execute_success",
        },
        db_path=db_path,
    )

    summary = summarize_router_events(db_path=db_path)

    assert summary["success"] is True
    assert summary["event_count"] == 2
    assert summary["unsatisfied"][0]["request"] == "missing_calendar_tool"
    assert summary["unsatisfied"][0]["outcome"] == "unknown_target"
    assert summary["top_targets"][0]["target_type"] == "skill"
    assert summary["top_targets"][0]["target"] == "skill:weather-lookup"
