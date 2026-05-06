import json

from agent.tool_router import ToolRouter, parse_router_arguments, parse_router_arguments_checked


def _tool(name, description, properties=None, required=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
            },
        },
    }


def test_router_search_uses_lexical_fallback_without_embeddings():
    router = ToolRouter(
        [
            _tool("web_search", "Search the web for current information", {"q": {"type": "string"}}),
            _tool("terminal", "Run shell commands", {"command": {"type": "string"}}),
        ],
        embedding_index=False,
    )

    result = router.search("current web lookup")

    assert result["success"] is True
    assert result["embedding_status"] == "disabled"
    assert result["matches"][0]["name"] == "web_search"


def test_router_search_returns_compact_matches():
    router = ToolRouter(
        [
            _tool(
                "verbose_tool",
                " ".join(["long description"] * 40),
                {f"param_{idx}": {"type": "string"} for idx in range(30)},
            )
        ],
        embedding_index=False,
    )

    result = router.search("verbose")
    match = result["matches"][0]

    assert len(match["description"]) <= 120
    assert len(match["parameters"]) == 12


def test_router_search_includes_skills():
    router = ToolRouter(
        [_tool("terminal", "Run shell commands", {"command": {"type": "string"}})],
        skill_entries=[
            {
                "name": "weather-lookup",
                "description": "Resolve local weather forecasts and nearby location context.",
                "category": "research",
            }
        ],
        embedding_index=False,
    )

    result = router.search("weekend weather near me")

    assert result["skill_count"] == 1
    assert result["matches"][0]["type"] == "skill"
    assert result["matches"][0]["name"] == "skill:weather-lookup"
    assert result["matches"][0]["skill_name"] == "weather-lookup"


def test_router_describe_returns_full_schema():
    router = ToolRouter(
        [_tool("terminal", "Run shell commands", {"command": {"type": "string"}}, ["command"])],
        embedding_index=False,
    )

    result = router.describe("terminal")

    assert result["success"] is True
    assert result["schema"]["name"] == "terminal"
    assert result["schema"]["parameters"]["required"] == ["command"]


def test_router_describe_skill_returns_load_hint():
    router = ToolRouter(
        [],
        skill_entries=[
            {
                "name": "weather-lookup",
                "description": "Resolve local weather forecasts.",
                "category": "research",
            }
        ],
        embedding_index=False,
    )

    result = router.describe("skill:weather-lookup")

    assert result["success"] is True
    assert result["type"] == "skill"
    assert result["skill_name"] == "weather-lookup"
    assert result["equivalent_tool"] == {
        "tool_name": "skill_view",
        "arguments": {"name": "weather-lookup"},
    }


def test_router_unknown_tool_returns_suggestions():
    router = ToolRouter(
        [_tool("search_files", "Search file contents", {"query": {"type": "string"}})],
        embedding_index=False,
    )

    result = router.describe("search")

    assert result["success"] is False
    assert result["suggestions"][0]["name"] == "search_files"


def test_parse_router_arguments_accepts_dict_and_json_string():
    assert parse_router_arguments({"q": "hello"}) == {"q": "hello"}
    assert parse_router_arguments(json.dumps({"q": "hello"})) == {"q": "hello"}
    assert parse_router_arguments("not json") == {}


def test_parse_router_arguments_checked_reports_malformed_arguments():
    parsed, error = parse_router_arguments_checked("not json")

    assert parsed == {}
    assert "arguments must be" in error
