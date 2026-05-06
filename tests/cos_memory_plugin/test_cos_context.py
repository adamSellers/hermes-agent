from plugins.context_engine import load_context_engine


def _conversation(turns: int):
    messages = [
        {"role": "system", "content": "primary system"},
        {"role": "system", "content": "secondary system"},
    ]
    for idx in range(1, turns + 1):
        messages.append({"role": "user", "content": f"question {idx}"})
        messages.append({"role": "assistant", "content": f"answer {idx}"})
    return messages


def test_load_context_engine_discovers_cos_context():
    engine = load_context_engine("cos-context")

    assert engine is not None
    assert engine.name == "cos-context"
    assert engine.get_tool_schemas() == []


def test_compress_keeps_hot_turns_and_digest_warm_turns():
    engine = load_context_engine("cos-context")
    messages = _conversation(35)

    compressed = engine.compress(messages)

    assert compressed[0]["content"] == "primary system"
    assert compressed[1]["content"] == "secondary system"
    digest = compressed[2]["content"]
    digest_lines = set(digest.splitlines())
    assert "[COMPRESSED CONVERSATION HISTORY" in digest
    assert "latest 20 warm turns" in digest
    assert "Turn 1 - question-1" not in digest_lines
    assert "Turn 25 - question-25" in digest_lines
    assert any(msg["content"] == "question 26" for msg in compressed)
    assert any(msg["content"] == "answer 35" for msg in compressed)
    assert not any(
        msg.get("role") == "user" and msg.get("content") == "question 25"
        for msg in compressed[3:]
    )
    assert engine.hot_zone_turns_current == 10
    assert engine.warm_zone_turns_current == 20
    assert engine.cold_zone_turns_total == 5
    assert engine.compression_count == 1


def test_preflight_and_manual_checks_use_hot_zone_boundary():
    engine = load_context_engine("cos-context")

    assert engine.should_compress_preflight(_conversation(10)) is False
    assert engine.has_content_to_compress(_conversation(10)) is False
    assert engine.should_compress_preflight(_conversation(11)) is True
    assert engine.has_content_to_compress(_conversation(11)) is True


def test_preflight_uses_token_threshold_for_short_fat_sessions():
    engine = load_context_engine("cos-context")
    engine.threshold_tokens = 100
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "read the log"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": "x" * 1000, "tool_call_id": "call_1"},
    ]

    assert engine.should_compress_preflight(messages) is True


def test_compress_compacts_large_hot_tool_output_without_warm_turns():
    engine = load_context_engine("cos-context")
    engine.threshold_tokens = 100
    engine.hot_tool_max_chars = 100
    engine.hot_tool_head_chars = 30
    engine.hot_tool_tail_chars = 20
    large_output = "A" * 500 + "TAIL_MARKER"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "inspect this"},
        {"role": "assistant", "content": "", "tool_calls": []},
        {"role": "tool", "content": large_output, "tool_call_id": "call_1"},
    ]

    compressed = engine.compress(messages, current_tokens=1000)

    assert len(compressed) == len(messages)
    assert "[Large tool output compacted by cos-context]" in compressed[-1]["content"]
    assert "TAIL_MARKER" in compressed[-1]["content"]
    assert len(compressed[-1]["content"]) < len(large_output)
    assert engine.hot_tool_compactions_count == 1
    assert engine.compression_count == 1


def test_existing_digest_block_is_preserved_and_capped():
    engine = load_context_engine("cos-context")
    existing_entries = "\n\n".join(
        f"Turn old-{idx}\n- Intent: old {idx}"
        for idx in range(1, 21)
    )
    messages = [
        {"role": "system", "content": "system"},
        {
            "role": "system",
            "content": (
                "[COMPRESSED CONVERSATION HISTORY - latest 20 warm turns]\n"
                f"{existing_entries}\n"
                "[END COMPRESSED CONVERSATION HISTORY]"
            ),
        },
    ]
    for idx in range(1, 16):
        messages.append({"role": "user", "content": f"fresh {idx}"})
        messages.append({"role": "assistant", "content": f"reply {idx}"})

    compressed = engine.compress(messages)
    digest = compressed[1]["content"]
    digest_lines = set(digest.splitlines())

    assert "latest 20 warm turns" in digest
    assert "Turn old-1" not in digest_lines
    assert "Turn old-10" in digest_lines
    assert "Turn 5 - fresh-5" in digest_lines
    assert "fresh 5" in digest
    assert any(msg["content"] == "fresh 15" for msg in compressed)
