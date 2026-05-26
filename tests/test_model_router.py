"""Unit tests for model_router.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from hypothesis import given, settings, strategies as st

import model_router


# ---------------------------------------------------------------------------
# _route()
# ---------------------------------------------------------------------------
def test_route_returns_deepseek_for_opus_model() -> None:
    tier, cfg = model_router._route("claude-opus-4-7")
    assert tier == "opus"
    assert cfg["name"] == "DeepSeek"


def test_route_returns_kimi_for_sonnet_model() -> None:
    tier, cfg = model_router._route("claude-sonnet-4")
    assert tier == "sonnet"
    assert cfg["name"] == "Kimi"


def test_route_returns_minimax_for_haiku_model() -> None:
    tier, cfg = model_router._route("claude-haiku-7")
    assert tier == "haiku"
    assert cfg["name"] == "MiniMax"


def test_route_raises_value_error_for_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model tier"):
        model_router._route("gpt-4")


def test_route_raises_value_error_for_non_string() -> None:
    with pytest.raises(ValueError, match="unknown model tier"):
        model_router._route(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _strip_thinking_blocks()
# ---------------------------------------------------------------------------
def test_strip_thinking_blocks_removes_thinking_block() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "...", "signature": ""},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    result = model_router._strip_thinking_blocks(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_blocks_removes_redacted_thinking() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "redacted_thinking", "data": "abc"},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    result = model_router._strip_thinking_blocks(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_blocks_removes_reasoning() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "reasoning": "..."},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    result = model_router._strip_thinking_blocks(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_blocks_preserves_text_only() -> None:
    messages = [
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]
    result = model_router._strip_thinking_blocks(messages)
    assert result[0]["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_blocks_leaves_string_content_unchanged() -> None:
    messages = [
        {"role": "assistant", "content": "hello"},
    ]
    result = model_router._strip_thinking_blocks(messages)
    assert result[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# _sanitize_thinking_blocks_deepseek()
# ---------------------------------------------------------------------------
def test_sanitize_replaces_thinking_with_empty_block() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "secret", "signature": "sig"},
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    assert result[0]["content"][0] == model_router._EMPTY_THINKING
    assert result[0]["content"][1] == {"type": "text", "text": "hello"}


def test_sanitize_prepends_empty_thinking_when_missing() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
            ],
        },
    ]
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    assert result[0]["content"][0] == model_router._EMPTY_THINKING
    assert result[0]["content"][1] == {"type": "text", "text": "hello"}


def test_sanitize_leaves_non_assistant_messages_unchanged() -> None:
    messages = [
        {"role": "user", "content": [{"type": "thinking", "thinking": "..."}]},
    ]
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    assert result[0]["content"][0]["type"] == "thinking"


def test_sanitize_leaves_string_content_unchanged() -> None:
    messages = [
        {"role": "assistant", "content": "hello"},
    ]
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    assert result[0]["content"] == "hello"


# ---------------------------------------------------------------------------
# _strip_cache_control()
# ---------------------------------------------------------------------------
def test_strip_cache_control_from_system_array() -> None:
    body = {
        "system": [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [],
    }
    model_router._strip_cache_control(body)
    assert "cache_control" not in body["system"][0]


def test_strip_cache_control_from_message_content() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
    }
    model_router._strip_cache_control(body)
    assert "cache_control" not in body["messages"][0]["content"][0]


def test_strip_cache_control_handles_missing_keys() -> None:
    body = {"messages": [{"role": "user", "content": "hi"}]}
    model_router._strip_cache_control(body)
    assert body["messages"][0]["content"] == "hi"


# ---------------------------------------------------------------------------
# _filter_sse_deepseek()
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_filter_sse_drops_thinking_events() -> None:
    events = [
        b'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hi"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        b'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"ok"}}\n\n',
        b'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
    ]

    async def upstream():
        for ev in events:
            yield ev

    writer = AsyncMock()
    await model_router._filter_sse_deepseek(upstream(), writer)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"thinking" not in written
    assert b"text_delta" in written
    assert b"ok" in written


@pytest.mark.asyncio
async def test_filter_sse_passes_non_thinking_events() -> None:
    events = [
        b'event: message_start\ndata: {"type":"message_start"}\n\n',
        b'event: ping\ndata: {"type":"ping"}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]

    async def upstream():
        for ev in events:
            yield ev

    writer = AsyncMock()
    await model_router._filter_sse_deepseek(upstream(), writer)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"message_start" in written
    assert b"ping" in written
    assert b"message_stop" in written


@pytest.mark.asyncio
async def test_filter_sse_handles_multi_chunk_buffering() -> None:
    # First chunk has thinking start + delta, plus text start
    chunk1 = (
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}\n\n'
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hello"}}\n\n'
        b"event: content_block_start\n"
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n'
    )
    # Second chunk finishes the text block
    chunk2 = (
        b"event: content_block_delta\n"
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"world"}}\n\n'
    )

    async def upstream():
        yield chunk1
        yield chunk2

    writer = AsyncMock()
    await model_router._filter_sse_deepseek(upstream(), writer)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"thinking_delta" not in written
    assert b"text_delta" in written
    assert b"world" in written


@pytest.mark.asyncio
async def test_filter_sse_handles_partial_json_lines_gracefully() -> None:
    # Malformed JSON should be forwarded to be safe
    events = [
        b"event: content_block_start\ndata: {not valid json}\n\n",
    ]

    async def upstream():
        for ev in events:
            yield ev

    writer = AsyncMock()
    await model_router._filter_sse_deepseek(upstream(), writer)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    assert b"not valid json" in written


@pytest.mark.asyncio
async def test_filter_sse_forwards_trailing_bytes() -> None:
    # Incomplete event at stream end
    events = [
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"x"}}',
    ]

    async def upstream():
        for ev in events:
            yield ev

    writer = AsyncMock()
    await model_router._filter_sse_deepseek(upstream(), writer)

    written = b"".join(call.args[0] for call in writer.write.call_args_list)
    # The trailing incomplete event should be forwarded
    assert b"text_delta" in written


# ---------------------------------------------------------------------------
# _strip_thinking_from_response_body()
# ---------------------------------------------------------------------------
def test_strip_thinking_from_response_body_removes_thinking() -> None:
    data = {
        "content": [
            {"type": "thinking", "thinking": "...", "signature": ""},
            {"type": "text", "text": "hello"},
        ],
    }
    body = json.dumps(data).encode()
    result = model_router._strip_thinking_from_response_body(body)
    parsed = json.loads(result)
    assert parsed["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_from_response_body_passes_through_no_thinking() -> None:
    data = {"content": [{"type": "text", "text": "hello"}]}
    body = json.dumps(data).encode()
    result = model_router._strip_thinking_from_response_body(body)
    parsed = json.loads(result)
    assert parsed["content"] == [{"type": "text", "text": "hello"}]


def test_strip_thinking_from_response_body_returns_invalid_json_unchanged() -> None:
    body = b"not json"
    result = model_router._strip_thinking_from_response_body(body)
    assert result == b"not json"


def test_strip_thinking_from_response_body_returns_unicode_error_unchanged() -> None:
    body = b"\xff\xfe"
    result = model_router._strip_thinking_from_response_body(body)
    assert result == b"\xff\xfe"


# ---------------------------------------------------------------------------
# validate_config()
# ---------------------------------------------------------------------------
def test_validate_config_raises_on_invalid_port() -> None:
    with patch.object(model_router, "PROXY_PORT", 80):
        with pytest.raises(SystemExit, match="PROXY_PORT must be in 1024-65535"):
            model_router.validate_config()


def test_validate_config_raises_on_missing_keys() -> None:
    with patch.dict(model_router.ROUTING_TABLE, {}, clear=True):
        with patch.object(model_router, "PROXY_PORT", 9099):
            # Empty routing table means no missing keys, but let's test with
            # a fake table that references a missing env var.
            fake_table = {
                "test-": {
                    "key_env": "MISSING_TEST_KEY",
                    "name": "TestProvider",
                },
            }
            with patch.object(model_router, "ROUTING_TABLE", fake_table):
                with pytest.raises(SystemExit, match="Missing API keys for: TestProvider"):
                    model_router.validate_config()


def test_validate_config_passes_when_all_keys_present(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-deepseek")
    monkeypatch.setenv("KIMI_API_KEY", "fake-kimi")
    monkeypatch.setenv("MINIMAX_API_KEY", "fake-minimax")
    # Must reload ROUTING_TABLE or just call validate_config directly since
    # ROUTING_TABLE references the env vars by name.
    model_router.validate_config()


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------

# -- _route() ---------------------------------------------------------------


@given(
    prefix=st.text(min_size=0, max_size=20),
    suffix=st.text(min_size=0, max_size=20),
)
@settings(max_examples=50)
def test_route_opus_keyword(prefix: str, suffix: str) -> None:
    model = prefix + "opus" + suffix
    tier, cfg = model_router._route(model)
    assert tier == "opus"
    assert cfg["name"] == "DeepSeek"


@given(
    prefix=st.text(min_size=0, max_size=20),
    suffix=st.text(min_size=0, max_size=20),
)
@settings(max_examples=50)
def test_route_sonnet_keyword(prefix: str, suffix: str) -> None:
    model = prefix + "sonnet" + suffix
    tier, cfg = model_router._route(model)
    assert tier == "sonnet"
    assert cfg["name"] == "Kimi"


@given(
    prefix=st.text(min_size=0, max_size=20),
    suffix=st.text(min_size=0, max_size=20),
)
@settings(max_examples=50)
def test_route_haiku_keyword(prefix: str, suffix: str) -> None:
    model = prefix + "haiku" + suffix
    m = model.lower()
    # If the generated string also contains "sonnet", _route() will match sonnet first
    if "sonnet" in m:
        tier, cfg = model_router._route(model)
        assert tier == "sonnet"
        assert cfg["name"] == "Kimi"
    else:
        tier, cfg = model_router._route(model)
        assert tier == "haiku"
        assert cfg["name"] == "MiniMax"


@given(model=st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_route_no_keyword_raises(model: str) -> None:
    m = model.lower()
    assume = "opus" not in m and "sonnet" not in m and "haiku" not in m
    if not assume:
        return
    with pytest.raises(ValueError, match="unknown model tier"):
        model_router._route(model)


@given(model=st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_route_idempotent(model: str) -> None:
    try:
        r1 = model_router._route(model)
        r2 = model_router._route(model)
    except ValueError:
        return
    assert r1 == r2


# -- _strip_thinking_blocks() -----------------------------------------------

content_block = st.one_of(
    st.fixed_dictionaries(
        {"type": st.just("text"), "text": st.text(min_size=0, max_size=50)}
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("thinking"),
            "thinking": st.text(min_size=0, max_size=50),
            "signature": st.text(min_size=0, max_size=20),
        }
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("redacted_thinking"),
            "data": st.text(min_size=0, max_size=50),
        }
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("reasoning"),
            "reasoning": st.text(min_size=0, max_size=50),
        }
    ),
)

message_with_list_content = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["user", "assistant", "system"]),
        "content": st.lists(content_block, min_size=0, max_size=10),
    }
)

message_with_string_content = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["user", "assistant", "system"]),
        "content": st.text(min_size=0, max_size=100),
    }
)

message_strategy = st.one_of(
    message_with_list_content,
    message_with_string_content,
    st.integers(),
    st.text(),
)


@given(messages=st.lists(message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_strip_thinking_blocks_output_never_longer(messages: list) -> None:
    result = model_router._strip_thinking_blocks(messages)
    assert len(result) <= len(messages)


@given(messages=st.lists(message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_strip_thinking_blocks_idempotent(messages: list) -> None:
    once = model_router._strip_thinking_blocks(messages)
    twice = model_router._strip_thinking_blocks(once)
    assert once == twice


@given(messages=st.lists(message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_strip_thinking_blocks_no_thinking_remains(messages: list) -> None:
    result = model_router._strip_thinking_blocks(messages)
    for msg in result:
        if isinstance(msg, dict) and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict):
                    assert block.get("type") not in model_router._THINKING_TYPES


@given(messages=st.lists(message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_strip_thinking_blocks_non_dict_passthrough(messages: list) -> None:
    result = model_router._strip_thinking_blocks(messages)
    for orig, out in zip(messages, result):
        if not isinstance(orig, dict):
            assert orig == out


# -- _strip_cache_control() -------------------------------------------------

cache_control_block = st.one_of(
    st.fixed_dictionaries(
        {"type": st.just("text"), "text": st.text(min_size=0, max_size=30)}
    ),
    st.fixed_dictionaries(
        {
            "type": st.just("text"),
            "text": st.text(min_size=0, max_size=30),
            "cache_control": st.fixed_dictionaries(
                {"type": st.just("ephemeral")}
            ),
        }
    ),
)

message_body = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["user", "assistant", "system"]),
        "content": st.one_of(
            st.text(min_size=0, max_size=50),
            st.lists(cache_control_block, min_size=0, max_size=5),
        ),
    }
)

request_body = st.fixed_dictionaries(
    {
        "system": st.one_of(
            st.none(),
            st.lists(cache_control_block, min_size=0, max_size=5),
        ),
        "messages": st.lists(message_body, min_size=0, max_size=10),
    }
)


def _has_cache_control(obj):
    """Return True if any dict in the nested structure has a cache_control key."""
    if isinstance(obj, dict):
        if "cache_control" in obj:
            return True
        return any(_has_cache_control(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_cache_control(item) for item in obj)
    return False


@given(body=request_body)
@settings(max_examples=50)
def test_strip_cache_control_removes_all(body: dict) -> None:
    model_router._strip_cache_control(body)
    assert not _has_cache_control(body)


@given(body=request_body)
@settings(max_examples=50)
def test_strip_cache_control_no_op_when_absent(body: dict) -> None:
    if _has_cache_control(body):
        return
    original = json.dumps(body, sort_keys=True)
    model_router._strip_cache_control(body)
    after = json.dumps(body, sort_keys=True)
    assert original == after


# -- _sanitize_thinking_blocks_deepseek() -----------------------------------

assistant_message_with_list = st.fixed_dictionaries(
    {
        "role": st.just("assistant"),
        "content": st.lists(content_block, min_size=0, max_size=10),
    }
)

assistant_message_with_string = st.fixed_dictionaries(
    {
        "role": st.just("assistant"),
        "content": st.text(min_size=0, max_size=100),
    }
)

non_assistant_message = st.fixed_dictionaries(
    {
        "role": st.sampled_from(["user", "system"]),
        "content": st.one_of(
            st.text(min_size=0, max_size=100),
            st.lists(content_block, min_size=0, max_size=5),
        ),
    }
)

deepseek_message_strategy = st.one_of(
    assistant_message_with_list,
    assistant_message_with_string,
    non_assistant_message,
    st.integers(),
    st.text(),
)


@given(messages=st.lists(deepseek_message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_sanitize_assistant_has_thinking_block(messages: list) -> None:
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    for msg in result:
        if (
            isinstance(msg, dict)
            and msg.get("role") == "assistant"
            and isinstance(msg.get("content"), list)
        ):
            assert any(
                isinstance(b, dict) and b.get("type") == "thinking"
                for b in msg["content"]
            )


@given(messages=st.lists(deepseek_message_strategy, min_size=0, max_size=20))
@settings(max_examples=50)
def test_sanitize_non_assistant_unchanged(messages: list) -> None:
    result = model_router._sanitize_thinking_blocks_deepseek(messages)
    for orig, out in zip(messages, result):
        if isinstance(orig, dict) and orig.get("role") != "assistant":
            assert orig == out
