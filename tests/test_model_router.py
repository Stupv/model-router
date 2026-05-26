"""Unit tests for model_router.py."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

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
