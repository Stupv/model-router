#!/usr/bin/env python3
"""Configurable routing proxy — maps Claude model tiers to upstream LLM providers.

A single ANTHROPIC_BASE_URL, multiple backends. The router inspects the
incoming model name and forwards to the matching upstream with the correct
auth scheme, headers, and body transformations. The upstream just needs to
speak an Anthropic-compatible wire format.

Capabilities:
  - Tiered model routing (prefix-based, configurable)
  - Per-upstream auth (x-api-key / bearer)
  - Per-upstream timeouts and connect timeouts
  - Thinking / reasoning block sanitisation (strip, preserve, or stub per upstream)
  - Anthropic-specific field stripping (cache_control, reasoning_effort)
  - SSE stream filtering for thinking-block events
  - Health endpoint (GET /health) with per-upstream key status
  - Concurrency bounding (semaphore — 503 when at capacity)
  - Graceful shutdown on SIGTERM / SIGINT
  - Startup config validation (port range, required keys)

Usage:
  PROXY_PORT=9099 UPSTREAM_A_KEY=... UPSTREAM_B_KEY=... python3 model_router.py

Point your Anthropic client at it:
  ANTHROPIC_BASE_URL=http://127.0.0.1:9099
  ANTHROPIC_API_KEY=any-non-empty-string

See README.md for a wrapper script and systemd service template.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from typing import Any, TypedDict

import aiohttp
from aiohttp import web


class RouteConfig(TypedDict):
    upstream: str
    target_path: str
    model_id: str
    key_env: str
    auth_type: str
    name: str
    timeout_s: int
    connect_s: int


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s model-router %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PROXY_PORT = int(os.environ.get("PROXY_PORT", "9099"))

ROUTING_TABLE: dict[str, RouteConfig] = {
    "claude-opus-": {
        "upstream": "https://api.deepseek.com/anthropic",
        "target_path": "/v1/messages",
        "model_id": "deepseek-v4-pro",
        "key_env": "DEEPSEEK_API_KEY",
        "auth_type": "x-api-key",
        "name": "DeepSeek",
        "timeout_s": 300,
        "connect_s": 15,
    },
    "claude-sonnet-": {
        "upstream": "https://api.kimi.com/coding",
        "target_path": "/v1/messages",
        "model_id": "kimi-for-coding",
        "key_env": "KIMI_API_KEY",
        "auth_type": "x-api-key",
        "name": "Kimi",
        "timeout_s": 1800,
        "connect_s": 10,
    },
    "claude-haiku-": {
        "upstream": "https://api.minimax.io/anthropic",
        "target_path": "/v1/messages",
        "model_id": "MiniMax-M2.7",
        "key_env": "MINIMAX_API_KEY",
        "auth_type": "bearer",
        "name": "MiniMax",
        "timeout_s": 180,
        "connect_s": 10,
    },
}

# Anthropic content block types that carry reasoning state upstreams require
# to round-trip but Claude Code strips for third-party endpoints.
_THINKING_TYPES = frozenset({"thinking", "redacted_thinking", "reasoning"})
_EMPTY_THINKING = {"type": "thinking", "thinking": "", "signature": ""}

_shutdown_event = asyncio.Event()


def _strip_thinking_blocks(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove thinking/reasoning content blocks from message history.

    Kimi: requires `reasoning_content` to round-trip. DeepSeek: requires
    `content[].thinking` to round-trip. Both 400 when Claude Code strips them.
    By removing these blocks before they ever reach the upstream, we ensure
    the conversation history never references state the upstream can't find.
    """
    out = []
    for msg in messages:
        if not isinstance(msg, dict):
            out.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            filtered = [
                b for b in content if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
            ]
            out.append({**msg, "content": filtered})
        else:
            # String content — no thinking blocks possible
            out.append(msg)
    return out


def _sanitize_thinking_blocks_deepseek(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ensure assistant messages have thinking blocks for DeepSeek.

    DeepSeek requires thinking blocks to round-trip for tool-use contexts.
    We replace actual thinking content with empty blocks to prevent context
    bloat while satisfying the API's validation requirements.
    """
    out = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            out.append(msg)
            continue
        content = msg.get("content")
        if isinstance(content, list):
            sanitized = []
            has_thinking = False
            for b in content:
                if isinstance(b, dict) and b.get("type") in _THINKING_TYPES:
                    sanitized.append(_EMPTY_THINKING.copy())
                    has_thinking = True
                else:
                    sanitized.append(b)
            if not has_thinking:
                sanitized.insert(0, _EMPTY_THINKING.copy())
            out.append({**msg, "content": sanitized})
        else:
            # String content — leave as-is
            out.append(msg)
    return out


def _strip_thinking_from_response_body(body: bytes) -> bytes:
    """Remove thinking/reasoning blocks from a non-streaming JSON response body."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return body

    if isinstance(data, dict) and isinstance(data.get("content"), list):
        original = len(data["content"])
        data["content"] = [
            b
            for b in data["content"]
            if not (isinstance(b, dict) and b.get("type") in _THINKING_TYPES)
        ]
        if len(data["content"]) < original:
            log.info(
                "stripped %d thinking block(s) from response body",
                original - len(data["content"]),
            )
        return json.dumps(data, ensure_ascii=False).encode("utf-8")
    return body


async def _filter_sse_deepseek(
    upstream_reader: Any,
    client_writer: web.StreamResponse,
) -> None:
    """Filter thinking blocks out of a DeepSeek SSE stream before forwarding.

    We buffer incoming chunks, split on double-newline to find complete SSE
    events, parse each event's JSON payload, and drop events belonging to
    thinking blocks. This prevents Claude Code from ever seeing thinking
    blocks, which avoids context-window bloat and session resets.
    """
    buf = bytearray()
    thinking_indices: set[int] = set()
    dropped_events = 0

    async for chunk in upstream_reader:
        buf.extend(chunk)

        while True:
            sep = buf.find(b"\n\n")
            if sep == -1:
                break

            event = bytes(buf[:sep])
            del buf[: sep + 2]

            should_forward = True
            event_text = event.decode("utf-8", errors="replace")
            data_lines = []
            for line in event_text.split("\n"):
                if line.startswith("data:"):
                    data_lines.append(line[5:].strip())

            if data_lines:
                try:
                    payload = json.loads("\n".join(data_lines))
                    msg_type = payload.get("type")

                    if msg_type == "content_block_start":
                        idx = payload.get("index")
                        block_type = payload.get("content_block", {}).get("type")
                        if block_type in _THINKING_TYPES:
                            thinking_indices.add(idx)
                            should_forward = False
                            dropped_events += 1

                    elif msg_type == "content_block_delta":
                        idx = payload.get("index")
                        if idx in thinking_indices:
                            should_forward = False
                            dropped_events += 1

                    elif msg_type == "content_block_stop":
                        idx = payload.get("index")
                        if idx in thinking_indices:
                            thinking_indices.discard(idx)
                            should_forward = False
                            dropped_events += 1
                except json.JSONDecodeError:
                    pass  # malformed JSON — forward to be safe

            if should_forward:
                await client_writer.write(event + b"\n\n")

    # Forward any trailing bytes (incomplete event at stream end)
    if buf:
        await client_writer.write(bytes(buf))

    if dropped_events:
        log.info(
            "dropped %d thinking-related SSE event(s) for DeepSeek",
            dropped_events,
        )


def _strip_cache_control(body: dict[str, Any]) -> None:
    """Remove cache_control from all content blocks.

    Prompt caching is Anthropic-only.
    """

    def _clean(blocks: Any) -> None:
        if isinstance(blocks, list):
            for block in blocks:
                if isinstance(block, dict):
                    block.pop("cache_control", None)

    # system is a list of content blocks
    _clean(body.get("system"))
    # messages contain content lists
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            _clean(msg.get("content"))


def _route(model: str) -> tuple[str, RouteConfig]:
    """Return (tier_name, cfg) for a model string. Fuzzy-matches on tier keyword."""
    if not isinstance(model, str):
        raise ValueError(f"unknown model tier: {model}")
    m = model.lower()
    if "opus" in m:
        return "opus", ROUTING_TABLE["claude-opus-"]
    elif "sonnet" in m:
        return "sonnet", ROUTING_TABLE["claude-sonnet-"]
    elif "haiku" in m:
        return "haiku", ROUTING_TABLE["claude-haiku-"]
    else:
        raise ValueError(f"unknown model tier: {model}")


def _build_forward_headers(cfg: RouteConfig, raw_request_headers: Any) -> dict[str, str]:
    """Build headers to forward to the upstream based on auth_type."""
    api_key = os.environ.get(cfg["key_env"], "")
    if cfg["auth_type"] == "x-api-key":
        return {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
    elif cfg["auth_type"] == "bearer":
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    else:
        raise ValueError(f"unknown auth_type: {cfg['auth_type']}")


async def _proxy(request: web.Request) -> web.StreamResponse | web.Response:
    t0 = time.monotonic()

    sem = request.app["sem"]
    # asyncio.Semaphore._value is safe in single-threaded asyncio — no yield
    # between check and acquire
    if sem._value <= 0:
        return web.Response(
            body=json.dumps(
                {
                    "error": {"type": "proxy_error", "message": "rate limit exceeded"},
                }
            ),
            status=503,
            content_type="application/json",
        )
    await sem.acquire()

    try:
        raw = await request.read()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            body = {}

        if not isinstance(body, dict):
            return web.Response(
                body=json.dumps(
                    {
                        "error": {"type": "proxy_error", "message": "invalid request body"},
                    }
                ),
                status=400,
                content_type="application/json",
            )

        model = body.get("model")
        if not isinstance(model, str) or len(model) > 128:
            return web.Response(
                body=json.dumps(
                    {
                        "error": {
                            "type": "proxy_error",
                            "message": "missing or invalid model field",
                        },
                    }
                ),
                status=400,
                content_type="application/json",
            )

        try:
            tier, cfg = _route(model)
        except ValueError as exc:
            return web.Response(
                body=json.dumps(
                    {
                        "error": {"type": "proxy_error", "message": str(exc)},
                    }
                ),
                status=400,
                content_type="application/json",
            )

        original_model = model
        body["model"] = cfg["model_id"]
        is_stream = body.get("stream", False)

        # Only patch POST/PUT/PATCH bodies — GET/HEAD must not receive a body
        if request.method in ("POST", "PUT", "PATCH") and isinstance(body, dict):
            # Strip Claude Code-specific fields that third-party upstreams reject
            body.pop("reasoning_effort", None)

            # DeepSeek V4 auto-enables reasoning for agent scenarios (full tool
            # list + system prompt) and sets reasoning_effort to 'max' on the
            # backend. Injecting thinking:disabled conflicts with this implicit
            # setting. For DeepSeek we let the upstream manage thinking natively
            # and preserve thinking blocks so multi-turn tool loops round-trip.
            if cfg["name"] != "DeepSeek":
                body["thinking"] = {"type": "disabled"}

            # Strip cache_control from all content blocks (prompt caching is
            # Anthropic-only and not supported by third-party upstreams)
            _strip_cache_control(body)
            if isinstance(body.get("messages"), list):
                if cfg["name"] == "DeepSeek":
                    # DeepSeek requires thinking blocks for tool-use contexts.
                    # Replace actual thinking with empty blocks to prevent
                    # bloat while keeping the structure required by the API.
                    original_count = sum(
                        1
                        for m in body["messages"]
                        if isinstance(m, dict) and isinstance(m.get("content"), list)
                        for b in m["content"]
                        if isinstance(b, dict) and b.get("type") in _THINKING_TYPES
                    )
                    body["messages"] = _sanitize_thinking_blocks_deepseek(
                        body["messages"],
                    )
                    if original_count:
                        log.info(
                            "sanitized %d thinking block(s) for DeepSeek",
                            original_count,
                        )
                else:
                    original_count = sum(
                        1
                        for m in body["messages"]
                        if isinstance(m, dict) and isinstance(m.get("content"), list)
                        for b in m["content"]
                        if isinstance(b, dict) and b.get("type") in _THINKING_TYPES
                    )
                    body["messages"] = _strip_thinking_blocks(body["messages"])
                    if original_count:
                        log.info(
                            "stripped %d thinking block(s) from history",
                            original_count,
                        )

        upstream_url = cfg["upstream"] + cfg["target_path"]
        # Do not forward client query params — ?beta=true etc. are
        # Claude Code-specific
        log.debug(
            "outgoing body (no msgs): %s | reasoning_effort=%s",
            json.dumps(
                {k: v for k, v in body.items() if k not in ("messages", "system")},
                ensure_ascii=False,
            )[:300],
            body.get("reasoning_effort", "ABSENT"),
        )

        fwd_headers = _build_forward_headers(cfg, request.headers)
        timeout = aiohttp.ClientTimeout(
            total=cfg["timeout_s"],
            connect=cfg["connect_s"],
        )

        session = request.app["session"]

        send_body = (
            json.dumps(body, ensure_ascii=False)
            if request.method in ("POST", "PUT", "PATCH")
            else None
        )

        try:
            async with session.request(
                request.method,
                upstream_url,
                data=send_body,
                headers=fwd_headers,
                timeout=timeout,
            ) as upstream_resp:
                overhead_ms = (time.monotonic() - t0) * 1000
                log.info(
                    "→ %s %s(%s) model=%s→%s stream=%s status=%d overhead=%.1fms",
                    request.method,
                    tier,
                    cfg["name"],
                    original_model,
                    cfg["model_id"],
                    is_stream,
                    upstream_resp.status,
                    overhead_ms,
                )

                if upstream_resp.status >= 400:
                    err_body = await upstream_resp.read()
                    log.warning(
                        "upstream %d: %s",
                        upstream_resp.status,
                        err_body[:500],
                    )
                    return web.Response(
                        body=err_body,
                        status=upstream_resp.status,
                        content_type=upstream_resp.content_type or "application/json",
                    )

                if is_stream:
                    resp = web.StreamResponse(
                        status=upstream_resp.status,
                        headers={"Content-Type": upstream_resp.content_type},
                    )
                    await resp.prepare(request)
                    if cfg["name"] == "DeepSeek":
                        await _filter_sse_deepseek(
                            upstream_resp.content.iter_any(),
                            resp,
                        )
                    else:
                        async for chunk in upstream_resp.content.iter_any():
                            await resp.write(chunk)
                    await resp.write_eof()
                    return resp
                else:
                    resp_body = await upstream_resp.read()
                    if cfg["name"] == "DeepSeek":
                        resp_body = _strip_thinking_from_response_body(resp_body)
                    return web.Response(
                        body=resp_body,
                        status=upstream_resp.status,
                        content_type=upstream_resp.content_type or "application/json",
                    )

        except aiohttp.ClientError as exc:
            log.error("upstream error: %s", exc)
            return web.Response(
                body=json.dumps(
                    {
                        "error": {"type": "proxy_error", "message": str(exc)},
                    }
                ),
                status=502,
                content_type="application/json",
            )

    finally:
        sem.release()


async def _health(_: web.Request) -> web.Response:
    """Return health status for all configured upstream tiers."""
    tiers: dict[str, dict[str, Any]] = {}
    all_present = True
    for prefix, cfg in ROUTING_TABLE.items():
        key_present: bool = bool(os.environ.get(cfg["key_env"]))
        if not key_present:
            all_present = False
        tiers[prefix.replace("claude-", "").rstrip("-")] = {
            "upstream": cfg["name"],
            "key_present": key_present,
        }

    status = 200 if all_present else 503
    return web.Response(
        body=json.dumps({"status": "ok", "tiers": tiers}),
        status=status,
        content_type="application/json",
    )


def validate_config() -> None:
    """Validate proxy port and required API keys."""
    if not (1024 <= PROXY_PORT <= 65535):
        raise SystemExit(f"PROXY_PORT must be in 1024-65535, got {PROXY_PORT}")

    missing: list[str] = []
    confirmed: list[str] = []
    for cfg in ROUTING_TABLE.values():
        key: str = os.environ.get(cfg["key_env"], "")
        if not key:
            missing.append(cfg["name"])
        else:
            confirmed.append(cfg["name"])

    if missing:
        raise SystemExit(f"Missing API keys for: {', '.join(missing)}")

    log.info("Config validated: %s", ", ".join(f"{name} confirmed" for name in confirmed))


def _on_signal(signum: int) -> None:
    name = signal.Signals(signum).name
    log.info("Received %s, initiating graceful shutdown...", name)
    _shutdown_event.set()


async def main() -> None:
    """Start the model-router HTTP server and wait for shutdown signal."""
    validate_config()

    connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
    session = aiohttp.ClientSession(connector=connector)

    try:
        app = web.Application(client_max_size=10 * 1024 * 1024)
        app["session"] = session
        app["sem"] = asyncio.Semaphore(50)

        app.router.add_get("/health", _health)
        app.router.add_route("*", "/{path_info:.*}", _proxy)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", PROXY_PORT)
        await site.start()

        log.info("listening on http://127.0.0.1:%d", PROXY_PORT)

        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, _on_signal, signal.SIGTERM)
        loop.add_signal_handler(signal.SIGINT, _on_signal, signal.SIGINT)

        _shutdown_event.clear()
        await _shutdown_event.wait()

        log.info("Shutting down gracefully...")
        await runner.cleanup()
    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(main())
