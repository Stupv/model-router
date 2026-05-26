# Model Router

A lightweight, configurable proxy that routes Anthropic-format API requests to different backends based on the model name. Works with any LLM provider that exposes an Anthropic-compatible endpoint.

## Intent

Claude Code, and any client built on the Anthropic Messages API, sends all requests to a single `ANTHROPIC_BASE_URL`. But not every model tier needs the same backend. Model Router sits between the client and your upstreams, inspecting the `model` field and forwarding each request to the right provider — transparently, with zero client-side changes.

The client never knows it's talking to anything other than Anthropic. The router handles auth scheme translation, header normalisation, and per-upstream body sanitisation so every request lands in a shape the target provider accepts.

### Example setup (shipped default)

| Model pattern | Routes to |
|---|---|
| `claude-opus-*` | DeepSeek API (`/anthropic`) |
| `claude-sonnet-*` | Kimi API (`/coding`) |
| `claude-haiku-*` | MiniMax API (`/anthropic`) |

This is just the working example baked into the default routing table. Swap in any Anthropic-compatible provider — OpenAI, Groq, LiteLLM, a local vLLM instance, whatever speaks the wire format.

## What it handles

- **Auth scheme translation.** Different providers use different auth: `x-api-key`, `Bearer`, custom headers. The router maps each upstream's scheme automatically.
- **Thinking / reasoning block compatibility.** Some providers require thinking blocks for tool-use round-tripping; others reject them with 400s. The router can strip, sanitise, or preserve per upstream so multi-turn conversations survive.
- **Anthropic-specific field stripping.** `cache_control` (prompt caching) and `reasoning_effort` are Anthropic-only — the router removes them before forwarding to third-party upstreams.
- **SSE stream filtering.** If an upstream emits event types your client doesn't understand (e.g. thinking blocks in the stream), the router filters them in real time.
- **Single upstream failure doesn't take down the proxy.** Each upstream has independent timeouts and the health endpoint reports per-upstream status.
- **Concurrency bounding.** A configurable semaphore caps in-flight requests; excess returns 503 rather than overwhelming upstreams.
- **Graceful shutdown.** SIGTERM/SIGINT drain in-flight requests cleanly.
- **Config validation at startup.** Missing keys or invalid ports fail fast.

## Setup

### Prerequisites

- Python 3.12+
- One or more LLM providers with Anthropic-compatible APIs
- API keys for each provider you want to route to

### Install

```bash
git clone https://github.com/Stupv/model-router.git
cd model-router
pip install aiohttp
```

That's it — the only runtime dependency is `aiohttp`.

### Configure

Set an API key for each upstream in your routing table. With the defaults:

```bash
export PROXY_PORT=9099
export DEEPSEEK_API_KEY=sk-your-deepseek-key
export KIMI_API_KEY=sk-your-kimi-key
export MINIMAX_API_KEY=sk-your-minimax-key
```

### Run

```bash
python3 model_router.py
# listening on http://127.0.0.1:9099
```

### Try it safely without touching your config

Before wiring it into your daily setup, test with a wrapper script that isolates the environment variables to a single session. When the session exits, your real config is untouched.

```bash
#!/usr/bin/env bash
# test-model-router.sh — launch a Claude Code session routed through the proxy.
# Set your API keys before running, or source them from a .env file.

set -euo pipefail

# --- config -------------------------------------------------
PROXY_PORT="${PROXY_PORT:-9099}"
# Uncomment and adjust to your setup:
# export DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
# export KIMI_API_KEY="${KIMI_API_KEY:-}"
# export MINIMAX_API_KEY="${MINIMAX_API_KEY:-}"
# ------------------------------------------------------------

# Health check — bail early if the proxy isn't running
if ! curl -sf "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null; then
    echo "❌ Model Router not running on port ${PROXY_PORT}" >&2
    echo "   Start it first: PROXY_PORT=${PROXY_PORT} python3 model_router.py &" >&2
    exit 1
fi

echo "✓ Router healthy — routing through http://127.0.0.1:${PROXY_PORT}"

# Isolate: these env vars live only for this command
ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}" \
ANTHROPIC_API_KEY="proxy-passthrough" \
    exec claude "$@"
```

Save as `test-model-router.sh`, make it executable, and run:

```bash
chmod +x test-model-router.sh
./test-model-router.sh
```

The environment variables vanish when Claude exits — your global `claude` config never sees them. If something breaks, the proxy isn't in the path for your normal sessions.

### Point your client at it permanently

Once you're comfortable it works, wire it in:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
export ANTHROPIC_API_KEY=proxy-passthrough  # any non-empty value
```

Add those to your shell profile (`.bashrc`, `.zshrc`) or Claude Code's environment config for permanent routing.

### Health check

```bash
curl http://127.0.0.1:9099/health
```

```json
{
  "status": "ok",
  "tiers": {
    "opus":   {"upstream": "DeepSeek", "key_present": true},
    "sonnet": {"upstream": "Kimi",     "key_present": true},
    "haiku":  {"upstream": "MiniMax",  "key_present": true}
  }
}
```

Returns HTTP 503 if any configured upstream is missing its API key.

### As a systemd service

```ini
[Unit]
Description=Model Router Proxy
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/path/to/.venv/bin/python3 /opt/model-router/model_router.py
EnvironmentFile=/opt/model-router/.env
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now model-router
```

## How it works

1. Client sends `POST /v1/messages` with `"model": "claude-sonnet-4-6"`.
2. `_route()` inspects the model string — sees `"sonnet"` → matches the sonnet tier.
3. `_build_forward_headers()` constructs the correct auth headers for that upstream.
4. The body is transformed: model ID is swapped to the upstream's native model name, incompatible fields are stripped, thinking blocks are sanitised per the upstream's requirements.
5. The request is proxied upstream. The response streams back to the client unchanged.
6. If the upstream returns an error (≥400), it's passed through to the client so error handling works as normal.

## Writing your own routing table

Edit the `ROUTING_TABLE` dict in `model_router.py`. Each entry is keyed by a model-name prefix and maps to:

| Field | Description |
|---|---|
| `upstream` | Base URL of the upstream API |
| `target_path` | Path appended to `upstream` (usually `/v1/messages`) |
| `model_id` | Native model name to swap into the request |
| `key_env` | Environment variable holding the API key |
| `auth_type` | `"x-api-key"` or `"bearer"` |
| `name` | Human-readable label (used in logs and health endpoint) |
| `timeout_s` | Total request timeout in seconds |
| `connect_s` | Connection timeout in seconds |

The routing is prefix-based: any model string containing the prefix keyword (`opus`, `sonnet`, `haiku`) matches that tier. Unknown tiers return HTTP 400.

If your upstream needs custom body transforms (e.g. different thinking-block handling), the transformation logic is in `_proxy()` — it's straightforward to extend.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
