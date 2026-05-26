# Model Router

A model-aware routing proxy that maps Claude Code model tiers to third-party upstreams with Anthropic-compatible APIs.

## Intent

Claude Code sends all requests to a single `ANTHROPIC_BASE_URL`. But not every workload needs the same backend. Model Router intercepts those requests and routes them transparently based on the model name:

| Claude Tier | Upstream | Why |
|---|---|---|
| `claude-opus-*` | DeepSeek V4 Pro | Heavy reasoning, multi-turn tool loops |
| `claude-sonnet-*` | Kimi for Coding | General development, code review |
| `claude-haiku-*` | MiniMax M2.7 | Fast, cheap, simple completions |

Claude Code never knows it's talking to anything other than Anthropic. The router handles auth schemes, header translation, and thinking-block sanitisation per upstream.

## Outcomes

- **One proxy, three backends.** A single `ANTHROPIC_BASE_URL` gives Claude Code access to DeepSeek, Kimi, and MiniMax behind the scenes.
- **Thinking-block compatibility.** Each upstream handles reasoning/thinking blocks differently. DeepSeek requires them for tool-use round-tripping; Kimi and MiniMax reject them with 400s. The router sanitises or preserves per upstream so multi-turn conversations don't break.
- **Prompt-cache stripping.** Anthropic's `cache_control` is stripped before forwarding — third-party upstreams don't support it.
- **SSE stream filtering.** DeepSeek emits thinking blocks in its SSE stream that Claude Code doesn't understand and that bloat context windows. The router filters them in real time, event by event.
- **Graceful shutdown.** SIGTERM/SIGINT trigger a clean drain of in-flight requests before exit.
- **Config validation at startup.** Missing API keys or invalid port ranges fail fast with a clear message.
- **Health endpoint.** `GET /health` returns per-upstream status including whether API keys are present.
- **Concurrency bounding.** A semaphore caps in-flight requests at 50; excess returns 503 rather than overloading upstreams.
- **43 passing tests.** Unit tests cover routing, thinking-block stripping/sanitisation, SSE filtering, cache-control removal, config validation, and response-body sanitisation.

## Setup

### Prerequisites

- Python 3.12+
- API keys for the upstreams you want to route to (all three required)

### Install

```bash
# Clone
git clone https://github.com/Stupv/model-router.git
cd model-router

# Create venv and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Or just the single dependency directly:
pip install aiohttp
```

### Configure

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

### Point Claude Code at it

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:9099
export ANTHROPIC_API_KEY=proxy-passthrough  # any non-empty value
```

Claude Code will now route `claude-opus-*` → DeepSeek, `claude-sonnet-*` → Kimi, `claude-haiku-*` → MiniMax.

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

1. Claude Code sends a request to `http://127.0.0.1:9099/v1/messages` with `"model": "claude-sonnet-4-6"`.
2. `_route()` fuzzy-matches the model string: sees `"sonnet"` → Kimi tier.
3. `_build_forward_headers()` constructs auth headers for Kimi's `x-api-key` scheme.
4. The body is transformed: `model` is swapped to `kimi-for-coding`, `thinking` is set to disabled, `cache_control` and `reasoning_effort` are stripped, thinking blocks are removed from message history.
5. The request is proxied to `https://api.kimi.com/coding/v1/messages`.
6. The response is streamed or returned verbatim back to Claude Code.

For DeepSeek the flow differs: thinking blocks are *preserved* (sanitised to empty stubs) to satisfy the API's tool-use round-trip requirement, and thinking events are filtered out of the SSE stream on the way back.

## Routing table

The routing is prefix-based on the model string. Any model containing `opus` routes to DeepSeek, `sonnet` to Kimi, `haiku` to MiniMax. Unknown tiers return HTTP 400.

To change upstreams or add new ones, edit the `ROUTING_TABLE` dict in `model_router.py`.

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT
