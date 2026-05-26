#!/usr/bin/env bash
# test-model-router.sh — launch a Claude Code session routed through the proxy.
# Your real ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY are never modified — the
# overrides live only for the lifetime of the `claude` process.
#
# Usage:
#   chmod +x test-model-router.sh
#   ./test-model-router.sh
#   ./test-model-router.sh -p "write a haiku about proxies"
#
# API keys are read from the environment. Set them before running, or create a
# .env file in the model-router directory and this script will source it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROXY_PORT="${PROXY_PORT:-9099}"

# Source .env if present (safe: no secrets in the repo)
if [ -f "$SCRIPT_DIR/.env" ]; then
    set -a
    # shellcheck source=/dev/null
    source "$SCRIPT_DIR/.env"
    set +a
fi

# Health check — bail early if the proxy isn't running
if ! curl -sf "http://127.0.0.1:${PROXY_PORT}/health" > /dev/null 2>&1; then
    echo "❌ Model Router not running on port ${PROXY_PORT}" >&2
    echo "   Start it first:" >&2
    echo "     PROXY_PORT=${PROXY_PORT} python3 model_router.py &" >&2
    exit 1
fi

echo "✓ Router healthy — routing through http://127.0.0.1:${PROXY_PORT}"
echo "  (your real Anthropic config is untouched — these env vars are session-only)"
echo ""

# env vars live only for this process tree
exec env \
    ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}" \
    ANTHROPIC_API_KEY="proxy-passthrough" \
    claude "$@"
