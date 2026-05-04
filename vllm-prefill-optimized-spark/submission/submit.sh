#!/bin/bash
# Usage: LOCALMAXXING_API_KEY=*** ./submit.sh
# Submits the Qwen3.5-27B NVFP4 vLLM prefill-optimized AR headline to localmaxxing.
set -euo pipefail

if [[ -z "${LOCALMAXXING_API_KEY:-}" ]]; then
    echo "ERROR: set LOCALMAXXING_API_KEY (looks like bhk_<40hex>)"
    echo "Get one at https://localmaxxing.com → sign in with GitHub → dashboard → API Keys"
    exit 1
fi

cd "$(dirname "$0")"

submit() {
    local body_file="$1"
    local label="$2"
    echo "==> Submitting ${label} (${body_file})..."
    local response
    response=$(curl -sS -w "\n--HTTP %{http_code}--" \
        -X POST https://www.localmaxxing.com/api/benchmarks \
        -H "Authorization: Bearer ${LOCALMAXXING_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "@${body_file}")
    echo "${response}"
    local code
    code=$(echo "${response}" | tail -n1 | sed 's/[^0-9]//g')
    if [[ "${code}" != "201" ]]; then
        echo "ERROR: HTTP ${code}. Aborting."
        exit 1
    fi
}

submit headline.json "Qwen3.5-27B NVFP4 vLLM AR headline (pp128/tg128/c1)"
echo "Done. Check https://localmaxxing.com/leaderboard"
