#!/usr/bin/env bash
# Usage: LOCALMAXXING_API_KEY=bhk_xxx ./submit.sh headline-ddtree-sherlock-thinkON.json
set -euo pipefail
body="${1:-headline-ddtree-sherlock-thinkON.json}"
if [[ -z "${LOCALMAXXING_API_KEY:-}" ]]; then
  echo "ERROR: set LOCALMAXXING_API_KEY (do not paste it into chat)." >&2
  exit 1
fi
if [[ ! -f "$body" ]]; then
  echo "ERROR: no such payload: $body" >&2
  exit 1
fi
response=$(curl -sS -w "
--HTTP %{http_code}--"   -X POST https://www.localmaxxing.com/api/benchmarks   -H "Authorization: Bearer ${LOCALMAXXING_API_KEY}"   -H "Content-Type: application/json"   -d "@${body}")
echo "$response"
code=$(echo "$response" | tail -n1 | sed 's/[^0-9]//g')
if [[ "$code" != "201" ]]; then
  echo "ERROR: HTTP $code" >&2
  exit 1
fi
