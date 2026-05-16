#!/usr/bin/env bash
set -euo pipefail
: "${LOCALMAXXING_API_KEY:?set LOCALMAXXING_API_KEY}"
file="${1:?payload json}"
curl -sS -A 'Mozilla/5.0' -X POST 'https://www.localmaxxing.com/api/benchmarks' \
  -H "Authorization: Bearer ${LOCALMAXXING_API_KEY}" \
  -H 'Content-Type: application/json' \
  --data-binary @"$file"
