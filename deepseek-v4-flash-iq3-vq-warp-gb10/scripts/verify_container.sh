#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
OUT="${OUT:-receipts}"
CONTAINER_NAME="${CONTAINER_NAME:-ds4-iq3-repro}"
mkdir -p "$OUT"
curl -fsS "${BASE_URL%/v1}/health" >/dev/null
python3 scripts/bench_live_tps.py \
  --base-url "$BASE_URL" --max-tokens 32 --warmup 1 --warmup-tokens 64 \
  --reps 1 --output "$OUT/container-greedy32.json"
python3 scripts/bench_live_tps.py \
  --base-url "$BASE_URL" --max-tokens 64 --warmup 2 --warmup-tokens 64 \
  --reps 5 --output "$OUT/container-5x64.json"
raw_log="$(mktemp "$OUT/.container-server.raw.XXXXXX")"
trap 'rm -f "$raw_log"' EXIT
docker logs "$CONTAINER_NAME" >"$raw_log" 2>&1
: >"$OUT/container-sentinels.log"
for sentinel in \
  'IQ3 CUDA WARP-GEMV ON-PATH sentinel' \
  'DECODE-GRAPH ON-PATH sentinel' \
  'IQ3 exact VQ ON-PATH sentinel'
do
  grep -F "$sentinel" "$raw_log" \
    | tee -a "$OUT/container-sentinels.log" >/dev/null
  printf 'PASS %s\n' "$sentinel"
done
python3 scripts/scrub_log.py "$raw_log" "$OUT/container-server.log"
