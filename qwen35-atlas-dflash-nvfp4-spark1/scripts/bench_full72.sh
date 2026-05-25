#!/usr/bin/env bash
set -euo pipefail
: "${BASE_URL:=http://127.0.0.1:9156}"
: "${MODE:=dflash}"
: "${OUT:=results/live_${MODE}_benchmark_q35_nvfp4_full72_c1.json}"
python3 scripts/bench_atlas.py \
  --base-url "$BASE_URL" \
  --prompts prompts/atlas_diverse_72.jsonl \
  --output "$OUT" \
  --label "q35_nvfp4_full72_c1_${MODE}" \
  --mode "$MODE" \
  --max-tokens 160 \
  --temperature 0.0 \
  --concurrency 1 \
  --timeout 300 \
  --min-prompts 64
