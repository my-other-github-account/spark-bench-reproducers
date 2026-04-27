#!/bin/bash
# bench.sh — run llama-benchy at pp=128, tg=128, depth=0, c=1, n=30 trials.
# Defaults match the localmaxxing.com headline submission for this recipe.
# Result JSON written to $OUT (default /repro/result.json).
#
# Headline expected on a healthy DGX Spark GB10 with NVFP4 + DFlash, thinking-ON:
#   tg_throughput mean ~= 32 tok/s   (median ~30, std ~5-12)
#   ttfr (ms)     mean ~= 300        (after dropping cold-start outlier)
#   pp_throughput mean ~= 420 tok/s  (small prefill is overhead-bound)
#
# Override defaults via env vars:
#   PP=2048 TG=128 RUNS=30 bash bench.sh    # large-prefill variant
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT="${OUT:-/repro/result.json}"
RUNS="${RUNS:-30}"
PP="${PP:-128}"
TG="${TG:-128}"
DEPTH="${DEPTH:-0}"

uvx llama-benchy \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model qwen36-27b \
  --tokenizer "$MODELS_DIR/Qwen3.6-27B-NVFP4" \
  --concurrency 1 \
  --pp "$PP" --tg "$TG" --depth "$DEPTH" \
  --skip-coherence \
  --runs "$RUNS" \
  --save-result "$OUT" \
  --format json

echo
echo "[bench] Saved to $OUT (pp=$PP, tg=$TG, depth=$DEPTH, n=$RUNS)"
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
b = json.load(open(sys.argv[1]))["benchmarks"][0]
tg = b["tg_throughput"]
ttfr = b["ttfr"]            # values are already in milliseconds
pp = b["pp_throughput"]
tg_v = tg["values"]
ttfr_v = ttfr["values"]

# Drop the first sample as cold-start warmup (it can be 10-20x median).
tg_warm = tg_v[1:] if len(tg_v) > 1 else tg_v
ttfr_warm = ttfr_v[1:] if len(ttfr_v) > 1 else ttfr_v

print(f"prefill / output     : {b['prompt_size']} / {b['response_size']}")
print(f"tg_throughput (all)  : mean {tg['mean']:.2f} median {st.median(tg_v):.2f} std {tg['std']:.2f} n={len(tg_v)}")
print(f"tg_throughput (warm) : mean {st.mean(tg_warm):.2f} median {st.median(tg_warm):.2f} std {st.pstdev(tg_warm):.2f} n={len(tg_warm)}")
print(f"ttfr ms (all)        : mean {st.mean(ttfr_v):.0f} median {st.median(ttfr_v):.0f} max {max(ttfr_v):.0f}")
print(f"ttfr ms (warm)       : mean {st.mean(ttfr_warm):.0f} median {st.median(ttfr_warm):.0f} max {max(ttfr_warm):.0f}")
print(f"pp_throughput        : mean {pp['mean']:.1f} tok/s")
PY
