#!/bin/bash
# bench-codegen.sh — code corpus variant. Defaults match bench.sh (pp=128, tg=128).
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT="${OUT:-/repro/result-codegen.json}"
RUNS="${RUNS:-30}"
PP="${PP:-128}"
TG="${TG:-128}"
DEPTH="${DEPTH:-0}"
# Large vLLM source file (gpu_model_runner.py, ~7k lines, ~311 KB). This file
# contains the DFlash drafter integration so it stresses dense Python+CUDA
# code with realistic LLM-engineering vocabulary.
BOOK_URL="${BOOK_URL:-https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py}"

uvx llama-benchy \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model qwen36-35b-a3b \
  --tokenizer "$MODELS_DIR/Qwen3.6-35B-A3B-NVFP4" \
  --book-url "$BOOK_URL" \
  --concurrency 1 \
  --pp "$PP" --tg "$TG" --depth "$DEPTH" \
  --skip-coherence \
  --runs "$RUNS" \
  --save-result "$OUT" \
  --format json

echo
echo "[bench-codegen] Saved to $OUT (corpus: $BOOK_URL, pp=$PP, tg=$TG)"
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
b = json.load(open(sys.argv[1]))["benchmarks"][0]
tg = b["tg_throughput"]
ttfr = b["ttfr"]
tg_v = tg["values"]
ttfr_v = ttfr["values"]
tg_warm = tg_v[1:] if len(tg_v) > 1 else tg_v
ttfr_warm = ttfr_v[1:] if len(ttfr_v) > 1 else ttfr_v
print(f"tg_throughput (warm) : mean {st.mean(tg_warm):.2f} median {st.median(tg_warm):.2f} std {st.pstdev(tg_warm):.2f} n={len(tg_warm)}")
print(f"ttfr ms       (warm) : mean {st.mean(ttfr_warm):.0f} median {st.median(ttfr_warm):.0f}")
print(f"pp_throughput        : mean {b['pp_throughput']['mean']:.1f} tok/s")
PY
