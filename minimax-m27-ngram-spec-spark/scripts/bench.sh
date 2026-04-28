#!/usr/bin/env bash
# bench.sh — run llama-benchy at the EXACT settings used for the leaderboard
# headline. Sherlock prose corpus (llama-benchy default), pp=128, tg=128,
# n=30, single-stream, depth=0, NO warmup, NO coherence check.
#
# Result JSON written to $OUT (default /repro/results/result.json).
#
# Headline ngram-simple result on a healthy DGX Spark GB10:
#   tg/s warm  : median ≈ 30.98  mean ≈ 31.00  std ≈ 4.00  n=29
#   pp tok/s   : ≈ 113
#   ttfr ms    : ≈ 783 (median, warm)
#
# Override defaults via env vars:
#   OUT=/repro/results/result-codegen.json BENCH_TYPE=codegen bash bench.sh
#   OUT=/repro/results/result-thinkOFF.json THINK=off bash bench.sh
set -euo pipefail

TOKENIZER_DIR="${TOKENIZER_DIR:-/models/MiniMax-M2.7-tokenizer}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8012}"
OUT="${OUT:-/repro/results/result.json}"
RUNS="${RUNS:-30}"
PP="${PP:-128}"
TG="${TG:-128}"
DEPTH="${DEPTH:-0}"
THINK="${THINK:-on}"          # on | off — controls Jinja chat-template kwarg
BENCH_TYPE="${BENCH_TYPE:-sherlock}"  # sherlock (default) | codegen

mkdir -p "$(dirname "$OUT")"

# Build llama-benchy args
ARGS=(
  --base-url "http://${HOST}:${PORT}/v1"
  --model MiniMax-M2.7-UD-IQ4_XS
  --tokenizer "$TOKENIZER_DIR"
  --pp "$PP" --tg "$TG" --depth "$DEPTH"
  --concurrency 1 --runs "$RUNS"
  --no-warmup --skip-coherence --no-cache
  --latency-mode api
  --save-result "$OUT" --format json
)

# Codegen corpus — point at a large vLLM source file (~7000 lines)
if [[ "$BENCH_TYPE" == "codegen" ]]; then
  ARGS+=(--book-url "https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/v1/worker/gpu_model_runner.py")
fi

# Thinking mode — pass through to chat completion request
if [[ "$THINK" == "off" ]]; then
  # llama-benchy 0.3.6+ supports --extra-body for OpenAI-compat extras;
  # MiniMax-M2 toggles thinking via chat_template_kwargs.enable_thinking.
  ARGS+=(--extra-body '{"chat_template_kwargs":{"enable_thinking":false}}')
fi

echo "[bench] benchmark: $BENCH_TYPE  pp=$PP tg=$TG depth=$DEPTH runs=$RUNS think=$THINK"
echo "[bench] out: $OUT"

uvx llama-benchy "${ARGS[@]}"

echo
python3 - "$OUT" <<'PY'
import json, statistics as st, sys
b = json.load(open(sys.argv[1]))["benchmarks"][0]
tg = b["tg_throughput"]["values"]
pp = b["pp_throughput"]
ttfr = b.get("e2e_ttft", b.get("ttfr"))["values"]   # ms
# Drop cold-start sample
tg_warm = tg[1:] if len(tg) > 1 else tg
ttfr_warm = ttfr[1:] if len(ttfr) > 1 else ttfr
print(f"prefill / output    : {b['prompt_size']} / {b['response_size']}")
print(f"tg_throughput (all) : mean {b['tg_throughput']['mean']:.2f} median {st.median(tg):.2f} std {b['tg_throughput']['std']:.2f} n={len(tg)}")
print(f"tg_throughput (warm): mean {st.mean(tg_warm):.2f} median {st.median(tg_warm):.2f} std {st.pstdev(tg_warm):.2f} n={len(tg_warm)}")
print(f"ttfr ms (warm)      : median {st.median(ttfr_warm):.0f} mean {st.mean(ttfr_warm):.0f}")
print(f"pp_throughput       : mean {pp['mean']:.1f} tok/s")
PY
