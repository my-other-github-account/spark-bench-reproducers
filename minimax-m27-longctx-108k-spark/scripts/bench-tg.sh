#!/usr/bin/env bash
# bench-tg.sh — single tg cell driver. Run llama-benchy at given depth/tg/runs.
# Honest mode: --no-cache (each request unique sherlock prose; no cache cheating).
#
# Must be run INSIDE the running server container via `docker exec`, because
# uvx is only inside the image, not on the host.
#
# Usage (inside the container):
#   bash /repro/scripts/bench-tg.sh <depth> <runs> [tg]
# e.g.
#   bash /repro/scripts/bench-tg.sh 0       30 128    # d=0 tg128 n=30
#   bash /repro/scripts/bench-tg.sh 100100   5 128    # d=100K tg128 n=5
#   bash /repro/scripts/bench-tg.sh 100100   3 2048   # d=100K tg2048 n=3
#
# From the host:
#   docker exec mm-srv bash /repro/scripts/bench-tg.sh 100100 5 128
set -euo pipefail

DEPTH="${1:?usage: bench-tg.sh depth runs [tg]}"
RUNS="${2:?need runs}"
TG="${3:-128}"
PP="${PP:-128}"
HOST="${HOST_BIND:-127.0.0.1}"
PORT="${PORT:-18080}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/models/MiniMax-M2.7-tokenizer}"
MODEL_NAME="${MODEL_NAME:-MiniMax-M2.7-UD-IQ4_XS}"
OUT="${OUT:-/repro/results/bench-d${DEPTH}-tg${TG}-n${RUNS}.json}"

mkdir -p "$(dirname "$OUT")"

# Sanity: server must be up
if ! curl -fsS -m 3 "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "[bench-tg] ERROR: no llama-server on ${HOST}:${PORT}" >&2
  exit 1
fi

echo "[bench-tg] depth=$DEPTH tg=$TG pp=$PP runs=$RUNS  →  $OUT"

# llama-benchy invocation (--no-cache + --skip-coherence + --no-warmup = honest)
uvx llama-benchy \
  --base-url "http://${HOST}:${PORT}/v1" \
  --model "$MODEL_NAME" \
  --tokenizer "$TOKENIZER_DIR" \
  --pp "$PP" --tg "$TG" --depth "$DEPTH" \
  --concurrency 1 --runs "$RUNS" \
  --no-warmup --skip-coherence --no-cache \
  --latency-mode api \
  --save-result "$OUT" --format json

echo
python3 - "$OUT" "$DEPTH" <<'PY'
import json, statistics as st, sys
out, depth = sys.argv[1], int(sys.argv[2])
b = json.load(open(out))["benchmarks"][0]
tg = b["tg_throughput"]["values"]
pp_obj = b["pp_throughput"]
ttfr = b.get("e2e_ttft", b.get("ttfr"))["values"]
tg_warm = tg[1:] if len(tg) > 1 else tg
ttfr_warm = ttfr[1:] if len(ttfr) > 1 else ttfr
print(f"depth               : {depth}  prefill/output: {b['prompt_size']}/{b['response_size']}")
print(f"tg_throughput (all) : mean {b['tg_throughput']['mean']:.2f}  median {st.median(tg):.2f}  std {b['tg_throughput']['std']:.2f}  n={len(tg)}")
if len(tg_warm) > 0:
    print(f"tg_throughput (warm): mean {st.mean(tg_warm):.2f}  median {st.median(tg_warm):.2f}  std {st.pstdev(tg_warm):.2f}  n={len(tg_warm)}")
print(f"ttfr ms (warm)      : median {st.median(ttfr_warm):.0f}  mean {st.mean(ttfr_warm):.0f}")
print(f"pp_throughput       : mean {pp_obj['mean']:.1f} tok/s  (depth={depth} so this is the {depth or 'cold'}-prefill rate)")
PY
