#!/usr/bin/env bash
# Run the exact full-context deployment reproduction: PP2048 prefill, Sherlock TG128 decode, and 258048-token prompt.
set -euo pipefail
IMAGE="${IMAGE:-qwen36-flashqla-dflash-longctx-spark:repro}"
CONTAINER="${CONTAINER:-qwen36-flashqla-dflash-longctx}"
MODELS_DIR_HOST="${MODELS_DIR_HOST:-$HOME/models}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="${OUTDIR:-$PWD/results/repro-$STAMP-fullctx}"
mkdir -p "$OUTDIR"
sudo docker rm -f "$CONTAINER" 2>/dev/null || true
sync; printf 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null || true; sudo swapoff -a && sudo swapon -a || true
sudo docker run -d --gpus all --name "$CONTAINER" --network=host --ipc=host \
  -v "$PWD:/repro" -v "$MODELS_DIR_HOST:/models:ro" \
  -e MAX_MODEL_LEN=262144 -e MAX_NUM_BATCHED_TOKENS=8192 -e NUM_SPECULATIVE_TOKENS=8 \
  "$IMAGE" > "$OUTDIR/container.id"
for i in $(seq 1 300); do
  curl -fsS http://127.0.0.1:8000/v1/models > "$OUTDIR/models.json" 2>/dev/null && break
  sleep 5
  [ "$i" != 300 ] || { sudo docker logs "$CONTAINER" > "$OUTDIR/server-timeout.log" 2>&1 || true; exit 1; }
done
sudo docker logs "$CONTAINER" > "$OUTDIR/server-ready.log" 2>&1 || true
sudo docker run --rm --network=host -v "$PWD:/repro" -v "$MODELS_DIR_HOST:/models:ro" "$IMAGE" \
  bash -lc "RUNS=30 WARMUP_RUNS=2 OUT=/repro/results/$(basename "$OUTDIR")/prefill-pp2048-tg32-c1-n30-fullctx.json WARMUP_OUT=/repro/results/$(basename "$OUTDIR")/prefill-warmup-fullctx.json /repro/scripts/bench.sh" | tee "$OUTDIR/prefill.stdout.log"
sudo docker run --rm --network=host -v "$PWD:/repro" -v "$MODELS_DIR_HOST:/models:ro" "$IMAGE" \
  bash -lc "RUNS=30 PP=128 TG=128 OUT=/repro/results/$(basename "$OUTDIR")/sherlock-pp128-tg128-c1-n30-fullctx.json /repro/scripts/bench-sherlock.sh" | tee "$OUTDIR/sherlock.stdout.log"
sudo docker run --rm --network=host -v "$PWD:/repro" -v "$MODELS_DIR_HOST:/models:ro" -e OUTDIR="/repro/results/$(basename "$OUTDIR")" "$IMAGE" \
  python3 /repro/scripts/long_context_probe.py | tee "$OUTDIR/longctx.stdout.log"
sudo docker logs "$CONTAINER" > "$OUTDIR/server-after-all.log" 2>&1 || true
python3 scripts/summarize_results.py "$OUTDIR/prefill-pp2048-tg32-c1-n30-fullctx.json" || true
python3 - <<PY | tee "$OUTDIR/summary.txt"
import json, pathlib, statistics as st
out=pathlib.Path('$OUTDIR')
print('OUTDIR', out)
for name in ['prefill-pp2048-tg32-c1-n30-fullctx.json','sherlock-pp128-tg128-c1-n30-fullctx.json']:
 p=out/name; print('---', name)
 b=json.load(open(p))['benchmarks'][0]
 for k in ['pp_throughput','tg_throughput']:
  m=b[k]; print(k, 'mean', m['mean'], 'median', st.median(m['values']), 'n', len(m['values']))
print('--- longctx')
print((out/'summary.json').read_text())
PY
