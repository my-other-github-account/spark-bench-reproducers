#!/usr/bin/env bash
set -euo pipefail
VARIANT=${1:?usage: run_real_plane_gate.sh l3|l3b|l3c|l3d /path/to/layer_042 output.json}
PREFIX=${2:?provide real-plane prefix}
OUTPUT=${3:?provide output JSON}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
HARNESS="$ROOT/../concurrency-cliff/bench/microbench_vq_warp_m4.py"
MODULE=${MODULE:?set MODULE to the compatible moe_vq_triton.py runtime patch}
KERNEL="$ROOT/artifacts/kernel/vq_warp_${VARIANT}"
PYTHONPATH="$KERNEL${PYTHONPATH:+:$PYTHONPATH}" \
python "$HARNESS" --module "$MODULE" --prefix "$PREFIX" --output "$OUTPUT" \
  --warmup 5 --reps 15
