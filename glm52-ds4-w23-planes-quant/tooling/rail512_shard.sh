#!/usr/bin/env bash
# Export arm4 codebooks into local VQ3U planes and evaluate one fleet shard.
# Usage: MISSION_ROOT=/path/to/missions bash rail512_shard.sh "0,1,..." spark-N
set -euo pipefail

[[ $# -eq 2 ]] || { echo "usage: $0 <comma-window-list> <shard-tag>" >&2; exit 2; }
: "${MISSION_ROOT:?set MISSION_ROOT to the campaign mission parent}"
WINS="$1"
TAG="$2"
PY="${PY:-python3}"
REPAIR_ROOT="${REPAIR_ROOT:-$MISSION_ROOT/BINREPAIR_t_2956f863}"
RAIL_ROOT="${RAIL_ROOT:-$MISSION_ROOT/RAIL512}"
ARM4_CKPT="${ARM4_CKPT:-$RAIL_ROOT/arm4.best.pt}"
BASE_PLANES_DIR="${BASE_PLANES_DIR:-$REPAIR_ROOT/planes}"
EXPORTED_PLANES_DIR="${EXPORTED_PLANES_DIR:-$MISSION_ROOT/SERVED_AB/planes_arm4}"
REF_LEDGER="${REF_LEDGER:-$RAIL_ROOT/ledger_512.json}"
RUN_PILOT="${RUN_PILOT:-$REPAIR_ROOT/code/run_pilot.sh}"
EXPORT_TOOL="${EXPORT_TOOL:-$(cd "$(dirname "$0")" && pwd)/export_arm4.py}"
LOG="$RAIL_ROOT/${TAG}.log"
mkdir -p "$RAIL_ROOT"

say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG"; }

if [[ ! -f "$EXPORTED_PLANES_DIR/EXPORT_META.json" ]]; then
  say "exporting arm4 codebooks"
  "$PY" "$EXPORT_TOOL" \
    --checkpoint "$ARM4_CKPT" \
    --base-planes "$BASE_PLANES_DIR" \
    --output "$EXPORTED_PLANES_DIR" >>"$LOG" 2>&1
fi
[[ -f "$EXPORTED_PLANES_DIR/vq3u_layer_042.pt" ]] || {
  say "export is incomplete"; exit 41;
}
FIRST="${WINS%%,*}"
say "starting shard first=$FIRST"
BR_TAG="rail512_${TAG}" \
BR_VQ3B_DIR="$EXPORTED_PLANES_DIR" \
BR_TRAIN="$WINS" \
BR_PROBE="$FIRST" \
BR_STEPS=0 \
BR_GRADCHECK=0 \
BR_LR=1e-9 \
BR_REF_KLD="$REF_LEDGER" \
BR_OUTDIR="$RAIL_ROOT" \
BR_MAX_HOURS="${BR_MAX_HOURS:-8}" \
  bash "$RUN_PILOT" >>"$LOG" 2>&1
say "shard complete"
