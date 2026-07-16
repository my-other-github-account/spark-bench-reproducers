#!/usr/bin/env bash
# Launch the orthogonal output-scale repair pilot on a host already staged for
# BINREPAIR.  This script never takes or overwrites a host claim; the caller
# must first acquire the canonical claim for t_7a65a4c6.
set -euo pipefail

TASK=t_7a65a4c6
PARENT=t_2956f863
ROOT="$HOME/missions/ALTREPAIR_${TASK}"
PARENT_ROOT="$HOME/missions/BINREPAIR_${PARENT}"
PY=${PY:-$HOME/humming_env/bin/python3}
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"

[[ -f "$ROOT/code/output_scale_e2e.py" ]] || {
  echo "REFUSE: missing output-scale trainer" >&2; exit 3;
}
[[ -f "$PARENT_ROOT/code/binrepair_e2e.py" ]] || {
  echo "REFUSE: missing proven BINREPAIR base harness" >&2; exit 3;
}
[[ -f "$PARENT_ROOT/code/DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json" ]] || {
  echo "REFUSE: missing target manifest" >&2; exit 3;
}
[[ -f "$PARENT_ROOT/delta/DELTA_PACK.COMPLETE" ]] || {
  echo "REFUSE: incomplete delta pack" >&2; exit 3;
}

owner="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner",""))' "$CLAIM")"
[[ "$owner" == "$TASK" ]] || {
  echo "REFUSE: claim owner=$owner expected=$TASK" >&2; exit 3;
}

# Instant artifact gate only: all paths exist, all sizes are nonzero, and one
# plane header loads.  Step-0 ledger parity is the authoritative integrity gate.
"$PY" - "$PARENT_ROOT/planes" <<'PY'
import sys
from pathlib import Path
import torch
root = Path(sys.argv[1])
paths = [root / f"vq3u_layer_{layer:03d}.pt" for layer in range(43)]
bad = [str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0]
if bad:
    raise SystemExit(f"REFUSE: missing/empty planes: {bad}")
sample = torch.load(paths[0], map_location="cpu", mmap=True, weights_only=True)
expected = {"cb13", "cb2", "codes13", "codes2", "sc13", "sc2"}
missing = expected - set(sample)
if missing:
    raise SystemExit(f"REFUSE: sample plane missing keys: {sorted(missing)}")
print("instant artifact gate PASS: 43 files + L000 header", flush=True)
PY

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export PYTHONHASHSEED=0
export ALT_BINREPAIR_BASE="$PARENT_ROOT/code/binrepair_e2e.py"
export ALT_GAIN_CLAMP=${ALT_GAIN_CLAMP:-0.25}
seed="$PARENT_ROOT/out/BINREPAIR_arm4_all43_lr1e2.latest.pt"
if [[ -f "$seed" ]]; then
  export ALT_BASELINE_CKPT=${ALT_BASELINE_CKPT:-$seed}
fi
export BR_MANIFEST="$PARENT_ROOT/code/DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json"
export BR_DELTA_DIR="$PARENT_ROOT/delta"
export BR_VQ3B_DIR="$PARENT_ROOT/planes"
export BR_TRAINABLE=${BR_TRAINABLE:-$(seq -s, 0 42)}
export BR_PROBE=4,84,160,236,304,373,442,511
export BR_TRAIN=7,44,86,118,151,186,217,250,282,313,348,377,409,441,472,505
export BR_STEPS=${BR_STEPS:-16}
export BR_LR=${BR_LR:-1e-2}
export BR_BATCH=${BR_BATCH:-2}
export BR_PROBE_EVERY=${BR_PROBE_EVERY:-8}
export BR_EARLY_STOP=${BR_EARLY_STOP:-2}
export BR_MAX_HOURS=${BR_MAX_HOURS:-4}
export BR_OUTDIR="$ROOT/out"
export BR_TAG=${BR_TAG:-output_scale_all43_lr1e2_b2}
export BR_TEACH="$HOME/missions/DS4_TEACHER/t8192_eval"
export BR_CORPUS="$HOME/servedab/windows_ds4_eval.json"
export BR_REF_KLD="$PARENT_ROOT/code/ledger_ref.json"
export BR_GRADCHECK=1
export BR_CACHE_ONLY=0

mkdir -p "$ROOT/out"
exec "$PY" -u "$ROOT/code/output_scale_e2e.py"
