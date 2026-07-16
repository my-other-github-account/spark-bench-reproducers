#!/usr/bin/env bash
# BINREPAIR pilot launch (t_2956f863). Run ONLY after:
#   1. ~/missions/BINREPAIR_t_2956f863/STAGE.COMPLETE exists (planes verified)
#   2. HOST_CLAIM at ~/missions/LP4_BLOCKWISE/HOST_CLAIM.json owner == t_2956f863
#   3. GPU is free (no vllm/e2e/ab64 co-tenant; free mem > 100G)
set -euo pipefail
TASK=t_2956f863
ROOT="$HOME/missions/BINREPAIR_${TASK}"
PY=${PY:-$HOME/humming_env/bin/python3}
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"

[[ -f "$ROOT/STAGE.COMPLETE" ]] || { echo "REFUSE: staging not verified" >&2; exit 3; }
owner="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner",""))' "$CLAIM")"
[[ "$owner" == "$TASK" ]] || { echo "REFUSE: claim owner=$owner expected=$TASK" >&2; exit 3; }
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0)

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"   # nvcc lesson from t_880fdca2
export BR_MANIFEST="$ROOT/code/DUALVQ_K4096MENU_IQ3_BIN_MANIFEST.json"
export BR_DELTA_DIR="$ROOT/delta"
export BR_VQ3B_DIR="${BR_VQ3B_DIR:-$ROOT/planes}"
export BR_TRAINABLE=${BR_TRAINABLE:-23,33,41}
# Eval-corpus windows DISJOINT from WIDEDATA t_c65a87a6 manifest (train/probes/reserved)
export BR_PROBE="${BR_PROBE:-4,84,160,236,304,373,442,511}"
export BR_TRAIN="${BR_TRAIN:-7,44,86,118,151,186,217,250,282,313,348,377,409,441,472,505}"
export BR_STEPS=${BR_STEPS:-48} BR_LR=${BR_LR:-1e-2} BR_BATCH=${BR_BATCH:-2}
export BR_PROBE_EVERY=8 BR_EARLY_STOP=3 BR_MAX_HOURS=${BR_MAX_HOURS:-12}
export BR_OUTDIR="${BR_OUTDIR:-$ROOT/out}" BR_TAG=${BR_TAG:-pilot1}
export BR_TEACH="$HOME/missions/DS4_TEACHER/t8192_eval"
export BR_CORPUS="$HOME/servedab/windows_ds4_eval.json"
export BR_REF_KLD="${BR_REF_KLD:-$ROOT/code/ledger_ref.json}"
export BR_GRADCHECK=1 BR_CACHE_ONLY=${BR_CACHE_ONLY:-0}

mkdir -p "$ROOT/out"
exec "$PY" -u "$ROOT/code/binrepair_e2e.py"
