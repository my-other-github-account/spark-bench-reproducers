#!/usr/bin/env bash
# One-shot guarded launch for the fresh-tag RMSNorm replication.
set -euo pipefail

TASK=t_7a65a4c6
ROOT="$HOME/missions/ALTREPAIR_${TASK}"
OUT="$ROOT/out"
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"
PY=${PY:-$HOME/humming_env/bin/python3}
SOURCE_TAG=BINREPAIR_rmsnorm_all_lr1e4_b2
TAG=rmsnorm_all_lr1e4_b2_rep1_rot8
PREFIX="$OUT/BINREPAIR_${TAG}"
TOMBSTONE="$PREFIX.tombstone.json"
PIDFILE="$PREFIX.pid"
LOG="$OUT/${TAG}.log"
ROTATED_TRAIN=282,313,348,377,409,441,472,505,7,44,86,118,151,186,217,250
SEED="$OUT/${SOURCE_TAG}.latest.pt"

owner="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner",""))' "$CLAIM")"
[[ "$owner" == "$TASK" ]] || { echo "REFUSE: owner=$owner expected=$TASK" >&2; exit 3; }
[[ -s "$OUT/${SOURCE_TAG}.final.json" ]] || { echo "REFUSE: source final absent" >&2; exit 3; }
[[ -s "$SEED" ]] || { echo "REFUSE: baseline seed absent" >&2; exit 3; }
[[ ! -e "$TOMBSTONE" ]] || { echo "REFUSE: tombstone exists: $TOMBSTONE" >&2; exit 3; }
[[ ! -e "$PREFIX.final.json" ]] || { echo "ALREADY COMPLETE: $PREFIX.final.json"; exit 0; }

if pgrep -af "$ROOT/code/norm_tune_e2e.py" >/dev/null; then
  echo "REFUSE: norm trainer already active" >&2
  pgrep -af "$ROOT/code/norm_tune_e2e.py" >&2 || true
  exit 3
fi

# A fresh tag must not silently inherit optimizer/model state. Baseline is loaded
# from SOURCE_TAG only; step-0 still performs the full independent 8-probe panel.
if [[ -e "$PREFIX.latest.pt" || -e "$PREFIX.status.json" || -e "$PREFIX.jsonl" ]]; then
  echo "REFUSE: fresh-tag artifacts already exist; inspect before resume" >&2
  exit 3
fi

mkdir -p "$OUT"
nohup bash -lc '
  set -uo pipefail
  ROOT="$HOME/missions/ALTREPAIR_t_7a65a4c6"
  OUT="$ROOT/out"
  TAG=rmsnorm_all_lr1e4_b2_rep1_rot8
  export ALT_BASELINE_CKPT="$OUT/BINREPAIR_rmsnorm_all_lr1e4_b2.latest.pt"
  export BR_TAG="$TAG"
  export BR_TRAIN=282,313,348,377,409,441,472,505,7,44,86,118,151,186,217,250
  export BR_STEPS=16 BR_PROBE_EVERY=8 BR_MAX_HOURS=5 BR_LR=1e-4 BR_BATCH=2
  bash "$ROOT/code/run_norm_tune.sh"
  rc=$?
  if (( rc != 0 )); then
    "$HOME/humming_env/bin/python3" - "$OUT/BINREPAIR_${TAG}.tombstone.json" "$rc" <<'"'"'PY'"'"'
import json, os, pathlib, sys, tempfile, time
p = pathlib.Path(sys.argv[1])
payload = {"state": "failed", "exit_code": int(sys.argv[2]), "ts": time.time()}
fd, tmp = tempfile.mkstemp(prefix=p.name + ".", dir=p.parent)
with os.fdopen(fd, "w") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, p)
PY
  fi
  exit "$rc"
' >"$LOG" 2>&1 </dev/null &
pid=$!
printf '%s\n' "$pid" >"$PIDFILE.tmp"
mv "$PIDFILE.tmp" "$PIDFILE"
sleep 2
kill -0 "$pid" 2>/dev/null || { echo "FAIL: replication wrapper exited immediately; inspect $LOG" >&2; exit 4; }
printf 'LAUNCHED tag=%s pid=%s log=%s\n' "$TAG" "$pid" "$LOG"
