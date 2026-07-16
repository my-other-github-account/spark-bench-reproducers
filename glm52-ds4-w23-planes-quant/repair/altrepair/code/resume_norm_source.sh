#!/usr/bin/env bash
# Durable one-shot resume of the source RMSNorm run after checkpoint-before-probe death.
set -euo pipefail
TASK=t_7a65a4c6
ROOT="$HOME/missions/ALTREPAIR_${TASK}"
OUT="$ROOT/out"
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"
PY=${PY:-$HOME/humming_env/bin/python3}
TAG=rmsnorm_all_lr1e4_b2
PREFIX="$OUT/BINREPAIR_${TAG}"
TOMBSTONE="$PREFIX.tombstone.json"
PIDFILE="$PREFIX.resume.pid"
LOG="$OUT/${TAG}.resume.log"

owner="$($PY -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner",""))' "$CLAIM")"
[[ "$owner" == "$TASK" ]] || { echo "REFUSE: owner=$owner expected=$TASK" >&2; exit 3; }
[[ ! -e "$PREFIX.final.json" ]] || { echo "ALREADY COMPLETE"; exit 0; }
[[ ! -e "$TOMBSTONE" ]] || { echo "REFUSE: tombstone exists" >&2; exit 3; }
[[ -s "$PREFIX.latest.pt" ]] || { echo "REFUSE: latest checkpoint absent" >&2; exit 3; }
if pgrep -af "$ROOT/code/norm_tune_e2e.py" >/dev/null; then
  echo "REFUSE: norm trainer already active" >&2
  pgrep -af "$ROOT/code/norm_tune_e2e.py" >&2 || true
  exit 3
fi

"$PY" - "$PREFIX.latest.pt" <<'PY'
import sys, torch
p = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert p["format"] == "altrepair-rmsnorm-v1", p.get("format")
assert p["mechanism"] == "rmsnorm-gamma", p.get("mechanism")
assert p["next_step"] == 24, p.get("next_step")
assert p["steps_target"] == 24, p.get("steps_target")
print("resume checkpoint gate PASS: next_step=24 post-update panel pending", flush=True)
PY

nohup bash -lc '
  set -uo pipefail
  ROOT="$HOME/missions/ALTREPAIR_t_7a65a4c6"
  OUT="$ROOT/out"
  TAG=rmsnorm_all_lr1e4_b2
  export BR_TAG="$TAG" BR_STEPS=24 BR_PROBE_EVERY=8 BR_MAX_HOURS=6 BR_LR=1e-4 BR_BATCH=2
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
kill -0 "$pid" 2>/dev/null || { echo "FAIL: resume wrapper exited immediately; inspect $LOG" >&2; exit 4; }
printf 'RESUMED tag=%s pid=%s log=%s\n' "$TAG" "$pid" "$LOG"
