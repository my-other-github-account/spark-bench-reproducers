#!/usr/bin/env bash
set -uo pipefail
TASK=t_2956f863
ROOT="$HOME/missions/BINREPAIR_${TASK}"
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"
PY="$HOME/humming_env/bin/python3"
LOG="$ROOT/launcher.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
# stage verify (md5 all 43 planes vs canonical receipts)
if [ ! -f "$ROOT/STAGE.COMPLETE" ]; then
  "$PY" "$ROOT/code/stage_verify.py" >> "$LOG" 2>&1 || { say "STAGE_VERIFY FAILED"; exit 31; }
fi
say "stage verified"
owner=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"owner\",\"\"))" "$CLAIM" 2>/dev/null || echo "")
case "$owner" in
  "$TASK"|"${TASK}-support"|"") : ;;
  *) say "foreign claim $owner, refusing"; exit 32 ;;
esac
"$PY" - "$TASK" "$CLAIM" <<PYEOF
import json, os, sys, tempfile, time
task, path = sys.argv[1], sys.argv[2]
claim = {"owner": task, "host": os.uname().nodename,
         "purpose": "BINREPAIR arm7: ALL-43-layer vq3b codebooks (1.41M params) lr 1e-2 STEPS96",
         "claimed_at_epoch": time.time(), "no_services": True,
         "contact": "orchestrator-host2 kanban t_2956f863"}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w") as h: json.dump(claim, h, indent=2)
os.replace(tmp, path)
print("CLAIMED arm7")
PYEOF
say "claim taken"
for i in $(seq 1 10); do
  sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true
  used=$(free -g | awk "/^Mem:/{print \$3}")
  say "drop_caches pass $i used=${used}G"
  [ "$used" -lt 14 ] && break
  sleep 15
done
say "launching arm7"
cd "$ROOT"
BR_TAG=arm7_all43 BR_LR=1e-2 BR_TRAINABLE=$(seq -s, 0 42) BR_STEPS=96 BR_MAX_HOURS=14 \
  bash "$ROOT/code/run_pilot.sh" >> "$ROOT/out/arm7.log" 2>&1
rc=$?
say "arm7 exited rc=$rc"
exit $rc
