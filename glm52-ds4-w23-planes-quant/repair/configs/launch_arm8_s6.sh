#!/usr/bin/env bash
set -uo pipefail
TASK=t_2956f863
ROOT="$HOME/missions/BINREPAIR_${TASK}"
CLAIM="$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json"
PY="$HOME/humming_env/bin/python3"
LOG="$ROOT/launcher.log"
PURPOSE="BINREPAIR arm8: all-43L lr1e-2 x 64w (driver-authorized, high-LR x more-data combo)"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

if [ ! -f "$ROOT/STAGE.COMPLETE" ]; then
  "$PY" "$ROOT/code/stage_verify.py" >> "$LOG" 2>&1 || { say "arm8 STAGE_VERIFY FAILED"; exit 31; }
fi
say "arm8 stage verified"

owner=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("owner",""))' "$CLAIM" 2>/dev/null || echo "")
case "$owner" in
  "$TASK"|"${TASK}-support"|""|"UNCLAIMED") : ;;
  *) say "arm8 foreign claim $owner, refusing"; exit 32 ;;
esac

if pgrep -af '[b]inrepair_e2e.py|[e]2e_fast|[v]q8|[n]orm_tune' >/dev/null 2>&1; then
  say "arm8 refuse: conflicting repair process already alive"
  exit 33
fi

"$PY" - "$TASK" "$CLAIM" "$PURPOSE" <<'PYEOF'
import json, os, sys, tempfile, time
task, path, purpose = sys.argv[1:]
claim = {"owner": task, "host": os.uname().nodename,
         "purpose": purpose,
         "claimed_at_epoch": time.time(), "no_services": True,
         "contact": "kanban t_2f5d8c48; parent mission t_2956f863"}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w") as h:
    json.dump(claim, h, indent=2)
    h.flush()
    os.fsync(h.fileno())
os.replace(tmp, path)
print("CLAIMED arm8")
PYEOF
say "arm8 claim taken"

release_claim() {
  "$PY" - "$TASK" "$CLAIM" "$PURPOSE" <<'PYEOF'
import json, os, sys, tempfile, time
task, path, purpose = sys.argv[1:]
try:
    old = json.load(open(path))
except Exception:
    old = {}
if old.get("owner") == task and old.get("purpose") == purpose:
    claim = {"owner": "UNCLAIMED", "host": os.uname().nodename,
             "purpose": "released after BINREPAIR arm8 exit",
             "released_at_epoch": time.time(), "no_services": True,
             "contact": "kanban t_2f5d8c48"}
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
    with os.fdopen(fd, "w") as h:
        json.dump(claim, h, indent=2)
        h.flush()
        os.fsync(h.fileno())
    os.replace(tmp, path)
    print("RELEASED arm8")
else:
    print("NOT_RELEASED owner/purpose changed")
PYEOF
}
trap release_claim EXIT

for i in $(seq 1 10); do
  sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true
  used=$(free -g | awk '/^Mem:/{print $3}')
  say "arm8 drop_caches pass $i used=${used}G"
  [ "$used" -lt 14 ] && break
  sleep 15
done

say "launching arm8 all43 lr1e-2 64w steps96"
cd "$ROOT"
BR_TAG=arm8_all43_lr1e2_64w \
BR_LR=1e-2 \
BR_TRAINABLE=$(seq -s, 0 42) \
BR_TRAIN=0,7,10,20,29,39,44,51,61,71,82,86,94,104,114,118,126,135,146,151,157,168,178,186,189,199,209,217,221,230,243,250,254,266,275,282,288,298,311,313,321,332,342,348,353,365,376,377,388,397,409,410,420,431,441,444,455,465,472,477,487,498,505,509 \
BR_STEPS=96 \
BR_MAX_HOURS=14 \
  bash "$ROOT/code/run_pilot.sh" >> "$ROOT/out/arm8_all43_lr1e2_64w.log" 2>&1
rc=$?
say "arm8 exited rc=$rc"
exit $rc
