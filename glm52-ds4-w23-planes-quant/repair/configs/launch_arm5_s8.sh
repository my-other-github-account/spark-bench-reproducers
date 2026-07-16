#!/usr/bin/env bash
set -uo pipefail
TASK=t_2956f863
ROOT="$HOME/missions/BINREPAIR_${TASK}"
PY="$HOME/humming_env/bin/python3"
LOG="$ROOT/launcher.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
busy=$(pgrep -af "vllm serve|ab64|e2e_fast|served_nll" | grep -v grep | wc -l)
[ "$busy" -eq 0 ] || { say "arm5 refuse: host busy"; exit 32; }
"$PY" - "$TASK" "$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json" <<PYEOF
import json, os, sys, tempfile, time
task, path = sys.argv[1], sys.argv[2]
claim = {"owner": task, "host": os.uname().nodename,
         "purpose": "BINREPAIR arm5: all-43L vq3b, 64 train windows, lr 3e-3",
         "claimed_at_epoch": time.time(), "no_services": True,
         "contact": "orchestrator-host2 kanban t_2956f863"}
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path))
with os.fdopen(fd, "w") as h: json.dump(claim, h, indent=2)
os.replace(tmp, path)
print("CLAIMED arm5")
PYEOF
for i in 1 2 3 4 5; do sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true; used=$(free -g | awk "/^Mem:/{print \$3}"); say "arm5 drop_caches $i used=${used}G"; [ "$used" -lt 14 ] && break; sleep 15; done
say "launching arm5 (all43 lr3e-3 64w)"
cd "$ROOT"
BR_TAG=arm5_all43_64w BR_LR=3e-3 BR_TRAINABLE=$(seq -s, 0 42) \
BR_TRAIN=0,7,10,20,29,39,44,51,61,71,82,86,94,104,114,118,126,135,146,151,157,168,178,186,189,199,209,217,221,230,243,250,254,266,275,282,288,298,311,313,321,332,342,348,353,365,376,377,388,397,409,410,420,431,441,444,455,465,472,477,487,498,505,509 \
BR_STEPS=96 BR_MAX_HOURS=14 \
  bash "$ROOT/code/run_pilot.sh" >> "$ROOT/out/arm5.log" 2>&1
rc=$?; say "arm5 exited rc=$rc"; exit $rc
