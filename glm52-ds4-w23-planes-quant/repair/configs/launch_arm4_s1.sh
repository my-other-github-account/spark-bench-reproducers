#!/usr/bin/env bash
set -uo pipefail
TASK=t_2956f863
ROOT="$HOME/missions/BINREPAIR_${TASK}"
PY="$HOME/humming_env/bin/python3"
LOG="$ROOT/launcher.log"
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
owner=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(\"owner\",\"\"))" "$HOME/missions/LP4_BLOCKWISE/HOST_CLAIM.json" 2>/dev/null || echo "")
[ "$owner" = "$TASK" ] || { say "arm4 refuse: claim owner=$owner"; exit 32; }
for i in 1 2 3 4 5; do sudo -n sh -c "sync; echo 3 > /proc/sys/vm/drop_caches" 2>/dev/null || true; used=$(free -g | awk "/^Mem:/{print \$3}"); say "arm4 drop_caches $i used=${used}G"; [ "$used" -lt 14 ] && break; sleep 15; done
say "launching arm4 (all43 lr1e-2 16w)"
cd "$ROOT"
BR_TAG=arm4_all43_lr1e2 BR_LR=1e-2 BR_TRAINABLE=$(seq -s, 0 42) BR_MAX_HOURS=14 \
  bash "$ROOT/code/run_pilot.sh" >> "$ROOT/out/arm4.log" 2>&1
rc=$?; say "arm4 exited rc=$rc"; exit $rc
