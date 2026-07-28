#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPARK_HOME}/missions/P948_SPECULATIVE_REPAIR_t_b1aa61d3_s3
LABEL=SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL
mkdir -p "$ROOT"/{logs,run,receipts}
exec >>"$ROOT/logs/${LABEL}__P948_CONTROLLER.stdout.log" 2>>"$ROOT/logs/${LABEL}__P948_CONTROLLER.stderr.log"
printf '{"schema":"p948-controller-progress-v1","state":"PREPARING","speculative_label":"%s","pid":%s,"updated_unix":%s}\n' "$LABEL" "$$" "$(date +%s)" >"$ROOT/run/.P948_CONTROLLER_PROGRESS.json.tmp.$$"
mv "$ROOT/run/.P948_CONTROLLER_PROGRESS.json.tmp.$$" "$ROOT/run/P948_CONTROLLER_PROGRESS.json"
${SPARK_HOME}/humming_env/bin/python3 -u "$ROOT/code/prepare_p948.py"
printf '{"schema":"p948-controller-progress-v1","state":"LAUNCHING_UPDATE_001","speculative_label":"%s","pid":%s,"updated_unix":%s}\n' "$LABEL" "$$" "$(date +%s)" >"$ROOT/run/.P948_CONTROLLER_PROGRESS.json.tmp.$$"
mv "$ROOT/run/.P948_CONTROLLER_PROGRESS.json.tmp.$$" "$ROOT/run/P948_CONTROLLER_PROGRESS.json"
set +e
"$ROOT/code/run_p948_speculative_repair.sh" 1
rc=$?
set -e
export rc
${SPARK_HOME}/humming_env/bin/python3 - <<'PY'
import json,os,tempfile,time
from pathlib import Path
p=Path('${SPARK_HOME}/missions/P948_SPECULATIVE_REPAIR_t_b1aa61d3_s3/run/P948_CONTROLLER_PROGRESS.json')
obj={'schema':'p948-controller-progress-v1','state':'UPDATE_001_EXITED','trainer_rc':int(os.environ['rc']),'speculative_label':'SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL','pid':os.getpid(),'updated_unix':time.time()}
fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
with os.fdopen(fd,'w') as f:json.dump(obj,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PY
exit "$rc"
