#!/usr/bin/env bash
set -euo pipefail
ROOT=$HOME/run-bundles/P656_EARLY8_PUBLIC_TASK_s6
TASK=PUBLIC_TASK
PY=$HOME/humming_env/bin/python3
RUNNER="$ROOT/code/p656_early8.py"
LOG="$ROOT/logs/P656_EARLY8.log"
PID_JSON="$ROOT/run/P656_EARLY8_PID.json"
mkdir -p "$ROOT"/{logs,run,receipts,out,scratch}
"$PY" - "$ROOT" "$TASK" <<'PY'
import hashlib,json,subprocess,sys
from pathlib import Path
root,task=Path(sys.argv[1]),sys.argv[2]
raw=Path('$HOME/HOST_CLAIM.json').read_bytes(); claim=json.loads(raw)
expected={'host':'compute-node-6','owner':task,'task':task,'task_id':task,'mission':str(root)}
drift={k:(claim.get(k),v) for k,v in expected.items() if claim.get(k)!=v}
if drift or claim.get('status') not in ('CLAIMED','ACTIVE'): raise RuntimeError((drift,claim.get('status')))
apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True).stdout.strip().splitlines()
if apps: raise RuntimeError(f'foreign compute apps: {apps}')
for path in (root/'receipts/RAIL_EARLY8.json',root/'receipts/P656_EARLY8.json',root/'out/P640_FINAL_early8',root/'scratch/P640_FINAL_early8'):
 if path.exists(): raise RuntimeError(f'once-only target exists: {path}')
print(json.dumps({'status':'PASS_PRELAUNCH','claim_sha256':hashlib.sha256(raw).hexdigest(),'gpu_compute_apps':[]},sort_keys=True))
PY
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
setsid nohup "$PY" -u "$RUNNER" </dev/null >>"$LOG" 2>&1 &
PID=$!
sleep 3
kill -0 "$PID"
PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
SID=$(ps -o sid= -p "$PID" | tr -d ' ')
export ROOT TASK PY RUNNER LOG PID PGID SID PID_JSON
"$PY" - <<'PY'
import hashlib,json,os,time
from pathlib import Path
root=Path(os.environ['ROOT']); out=Path(os.environ['PID_JSON']); runner=Path(os.environ['RUNNER'])
raw=Path('$HOME/HOST_CLAIM.json').read_bytes()
x={'schema':'p656-detached-early8-launch-v1','status':'RUNNING','task_id':os.environ['TASK'],'host':'compute-node-6','pid':int(os.environ['PID']),'pgid':int(os.environ['PGID']),'sid':int(os.environ['SID']),'python':os.environ['PY'],'runner':str(runner),'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'log':os.environ['LOG'],'claim_sha256':hashlib.sha256(raw).hexdigest(),'no_service_manager':True,'no_tailscale':True,'started_unix':time.time()}
tmp=out.with_name('.'+out.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(tmp,out);print(json.dumps(x,sort_keys=True))
PY
