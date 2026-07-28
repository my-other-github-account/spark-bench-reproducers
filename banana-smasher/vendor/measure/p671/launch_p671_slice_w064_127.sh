#!/usr/bin/env bash
set -euo pipefail
ROOT=$HOME/run-bundles/P671_FULL512_PUBLIC_TASK_s8
RUN_ID=P640_SLICE_W064_127
PID=$$
PGID=$(ps -o pgid= -p $$ | tr -d ' ')
SID=$(ps -o sid= -p $$ | tr -d ' ')
LOG="$ROOT/logs/slice_w064_127.log"
printf '%s\n' "$PID" > "$ROOT/run/SLICE_W064_127_PID"
PID="$PID" PGID="$PGID" SID="$SID" ROOT="$ROOT" python3 -c 'import json,os,time,pathlib,uuid; r=pathlib.Path(os.environ["ROOT"]); p=r/"PROGRESS.json"; t=p.with_name("."+p.name+"."+uuid.uuid4().hex+".tmp"); x={"schema":"p671-progress-v1","status":"SLICE_W064_127_LAUNCHED","task_id":"PUBLIC_TASK","host":"compute-node-8","mission":str(r),"pid":int(os.environ["PID"]),"pgid":int(os.environ["PGID"]),"sid":int(os.environ["SID"]),"log":str(r/"logs/slice_w064_127.log"),"mode":"slice_w064_127","run_id":"P640_SLICE_W064_127","window_start":64,"window_end":127,"window_count":64,"microbatch":8,"microbatch_policy":"MB8: highest remaining after sealed P660 MB16 OOM; measured safe by this compute-node-8 run","updated_unix":time.time()}; t.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n"); os.replace(t,p)'
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec $HOME/humming_env/bin/python3 -u "$ROOT/code/p671_slice_w064_127.py" --mode slice_w064_127
