#!/usr/bin/env bash
set -euo pipefail
umask 077
TASK=task-redacted
MISSION=${SPARK_HOME}/missions/IQ3_BATCH_INVARIANCE_t_fafb2fe1
STARTED=$(python3 -c 'import time;print(time.time())')
SERVER_PID=''
atomic_status(){ python3 - "$MISSION/run/STATUS.json" "$1" "$2" <<'PY'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);o={'schema':'iq3-batching-invariance-status-v1','task':'task-redacted','state':sys.argv[2],'detail':sys.argv[3],'updated_epoch':time.time()};tmp=p.with_name(p.name+f'.tmp.{os.getpid()}')
with tmp.open('w') as f:json.dump(o,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PY
}
claim_guard(){ python3 - <<'PY'
import json
v=json.load(open('${SPARK_HOME}/HOST_CLAIM.json'))
assert v.get('owner')=='task-redacted' and v.get('mission')=='${SPARK_HOME}/missions/IQ3_BATCH_INVARIANCE_t_fafb2fe1',v
PY
}
stop_server(){
 if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
  CMD=$(tr '\0' ' ' < "/proc/$SERVER_PID/cmdline")
  [[ "$CMD" == *"llama-server"* && "$CMD" == *"UD-IQ3_XXS"* ]] || { echo "REFUSE unexpected server $SERVER_PID $CMD" >&2; return 90; }
  kill "$SERVER_PID"
  for _ in $(seq 1 120); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
  if kill -0 "$SERVER_PID" 2>/dev/null; then kill -9 "$SERVER_PID"; fi
  wait "$SERVER_PID" 2>/dev/null || true
 fi
 SERVER_PID=''
 for _ in $(seq 1 120); do
  APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
  [[ -z "${APPS//[[:space:]]/}" ]] && return 0
  sleep 1
 done
 echo "GPU failed to empty" >&2; return 91
}
cleanup(){ rc=$?; stop_server || true; if [[ $rc -ne 0 ]]; then atomic_status failed "pipeline rc=$rc" || true; fi; exit $rc; }
trap cleanup EXIT INT TERM
mkdir -p "$MISSION/logs" "$MISSION/run" "$MISSION/receipts" "$MISSION/results"
claim_guard
atomic_status running 'pre-registration sealed; launching arm A parallel1/client1'
launch_arm(){
 ARM=$1; PAR=$2; PORT=$3
 claim_guard
 APPS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
 [[ -z "${APPS//[[:space:]]/}" ]] || { echo "GPU busy before arm $ARM: $APPS" >&2; return 77; }
 bash "$MISSION/code/start_server.sh" "$ARM" "$PAR" "$PORT" > "$MISSION/logs/server_${ARM}.log" 2>&1 &
 SERVER_PID=$!; echo "$SERVER_PID" > "$MISSION/run/server_${ARM}.pid"
 python3 - "$ARM" "$PORT" "$SERVER_PID" "$STARTED" "$MISSION" <<'PY'
import json,os,pathlib,subprocess,sys,time,urllib.request
arm,port,pid,claimed,mission=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),pathlib.Path(sys.argv[5]);deadline=time.time()+1200;cuda_epoch=None
while time.time()<deadline:
 if not pathlib.Path(f'/proc/{pid}').exists():raise RuntimeError(f'server {arm} died before ready')
 out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],text=True).strip()
 if out and cuda_epoch is None:cuda_epoch=time.time()
 try:
  with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:
   if r.status==200:
    now=time.time();receipt={'schema':'iq3-batching-invariance-cuda-gate-v1','status':'PASS','task':'task-redacted','arm':arm,'server_pid':pid,'claim_started_epoch':claimed,'cuda_first_seen_epoch':cuda_epoch,'api_ready_epoch':now,'claim_to_cuda_seconds':None if cuda_epoch is None else cuda_epoch-claimed,'claim_to_api_ready_seconds':now-claimed,'cuda_within_20m':cuda_epoch is not None and cuda_epoch-claimed<=1200}
    if not receipt['cuda_within_20m']:raise RuntimeError(f'CUDA >20m gate failed {receipt}')
    p=mission/f'receipts/CUDA_GATE_{arm}.json';tmp=p.with_name(p.name+f'.tmp.{os.getpid()}');tmp.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');os.replace(tmp,p);print(json.dumps(receipt,sort_keys=True));raise SystemExit
 except Exception as e:
  if isinstance(e,SystemExit):raise
 time.sleep(2)
raise RuntimeError(f'server {arm} readiness timeout')
PY
}
launch_arm A 1 8331
atomic_status running 'arm A ready; generating ordered 10 rows serially'
python3 "$MISSION/code/generate_spot.py" --arm A --port 8331 --concurrency 1 | tee "$MISSION/logs/generate_A.log"
stop_server
atomic_status running 'arm A sealed and stopped; launching arm B parallel4/client4'
launch_arm B 4 8332
atomic_status running 'arm B ready; generating same ordered 10 rows concurrency4'
python3 "$MISSION/code/generate_spot.py" --arm B --port 8332 --concurrency 4 | tee "$MISSION/logs/generate_B.log"
stop_server
atomic_status running 'both arms sealed; comparing visible bytes/tokens/reasons/runtime fingerprints'
python3 "$MISSION/code/compare_spot.py" | tee "$MISSION/logs/compare.log"
python3 - "$MISSION/run/RESUME.json" <<'PY'
import json,os,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);o={'schema':'iq3-batching-invariance-resume-v1','task':'task-redacted','state':'GENERATION_AND_COMPARE_COMPLETE','next':'score selected arm A/B samples with pinned EvalPlus 26d6d00, then final merge/release','comparison':'${SPARK_HOME}/missions/IQ3_BATCH_INVARIANCE_t_fafb2fe1/receipts/GENERATION_COMPARISON.json','updated_epoch':time.time()};tmp=p.with_name(p.name+f'.tmp.{os.getpid()}');tmp.write_text(json.dumps(o,indent=2,sort_keys=True)+'\n');os.replace(tmp,p)
PY
atomic_status sealed 'generation and byte comparison complete; GPU empty; selected EvalPlus scoring next'
trap - EXIT INT TERM
exit 0
