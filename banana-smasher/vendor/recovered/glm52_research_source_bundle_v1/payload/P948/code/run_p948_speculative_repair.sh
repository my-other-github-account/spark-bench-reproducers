#!/usr/bin/env bash
set -euo pipefail
ROOT=${SPARK_HOME}/missions/P948_SPECULATIVE_REPAIR_t_b1aa61d3_s3
TASK=task-redacted
CLAIM_SHA=6eb7d061bc57a364404eeaf1de22eb0f67e14bb59c888954337cbeb015593b8c
STOP_AFTER="${1:?usage: run_p911_repair.sh STOP_AFTER_UPDATE}"
[[ "$STOP_AFTER" == "1" || "$STOP_AFTER" == "24" ]]
mkdir -p "$ROOT"/{checkpoints,logs,receipts,rollback,run/basic_harness,stage_cache}
exec 9>"$ROOT/run/P911.lock"
flock -n 9 || { echo 'P911 already active' >&2; exit 73; }
if [[ "$STOP_AFTER" == "24" ]]; then
  [[ -f "$ROOT/receipts/FIRST_UNIT_ACCEPTANCE.json" ]] || { echo 'FIRST_UNIT_ACCEPTANCE missing' >&2; exit 74; }
fi
export PYTHONPATH="$ROOT/code"
export PYTHONHASHSEED=0
export CUDA_MODULE_LOADING=EAGER
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export GENESIS_REPAIR_DETERMINISTIC=0
export GENESIS_REPAIR_REQUIRE_DETERMINISTIC_REDUCTION=0
export GENESIS_REPAIR_FUSED_ADAM=0
export GENESIS_REPAIR_DEQ_CHUNK=1
export GENESIS_REPAIR_NATIVE_CHUNK=1
export GENESIS_REPAIR_EXPERT_RESIDENT_SCOPE=4
export GENESIS_REPAIR_EVICT=1
export GENESIS_REPAIR_CHECKPOINT=1
export GENESIS_REPAIR_MICROBATCH=4
export GENESIS_REPAIR_DIRECTIONAL_BATCH=4
export GENESIS_REPAIR_MEM_FLOOR_BYTES=8589934592
export GENESIS_REPAIR_COMPILE_VQ=0
export GENESIS_REPAIR_FIRST_RESUME_GATE_SECONDS=7200.0
export GENESIS_SPECULATIVE_SKIP_PREFLIGHT_INSTRUMENTS=1
export GENESIS_REPAIR_CADENCE_MIN_SECONDS=480.0
export GENESIS_REPAIR_CADENCE_MAX_SECONDS=7200.0
export GENESIS_REPAIR_CODE_GUARD_KLD=0.045
export GENESIS_REPAIR_DEVICE=cuda
export GENESIS_REPAIR_ROOT="$ROOT"
export GENESIS_SPECULATIVE_LABEL=SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL
export GENESIS_REPAIR_CLAIM_MISSION="$ROOT"
export GENESIS_REPAIR_EXPECTED_CLAIM_SHA256="$CLAIM_SHA"
export GENESIS_REPAIR_STOP_AFTER_UPDATE="$STOP_AFTER"
export GENESIS_REPAIR_UPDATE0_SMOKE_ONLY=0
export GENESIS_REPAIR_DIRECTIONAL_SPEC="$ROOT/inputs/P680_TRAIN8_DIRECTIONAL_SPEC.json"
export GENESIS_HOST_CLAIM=${SPARK_HOME}/HOST_CLAIM.json
export GENESIS_TASK_ID="$TASK"
export GENESIS_PHYSICAL_PACKAGE=${SPARK_HOME}/missions/P875_QTIP3_FORTRESS_t_67604030_s3/base_wire
export GENESIS_ASSIGNMENT="$ROOT/inputs/WIRE_C_V2_RECONSTRUCTED_ASSIGNMENT.json"
export GENESIS_BASE_ASSIGNMENT="$ROOT/inputs/BASE_ASSIGNMENT.json"
export GENESIS_F521_ASSIGNMENT="$GENESIS_ASSIGNMENT"
export GENESIS_F521_MANIFEST="$ROOT/inputs/P948_SPECULATIVE_MANIFEST.json"
export GENESIS_F521_SOURCE_PLAN="$ROOT/inputs/P948_SPECULATIVE_SOURCE_PLAN.json"
export GENESIS_F521_STAGE_ROOT="$ROOT/stage_cache"
export GENESIS_F521_SLICES_ROOT="$ROOT/slices"
export GENESIS_F521_CODEBOOK_RECEIPT="$ROOT/receipts/P948_CODEBOOK_SET_184.json"
export GENESIS_F521_CODEBOOK_ROOT="$ROOT/pinned_codebooks"
export GENESIS_F521_CHECKPOINT_ROOT=${SPARK_HOME}/missions/QTIP_PROOF1_SHARD_t_a305e412_s3/source_model
export GENESIS_F521_NATIVE_MEMBERS_RECEIPT="$ROOT/receipts/NATIVE_CHECKPOINT_MEMBERS_20_OF_20.json"
export GENESIS_F521_TEACHERS_ALL40="$ROOT/inputs/SELECTED_TEACHER_ASSETS_ALL40.json"
export GENESIS_F521_TEACHER9_RECEIPT="$ROOT/receipts/TEACHER_ASSETS_9_OF_9_CLOSED.json"
export GENESIS_F521_APPROVAL="$ROOT/approvals/EXECUTION_APPROVAL.json"
export GENESIS_F521_APPROVAL_VALIDATION="$ROOT/receipts/EXECUTION_APPROVAL_VALIDATED.json"
export GENESIS_F521_UPDATE_PLAN="$ROOT/staging/UPDATE_PLAN.json"
export GENESIS_QTIP2_TLUT=${SPARK_HOME}/missions/P875_QTIP3_FORTRESS_t_67604030_s3/inputs/P641_QTIP_TLUT_SOURCE.pt
export GENESIS_QTIP2_SOURCE_CODE="$ROOT/code/p605r_run_qtip_anchor.py"
export GENESIS_QTIP2_KERNEL_CODE="$ROOT/code/qtip_kernel_decompress.py"
export GENESIS_REPAIR_CONFIG="$ROOT/inputs/F521_BASIC_REPAIR_CONFIG.json"
export COMBO_BINREPAIR_BASE="$ROOT/code/canonical_binrepair_e2e.py"
export BR_BASE_HARNESS="$ROOT/code/base_binrepair_e2e.py"
export BR_WIRE_DIR=${SPARK_HOME}/missions/MTP_T3G_t_6036af62/wire_v4-step32
export BR_STEPS=64
export BR_LR=1e-2
export BR_CACHE_ONLY=0
export BR_TEACH=${SPARK_HOME}/missions/P635_FOCUSED_t_0aeca305_s3/inputs/BASIC_COMBINED_REFS
export BR_DELTA_DIR=${SPARK_HOME}/missions/BINREPAIR_t_2956f863/delta
export BR_MANIFEST="$GENESIS_ASSIGNMENT"
export BR_MAX_HOURS=8
export BR_TRAINABLE=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42
export BR_TRAIN="$(python3 -c 'import json; print(",".join(map(str,json.load(open("${SPARK_HOME}/missions/P948_SPECULATIVE_REPAIR_t_b1aa61d3_s3/inputs/F521_BASIC_REPAIR_CONFIG.json"))["train_combined_wins"])))')"
export BR_CORPUS=${SPARK_HOME}/missions/P635_FOCUSED_t_0aeca305_s3/inputs/BASIC_COMBINED_768.json
export BR_PROBE=278,279
export BR_GRADCHECK=0
export BR_VQ3B_DIR=${SPARK_HOME}/missions/BINREPAIR_t_2956f863/planes
export BR_OUTDIR="$ROOT/run/basic_harness"
export BR_CACHE_BATCH=40
export BR_BATCH=4
export BR_PROBE_EVERY=8
export BR_TAG=p911_f521_wire_c_repair
export BR_EARLY_STOP=999
unset GENESIS_REPAIR_STATE_SEED GENESIS_REPAIR_CANARY_SEED

python3 - <<'PY'
import hashlib,json,os,subprocess
p='${SPARK_HOME}/HOST_CLAIM.json'; raw=open(p,'rb').read(); d=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==os.environ['GENESIS_REPAIR_EXPECTED_CLAIM_SHA256']
assert d.get('owner')==os.environ['GENESIS_TASK_ID'] and d.get('host')=='spark-3'
assert not subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip()
sv=os.statvfs('/'); assert sv.f_bavail*sv.f_frsize>=8*1024**3
mem=next(int(x.split()[1])*1024 for x in open('/proc/meminfo') if x.startswith('MemAvailable:')); assert mem>=8*1024**3
PY

GUARD=""
TRAINER=""
cleanup() {
  if [[ -n "$GUARD" ]] && kill -0 "$GUARD" 2>/dev/null; then kill "$GUARD" 2>/dev/null || true; wait "$GUARD" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM
${SPARK_HOME}/humming_env/bin/python3 -u "$ROOT/code/p911_resource_guard.py" >>"$ROOT/logs/SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL__resource_guard.log" 2>&1 &
GUARD=$!
printf '%s\n' "$GUARD" > "$ROOT/run/RESOURCE_GUARD.pid"
${SPARK_HOME}/humming_env/bin/python3 -u "$ROOT/code/genesis_basic_repair.py" >>"$ROOT/logs/SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL__P911_TRAINER.stdout.log" 2>>"$ROOT/logs/SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL__P911_TRAINER.stderr.log" &
TRAINER=$!
printf '%s\n' "$TRAINER" > "$ROOT/run/TRAINER.pid"
export STOP_AFTER TRAINER GUARD
python3 - <<'PY'
import json,os,tempfile,time
from pathlib import Path
root=Path(os.environ['GENESIS_REPAIR_ROOT']); p=root/f"run/PHASE_{int(os.environ['STOP_AFTER']):03d}_LAUNCH.json"
def proc(pid):
 s=open(f'/proc/{pid}/stat').read().split(); return {'pid':pid,'pgid':int(s[4]),'sid':int(s[5]),'cmdline':open(f'/proc/{pid}/cmdline','rb').read().replace(b'\0',b' ').decode()}
obj={'schema':'p911-phase-launch-v1','speculative_label':'SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL','task_id':os.environ['GENESIS_TASK_ID'],'stop_after_update':int(os.environ['STOP_AFTER']),'trainer':proc(int(os.environ['TRAINER'])),'guard':proc(int(os.environ['GUARD'])),'claim_sha256':os.environ['GENESIS_REPAIR_EXPECTED_CLAIM_SHA256'],'launched_unix':time.time()}
fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
with os.fdopen(fd,'w') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PY
set +e
wait "$TRAINER"
RC=$?
set -e
cleanup
GUARD=""
if [[ "$RC" -eq 0 ]]; then
  ${SPARK_HOME}/humming_env/bin/python3 -u "$ROOT/code/p911_acceptance_sweep.py" --through "$STOP_AFTER" | tee -a "$ROOT/logs/SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL__P911_ACCEPTANCE.log"
fi
export RC
python3 - <<'PY'
import json,os,tempfile,time
from pathlib import Path
root=Path(os.environ['GENESIS_REPAIR_ROOT']); stop=int(os.environ['STOP_AFTER']); p=root/f'run/PHASE_{stop:03d}_EXIT.json'; obj={'schema':'p911-phase-exit-v1','speculative_label':'SPECULATIVE_REVOKED_UNTIL_P943_TERMINAL','stop_after_update':stop,'trainer_rc':int(os.environ['RC']),'completed_unix':time.time()}
fd,tmp=tempfile.mkstemp(prefix='.'+p.name+'.',dir=p.parent)
with os.fdopen(fd,'w') as f:json.dump(obj,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
os.replace(tmp,p)
PY
exit "$RC"
