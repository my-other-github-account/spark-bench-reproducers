#!/usr/bin/env bash
set -euo pipefail
umask 077
if [[ $# -ne 5 ]]; then
  echo "usage: $0 TASK_ID MISSION DATASET_NAME RAW_JSONL OUT_DIR" >&2
  exit 64
fi
TASK_ID=$1
MISSION=$2
DATASET=$3
RAW=$4
OUT=$5
[[ "$DATASET" == humaneval || "$DATASET" == mbpp ]] || { echo "invalid dataset" >&2; exit 64; }
CLAIM_PATH=${P968_HOST_CLAIM:-/run/p968/HOST_CLAIM.json}
python3 -c 'import json,sys; d=json.load(open(sys.argv[3])); assert d.get("owner")==sys.argv[1] and d.get("mission")==sys.argv[2] and d.get("status")=="CLAIMED",d' "$TASK_ID" "$MISSION" "$CLAIM_PATH"
[[ -f "$RAW" ]] || { echo "missing raw JSONL: $RAW" >&2; exit 66; }
mkdir -p "$OUT" "$MISSION/cache/evalplus" "$MISSION/logs"
RAW_REL=$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1],sys.argv[2]))' "$RAW" "$MISSION")
OUT_REL=$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1],sys.argv[2]))' "$OUT" "$MISSION")
SAN="${RAW%.jsonl}-sanitized.jsonl"
SAN_REL=$(python3 -c 'import os,sys; print(os.path.relpath(sys.argv[1],sys.argv[2]))' "$SAN" "$MISSION")
rm -f "$SAN"
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
else
  DOCKER=(sudo -n docker)
fi
IMAGE_ID=$("${DOCKER[@]}" image inspect evalplus:26d6d00 --format '{{.Id}}')
COMMON=("${DOCKER[@]}" run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/workspace/evaluser --cap-drop ALL --security-opt no-new-privileges --pids-limit 1024 --memory 16g --cpus 16 --tmpfs /tmp:rw,nosuid,noexec,size=4g --tmpfs /run:rw,nosuid,noexec,size=64m -v "$MISSION:/work" -v "$MISSION/cache/evalplus:/workspace/evaluser/.cache/evalplus" evalplus:26d6d00)
"${COMMON[@]}" python -m evalplus.sanitize --samples "/work/$RAW_REL" >"$MISSION/logs/sanitize.$DATASET.log" 2>&1
[[ -f "$SAN" ]] || { echo "sanitizer did not create $SAN" >&2; exit 70; }
python3 - "$SAN" "$RAW" "$OUT/SANITIZE_RECEIPT.json" "$IMAGE_ID" <<'PY'
import hashlib,json,pathlib,sys,time
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
san,raw,out,image=sys.argv[1:]
value={'schema':'p968-sanitize-receipt-v1','status':'PASS','raw':raw,'raw_sha256':sha(raw),'sanitized':san,'sanitized_sha256':sha(san),'image':'evalplus:26d6d00','image_id':image,'created_epoch':time.time()}
pathlib.Path(out).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
PY
for CELL in canonical min8 factor8 relaxed; do
  case "$CELL" in
    canonical) MIN=4; FACTOR=4;;
    min8) MIN=8; FACTOR=4;;
    factor8) MIN=4; FACTOR=8;;
    relaxed) MIN=8; FACTOR=8;;
  esac
  RESULT="$OUT/${CELL}.eval_results.json"
  rm -f "$RESULT"
  "${COMMON[@]}" python -m evalplus.evaluate --dataset "$DATASET" --samples "/work/$SAN_REL" --test-details True --min-time-limit "$MIN" --gt-time-limit-factor "$FACTOR" --output-file "/work/$OUT_REL/${CELL}.eval_results.json" >"$MISSION/logs/evaluate.$DATASET.$CELL.log" 2>&1
  [[ -f "$RESULT" ]] || { echo "evaluator did not create $RESULT" >&2; exit 71; }
done
python3 - "$MISSION" "$DATASET" "$RAW" "$SAN" "$OUT" "$IMAGE_ID" <<'PY'
import hashlib,json,pathlib,sys,time
mission,dataset,raw,san,out,image=sys.argv[1:]
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
rows={}
for cell in ['canonical','min8','factor8','relaxed']:
 p=pathlib.Path(out)/(cell+'.eval_results.json'); rows[cell]={'path':str(p),'sha256':sha(p)}
value={'schema':'p968-score-matrix-receipt-v1','status':'PASS','dataset':dataset,'evalplus_commit':'26d6d00bb1fd0fa37f39c99d5290da67891d1c5e','image':'evalplus:26d6d00','image_id':image,'network':'none','test_details':True,'raw':{'path':raw,'sha256':sha(raw)},'sanitized':{'path':san,'sha256':sha(san)},'cells':rows,'timing_cells':{'canonical':{'min_time_limit':4.0,'gt_time_limit_factor':4.0},'min8':{'min_time_limit':8.0,'gt_time_limit_factor':4.0},'factor8':{'min_time_limit':4.0,'gt_time_limit_factor':8.0},'relaxed':{'min_time_limit':8.0,'gt_time_limit_factor':8.0}},'created_epoch':time.time()}
p=pathlib.Path(out)/'SCORE_MATRIX_RECEIPT.json'; p.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':'PASS','receipt':str(p),'sha256':sha(p)},sort_keys=True))
PY
