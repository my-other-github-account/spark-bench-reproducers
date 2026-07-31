#!/usr/bin/env bash
# Redirect targets are task-owned; only the docker CLI requires sudo.
# shellcheck disable=SC2024
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACK="${PACK:?set PACK to the sealed external model+wire artifact}"
IMAGE="${IMAGE:-genesis-serve:golden}"
BUILD_RECEIPT="${BUILD_RECEIPT:-$HERE/receipts/IMAGE_BUILD_RECEIPT.json}"
MISSION="${MISSION:-/home/dnola/missions/P1321_BOX10_GOLDEN_t_73d48597_s8}"
EXPECTED_OWNER="${EXPECTED_OWNER:-t_73d48597}"
GOOD=t_73d48597-golden-validate
BAD=t_73d48597-golden-refusal
BAD_PACK="${PACK}.missing-pack-complete.t_73d48597"
mkdir -p "$MISSION"/{logs,receipts,results}

cleanup() {
  sudo -n docker rm -f "$BAD" "$GOOD" >/dev/null 2>&1 || true
  rm -rf "$BAD_PACK"
}
trap cleanup EXIT INT TERM

[[ -f "$BUILD_RECEIPT" ]] || { echo "missing build receipt: $BUILD_RECEIPT" >&2; exit 2; }
expected_image_id="$(python3 - "$BUILD_RECEIPT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
if d.get('status') != 'PASS' or not str(d.get('image_id','')).startswith('sha256:'):
    raise SystemExit('invalid image build receipt')
print(d['image_id'])
PY
)"
actual_image_id="$(sudo -n docker image inspect "$IMAGE" --format '{{.Id}}')"
[[ "$actual_image_id" == "$expected_image_id" ]] || {
  echo "validated image differs from build receipt: $actual_image_id != $expected_image_id" >&2
  exit 2
}
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]] || {
  echo "GPU is not empty before validation" >&2; exit 2;
}
if command -v ss >/dev/null && ss -ltnH 'sport = :8000' | grep -q .; then
  echo "port 8000 is already listening" >&2; exit 2
fi

python3 - "$EXPECTED_OWNER" "$PACK" <<'PY'
import hashlib,json,sys
from pathlib import Path
owner,raw=sys.argv[1:]
claim=json.load(open('/home/dnola/HOST_CLAIM.json'))
if claim.get('owner') != owner or claim.get('state') != 'CLAIMED':
    raise SystemExit(f"host claim mismatch: {claim.get('owner')} {claim.get('state')}")
pack=Path(raw).resolve()
manifest=pack/'BS_PACK_MANIFEST.json'
if hashlib.sha256(manifest.read_bytes()).hexdigest() != '4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a':
    raise SystemExit('external pack manifest digest mismatch')
config=json.load(open(pack/'config.json'))
q=config.get('quantization_config') or {}
if q.get('moe_quant_algo') != 'IQ3_WIRE' or q.get('moe_pack_root') != 'wire_v4-step32':
    raise SystemExit(f'quantization_config mismatch: {q}')
for rel in ('wire_v4-step32/PACK_MANIFEST.json','wire_v4-step32/PACK_COMPLETE'):
    p=pack/rel
    if not p.is_file() or p.is_symlink():
        raise SystemExit(f'missing regular pack member: {rel}')
PY
sudo -n docker image inspect "$IMAGE" > "$MISSION/receipts/IMAGE_INSPECT_PREVALIDATION.json"

# Refusal path: identical hard-linked artifact metadata, with only PACK_COMPLETE removed.
sudo -n docker rm -f "$BAD" "$GOOD" >/dev/null 2>&1 || true
rm -rf "$BAD_PACK"
cp -al "$PACK" "$BAD_PACK"
rm -f "$BAD_PACK/wire_v4-step32/PACK_COMPLETE"
sudo -n docker run -d --name "$BAD" --label io.genesis.task=t_73d48597 \
  --device nvidia.com/gpu=0 --ipc=host --ulimit memlock=-1:-1 \
  --memory 110g --memory-swap 110g -v "$BAD_PACK:/model:ro" "$IMAGE" >/dev/null
for _ in $(seq 1 180); do
  state="$(sudo -n docker inspect "$BAD" --format '{{.State.Running}}')"
  [[ "$state" == false ]] && break
  sleep 1
done
if [[ "$(sudo -n docker inspect "$BAD" --format '{{.State.Running}}')" != false ]]; then
  sudo -n docker stop -t 10 "$BAD" >/dev/null
  echo "refusal container did not exit within 180s" >&2
  exit 6
fi
sudo -n docker logs "$BAD" >"$MISSION/logs/refusal.log" 2>&1 || true
python3 - "$MISSION/logs/refusal.log" <<'PY'
import sys
text=open(sys.argv[1],errors='ignore').read()
needle='IQ3_WIRE pack is missing PACK_COMPLETE'
if needle not in text:
    raise SystemExit(f"expected quant-method refusal not found: {needle}")
if 'Application startup complete' in text:
    raise SystemExit('refusal path reached successful service startup')
print('REFUSAL_PATH_PASS')
PY
sudo -n docker rm "$BAD" >/dev/null
rm -rf "$BAD_PACK"

# Happy path: ordinary image CMD == `vllm serve /model ...`.
sudo -n docker run -d --name "$GOOD" --label io.genesis.task=t_73d48597 \
  --device nvidia.com/gpu=0 --ipc=host --ulimit memlock=-1:-1 \
  --memory 110g --memory-swap 110g -p 8000:8000 \
  -v "$PACK:/model:ro" "$IMAGE" >/dev/null
ready=0
for _ in $(seq 1 1200); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi
  if [[ "$(sudo -n docker inspect "$GOOD" --format '{{.State.Running}}')" != true ]]; then break; fi
  sleep 1
done
if (( ready == 0 )); then
  sudo -n docker logs "$GOOD" >"$MISSION/logs/server-start-failed.log" 2>&1 || true
  sudo -n docker rm -f "$GOOD" >/dev/null 2>&1 || true
  echo "golden container failed readiness" >&2
  exit 7
fi
curl -fsS http://127.0.0.1:8000/v1/models > "$MISSION/receipts/MODELS.json"
sudo -n docker inspect "$GOOD" > "$MISSION/receipts/CONTAINER_INSPECT.json"
sudo -n docker exec -i "$GOOD" python - <<'PY' > "$MISSION/receipts/RUNTIME_VERSIONS.json"
import hashlib, json
from pathlib import Path
import torch, vllm
rows={}
for raw in ['/opt/genesis/runtime_cubins/vq_warp_gemv/_C.so','/model/bs_runtime_assets/dense_patch.safetensors','/opt/genesis/WHEEL_MANIFEST.json','/opt/genesis/RUNTIME_CACHE_MANIFEST.json']:
 p=Path(raw); rows[raw]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
print(json.dumps({'python':__import__('sys').version,'torch':torch.__version__,'vllm':vllm.__version__,'critical':rows},sort_keys=True))
PY

# Eyeball gate precedes all performance warmups/measurement. Preserve the first
# three raw stock-API generations verbatim (bounded to 300 chars in receipt).
python3 - "$MISSION/receipts/MODELS.json" "$MISSION/receipts/FIRST3_RAW_GENERATIONS.json" "$MISSION/receipts/IMAGE_INSPECT_PREVALIDATION.json" <<'PY'
import json, sys, time, urllib.request
models_path, output_path, inspect_path = sys.argv[1:]
model = json.load(open(models_path))["data"][0]["id"]
image_id = json.load(open(inspect_path))[0]["Id"]
prompts = [
    "In one clear sentence, explain why water freezes when sufficiently cooled.",
    "Write a Python function named add that returns the sum of two numbers.",
    "Name three primary colors and nothing else.",
]
rows = []
for index, prompt in enumerate(prompts, 1):
    body = {"model": model, "prompt": prompt, "max_tokens": 96, "temperature": 0.0}
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.load(response)
        if response.status != 200:
            raise SystemExit(f"raw generation {index} HTTP {response.status}")
    text = str(payload["choices"][0].get("text", ""))
    if not text.strip():
        raise SystemExit(f"raw generation {index} was empty")
    rows.append({"index": index, "prompt": prompt, "http_status": 200, "text": text[:300]})
receipt = {
    "schema": "genesis-golden-first3-raw-v1",
    "created_unix": time.time(),
    "task_id": "t_73d48597",
    "image_id": image_id,
    "provenance": "P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a; PUBLIC_CANON_IQ3_WIRE, NOT P943 native TRUE-C",
    "rows": rows,
}
open(output_path, "w").write(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
print(json.dumps(receipt, sort_keys=True))
PY

# Excluded in-container warmups, then mark the Docker-log boundary.
sudo -n docker exec "$GOOD" python /opt/genesis/bin/golden_perf_check.py \
  --warm-only --output /tmp/GOLDEN_WARMUPS.json
sudo -n docker cp "$GOOD:/tmp/GOLDEN_WARMUPS.json" "$MISSION/receipts/GOLDEN_WARMUPS.json" >/dev/null
sudo -n docker logs "$GOOD" > "$MISSION/logs/server-before-measured.log" 2>&1
measure_line="$(wc -l < "$MISSION/logs/server-before-measured.log" | tr -d ' ')"

set +e
sudo -n docker exec "$GOOD" python /opt/genesis/bin/golden_perf_check.py \
  --c1-warmups 0 --c2-warmups 0 --c4-warmups 0 \
  --c1-rows 3 --c2-rows 3 --c4-rows 3 \
  --output /tmp/GOLDEN_IN_CONTAINER_C1X3_C2X3_C4X3.json
perf_rc=$?
set -e
sudo -n docker cp "$GOOD:/tmp/GOLDEN_IN_CONTAINER_C1X3_C2X3_C4X3.json" "$MISSION/results/GOLDEN_IN_CONTAINER_C1X3_C2X3_C4X3.json" >/dev/null
sudo -n docker logs "$GOOD" > "$MISSION/logs/server.log" 2>&1
sudo -n docker stop -t 60 "$GOOD" >/dev/null
sudo -n docker rm "$GOOD" >/dev/null

python3 - "$MISSION" "$PACK" "$IMAGE" "$measure_line" "$perf_rc" <<'PY'
import hashlib,json,statistics,subprocess,sys,time
from pathlib import Path
mission,pack,image,measure_line,perf_rc=sys.argv[1:]
mission=Path(mission); pack=Path(pack); measure_line=int(measure_line); perf_rc=int(perf_rc)
def sha(path):
 p=Path(path); return {'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
perf=json.load(open(mission/'results/GOLDEN_IN_CONTAINER_C1X3_C2X3_C4X3.json'))
logs=(mission/'logs/server.log').read_text(errors='ignore').splitlines()
measured=logs[measure_line:]
jit_patterns=('triton compiler','compiling','jit compilation','generating new triton','building extension')
jit_hits=[{'line':measure_line+i+1,'text':line} for i,line in enumerate(measured) if any(p in line.lower() for p in jit_patterns)]
marker_patterns=('IQ3 CUDA WARP-GEMV ON-PATH sentinel','VQ DISPATCH PROBE path=cuda_warp_m4','Application startup complete')
markers={p:[{'line':i+1,'text':line} for i,line in enumerate(logs) if p in line] for p in marker_patterns}
inspect=json.load(open(mission/'receipts/IMAGE_INSPECT_PREVALIDATION.json'))[0]
config=inspect['Config']
checks={
 'perf_exit_zero':perf_rc==0,
 'perf_ready':perf.get('status')=='READY',
 'three_c1_rows':len(perf.get('measured',{}).get('c1',[]))==3,
 'three_c2_rows':len(perf.get('measured',{}).get('c2',[]))==3,
 'three_c4_rows':len(perf.get('measured',{}).get('c4',[]))==3,
 'no_measured_jit':not jit_hits,
 'all_on_path_markers':all(markers[p] for p in marker_patterns),
 'stock_no_entrypoint':config.get('Entrypoint') in (None,[]),
 'stock_vllm_cmd':config.get('Cmd',[])[:3]==['vllm','serve','/model'],
}
receipt={
 'schema':'genesis-golden-in-container-validation-v1','status':'PASS' if all(checks.values()) else 'FAIL',
 'created_unix':time.time(),'task_id':'t_73d48597','host':__import__('socket').gethostname(),
 'provenance':'P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a',
 'truth_label':'PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C',
 'image':image,'image_id':inspect['Id'],'repo_digests':inspect.get('RepoDigests') or [],
 'pack':str(pack),'pack_manifest':sha(pack/'BS_PACK_MANIFEST.json'),
 'checks':checks,'summary':perf.get('summary'),'perf_receipt':sha(mission/'results/GOLDEN_IN_CONTAINER_C1X3_C2X3_C4X3.json'),
 'warmup_receipt':sha(mission/'receipts/GOLDEN_WARMUPS.json'),'refusal_log':sha(mission/'logs/refusal.log'),
 'server_log':sha(mission/'logs/server.log'),'measured_log_start_line':measure_line+1,
 'measured_jit_hits':jit_hits,'on_path_markers':markers,
}
out=mission/'receipts/GOLDEN_VALIDATION.json'; out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':receipt['status'],'summary':receipt['summary'],'receipt':str(out)},sort_keys=True))
if receipt['status']!='PASS': raise SystemExit(1)
PY

# Postcondition: only this script's exact containers are gone; host claim remains.
[[ -z "$(sudo -n docker ps -a --filter name=^/${GOOD}$ --filter name=^/${BAD}$ -q)" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]
printf 'VALIDATION_PASS receipt=%s\n' "$MISSION/receipts/GOLDEN_VALIDATION.json"
