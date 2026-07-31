#!/usr/bin/env bash
# Redirect targets are task-owned; only the docker CLI requires sudo.
# shellcheck disable=SC2024
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_ID="${TASK_ID:-BOX10-BUILD}"
P1268="${P1268:-/work/build/artifacts/P1268_C2_CANON_VERBATIM_INTERNAL-ID_s8}"
P1321_ROOT="${P1321_ROOT:-/work/build/artifacts/GLM52_HUMMING_W3_P1321_s8/p1321}"
P1321_STAGE="${P1321_STAGE:-/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate}"
P1321_SOURCE="${P1321_SOURCE:-/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1}"
VENV="${VENV:-/work/build/venvs/vllm-moet}"
CUBINS_W2="${CUBINS_W2:-/work/build/Dev/vLLM-Moet/kernels/cubins-sm120}"
CUBINS_W3="${CUBINS_W3:-/work/build/ds4w3/cubins_e43}"
TRITON_CACHE="${TRITON_CACHE:-/work/build/.triton/cache}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-/work/build/.cache/vllm/flashinfer_autotune_cache}"
FLASHINFER_JIT_CACHE="${FLASHINFER_JIT_CACHE:-/work/build/.cache/flashinfer}"
IMAGE="${IMAGE:-genesis-serve:golden}"
BASE_ALIAS="genesis-serve:p1135n-b66edfa3-closure"
BASE_ID="sha256:b66edfa3811486df5ad61f513861a08e99b7b7ffe18edf5c1f4ed494567631fe"
OUT="${OUT:-$HERE/receipts}"
REGISTRY="${REGISTRY:-}"
mkdir -p "$OUT"

command -v docker >/dev/null
command -v nvidia-smi >/dev/null
[[ "$(uname -m)" == "aarch64" ]] || { echo "golden image requires aarch64 Spark, got $(uname -m)" >&2; exit 2; }
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')"
[[ "$compute_cap" == 12.* ]] || { echo "golden image requires Blackwell compute capability 12.x, got $compute_cap" >&2; exit 2; }

PYOVERLAY="$P1321_STAGE/runtime/pyoverlay"
M4="$P1321_STAGE/kernel/vq_warp_public/vq_warp_gemv"

for required in \
  "$VENV/bin/python" \
  "$PYOVERLAY/vllm/models/deepseek_v4/quant_config.py" \
  "$M4/__init__.py" \
  "$P1268/receipts/P1268_C1_C2_RESULT.json" \
  "$P1321_ROOT/C_LADDER_FULL_SEAL.json" \
  "$P1321_ROOT/WINNING_BOOT_CONFIG.json" \
  "$P1321_ROOT/receipts/BOOT_CONFIG_FREEZE.json" \
  "$P1321_ROOT/receipts/P1321_SOURCE_ADMISSION_READBACK.json" \
  "$P1321_SOURCE/vq_warp_m4/csrc/vq_warp_gemv.cu" \
  "$P1321_STAGE/artifacts/dense_patch.safetensors" \
  "$CUBINS_W2" "$CUBINS_W3" "$TRITON_CACHE" "$FLASHINFER_CACHE" "$FLASHINFER_JIT_CACHE"; do
  [[ -e "$required" ]] || { echo "missing build input: $required" >&2; exit 3; }
done
mapfile -t m4_bins < <(python3 - "$M4" <<'PY'
from pathlib import Path
import sys
for path in sorted(Path(sys.argv[1]).glob('_C*.so')):
    print(path)
PY
)
[[ "${#m4_bins[@]}" -eq 1 ]] || { echo "expected one P1321 extension, got ${#m4_bins[@]}" >&2; exit 3; }
[[ "$(sha256sum "${m4_bins[0]}" | cut -d' ' -f1)" == "b98e7917881bc846b4f3ad3d1da8671a87fb9021f28ce6bd33f6c3b97c4135e5" ]] || {
  echo "P1321 vector-M4 extension hash drift" >&2; exit 3;
}

actual_base="$(sudo -n docker image inspect genesis-serve:golden-bare --format '{{.Id}}')"
[[ "$actual_base" == "$BASE_ID" ]] || { echo "base drift: $actual_base != $BASE_ID" >&2; exit 4; }
sudo -n docker tag "$BASE_ID" "$BASE_ALIAS"

python3 "$HERE/make_wheel_manifest.py" \
  --python "$VENV/bin/python" \
  --venv "$VENV" \
  --pyoverlay "$PYOVERLAY" \
  --m4-root "$M4" \
  --output "$HERE/WHEEL_MANIFEST.json"
python3 "$HERE/make_runtime_cache_manifest.py" \
  --context cubins_w2 "$CUBINS_W2" \
  --context cubins_w3 "$CUBINS_W3" \
  --context triton_cache "$TRITON_CACHE" \
  --context flashinfer_cache "$FLASHINFER_CACHE" \
  --context flashinfer_jit_cache "$FLASHINFER_JIT_CACHE" \
  --output "$HERE/RUNTIME_CACHE_MANIFEST.json"
cp "$P1268/receipts/P1268_C1_C2_RESULT.json" "$HERE/P1268_C1_C2_RESULT.json"
cp "$P1321_ROOT/C_LADDER_FULL_SEAL.json" "$HERE/C_LADDER_FULL_SEAL.json"
cp "$P1321_ROOT/WINNING_BOOT_CONFIG.json" "$HERE/WINNING_BOOT_CONFIG.json"
cp "$P1321_ROOT/receipts/BOOT_CONFIG_FREEZE.json" "$HERE/BOOT_CONFIG_FREEZE.json"

python3 - "$HERE" "$P1268" "$VENV" "$CUBINS_W2" "$CUBINS_W3" "$TRITON_CACHE" "$FLASHINFER_CACHE" "$FLASHINFER_JIT_CACHE" <<'PY'
import hashlib, json, sys
from pathlib import Path
here, p1268, venv, c2, c3, triton, flash, flash_jit = map(Path, sys.argv[1:])
critical = {
    "base_image_id": "sha256:b66edfa3811486df5ad61f513861a08e99b7b7ffe18edf5c1f4ed494567631fe",
    "p1268_result": p1268 / "receipts/P1268_C1_C2_RESULT.json",
    "p1268_launcher": p1268 / "run/launch_p1268.sh",
    "runtime_overlay_receipt": p1268 / "runtime/pyoverlay/RUNTIME_OVERLAY_RECEIPT.json",
    "quant_config_preimage": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/runtime/pyoverlay/vllm/models/deepseek_v4/quant_config.py"),
    "moe_w2_cubit": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/runtime/pyoverlay/vllm/model_executor/layers/quantization/utils/moe_w2_cubit.py"),
    "moe_vq_triton": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/runtime/pyoverlay/vllm/model_executor/layers/quantization/utils/moe_vq_triton.py"),
    "flashinfer_jit_core_preimage": venv / "lib/python3.12/site-packages/flashinfer/jit/core.py",
    "p1321_ladder_seal": Path("/work/build/artifacts/GLM52_HUMMING_W3_P1321_s8/p1321/C_LADDER_FULL_SEAL.json"),
    "p1321_winning_boot": Path("/work/build/artifacts/GLM52_HUMMING_W3_P1321_s8/p1321/WINNING_BOOT_CONFIG.json"),
    "p1321_freeze": Path("/work/build/artifacts/GLM52_HUMMING_W3_P1321_s8/p1321/receipts/BOOT_CONFIG_FREEZE.json"),
    "p1321_admission": Path("/work/build/artifacts/GLM52_HUMMING_W3_P1321_s8/p1321/receipts/P1321_SOURCE_ADMISSION_READBACK.json"),
    "m4_binary": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/kernel/vq_warp_public/vq_warp_gemv/_C.cpython-312-aarch64-linux-gnu.so"),
    "m4_source": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/vq_warp_m4/csrc/vq_warp_gemv.cu"),
    "m4_wrapper": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/kernel/vq_warp_public/vq_warp_gemv/__init__.py"),
    "dense_patch": Path("/work/build/artifacts/W3_PLANES_KERNEL/task_stages/P1321/vector_m4_v1/serving_candidate/artifacts/dense_patch.safetensors"),

    "python": venv / "bin/python",
}
def row(path):
    data=path.read_bytes(); return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
rows={name: row(path) for name,path in critical.items() if name != "base_image_id"}
# Context directory identity is represented by exact roots plus critical manifests;
# the image build itself seals every resulting layer by immutable image digest.
out={
 "schema":"genesis-golden-source-manifest-v3",
 "task_id":"BOX10-BUILD",
 "truth_label":"PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
 "base_image_id":critical["base_image_id"],
 "critical":rows,
 "contexts":["venv","p1321_pyoverlay","p1321_vector_m4","cubins_w2","cubins_w3","triton_cache","flashinfer_cache","flashinfer_jit_cache"],
 "p1321":{"ladder_seal_sha256":"be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7","winning_boot_sha256":"091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2","freeze_sha256":"cff72b34c5cd9d29a17d9a1842005febf5402141f6709c10f85a25cd8a61d707","admission":"scalar valid_m<4; vector valid_m==4"},
 "no_model_or_wire_context":True,
}
(here/"SOURCE_MANIFEST.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
PY

# Refuse accidental model/wire payloads in the main recipe context.
python3 - "$HERE" <<'PY'
import sys
from pathlib import Path
root=Path(sys.argv[1])
allowed={"P1268_C1_C2_RESULT.json"}
for p in root.rglob("*"):
    if not p.is_file() or p.name in allowed:
        continue
    low=str(p.relative_to(root)).lower()
    if p.suffix in {".safetensors", ".bin"} or "wire_v4" in low or "true_c_planes" in low:
        raise SystemExit(f"forbidden model/wire payload in recipe context: {p}")
PY

avail="$(df --output=avail -B1 / | tr -d ' ' | sed -n '2p')"
(( avail >= 42949672960 )) || { echo "root below 40 GiB build floor: $avail" >&2; exit 5; }

export BUILDKIT_PROGRESS=plain
sudo -n ionice -c2 -n7 nice -n 10 docker buildx build \
  --load --pull=false --network=none --provenance=false --progress=plain \
  --build-context "venv=$VENV" \
  --build-context "pyoverlay=$PYOVERLAY" \
  --build-context "m4=$M4" \
  --build-context "cubins_w2=$CUBINS_W2" \
  --build-context "cubins_w3=$CUBINS_W3" \
  --build-context "triton_cache=$TRITON_CACHE" \
  --build-context "flashinfer_cache=$FLASHINFER_CACHE" \
  --build-context "flashinfer_jit_cache=$FLASHINFER_JIT_CACHE" \
  --tag "$IMAGE" --file "$HERE/Dockerfile" "$HERE" \
  2>&1 | tee "$OUT/IMAGE_BUILD.log"

sudo -n docker image inspect "$IMAGE" > "$OUT/IMAGE_INSPECT.json"
python3 - "$OUT/IMAGE_INSPECT.json" "$OUT/IMAGE_BUILD_RECEIPT.json" "$IMAGE" "$compute_cap" <<'PY'
import hashlib,json,sys,time
source,out,image,cap=sys.argv[1:]
inspect=json.load(open(source))[0]
config=inspect["Config"]
if config.get("Entrypoint") not in (None, []):
    raise SystemExit(f"unexpected entrypoint: {config.get('Entrypoint')}")
if config.get("Cmd",[])[:3] != ["vllm","serve","/model"]:
    raise SystemExit(f"non-stock command: {config.get('Cmd')}")
labels=config.get("Labels") or {}
if labels.get("io.genesis.no-model-bytes") != "true":
    raise SystemExit("no-model label missing")
if labels.get("io.genesis.external-pack.manifest.sha256") != "4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a":
    raise SystemExit("external pack digest label mismatch")
if labels.get("io.genesis.p1321.ladder-seal.sha256") != "be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7":
    raise SystemExit("P1321 ladder seal label mismatch")
if labels.get("io.genesis.p1321.winning-boot.sha256") != "091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2":
    raise SystemExit("P1321 winning boot label mismatch")
receipt={
 "schema":"genesis-golden-image-build-v2","status":"PASS","created_unix":time.time(),
 "task_id":"BOX10-BUILD","provenance":"P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a",
 "image":image,"image_id":inspect["Id"],"repo_digests":inspect.get("RepoDigests") or [],
 "size":inspect["Size"],"architecture":inspect["Architecture"],"compute_capability":cap,
 "cmd":config["Cmd"],"entrypoint":config.get("Entrypoint"),"labels":labels,
 "inspect_sha256":hashlib.sha256(open(source,"rb").read()).hexdigest(),
 "truth_label":"PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
}
open(out,"w").write(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps(receipt,sort_keys=True))
PY

if [[ -n "$REGISTRY" ]]; then
  echo "REGISTRY inline push is disabled: use publish_local_registry.sh or push_local_registry_via_ssh.sh so publication is immutable-digest-bound" >&2
  exit 6
fi
printf 'PASS image=%s receipt=%s\n' "$IMAGE" "$OUT/IMAGE_BUILD_RECEIPT.json"
