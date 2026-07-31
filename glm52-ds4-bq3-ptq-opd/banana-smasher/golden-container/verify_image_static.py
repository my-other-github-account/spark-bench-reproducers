#!/usr/bin/env python3
"""Fail-closed static and full-manifest check run during image build."""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

EXPECTED = {
    "/opt/genesis/runtime_cubins/vq_warp_gemv/_C.so": "b98e7917881bc846b4f3ad3d1da8671a87fb9021f28ce6bd33f6c3b97c4135e5",
    "/opt/genesis/P1268_C1_C2_RESULT.json": "9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860",
    "/opt/genesis/C_LADDER_FULL_SEAL.json": "be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7",
    "/opt/genesis/WINNING_BOOT_CONFIG.json": "091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2",
    "/opt/genesis/BOOT_CONFIG_FREEZE.json": "cff72b34c5cd9d29a17d9a1842005febf5402141f6709c10f85a25cd8a61d707",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_row(root: Path, row: dict, label: str) -> None:
    rel = Path(str(row.get("path", "")))
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise SystemExit(f"unsafe {label} manifest path: {rel}")
    path = root / rel
    kind = row.get("kind", "file")
    if kind == "symlink":
        if not path.is_symlink() or os.readlink(path) != row.get("target"):
            raise SystemExit(f"{label} symlink mismatch: {rel}")
        return
    if kind != "file" or not path.is_file() or path.is_symlink():
        raise SystemExit(f"{label} regular file missing: {rel}")
    if path.stat().st_size != int(row.get("bytes", -1)):
        raise SystemExit(f"{label} size mismatch: {rel}")
    actual = sha256(path)
    if actual != row.get("sha256"):
        raise SystemExit(f"{label} hash mismatch {rel}: {actual} != {row.get('sha256')}")


for name, expected in EXPECTED.items():
    path = Path(name)
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"static hash mismatch {name}: {actual} != {expected}")

wheel = json.loads(Path("/opt/genesis/WHEEL_MANIFEST.json").read_text())
if wheel.get("schema") != "genesis-golden-wheel-manifest-v2" or not wheel.get("files"):
    raise SystemExit("wheel manifest schema/rows mismatch")
wheel_root = Path(wheel["destination_root"])
for row in wheel["files"]:
    verify_row(wheel_root, row, "wheel")

runtime = json.loads(Path("/opt/genesis/RUNTIME_CACHE_MANIFEST.json").read_text())
if runtime.get("schema") != "genesis-golden-runtime-cache-manifest-v1" or not runtime.get("files"):
    raise SystemExit("runtime-cache manifest schema/rows mismatch")
runtime_roots = {
    "cubins_w2": Path("/opt/genesis/runtime_cubins/cubins-sm120"),
    "cubins_w3": Path("/opt/genesis/runtime_cubins/cubins_e43"),
    "triton_cache": Path("/opt/genesis/cache/triton"),
    "flashinfer_cache": Path("/root/.cache/vllm/flashinfer_autotune_cache"),
    "flashinfer_jit_cache": Path("/root/.cache/flashinfer"),
}
if set(runtime.get("contexts") or []) != set(runtime_roots):
    raise SystemExit("runtime-cache context set mismatch")
for row in runtime["files"]:
    context = row.get("context")
    if context not in runtime_roots:
        raise SystemExit(f"unknown runtime context: {context}")
    verify_row(runtime_roots[context], row, f"runtime:{context}")

import torch
import vllm
if torch.__version__ != "2.11.0+cu130":
    raise SystemExit(f"torch version mismatch: {torch.__version__}")
if vllm.__version__ != "0.24.0":
    raise SystemExit(f"vLLM version mismatch: {vllm.__version__}")
quant = importlib.import_module("vllm.models.deepseek_v4.quant_config")
source = Path(quant.__file__).read_text()
for marker in (
    "IQ3_WIRE", "moe_pack_root", "os.environ.setdefault",
    '"VQ_WARP_M4_VECTOR": "1"', '"VLLM_MOE_VQ_CUDA_WARP_MAX_M": "4"',
    "4a4c15a52eaa8f87e4eb2f436da1580cb5e9addb15713d41bd9a74276731578a",
):
    if marker not in source:
        raise SystemExit(f"quant-method product default missing: {marker}")
attention_source = Path(
    "/work/build/venvs/vllm-moet/lib/python3.12/site-packages/"
    "vllm/models/deepseek_v4/attention.py"
).read_text()
if 'quant_cfg.get("moe_quant_algo", "")' not in attention_source:
    raise SystemExit("IQ3 dense-sidecar admission is not model-config driven")
if 'if os.environ.get("DS4_DENSE_PATCH")' in attention_source:
    raise SystemExit("IQ3 dense-sidecar admission still depends on env timing")
flashinfer_core_source = Path(
    "/work/build/venvs/vllm-moet/lib/python3.12/site-packages/flashinfer/jit/core.py"
).read_text()
for marker in (
    'if os.environ.get("FLASHINFER_DISABLE_JIT"):',
    'if not so_path.is_file():',
    'result = self.load(so_path)',
):
    if marker not in flashinfer_core_source:
        raise SystemExit(f"FlashInfer sealed-cache load patch missing: {marker}")
if os.environ.get("PYTHONPATH"):
    raise SystemExit("golden image must not require PYTHONPATH")
print(json.dumps({
    "status": "PASS", "torch": torch.__version__, "vllm": vllm.__version__,
    "wheel_rows_verified": len(wheel["files"]),
    "runtime_rows_verified": len(runtime["files"]),
    "quant_method_defaults": True, "flashinfer_sealed_cache_load": True,
    "pythonpath_empty": True,
    "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
    "p1321_split_admission": True,
}, sort_keys=True))
