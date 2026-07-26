#!/usr/bin/env python3
"""Once-only canonical full-512 GENESIS evaluator on compute-node-1.

The corrected physical wire remains read-only on compute-node-8. Exactly one layer is
streamed over QSFP into bounded local scratch, verified against the sealed layer
receipt, consumed, and retired before the next layer is admitted.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter

from full512_safety import (
    parse_gpu_snapshot,
    require_unoptimized,
    retire_scratch,
    validate_claim,
    validate_gate_bundle,
    validate_layer_coverage,
    validate_runtime_environment,
    validate_staged_layer,
    validate_preserved_outputs,
)
from rail_loading import PlaneStore

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/GENESIS_PRE_REPAIR_FULL512_PUBLIC_TASK_s1")
CLAIM = Path("$HOME/HOST_CLAIM.json")
SOURCE_HOST = "203.0.113.9"
REMOTE_PACKAGE = "$HOME/run-bundles/GENESIS_FANIN_PUBLIC_TASK_s8/package/wire43"
ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
BUILD_BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
COMPACT_SHA = "d9421f1f6d0e696608bb0ce9b09131e63790c18e9cd536e440b1884b727db00d"
BASELINE_SHA = "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88"
TEACHER_DONE_SHA = "6338af84f907a26dfdf0f784edc322aa672738542ed884b70e4d9b6e96aa33b0"
EXPECTED_STREAMED_CODE = 0.05212973475888538
EXPECTED_PHYSICAL_READER_SHA = "f4ec85784db337faa751ede6779c4fa0cc64ad971a3a1cea39ad91dc5100dcc4"
EXPECTED_ARTIFACT_HASHES = {
    "pass_marker_sha256": "98e3a621f0de12db7c78c72c0ceeaa17daff10e1d13ed68130c551521987b6ca",
    "receipt_sha256": "8735f6047d22582ee64f774a9081d25c5a7a50ee5ef25e3911ea401df885ba95",
    "wire_sha256": "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755",
}
STEP0_GLOBAL = 0.077061
IQ4_GLOBAL = 0.07204
IQ4_CODE = 0.054215965394205624
MODEL = Path("$HOME/models/hf/DeepSeek-V4-Flash")
MODEL_INDEX_SHA = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
PACKAGE = ROOT / "code/eval_package"
TEACHER = Path("$HOME/run-bundles/DS4_TEACHER")
CORPUS = TEACHER / "static/windows_ds4_eval.json"
LABELS = ROOT / "inputs/BQ3_STEP0_PER_CLASS.json"
COMPACT_MANIFEST = ROOT / "inputs/GENESIS_COMPACT_FANIN.json"
ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
PHYSICAL_MARKER = ROOT / "inputs/PHYSICAL_CODE76.json"
PHYSICAL_PASS_MARKER = ROOT / "inputs/PHYSICAL_CODE76.PASS.json"
WIRE_MANIFEST = ROOT / "inputs/WIRE_43_MANIFEST.json"
ORIGINAL_COMMAND_RECEIPT = ROOT / "run/COMMAND.json"
RESUME_COMMAND_RECEIPT = ROOT / "run/RESUME_COMMAND.json"
RESUME_AUTHORIZATION = ROOT / "run/RESUME_AUTHORIZED.json"
LAUNCH_ONCE = ROOT / "run/LAUNCH_ONCE.json"
RESUME_LAUNCH_ONCE = ROOT / "run/RESUME_LAUNCH_ONCE.json"
LOADER_SOURCE = ROOT / "code/rail_loading.py"
LOADER_SHA = "155310d1e6701d6cb2d1c04558514366a2304cb2a8d6d26402ed7c800b8b6c89"
LOADER_SENTINEL = ROOT / "run/ARM4_MMAP_SENTINEL.json"
COVERAGE_INCIDENT = ROOT / "receipts/COVERAGE_DUPLICATION_STOP_PUBLIC_TASK.json"
ORIGINAL_COMMAND_SHA = "a5ceec2c136d6b740d5222950609e01bac9141fb81ae963db609a72b04382f02"
ORIGINAL_READER_SHA = "57b4b1537bf2931f33f67b043b3d70b92c5559647e67cbb81b0976af413fdb69"
ORIGINAL_LATCH_SHA = "499b7ca2ba6965240d12df75b8da0c17199d1b9a04079bb79b27c46882d8fea0"
COVERAGE_INCIDENT_SHA = "96cca6425acd35df5125907d269f141ccebeac8a55f4a52dd134914881041415"
CACHE = ROOT / "scratch/wire43"
PHYSICAL_PACKAGE = CACHE
DISK_FLOOR = 20 * (1 << 30)
PROGRESS = ROOT / "run/FULL512_STREAM_PROGRESS.json"
ENV_CONTRACT = {
    "TWOBIN_KLD_STREAM_OUT": "1",
    "TWOBIN_ATTN_IMPL": "eager",
    "TWOBIN_LAYER_OVERLAP": "0",
    "ARM4_LOADER_MODE": "torch-mmap",
    "ARM4_LOADER_FALLBACK": "torch-eager",
    "FULL512_LOADER_PROGRESS_PATH": str(PROGRESS),
    "FULL512_LOADER_SHA256": LOADER_SHA,
    "FULL512_LOADER_INPUT_SHA256": EXPECTED_ARTIFACT_HASHES["wire_sha256"],
    "FULL512_LOADER_SENTINEL_PATH": str(LOADER_SENTINEL),
    "FULL512_TASK_ID": TASK,
    "FULLMENU_ASSEMBLY_BATCH": "8",
    "PYTHONHASHSEED": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "CUDA_MODULE_LOADING": "EAGER",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_exclusive_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    dfd = os.open(path.parent, os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)


def current_claim() -> tuple[bytes, dict]:
    raw = CLAIM.read_bytes()
    obj = json.loads(raw)
    validate_claim(obj, task_id=TASK, mission=str(ROOT), now=time.time())
    return raw, obj


def validate_resume_authorization(claim_sha: str) -> tuple[dict, str]:
    auth = json.loads(RESUME_AUTHORIZATION.read_text())
    exact = {
        "schema": "genesis-full512-resume-authorization-v1",
        "status": "AUTHORIZED_PRESERVE_64_RESUME_448",
        "task_id": TASK,
        "claim_sha256": claim_sha,
        "launch_once_sha256": ORIGINAL_LATCH_SHA,
        "original_command_sha256": ORIGINAL_COMMAND_SHA,
        "coverage_incident_sha256": COVERAGE_INCIDENT_SHA,
        "preserved_windows": list(range(64)),
        "preserved_completed_chunks": 1,
        "discarded_unsealed_layer_visits": [0, 1, 2, 3],
    }
    for key, expected in exact.items():
        if auth.get(key) != expected:
            raise RuntimeError(f"resume authorization mismatch: {key}")
    if sha256(LAUNCH_ONCE) != ORIGINAL_LATCH_SHA or sha256(ORIGINAL_COMMAND_RECEIPT) != ORIGINAL_COMMAND_SHA:
        raise RuntimeError("original once-only launch evidence drift")
    if sha256(COVERAGE_INCIDENT) != COVERAGE_INCIDENT_SHA:
        raise RuntimeError("coverage incident evidence drift")
    progress = json.loads(PROGRESS.read_text())
    if progress.get("completed_layers") != list(range(43)) + [0, 1, 2, 3]:
        raise RuntimeError("false-stop progress history drift")
    output_set_sha = validate_preserved_outputs(
        ROOT / "out/full512", ROOT / "out/full512/DONE.jsonl",
        expected_windows=list(range(64)),
    )
    if auth.get("preserved_output_set_sha256") != output_set_sha:
        raise RuntimeError("resume preserved-output set drift")
    return auth, output_set_sha


def gpu_snapshot(*, require_zero_util: bool, own_pid: int | None = None) -> dict[str, object]:
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout
    utilization = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout
    return parse_gpu_snapshot(apps, utilization, own_pid=own_pid, require_zero_util=require_zero_util)


def tier_params(tier: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"d(\d+)_k(\d+)", tier)
    if not match:
        raise RuntimeError(f"unsupported tier {tier}")
    d, k = map(int, match.groups())
    bits = int(math.log2(k))
    if 1 << bits != k:
        raise RuntimeError(f"non-power-of-two K {tier}")
    return d, k, bits


def assignment_ids(layer_map: dict, tier: str, projection: str) -> list[int]:
    return [
        expert for expert in range(256)
        if layer_map[str(expert)][projection] == tier
    ]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhysicalRawMmapStore(PlaneStore):
    """Pinned PlaneStore adapter for receipt-listed physical raw tensors."""

    def __init__(self, *, input_identity_sha256: str):
        mode = os.environ.get("ARM4_LOADER_MODE")
        if mode != "torch-mmap":
            raise RuntimeError(f"physical raw loader mode must be torch-mmap, got {mode!r}")
        super().__init__(mode=mode, device="cpu", cache_size=1, prefetch_workers=1)
        self.input_identity_sha256 = input_identity_sha256
        self._raw_maps: list[object] = []
        self._first_map_emitted = False

    def map(self, path: Path, *, layer: int, dtype: object, shape: tuple[int, ...]):
        import numpy as numpy_module

        if self.mode != "torch-mmap" or sha256(LOADER_SOURCE) != LOADER_SHA:
            raise RuntimeError("binding physical raw mmap loader identity drift")
        mapped = numpy_module.memmap(path, dtype=dtype, mode="r", shape=shape)
        self._raw_maps.append(mapped)
        if not self._first_map_emitted:
            sentinel = {
                "schema": "genesis-arm4-mmap-sentinel-v1",
                "status": "ACTIVE_ON_PATH",
                "task_id": TASK,
                "mode": self.mode,
                "fallback": os.environ.get("ARM4_LOADER_FALLBACK"),
                "backend": "numpy.memmap physical-wire raw tensors",
                "loader_source": str(LOADER_SOURCE),
                "loader_sha256": LOADER_SHA,
                "input_identity_sha256": self.input_identity_sha256,
                "first_layer": layer,
                "first_path": str(path),
                "created_unix": time.time(),
            }
            atomic_exclusive_json(LOADER_SENTINEL, sentinel)
            print(f"ARM4 load L{layer:03d} mode=torch-mmap input_sha256={self.input_identity_sha256}", flush=True)
            self._first_map_emitted = True
        return mapped

    def release_layer(self) -> None:
        self._raw_maps.clear()

    def close(self) -> None:
        self.release_layer()
        super().close()


class GenesisTierSource:
    """Production evaluator source backed only by the physical raw package."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.resolve() != COMPACT_MANIFEST.resolve() or sha256(self.manifest_path) != COMPACT_SHA:
            raise RuntimeError("compact manifest SHA/path drift")
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("status") != "PASS_SEALED" or self.manifest.get("assignment", {}).get("sha256") != ASSIGNMENT_SHA:
            raise RuntimeError("compact manifest content drift")
        self.rows = {int(row["layer"]): row for row in self.manifest["rows"]}
        if set(self.rows) != set(range(43)):
            raise RuntimeError("compact layer surface drift")
        if sha256(ASSIGNMENT) != ASSIGNMENT_SHA:
            raise RuntimeError("assignment file SHA drift")
        self.assignment = json.loads(ASSIGNMENT.read_text())["assignment"]
        wire = json.loads(WIRE_MANIFEST.read_text())
        self.wire_rows = {int(row["layer"]): row for row in wire["layers"]}
        if set(self.wire_rows) != set(range(43)):
            raise RuntimeError("wire manifest layer surface drift")
        self.completed: list[int] = list(range(43)) if RESUME_AUTHORIZATION.is_file() else []
        self.mmap_completed: list[int] = []
        self._active_mmap_count = 0
        self._raw_store = PhysicalRawMmapStore(input_identity_sha256=sha256(WIRE_MANIFEST))
        self.active_stage: Path | None = None
        CACHE.mkdir(parents=True, exist_ok=True)
        self._e2m1 = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            dtype=torch.float32, device="cuda",
        )

    def _cleanup_stage(self) -> None:
        if self.active_stage is not None and self.active_stage.exists():
            shutil.rmtree(self.active_stage)
        self.active_stage = None

    def _stage_remote(self, layer: int, row: dict) -> Path:
        self._cleanup_stage()
        stage = CACHE / f"layer_{layer:03d}"
        temp = CACHE / f".layer_{layer:03d}.partial"
        if stage.exists():
            shutil.rmtree(stage)
        if temp.exists():
            shutil.rmtree(temp)
        temp.mkdir()
        wire_row = self.wire_rows[layer]
        required_bytes = int(wire_row["physical_wire_bytes"])
        free = shutil.disk_usage(CACHE).free
        if free - required_bytes < DISK_FLOOR:
            raise RuntimeError(f"disk floor before remote stage L{layer}: free={free} required={required_bytes}")
        receipt_path = temp / "LAYER_RECEIPT.json"
        subprocess.run([
            "scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            f"{SOURCE_HOST}:{REMOTE_PACKAGE}/layer_{layer:03d}/LAYER_RECEIPT.json",
            str(receipt_path),
        ], check=True)
        if sha256(receipt_path) != wire_row["receipt_sha256"]:
            raise RuntimeError(f"L{layer} receipt SHA drift after QSFP stage")
        receipt = json.loads(receipt_path.read_text())
        if (
            receipt.get("schema") != "genesis-materialized-layer-v1"
            or receipt.get("status") != "PASS"
            or int(receipt.get("layer", -1)) != layer
            or receipt.get("assignment_sha256") != ASSIGNMENT_SHA
            or receipt.get("builder_sha256") != BUILD_BUILDER_SHA
            or int(receipt.get("physical_wire_bytes", -1)) != required_bytes
        ):
            raise RuntimeError(f"L{layer} receipt identity drift")
        receipt_files = receipt.get("files")
        if not isinstance(receipt_files, list) or not receipt_files:
            raise RuntimeError(f"L{layer} receipt file list missing")
        allowed_paths: list[str] = []
        for item in receipt_files:
            rel = item.get("path")
            rel_path = Path(rel) if isinstance(rel, str) else Path("/")
            if not isinstance(rel, str) or rel_path.is_absolute() or ".." in rel_path.parts or rel in allowed_paths:
                raise RuntimeError(f"L{layer} receipt path unsafe or duplicate: {rel}")
            allowed_paths.append(rel)
        files_from = ROOT / f"run/.layer_{layer:03d}.files-from"
        files_from.write_text("".join(f"{rel}\n" for rel in allowed_paths))
        try:
            command = [
                "rsync", "-a", "--partial", "--timeout=180", "--bwlimit=200000",
                "--files-from", str(files_from),
                "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4",
                f"{SOURCE_HOST}:{REMOTE_PACKAGE}/layer_{layer:03d}/", str(temp) + "/",
            ]
            subprocess.run(command, check=True)
        finally:
            files_from.unlink(missing_ok=True)
        stage_accounting = validate_staged_layer(
            temp, receipt, required_bytes=required_bytes,
            free_bytes_after=shutil.disk_usage(CACHE).free, floor=DISK_FLOOR,
        )
        os.replace(temp, stage)
        self.active_stage = stage
        atomic_json(ROOT / f"run/LAYER_{layer:03d}_STAGE.json", {
            "schema": "genesis-full512-layer-stage-v2", "task_id": TASK, "layer": layer,
            "receipt_sha256": wire_row["receipt_sha256"], **stage_accounting,
            "allowed_files": [item["path"] for item in receipt["files"]],
            "stage": str(stage), "created_unix": time.time(),
        })
        return stage

    @staticmethod
    def _launch_vq(codes: torch.Tensor, scales: torch.Tensor, codebook: torch.Tensor,
                   expert_ids: list[int], destination: torch.Tensor, d: int) -> None:
        from delta_pack_sources import _vq_dequant_write_kernel
        if not expert_ids:
            return
        codes_cuda = codes.to("cuda", non_blocking=False).contiguous()
        scales_cuda = scales.to("cuda", non_blocking=False).contiguous()
        codebook_cuda = codebook.to("cuda", non_blocking=False).float().contiguous()
        ids_cuda = torch.tensor(expert_ids, dtype=torch.int32, device="cuda")
        n_rows = 4096
        out_cols = int(destination.shape[2])
        code_width = int(codes_cuda.shape[2])
        scale_cols = int(scales_cuda.shape[2])
        if code_width * d != out_cols:
            raise RuntimeError(f"code shape drift {codes_cuda.shape} D={d} out={out_cols}")
        total = len(expert_ids) * n_rows * out_cols
        block = 4096
        grid = (triton.cdiv(total, block),)
        _vq_dequant_write_kernel[grid](
            codes_cuda, scales_cuda, codebook_cuda, ids_cuda, destination,
            n_rows, out_cols, code_width, d, scale_cols, total,
            BLOCK=block, num_warps=8,
        )
        torch.cuda.synchronize()
        del codes_cuda, scales_cuda, codebook_cuda, ids_cuda

    def _fill_pt_tiers(self, layer: int, row: dict, stage: Path | None,
                       gate_up: torch.Tensor, down: torch.Tensor, coverage: set[tuple[int, str]]) -> None:
        layer_map = self.assignment[str(layer)]
        for artifact in row["tier_source_artifacts"]:
            tier = artifact.get("tier")
            if not tier:
                continue
            d, k, _ = tier_params(tier)
            path = Path(artifact["path"]) if row["shard"] == "B" else stage / f"{tier}.pt"
            if not path.is_file() or path.stat().st_size != int(artifact["bytes"]):
                raise RuntimeError(f"tier source size drift L{layer} {tier}")
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
            meta = payload.get("meta", {})
            if int(meta.get("layer", -1)) != layer or meta.get("tier") != tier or meta.get("assignment_sha256") != ASSIGNMENT_SHA:
                raise RuntimeError(f"tier source metadata drift L{layer} {tier}")
            source_builder = meta.get("builder_sha256") or meta.get("canonical_reviewed_builder_sha256")
            if source_builder != BUILD_BUILDER_SHA:
                raise RuntimeError(f"tier source builder drift L{layer} {tier}: {source_builder}")
            for projection, suffix, destination in (("fused13", "13", gate_up), ("down", "2", down)):
                expected = assignment_ids(layer_map, tier, projection)
                actual = [int(value) for value in payload[f"expert_ids{suffix}"].tolist()]
                if actual != expected:
                    raise RuntimeError(f"tier expert identity drift L{layer} {tier} {projection}")
                if not actual:
                    continue
                batch = 8
                for start in range(0, len(actual), batch):
                    ids = actual[start:start + batch]
                    self._launch_vq(
                        payload[f"codes{suffix}"][start:start + batch],
                        payload[f"sc{suffix}"][start:start + batch],
                        payload[f"cb{suffix}"], ids, destination, d,
                    )
                    coverage.update((expert, projection) for expert in ids)
                torch.cuda.empty_cache()
            del payload
            gc.collect()

    @staticmethod
    def _unpack_codes(path: Path, bits: int, offset_values: int, count_values: int) -> torch.Tensor:
        if (offset_values * bits) % 8 or (count_values * bits) % 8:
            raise RuntimeError("packed code byte alignment drift")
        offset = offset_values * bits // 8
        count = count_values * bits // 8
        with path.open("rb") as handle:
            handle.seek(offset)
            packed = np.frombuffer(handle.read(count), dtype=np.uint8)
        if packed.nbytes != count:
            raise RuntimeError(f"short packed code read {path}")
        if bits == 8:
            values = packed
        else:
            bit_rows = np.unpackbits(packed, bitorder="little").reshape(-1, bits)
            weights = (1 << np.arange(bits, dtype=np.uint16))
            values = (bit_rows.astype(np.uint16, copy=False) * weights).sum(axis=1, dtype=np.uint16)
        return torch.from_numpy(np.array(values, dtype=np.int16, copy=True))

    def _fill_a(self, layer: int, row: dict, stage: Path,
                gate_up: torch.Tensor, down: torch.Tensor, coverage: set[tuple[int, str]]) -> None:
        layer_map = self.assignment[str(layer)]
        for tier in row["tier_identities"]:
            if tier == "native_mxfp4":
                continue
            d, k, bits = tier_params(tier)
            for projection, suffix, destination, cols in (
                ("fused13", "fused13", gate_up, 4096),
                ("down", "down", down, 2048),
            ):
                expected = assignment_ids(layer_map, tier, projection)
                if not expected:
                    continue
                ids_path = stage / f"{tier}.{suffix}.expert_ids.i16.bin"
                ids = np.fromfile(ids_path, dtype="<i2").astype(np.int64).tolist()
                # Shard-A serialized rows preserve the assignment JSON's
                # insertion order (lexicographic expert keys), not numeric
                # order. Codes are aligned to that stored order, so compare
                # sets but retain `ids` when launching the dequantizer.
                if sorted(ids) != expected or len(ids) != len(set(ids)):
                    raise RuntimeError(f"A expert identity drift L{layer} {tier} {projection}")
                cb = torch.from_numpy(self._raw_store.map(
                    stage / f"{tier}.{suffix}.codebook.fp16.bin",
                    layer=layer, dtype="<f2", shape=(k, d),
                ))
                self._active_mmap_count += 1
                scale_cols = cols // 32
                scales = self._raw_store.map(
                    stage / f"{tier}.{suffix}.scales.e8m0.bin",
                    layer=layer, dtype=np.uint8, shape=(len(ids), 4096, scale_cols),
                )
                self._active_mmap_count += 1
                values_per_expert = 4096 * (cols // d)
                codes_path = stage / f"{tier}.{suffix}.codes.le{bits}.bin"
                batch = 4
                for start in range(0, len(ids), batch):
                    batch_ids = ids[start:start + batch]
                    n_values = len(batch_ids) * values_per_expert
                    codes = self._unpack_codes(codes_path, bits, start * values_per_expert, n_values).reshape(len(batch_ids), 4096, cols // d)
                    scale_tensor = torch.from_numpy(np.array(scales[start:start + batch], copy=True))
                    self._launch_vq(codes, scale_tensor, cb, batch_ids, destination, d)
                    coverage.update((expert, projection) for expert in batch_ids)
                    del codes, scale_tensor
                del cb, scales
                gc.collect()
                torch.cuda.empty_cache()

    def _fill_native(self, layer: int, gate_up: torch.Tensor, down: torch.Tensor,
                     coverage: set[tuple[int, str]]) -> None:
        layer_map = self.assignment[str(layer)]
        layer_dir = PHYSICAL_PACKAGE / f"layer_{layer:03d}"
        # The shard builder serializes fused13 by concatenating packed w1/w3
        # along the packed row axis.  Its physical shape is therefore
        # [experts, 4096, 2048], not [experts, 2, 4096, 2048].  The latter
        # double-counts the fused axis and rejects the correctly sealed bytes.
        for projection, destination, packed_rows, packed_cols, scale_cols in (
            ("fused13", gate_up, 4096, 2048, 128),
            ("down", down, 4096, 1024, 64),
        ):
            expected = assignment_ids(layer_map, "native_mxfp4", projection)
            if not expected:
                continue
            prefix = f"native_mxfp4.{projection}"
            ids_path = layer_dir / f"{prefix}.expert_ids.i16.bin"
            weights_path = layer_dir / f"{prefix}.weights.mxfp4.bin"
            scales_path = layer_dir / f"{prefix}.scales.e8m0.bin"
            ids = np.fromfile(ids_path, dtype="<i2").astype(np.int64).tolist()
            if sorted(ids) != expected or len(ids) != len(set(ids)):
                raise RuntimeError(f"physical native identity drift L{layer} {projection}")
            expected_weight_bytes = len(ids) * packed_rows * packed_cols
            expected_scale_bytes = len(ids) * packed_rows * scale_cols
            actual_weight_bytes = weights_path.stat().st_size
            actual_scale_bytes = scales_path.stat().st_size
            if actual_weight_bytes != expected_weight_bytes or actual_scale_bytes != expected_scale_bytes:
                raise RuntimeError(
                    f"physical native size drift L{layer} {projection} "
                    f"weights={actual_weight_bytes}/{expected_weight_bytes} "
                    f"scales={actual_scale_bytes}/{expected_scale_bytes}"
                )
            weights = self._raw_store.map(
                weights_path, layer=layer, dtype=np.uint8,
                shape=(len(ids), packed_rows, packed_cols),
            )
            self._active_mmap_count += 1
            scales = self._raw_store.map(
                scales_path, layer=layer, dtype=np.uint8,
                shape=(len(ids), packed_rows, scale_cols),
            )
            self._active_mmap_count += 1
            for row, expert in enumerate(ids):
                packed = torch.from_numpy(np.array(weights[row], copy=True)).to("cuda")
                scale = torch.from_numpy(np.array(scales[row], copy=True)).to("cuda")
                nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).flatten(-2)
                weight = self._e2m1[nibbles.long()]
                weight = weight * torch.exp2(scale.float() - 127.0).repeat_interleave(32, dim=1)
                if tuple(weight.shape) != tuple(destination[expert].shape):
                    raise RuntimeError(
                        f"physical native shape drift L{layer} E{expert} {projection} "
                        f"weight={tuple(weight.shape)} destination={tuple(destination[expert].shape)}"
                    )
                if not bool(torch.isfinite(weight).all()):
                    raise RuntimeError(f"physical native nonfinite L{layer} E{expert} {projection}")
                destination[expert].copy_(weight.to(dtype=torch.bfloat16))
                coverage.add((expert, projection))
                del packed, scale, nibbles, weight
            del weights, scales
            gc.collect()
            torch.cuda.empty_cache()

    def fill_layer(self, layer: int, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        current_claim()
        self._active_mmap_count = 0
        row = self.rows[layer]
        coverage: set[tuple[int, str]] = set()
        started = time.time()
        try:
            layer_dir = self._stage_remote(layer, row)
            if not layer_dir.is_dir():
                raise FileNotFoundError(layer_dir)
            self._fill_a(layer, row, layer_dir, gate_up, down, coverage)
            self._fill_native(layer, gate_up, down, coverage)
            expected_coverage = {(expert, projection) for expert in range(256) for projection in ("fused13", "down")}
            if coverage != expected_coverage:
                missing = sorted(expected_coverage - coverage)
                duplicate_count = 512 - len(coverage)
                raise RuntimeError(f"physical coverage drift L{layer} missing={missing[:8]} delta={duplicate_count}")
            if os.environ.get("ARM4_LOADER_MODE") != "torch-mmap" or self._active_mmap_count <= 0:
                raise RuntimeError(f"binding mmap loader was not on-path for L{layer}")
            self.mmap_completed.append(layer)
            self.completed.append(layer)
        finally:
            self._raw_store.release_layer()
            retire_scratch(CACHE)
            self.active_stage = None
        completed_chunks = len(self.completed) // 43
        if self.completed and len(self.completed) % 43 == 0:
            validate_layer_coverage(self.completed, expected_chunks=completed_chunks)
        atomic_json(PROGRESS, {
            "schema": "genesis-physical-stream-progress-v2", "task_id": TASK, "status": "RUNNING",
            "completed_layers": list(self.completed), "completed_count": len(self.completed),
            "layer_visits": len(self.completed), "active_layer": None,
            "completed_chunks": completed_chunks,
            "completed_layers_by_chunk": [list(range(43)) for _ in range(completed_chunks)],
            "current_chunk_layers": self.completed[completed_chunks * 43:],
            "mmap_loader_mode": "torch-mmap",
            "mmap_loader_sha256": LOADER_SHA,
            "mmap_input_identity_sha256": self._raw_store.input_identity_sha256,
            "mmap_completed_layers": list(self.mmap_completed),
            "mmap_completed_count": len(self.mmap_completed),
            "last_layer": layer, "last_layer_seconds": time.time() - started,
            "physical_package": f"{SOURCE_HOST}:{REMOTE_PACKAGE}",
            "local_stage_retired": True,
            "transport": "QSFP rsync receipt-first allowlist, one layer at a time",
            "disk_free_bytes": shutil.disk_usage(ROOT).free,
            "updated_unix": time.time(),
        })
        print(f"[PhysicalWireSource] L{layer:03d} exact coverage=512 retired seconds={time.time()-started:.1f}", flush=True)

    def layer(self, layer: int):
        raise RuntimeError("scalar layer path forbidden; canonical builder must use fill_layer")

    def finish(self) -> None:
        self._cleanup_stage()
        self._raw_store.close()


def all_windows() -> tuple[list[int], dict[int, str]]:
    labels = json.loads(LABELS.read_text())
    rows = labels.get("per_window", [])
    classes = {int(row["win"]): str(row["source_class"]) for row in rows}
    wins = sorted(classes)
    counts = Counter(classes.values())
    if wins != list(range(512)) or set(counts) != {"agentic", "chat", "code", "multilingual", "prose", "reasoning"} or counts["code"] != 76:
        raise RuntimeError(f"full512 label surface drift: {counts}")
    return wins, classes


def reduce_outputs(out: Path, wins: list[int], classes: dict[int, str]) -> dict:
    rows = []
    values: list[torch.Tensor] = []
    by_class_values: dict[str, list[torch.Tensor]] = {}
    by_class_window_means: dict[str, list[float]] = {}
    for win in wins:
        path = out / f"kld_win{win}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        tensor = payload.get("kld")
        if payload.get("win") != win or payload.get("support") != 8192 or payload.get("cutoff") != 1024:
            raise RuntimeError(f"KLD identity drift win{win}")
        if not isinstance(tensor, torch.Tensor) or tensor.shape != (1024,) or not bool(torch.isfinite(tensor).all()) or bool((tensor < -1e-6).any()):
            raise RuntimeError(f"KLD finite/payload drift win{win}")
        value = tensor.double()
        mean = float(value.mean())
        source_class = classes[win]
        values.append(value)
        by_class_values.setdefault(source_class, []).append(value)
        by_class_window_means.setdefault(source_class, []).append(mean)
        row = {"win": win, "source_class": source_class, "mean": mean,
               "bytes": path.stat().st_size, "sha256": sha256(path)}
        if win >= 64:
            for field in (
                "loader_mode", "loader_sha256", "loader_sentinel_sha256",
                "input_identity_sha256", "loader_progress_sha256",
                "loader_chunk_receipt_sha256",
            ):
                if field not in payload:
                    raise RuntimeError(f"resumed output loader field missing win{win}: {field}")
                row[field] = payload[field]
        rows.append(row)
    joined = torch.cat(values)
    window_means = [float(row["mean"]) for row in rows]

    def summary(name: str, tensors: list[torch.Tensor], means: list[float]) -> dict:
        combined = torch.cat(tensors)
        se = statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else 0.0
        mean = float(combined.mean())
        return {
            "source_class": name,
            "mean": mean,
            "n_windows": len(means),
            "n_positions": int(combined.numel()),
            "window_mean_se": se,
            "window_mean_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        }

    global_row = summary("global", values, window_means)
    class_rows = {
        name: summary(name, by_class_values[name], by_class_window_means[name])
        for name in sorted(by_class_values)
    }
    return {"global": global_row, "by_class": class_rows, "per_window": rows}


def main() -> int:
    global np, torch, triton
    require_unoptimized()
    validate_runtime_environment(ENV_CONTRACT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--receipt", required=True, type=Path)
    args = ap.parse_args()
    started = time.time()
    claim_raw, _ = current_claim()
    claim_sha = hashlib.sha256(claim_raw).hexdigest()
    gpu_snapshot(require_zero_util=True)
    required = [
        args.manifest, ASSIGNMENT, PHYSICAL_PASS_MARKER, PHYSICAL_MARKER,
        WIRE_MANIFEST, LABELS, CORPUS,
        TEACHER / "t8192_eval/DONE.jsonl", MODEL / "config.json",
        MODEL / "model.safetensors.index.json", PACKAGE / "t8192_ds4_build_v3.py",
        PACKAGE / "delta_pack_sources.py",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.manifest.resolve() != COMPACT_MANIFEST.resolve():
        raise RuntimeError("manifest path drift")
    if sha256(args.manifest) != COMPACT_SHA or sha256(ASSIGNMENT) != ASSIGNMENT_SHA or sha256(LABELS) != BASELINE_SHA:
        raise RuntimeError("manifest/assignment/label SHA drift")
    if sha256(TEACHER / "t8192_eval/DONE.jsonl") != TEACHER_DONE_SHA:
        raise RuntimeError("teacher receipt drift")
    if sha256(MODEL / "model.safetensors.index.json") != MODEL_INDEX_SHA:
        raise RuntimeError("model index drift")
    if args.receipt.exists():
        raise RuntimeError(f"receipt already exists {args.receipt}")
    resume_auth = None
    preserved_output_set_sha = None
    if RESUME_AUTHORIZATION.is_file():
        resume_auth, preserved_output_set_sha = validate_resume_authorization(claim_sha)
    elif list(args.out.glob("kld_win*.pt")):
        raise RuntimeError(f"once-only output directory is not empty: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    passed = json.loads(PHYSICAL_PASS_MARKER.read_text())
    marker = json.loads(PHYSICAL_MARKER.read_text())
    wire = json.loads(WIRE_MANIFEST.read_text())
    hashes = {
        "pass_marker_sha256": sha256(PHYSICAL_PASS_MARKER),
        "receipt_sha256": sha256(PHYSICAL_MARKER),
        "wire_sha256": sha256(WIRE_MANIFEST),
    }
    gate = validate_gate_bundle(
        passed, marker, wire, hashes=hashes,
        expected_code=EXPECTED_STREAMED_CODE,
        expected_reader_sha256=EXPECTED_PHYSICAL_READER_SHA,
        expected_assignment_sha256=ASSIGNMENT_SHA,
        expected_compact_sha256=COMPACT_SHA,
        expected_builder_sha256=BUILD_BUILDER_SHA,
        expected_package=REMOTE_PACKAGE,
        expected_artifact_hashes=EXPECTED_ARTIFACT_HASHES,
    )
    command_receipt = RESUME_COMMAND_RECEIPT if resume_auth is not None else ORIGINAL_COMMAND_RECEIPT
    if not command_receipt.is_file():
        raise FileNotFoundError(command_receipt)
    command = json.loads(command_receipt.read_text())
    exact_command = {
        "schema": "genesis-full512-resume-command-v1" if resume_auth is not None else "genesis-full512-command-v2",
        "task_id": TASK,
        "cwd": str(ROOT),
        "environment_contract": ENV_CONTRACT,
        "physical_code76_pass_marker_sha256": hashes["pass_marker_sha256"],
        "physical_code76_marker_sha256": hashes["receipt_sha256"],
        "wire_manifest_sha256": hashes["wire_sha256"],
        "assignment_sha256": ASSIGNMENT_SHA,
        "compact_manifest_sha256": COMPACT_SHA,
        "physical_reader_expected_sha256": EXPECTED_PHYSICAL_READER_SHA,
        "remote_package": REMOTE_PACKAGE,
        "claim_sha256": claim_sha,
        "loader_source": str(LOADER_SOURCE),
        "loader_sha256": LOADER_SHA,
        "loader_mode": "torch-mmap",
    }
    if resume_auth is not None:
        exact_command.update({
            "resume_authorization_sha256": sha256(RESUME_AUTHORIZATION),
            "original_command_sha256": ORIGINAL_COMMAND_SHA,
            "launch_once_sha256": ORIGINAL_LATCH_SHA,
            "resume_launch_once": str(RESUME_LAUNCH_ONCE),
            "resume_launch_once_sha256": sha256(RESUME_LAUNCH_ONCE),
            "preserved_output_set_sha256": preserved_output_set_sha,
        })
    for key, expected in exact_command.items():
        if command.get(key) != expected:
            raise RuntimeError(f"command identity mismatch: {key}")
    if command.get("argv") != [
        sys.executable, "-u", str(Path(__file__)), "--manifest", str(args.manifest),
        "--out", str(args.out), "--receipt", str(args.receipt),
    ]:
        raise RuntimeError("command argv mismatch")
    if command.get("reader_sha256") != sha256(Path(__file__)):
        raise RuntimeError("command evaluator reader SHA mismatch")
    if command.get("builder_sha256") != sha256(PACKAGE / "t8192_ds4_build_v3.py"):
        raise RuntimeError("command builder SHA mismatch")
    if command.get("delta_source_sha256") != sha256(PACKAGE / "delta_pack_sources.py"):
        raise RuntimeError("command delta source SHA mismatch")
    if sha256(LOADER_SOURCE) != LOADER_SHA or command.get("loader_sha256") != LOADER_SHA:
        raise RuntimeError("binding loader source SHA mismatch")

    wire_rows = wire["layers"]
    exact_package_bytes = int(gate["actual_serialized_wire_bytes"])
    import numpy as _np
    import torch as _torch
    import triton as _triton
    np, torch, triton = _np, _torch, _triton
    sys.path.insert(0, str(PACKAGE))
    import t8192_ds4_build_v3 as builder
    builder.PlaneSource = GenesisTierSource
    wins, classes = all_windows()
    builder_wins = wins[64:] if resume_auth is not None else wins
    original_argv = sys.argv
    original_cwd = Path.cwd()
    try:
        sys.argv = [
            "t8192_ds4_build_v3.py", "--mode", "planes",
            "--planes-dir", str(args.manifest),
            "--ref-dir", str(TEACHER / "t8192_eval"),
            "--corpus", str(CORPUS),
            "--meta-dir", str(MODEL), "--local-dir", str(MODEL),
            "--out", str(args.out), "--cand-pos-limit", "1024",
            "--count", "512", "--chunk", "64", "--mb", "2",
            "--windows", ",".join(map(str, builder_wins)),
            "--tag", "GENESIS_PRE_REPAIR_PHYSICAL_FULL512",
        ]
        os.chdir(TEACHER)
        rc = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        retire_scratch(CACHE)
    if rc:
        raise RuntimeError(f"canonical builder rc={rc}")
    progress = json.loads(PROGRESS.read_text())
    validate_layer_coverage(progress.get("completed_layers"), expected_chunks=8)
    if progress.get("layer_visits") != 344 or progress.get("completed_chunks") != 8 or progress.get("local_stage_retired") is not True:
        raise RuntimeError("physical stream visit/retirement coverage drift")
    expected_resumed_mmap = list(range(43)) * 7
    if progress.get("mmap_completed_layers") != expected_resumed_mmap or progress.get("mmap_completed_count") != 301:
        raise RuntimeError("resumed mmap loader on-path coverage drift")
    if shutil.disk_usage(ROOT).free < DISK_FLOOR:
        raise RuntimeError("disk floor not restored after final layer retirement")

    reduced = reduce_outputs(args.out, wins, classes)
    global_row = reduced["global"]
    class_rows = reduced["by_class"]
    per_window = reduced["per_window"]
    loader_input_identity_sha = hashes["wire_sha256"]
    if progress.get("mmap_input_identity_sha256") != loader_input_identity_sha:
        raise RuntimeError("resumed mmap input identity drift")
    if not LOADER_SENTINEL.is_file():
        raise RuntimeError("immutable first-map loader sentinel missing")
    loader_sentinel_sha = sha256(LOADER_SENTINEL)
    chunk_receipts = []
    for chunk_index in range(1, 8):
        start = chunk_index * 64
        stop = start + 63
        chunk_path = ROOT / f"run/ARM4_MMAP_CHUNK_{start:03d}_{stop:03d}.json"
        if not chunk_path.is_file():
            raise RuntimeError(f"immutable mmap chunk receipt missing: {chunk_path}")
        chunk = json.loads(chunk_path.read_text())
        chunk_receipts.append({
            "windows": [start, stop],
            "path": str(chunk_path),
            "sha256": sha256(chunk_path),
            "loader_progress_sha256": chunk.get("loader_progress_sha256"),
        })
    loader_proof = {
        "mode": "torch-mmap",
        "loader_sha256": LOADER_SHA,
        "sentinel": str(LOADER_SENTINEL),
        "sentinel_sha256": loader_sentinel_sha,
        "input_identity_sha256": loader_input_identity_sha,
        "resumed_from_window": 64,
        "chunk_receipts": chunk_receipts,
    }
    mean = float(global_row["mean"])
    if not math.isfinite(mean) or len(per_window) != 512:
        raise RuntimeError("non-finite measured mean")
    manifest = json.loads(args.manifest.read_text())
    code_row = class_rows["code"]
    global_row["comparisons"] = {
        "step0_global": STEP0_GLOBAL, "delta_vs_step0": mean - STEP0_GLOBAL,
        "iq4_global": IQ4_GLOBAL, "delta_vs_iq4": mean - IQ4_GLOBAL,
    }
    code_row["comparisons"] = {
        "iq4_code": IQ4_CODE, "delta_vs_iq4": float(code_row["mean"]) - IQ4_CODE,
        "sealed_streamed_pre_repair_code76": EXPECTED_STREAMED_CODE,
        "delta_vs_sealed_streamed": float(code_row["mean"]) - EXPECTED_STREAMED_CODE,
    }
    window_output_set = [{"win": int(row["win"]), "sha256": row["sha256"]} for row in per_window]
    output_set_sha = hashlib.sha256(json.dumps(window_output_set, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    layer_receipt_set = [{"layer": int(row["layer"]), "receipt_sha256": row["receipt_sha256"]} for row in wire_rows]
    layer_receipt_set_sha = hashlib.sha256(json.dumps(layer_receipt_set, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    post_eval_gpu = gpu_snapshot(require_zero_util=False, own_pid=os.getpid())
    result = {
        "schema": "genesis-pre-repair-physical-full512-v2",
        "status": "PASS_FULL512_MEASURED", "measurement_label": "MEASURED",
        "task_id": TASK, "host": os.uname().nodename, "direction": "KL(teacher||candidate)",
        "contract": "once-only physical-package pre-repair full512", "windows": len(wins), "window_ids": wins,
        "window_ids_sha256": hashlib.sha256(",".join(map(str, wins)).encode()).hexdigest(),
        "positions": 512 * 1024, "support": 8192, "cutoff": 1024,
        "microbatch": 2, "chunk_size": 64, "attention": "eager",
        "ordered_coverage": {"first": 0, "last": 511, "count": 512, "exact_order": wins},
        "global": global_row, "by_class": class_rows, "code": code_row,
        "actual_serialized_wire_bytes": exact_package_bytes,
        "assignment_sha256": ASSIGNMENT_SHA, "build_builder_sha256": BUILD_BUILDER_SHA,
        "compact_manifest": str(args.manifest), "compact_manifest_sha256": COMPACT_SHA,
        "physical_package": f"{SOURCE_HOST}:{REMOTE_PACKAGE}",
        "physical_code76_pass_marker": str(PHYSICAL_PASS_MARKER),
        "physical_code76_pass_marker_sha256": hashes["pass_marker_sha256"],
        "physical_code76_marker": str(PHYSICAL_MARKER),
        "physical_code76_marker_sha256": hashes["receipt_sha256"],
        "wire_manifest": str(WIRE_MANIFEST), "wire_manifest_sha256": hashes["wire_sha256"],
        "layer_receipt_set": layer_receipt_set, "layer_receipt_set_sha256": layer_receipt_set_sha,
        "claim_sha256": claim_sha,
        "command_receipt": str(command_receipt), "command_receipt_sha256": sha256(command_receipt),
        "original_command_receipt": str(ORIGINAL_COMMAND_RECEIPT),
        "original_command_receipt_sha256": ORIGINAL_COMMAND_SHA,
        "launch_once": str(LAUNCH_ONCE),
        "launch_once_sha256": ORIGINAL_LATCH_SHA,
        "resume_launch_once": str(RESUME_LAUNCH_ONCE) if resume_auth is not None else None,
        "resume_launch_once_sha256": sha256(RESUME_LAUNCH_ONCE) if resume_auth is not None else None,
        "resume_authorization": str(RESUME_AUTHORIZATION) if resume_auth is not None else None,
        "resume_authorization_sha256": sha256(RESUME_AUTHORIZATION) if resume_auth is not None else None,
        "preserved_output_set_sha256": preserved_output_set_sha,
        "physical_reader_history": [
            {"windows": [0, 63], "reader_sha256": ORIGINAL_READER_SHA, "command_sha256": ORIGINAL_COMMAND_SHA},
            {"windows": [64, 511], "reader_sha256": sha256(Path(__file__)), "command_sha256": sha256(command_receipt)},
        ] if resume_auth is not None else [
            {"windows": [0, 511], "reader_sha256": sha256(Path(__file__)), "command_sha256": sha256(command_receipt)},
        ],
        "canonical_eval_builder": str(Path(builder.__file__)), "canonical_eval_builder_sha256": sha256(Path(builder.__file__)),
        "physical_reader": str(Path(__file__)), "physical_reader_sha256": sha256(Path(__file__)),
        "loader_proof": loader_proof,
        "delta_pack_sources_sha256": sha256(PACKAGE / "delta_pack_sources.py"),
        "teacher_done_sha256": TEACHER_DONE_SHA, "labels_baseline_sha256": BASELINE_SHA,
        "corpus_sha256": sha256(CORPUS), "model_index_sha256": MODEL_INDEX_SHA,
        "finite_gate": True,
        "physical_marker_gate": {
            "status": marker["status"], "measured_code_kld": gate["measured_code_kld"],
            "paired_tolerance": gate["tolerance"], "layers": list(range(43)),
        },
        "source_replay_exact_gate": manifest["verification"]["minimum_replay_exact_fraction"] == 1.0,
        "source_replay_exact_fraction": manifest["verification"]["minimum_replay_exact_fraction"],
        "coverage_gate": {
            "layers": 43, "projection_units_per_layer": 512,
            "chunks": 8, "layer_visits": 344,
            "stream_completed_layers": list(range(43)) * 8,
            "stream_completed_layers_by_chunk": [list(range(43)) for _ in range(8)],
            "exact_once_per_chunk_in_order": True,
        },
        "baseline_restoration_gate": {
            "persistent_model_mutated": False, "remote_physical_package_mutated": False,
            "local_layer_scratch_retired": not any(CACHE.iterdir()),
            "baseline_identity_sha256": BASELINE_SHA, "verdict": "PASS_NO_PERSISTENT_MUTATION",
        },
        "per_window": per_window, "window_output_set_sha256": output_set_sha,
        "gpu_snapshot_before_child_exit": post_eval_gpu,
        "elapsed_seconds": time.time() - started,
        "disk_free_bytes_after": shutil.disk_usage(ROOT).free, "completed_unix": time.time(),
    }
    atomic_json(args.receipt, result)
    print(json.dumps({
        "status": result["status"], "global_kld": mean, "code_kld": code_row["mean"],
        "delta_global_vs_step0": global_row["comparisons"]["delta_vs_step0"],
        "delta_global_vs_iq4": global_row["comparisons"]["delta_vs_iq4"],
        "delta_code_vs_iq4": code_row["comparisons"]["delta_vs_iq4"],
        "windows": len(wins), "positions": result["positions"], "elapsed_seconds": result["elapsed_seconds"],
        "receipt": str(args.receipt), "receipt_sha256": sha256(args.receipt),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
