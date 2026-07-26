#!/usr/bin/env python3
"""P651 final P640 RESPENT sparse-overlay rail on the exact P632 scorer.

The evaluator is imported byte-for-byte from P632. This file only provides a physical
source adapter: immutable current-wire cells are mmap-filled locally; exact P647
changed-cell artifacts are streamed layerwise from compute-node-6 and overlaid on GPU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("TORCH_LOGS", "")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

import torch

TASK = "PUBLIC_TASK"
MISSION = Path("$HOME/run-bundles/P651_STREAM_CONSUMER_PUBLIC_TASK_s7")
P632_ROOT = Path("$HOME/run-bundles/P632_DIRECTIONAL_PUBLIC_TASK_s7")
P632_SCORER = P632_ROOT / "code/p632_score.py"
P632_SCORER_SHA = "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f"
BASELINE_RECEIPT = MISSION / "receipts/BASELINE_PARITY.json"
INDEX_PATH = MISSION / "inputs/SPARSE_OVERLAY_INDEX.json"
INDEX_SHA = "63751b1b85fdbb4ad2a9f5ce95cbc5bf03d2f48b077161bb72483b37e0b49c19"
META = MISSION / "inputs/P640_FINAL_META_RESPENT"
P653_META = MISSION / "inputs/P653_ASSEMBLY_META"
P653_MANIFEST_SHA = "e03bc8919d51bbf1a9cf1f54f342e9f43dea625839ad8aad23578f7b8f9d98fa"
ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
MAP_SHA = "36d0841986d5781186f766b3815e4b3c6332eece2090d3e6d73e7e3ffa33dc07"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
WIRE_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
WIRE_BYTES = 101346521679
TLUT_PATH = MISSION / "inputs/P640_FINAL_META/qtip2/PINNED_TLUT.pt"
TLUT_FILE_SHA = "be7e69b5b18419afc333dc3ef7841bda2ed8207114eeae0ac5bcd7bcab79b93c"
TLUT_TENSOR_SHA = "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19"
QTIP_DECODER = Path("$HOME/run-bundles/QTIP_ANCHOR_WIRE_PUBLIC_TASK_s7/qtip-canonical/lib/utils/kernel_decompress.py")
QTIP_DECODER_SHA = "4d4d526ac69660b0793d0c133de5ba7532d714d041b8aca3497268f89e34add0"
PRE_REPAIR_GLOBAL = 0.1283743972596208
PREDICTED_DELTA = -0.01166995372286938
PREDICTED_MEASURED_GLOBAL = 0.11670444353675142
RAW_WITHOUT = 0.06708283585873699
RAW_WITH = 0.05541288213586761
CLASS_KEYS = ["multilingual", "prose", "reasoning", "chat", "agentic", "code"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def exact_claim() -> bytes:
    path = Path("$HOME/HOST_CLAIM.json")
    raw = path.read_bytes()
    obj = json.loads(raw)
    expected = {"host": "compute-node-7", "owner": TASK, "task_id": TASK, "mission": str(MISSION)}
    drift = {k: (obj.get(k), v) for k, v in expected.items() if obj.get(k) != v}
    if drift:
        raise RuntimeError(f"HOST_CLAIM drift: {drift}")
    return raw


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes(order="C")).hexdigest()


def load_module(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def fwht_inplace(x: torch.Tensor) -> torch.Tensor:
    n = int(x.shape[-1])
    if n <= 0 or n & (n - 1):
        raise RuntimeError(f"FWHT dimension must be power of two: {n}")
    h = 1
    while h < n:
        y = x.reshape(*x.shape[:-1], -1, 2 * h)
        a = y[..., :h].clone()
        b = y[..., h : 2 * h].clone()
        y[..., :h] = a + b
        y[..., h : 2 * h] = a - b
        x = y.reshape_as(x)
        h *= 2
    return x


def decode_qtip(payload: dict[str, Any], tlut: torch.Tensor, kernel_mod: Any, device: torch.device) -> torch.Tensor:
    geometry = payload["geometry"]
    if payload.get("schema") != "qtip-rate-rung-unit-v1":
        raise RuntimeError(f"unexpected QTIP schema {payload.get('schema')}")
    if (int(geometry["K"]), int(geometry["V"]), int(geometry["L"]), int(geometry["m"])) != (2, 2, 16, 2048):
        raise RuntimeError(f"QTIP geometry drift: {geometry}")
    if payload.get("tlut_sha256") != TLUT_TENSOR_SHA:
        raise RuntimeError("QTIP TLUT tensor pin drift")
    packed = payload["packed"].to(device=device, dtype=torch.int32)
    suw = payload["SUw"].to(device=device, dtype=torch.float32)
    sv = payload["SV"].to(device=device, dtype=torch.float32)
    had_left = payload["had_left"].to(device=device, dtype=torch.float32)
    had_right = payload["had_right"].to(device=device, dtype=torch.float32)
    tlut_dev = tlut.to(device=device, dtype=torch.float32)
    hatw = kernel_mod.decode_compressed(packed, tlut_dev, int(geometry["m"]), 2)
    w = fwht_inplace((had_left * hatw).contiguous())
    w.mul_(math.sqrt(1.0 / int(geometry["m"])))
    w = fwht_inplace((w.t() * had_right).contiguous()).t()
    w.mul_(math.sqrt(1.0 / int(geometry["n"])))
    w = (sv[:, None] * w) * suw[None, :]
    if tuple(w.shape) != (int(geometry["m"]), int(geometry["n"])) or not bool(torch.isfinite(w).all().item()):
        raise RuntimeError("QTIP reconstruction shape/finite failure")
    return w.to(torch.bfloat16)


class OverlayTransfer:
    def __init__(self, index: dict[str, Any], run_id: str, claim_raw: bytes):
        self.index = index
        self.run_id = run_id
        self.claim_raw = claim_raw
        self.root = MISSION / "scratch" / f"OVERLAY_{run_id}"
        self.receipts = MISSION / "run" / run_id / "transfer_receipts"
        self.root.mkdir(parents=True, exist_ok=True)
        self.receipts.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe(path: str) -> None:
        p = Path(path)
        if p.is_absolute() or ".." in p.parts or not path or path.startswith("."):
            raise RuntimeError(f"unsafe payload path: {path}")

    def stage(self, layer: int) -> Path:
        exact_claim()
        spec = self.index["layers"][str(layer)]
        final = self.root / f"L{layer:03d}"
        receipt = self.receipts / f"L{layer:03d}_TRANSFER.json"
        if final.exists() or receipt.exists():
            raise RuntimeError(f"once-only layer transfer destination exists: L{layer:03d}")
        temp = self.root / f".L{layer:03d}.{os.getpid()}.partial"
        temp.mkdir(parents=True)
        files = list(spec["files"])
        if not files:
            os.replace(temp, final)
            atomic_json(receipt, {
                "schema": "p651-sparse-overlay-layer-transfer-v1", "status": "PASS_NO_CHANGED_CELLS",
                "task_id": TASK, "run_id": self.run_id, "layer": layer, "stream_count": 0,
                "files": 0, "bytes": 0, "created_unix": time.time(),
            })
            return final
        roots = {x["source_root"] for x in files}
        if len(roots) != 1:
            raise RuntimeError(f"multi-root layer transfer forbidden: {roots}")
        source_root = next(iter(roots))
        for item in files:
            self._safe(str(item["path"]))
        stream_count = min(4, len(files))
        if len(files) >= 4 and stream_count < 4:
            raise RuntimeError(">=4 QSFP transfer streams required")
        partitions: list[list[dict[str, Any]]] = [[] for _ in range(stream_count)]
        loads = [0] * stream_count
        for item in sorted(files, key=lambda x: int(x["bytes"]), reverse=True):
            idx = min(range(stream_count), key=loads.__getitem__)
            partitions[idx].append(item)
            loads[idx] += int(item["bytes"])
        processes = []
        list_paths = []
        start = time.time()
        try:
            for idx, part in enumerate(partitions):
                list_path = temp / f".files.{idx}.txt"
                list_path.write_text("".join(f"{x['path']}\n" for x in part))
                list_paths.append(list_path)
                cmd = [
                    "rsync", "-a", "--partial", "--timeout=300", "--files-from", str(list_path),
                    "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                    f"{self.index['source_qsfp']}:{source_root}/", f"{temp}/",
                ]
                processes.append((idx, cmd, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)))
            failures = []
            for idx, cmd, proc in processes:
                stdout, stderr = proc.communicate()
                if proc.returncode:
                    failures.append({"stream": idx, "rc": proc.returncode, "stderr": stderr[-4000:], "stdout": stdout[-1000:]})
            if failures:
                raise RuntimeError(f"QSFP rsync failure: {failures}")
            for path in list_paths:
                path.unlink(missing_ok=True)
            verified = []
            for item in files:
                local = temp / item["path"]
                if not local.is_file() or local.stat().st_size != int(item["bytes"]):
                    raise RuntimeError(f"received file size drift: {item['path']}")
                got = sha256(local)
                if got != item["sha256"]:
                    raise RuntimeError(f"received file SHA drift: {item['path']} {got}")
                verified.append({"path": item["path"], "bytes": int(item["bytes"]), "sha256": got, "role": item["role"]})
            if exact_claim() != self.claim_raw:
                raise RuntimeError("claim bytes changed during overlay transfer")
            elapsed = time.time() - start
            os.replace(temp, final)
            atomic_json(receipt, {
                "schema": "p651-sparse-overlay-layer-transfer-v1", "status": "PASS_SHA256_VERIFIED",
                "task_id": TASK, "run_id": self.run_id, "layer": layer,
                "assignment_file_sha256": ASSIGNMENT_SHA, "assignment_map_sha256": MAP_SHA,
                "source_host": "compute-node-6", "source_qsfp": self.index["source_qsfp"], "source_root": source_root,
                "stream_count": stream_count, "partition_bytes": loads,
                "files": len(verified), "bytes": sum(x["bytes"] for x in verified),
                "elapsed_seconds": elapsed,
                "throughput_GBps": (sum(x["bytes"] for x in verified) / 1e9 / elapsed) if elapsed else None,
                "verified": verified, "created_unix": time.time(),
            })
            return final
        except Exception:
            shutil.rmtree(temp, ignore_errors=True)
            raise

    def retire(self, layer: int) -> None:
        path = self.root / f"L{layer:03d}"
        shutil.rmtree(path)
        if path.exists():
            raise RuntimeError(f"overlay scratch retirement failed: {path}")


class DirectCurrentWireSource:
    """Mixin behavior installed around the exact P632 GenesisTierSource."""

    def _direct_init(self, p: Any, base: Any, baseline: dict[str, Any], run_id: str) -> None:
        self._p651_p = p
        self._p651_base = base
        self._p651_baseline = baseline
        self._p651_run_id = run_id

    def _stage_remote(self, layer: int) -> Path:
        exact_claim()
        source = Path(self._p651_p.REMOTE_PACKAGE) / f"layer_{layer:03d}"
        receipt_path = Path(self._p651_p.REMOTE_PACKAGE) / "receipts" / f"L{layer:03d}_WIRE_COMPACT_LAYER.json"
        if not source.is_dir() or not receipt_path.is_file():
            raise RuntimeError(f"sealed current-wire source missing L{layer:03d}")
        receipt = json.loads(receipt_path.read_text())
        expected_receipt_sha = self._p651_p.WIRE_MANIFEST["layer_receipts"][str(layer)]["receipt_sha256"]
        if sha256(receipt_path) != expected_receipt_sha:
            raise RuntimeError(f"current-wire layer receipt SHA drift L{layer:03d}")
        if (
            receipt.get("status") != "PASS"
            or int(receipt.get("layer", -1)) != layer
            or receipt.get("assignment_sha256") != BASE_ASSIGNMENT_SHA
            or receipt.get("canonical_builder_sha256") != self._p651_p.BUILDER_SHA
            or int(receipt.get("physical_payload_bytes", -1)) != int(self._p651_p.WIRE_MANIFEST["layer_physical_payload_bytes"][str(layer)])
        ):
            raise RuntimeError(f"current-wire receipt authority drift L{layer:03d}")
        baseline_layer = MISSION / "run" / f"BASELINE_LAYER_{layer:03d}_STAGE.json"
        if not baseline_layer.is_file():
            raise RuntimeError(f"baseline-parity validated layer receipt missing L{layer:03d}")
        baseline_row = json.loads(baseline_layer.read_text())
        if baseline_row.get("schema") != "p651-physical-baseline-layer-stage-v1" or int(baseline_row.get("layer", -1)) != layer:
            raise RuntimeError(f"baseline layer validation receipt drift L{layer:03d}")
        expected_files = {str(x["path"]): int(x["bytes"]) for x in receipt["files"]}
        actual_files = {str(x.relative_to(source)): x.stat().st_size for x in source.rglob("*") if x.is_file()}
        if expected_files != actual_files:
            raise RuntimeError(f"current-wire source file/size drift L{layer:03d}")
        direct_receipt = MISSION / "run" / self._p651_run_id / f"L{layer:03d}_BASE_DIRECT.json"
        atomic_json(direct_receipt, {
            "schema": "p651-current-wire-readonly-direct-layer-v1", "status": "PASS_RECEIPT_AND_SIZE_BOUND",
            "task_id": TASK, "run_id": self._p651_run_id, "layer": layer,
            "source": str(source), "source_receipt": str(receipt_path), "source_receipt_sha256": expected_receipt_sha,
            "base_assignment_sha256": BASE_ASSIGNMENT_SHA, "base_wire_manifest_sha256": WIRE_SHA,
            "files": len(actual_files), "payload_bytes": sum(actual_files.values()),
            "baseline_parity_receipt_sha256": sha256(BASELINE_RECEIPT),
            "policy": "read-only direct mmap; zero base payload duplication",
            "created_unix": time.time(),
        })
        return source

    def _cleanup_stage(self, layer: int) -> None:
        self._p651_base.RawStore.assert_no_global_leaks()
        self._p651_base.assert_no_layer_mmaps(layer)
        self.active_stage = None


class SparseOverlaySource(DirectCurrentWireSource):
    def __init__(self, p: Any, base: Any, baseline: dict[str, Any], index: dict[str, Any], run_id: str, claim_raw: bytes):
        super().__init__(base.MANIFEST)
        self._direct_init(p, base, baseline, run_id)
        self.index = index
        self.run_id = run_id
        self.transfer = OverlayTransfer(index, run_id, claim_raw)
        self.prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="p651-overlay")
        self.tlut = torch.load(TLUT_PATH, map_location="cpu", weights_only=True)
        if not isinstance(self.tlut, torch.Tensor) or tuple(self.tlut.shape) != (512, 2) or tensor_sha(self.tlut) != TLUT_TENSOR_SHA:
            raise RuntimeError("pinned QTIP TLUT tensor drift")
        self.kernel_mod = load_module(f"p651_kernel_decompress_{run_id.lower()}", QTIP_DECODER)

    def _apply_vq(self, stage: Path, rows: list[dict[str, Any]], gate_up: torch.Tensor, down: torch.Tensor) -> int:
        grouped: dict[tuple[str, int, int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["projection"], int(row["d"]), int(row["k"]), row["codebook"]["path"])].append(row)
        applied = 0
        for (projection, d, k, codebook_rel), group in grouped.items():
            codebook_path = stage / codebook_rel
            codebook = torch.from_file(str(codebook_path), dtype=torch.float16, size=d * k).reshape(k, d)
            if not bool(torch.isfinite(codebook).all().item()):
                raise RuntimeError(f"non-finite codebook {codebook_path}")
            expected_cb_sha = group[0]["codebook"]["sha256"]
            if sha256(codebook_path) != expected_cb_sha:
                raise RuntimeError(f"codebook SHA drift before replay: {codebook_rel}")
            for offset in range(0, len(group), 8):
                batch = group[offset : offset + 8]
                codes, scales, experts = [], [], []
                for row in batch:
                    artifact_path = stage / row["artifact"]["path"]
                    payload = torch.load(artifact_path, map_location="cpu", mmap=True, weights_only=True)
                    meta = payload.get("meta") or {}
                    expected_identity = [int(row["layer"]), int(row["expert"]), row["projection"]]
                    if (
                        meta.get("schema") != "p647-genesis-overlay-cell-v1"
                        or meta.get("identity") != expected_identity
                        or meta.get("assignment_sha256") != ASSIGNMENT_SHA
                        or meta.get("assignment_map_sha256") != MAP_SHA
                        or meta.get("tier") != row["new"]
                        or meta.get("codebook_sha256") != expected_cb_sha
                        or meta.get("fp16_codebook_replay_exact") is not True
                    ):
                        raise RuntimeError(f"VQ payload metadata drift: {expected_identity}")
                    c, s = payload["codes"], payload["scales"]
                    if c.dtype != torch.int16 or s.dtype != torch.int8 or int(c.shape[0]) != d or int(s.shape[0]) != d:
                        raise RuntimeError(f"VQ tensor layout drift: {expected_identity}")
                    if not bool(torch.isfinite(s.float()).all().item()):
                        raise RuntimeError(f"VQ scales non-finite: {expected_identity}")
                    codes.append(c)
                    scales.append(s)
                    experts.append(int(row["expert"]))
                destination = gate_up if projection == "fused13" else down
                self._launch_vq(torch.stack(codes), torch.stack(scales), codebook, experts, destination, d)
                applied += len(batch)
        return applied

    def _apply_qtip(self, stage: Path, rows: list[dict[str, Any]], gate_up: torch.Tensor, down: torch.Tensor) -> int:
        applied = 0
        device = gate_up.device
        for row in rows:
            artifact_path = stage / row["artifact"]["path"]
            payload = torch.load(artifact_path, map_location="cpu", mmap=True, weights_only=True)
            identity = payload.get("identity")
            expected_identity = [int(row["layer"]), int(row["expert"]), row["projection"]]
            if identity != expected_identity or payload.get("tier") != "qtip2_2.0117":
                raise RuntimeError(f"QTIP identity/tier drift: expected {expected_identity}, got {identity}")
            weight = decode_qtip(payload, self.tlut, self.kernel_mod, device)
            expert = int(row["expert"])
            if row["projection"] == "fused13":
                if tuple(weight.shape) != tuple(gate_up[expert].shape):
                    raise RuntimeError(f"QTIP fused13 shape drift: {expected_identity}")
                gate_up[expert].copy_(weight)
            elif row["projection"] == "down":
                if tuple(weight.shape) != tuple(down[expert].shape):
                    raise RuntimeError(f"QTIP down shape drift: {expected_identity}")
                down[expert].copy_(weight)
            else:
                raise RuntimeError(f"invalid QTIP projection: {row['projection']}")
            del weight, payload
            applied += 1
        torch.cuda.synchronize()
        return applied

    def fill_layer(self, layer: int, gate_up: torch.Tensor, down: torch.Tensor) -> None:
        layer_spec = self.index["layers"][str(layer)]
        stage_future = self.prefetch.submit(self.transfer.stage, layer)
        super().fill_layer(layer, gate_up, down)
        stage = stage_future.result()
        rows = layer_spec["rows"]
        vq_rows = [x for x in rows if x["kind"] == "genesis_vq_rebuilt_cell"]
        qtip_rows = [x for x in rows if x["kind"] == "qtip2_exact_copy"]
        started = time.time()
        vq_applied = self._apply_vq(stage, vq_rows, gate_up, down)
        qtip_applied = self._apply_qtip(stage, qtip_rows, gate_up, down)
        if vq_applied != int(layer_spec["vq_cells"]) or qtip_applied != int(layer_spec["qtip2_cells"]):
            raise RuntimeError(f"overlay exact coverage drift L{layer:03d}")
        changed = vq_applied + qtip_applied
        if changed != int(layer_spec["changed_cells"]) or 512 - changed != int(layer_spec["unchanged_copythrough_cells"]):
            raise RuntimeError(f"overlay/copy-through partition drift L{layer:03d}")
        self.transfer.retire(layer)
        receipt = MISSION / "run" / self.run_id / f"L{layer:03d}_OVERLAY_APPLIED.json"
        atomic_json(receipt, {
            "schema": "p651-sparse-overlay-layer-applied-v1", "status": "PASS_EXACT_CHANGED_PLUS_COPYTHROUGH",
            "task_id": TASK, "run_id": self.run_id, "layer": layer,
            "assignment_file_sha256": ASSIGNMENT_SHA, "assignment_map_sha256": MAP_SHA,
            "base_assignment_sha256": BASE_ASSIGNMENT_SHA, "base_wire_manifest_sha256": WIRE_SHA,
            "changed_cells": changed, "vq_cells": vq_applied, "qtip2_cells": qtip_applied,
            "unchanged_copythrough_cells": 512 - changed,
            "row_set_sha256": hashlib.sha256(json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()).hexdigest(),
            "overlay_apply_seconds": time.time() - started, "scratch_retired": True,
            "created_unix": time.time(),
        })
        progress_path = MISSION / "run" / self.run_id / "PROGRESS.json"
        progress = json.loads((MISSION / "run" / self.run_id / "BASE_PROGRESS.json").read_text())
        progress.update({
            "schema": "p651-p640-sparse-overlay-rail-progress-v1", "task_id": TASK,
            "assignment_file_sha256": ASSIGNMENT_SHA, "assignment_map_sha256": MAP_SHA,
            "last_overlay_completed_layer": layer,
            "last_overlay_receipt_sha256": sha256(receipt),
            "changed_cells_completed_cumulative": sum(int(self.index["layers"][str(x)]["changed_cells"]) for x in range(layer + 1)),
            "updated_unix": time.time(),
        })
        atomic_json(progress_path, progress)
        print(f"[SparseOverlaySource] L{layer:03d} applied changed={changed} vq={vq_applied} qtip={qtip_applied} copythrough={512-changed}", flush=True)


def checked_preflight(p: Any, mode: str) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    claim_raw = exact_claim()
    if sha256(P632_SCORER) != P632_SCORER_SHA or sha256(INDEX_PATH) != INDEX_SHA:
        raise RuntimeError("P632 scorer or sparse resolver index SHA drift")
    if sha256(P653_META / "P653_EXACT_ASSEMBLED_WIRE_MANIFEST.json") != P653_MANIFEST_SHA:
        raise RuntimeError("P653 exact assembly manifest SHA drift")
    p653 = json.loads((P653_META / "P653_EXACT_ASSEMBLED_WIRE_MANIFEST.json").read_text())
    if (
        p653.get("status") != "PASS"
        or p653.get("assignment_file_sha256") != ASSIGNMENT_SHA
        or p653.get("assignment_map_sha256") != MAP_SHA
        or int(p653.get("exact_wire_bytes", -1)) != WIRE_BYTES
        or int(p653.get("changed_count", -1)) != 1411
        or int(p653.get("unchanged_count", -1)) != 20605
    ):
        raise RuntimeError("P653 exact assembly authority/count drift")
    index = json.loads(INDEX_PATH.read_text())
    if (
        index.get("status") != "PASS_FAIL_CLOSED_FINAL_RESPENT_ONLY"
        or index.get("assignment_file_sha256") != ASSIGNMENT_SHA
        or index.get("assignment_map_sha256") != MAP_SHA
        or index.get("base_wire_manifest_sha256") != WIRE_SHA
        or int(index.get("row_count", -1)) != 1411
        or index.get("kind_counts") != {"genesis_vq_rebuilt_cell": 1005, "qtip2_exact_copy": 406}
    ):
        raise RuntimeError("sparse resolver index authority/count drift")
    if not BASELINE_RECEIPT.is_file():
        raise RuntimeError("sealed baseline parity prerequisite missing")
    baseline = json.loads(BASELINE_RECEIPT.read_text())
    if (
        baseline.get("status") != "PASS_EXACT_BASELINE_PARITY"
        or baseline.get("pinned_p632_scorer_sha256") != P632_SCORER_SHA
        or baseline.get("wire_manifest_sha256") != WIRE_SHA
        or float(baseline.get("max_abs_delta", math.inf)) > 1e-12
    ):
        raise RuntimeError("baseline parity prerequisite drift")
    if sha256(TLUT_PATH) != TLUT_FILE_SHA or sha256(QTIP_DECODER) != QTIP_DECODER_SHA:
        raise RuntimeError("QTIP TLUT/decoder file pin drift")
    if shutil.disk_usage(MISSION).free < p.DISK_FLOOR:
        raise RuntimeError(f"disk floor failure: {shutil.disk_usage(MISSION).free} < {p.DISK_FLOOR}")
    if p.active_compute_processes():
        raise RuntimeError(f"GPU not exclusive at candidate start: {p.active_compute_processes()}")
    if exact_claim() != claim_raw:
        raise RuntimeError("claim changed during candidate preflight")
    return baseline, index, claim_raw


def evaluate(mode: str) -> dict[str, Any]:
    run_id = {"early8": "EARLY_8", "interim64": "INTERIM_64", "full512": "FINAL_512"}[mode]
    run_root = MISSION / "run" / run_id
    receipt_path = MISSION / "receipts" / f"P640_{run_id}_RAIL.json"
    if run_root.exists() or receipt_path.exists():
        raise RuntimeError(f"once-only candidate run exists: {run_id}")
    p = load_module(f"p632_exact_{run_id.lower()}", P632_SCORER)
    contract = p.preflight_contract(mode)
    if contract["failures"]:
        raise RuntimeError(f"P632 preflight failures: {contract['failures']}")
    baseline, index, claim_raw = checked_preflight(p, mode)
    run_root.mkdir(parents=True)
    cache = MISSION / "scratch" / f"BASE_DIRECT_{run_id}"
    cache.mkdir(parents=True)
    progress_base = run_root / "BASE_PROGRESS.json"
    sentinel = run_root / "SEAL.json"
    p.install_environment(mode)
    os.environ["TWOBIN_LAYER_OVERLAP"] = "1"
    os.environ["FULL512_LOADER_INPUT_SHA256"] = WIRE_SHA
    p.enforce_environment(mode)
    base = p.load_parent()
    p.configure_parent_module(base, cache=cache, progress=progress_base, sentinel=sentinel)
    base.PHYSICAL_PACKAGE = Path(p.REMOTE_PACKAGE)
    base.DISK_FLOOR = p.DISK_FLOOR

    class BoundSparseOverlaySource(SparseOverlaySource, base.GenesisTierSource):
        pass

    source = BoundSparseOverlaySource(p, base, baseline, index, run_id, claim_raw)
    artifact_rows, keys, run_indices = p.preflight_artifacts(mode)
    baseline_rows = p.selected_baseline_rows(run_indices)
    p.assert_gpu_only_source("pre_builder")
    builder = base.GENESISBuilder({"nocache": True, "precise": True, "source": source})
    p.assert_gpu_only_source("post_builder")
    wins = p.load_windows(run_indices)
    chunk_size = len(wins)
    mb = int(os.environ["TWOBIN_MB"])
    start_unix = time.time()
    outputs = []
    for offset in range(0, len(wins), chunk_size):
        chunk = wins[offset : offset + chunk_size]
        outputs.extend(builder.process(chunk, mb=mb))
        print(f"[{run_id}] windows {offset}:{offset+len(chunk)} complete", flush=True)
    source.prefetch.shutdown(wait=True, cancel_futures=False)
    current_rows, global_weighted, class_weighted = p.reduce_outputs(outputs, artifact_rows, keys)
    if len(current_rows) != len(baseline_rows) or not math.isfinite(global_weighted) or set(class_weighted) != set(CLASS_KEYS):
        raise RuntimeError("candidate reducer output coverage/finite failure")
    paired = p.paired_delta(current_rows, baseline_rows)
    transfer_receipts = sorted((run_root / "transfer_receipts").glob("L*_TRANSFER.json"))
    apply_receipts = sorted(run_root.glob("L*_OVERLAY_APPLIED.json"))
    direct_receipts = sorted(run_root.glob("L*_BASE_DIRECT.json"))
    if len(transfer_receipts) != 43 or len(apply_receipts) != 43 or len(direct_receipts) != 43:
        raise RuntimeError("candidate per-layer receipt coverage failure")
    transfer_docs = [json.loads(x.read_text()) for x in transfer_receipts]
    streams_for_payload = [int(x["stream_count"]) for x in transfer_docs if int(x["files"]) >= 4]
    if not streams_for_payload or min(streams_for_payload) < 4:
        raise RuntimeError("4-stream QSFP transfer invariant not demonstrated")
    measured_delta = global_weighted - PRE_REPAIR_GLOBAL
    prediction_error = global_weighted - PREDICTED_MEASURED_GLOBAL
    status = "PASS_MEASURED_PRE_REPAIR_IMPROVEMENT" if measured_delta < 0.0 else "FAIL_NO_PRE_REPAIR_IMPROVEMENT"
    receipt = {
        "schema": "p651-p640-pre-repair-undosed-sparse-overlay-rail-v1",
        "status": status, "task_id": TASK, "host": "compute-node-7", "mode": mode, "run_id": run_id,
        "label": "PRE_REPAIR_UNDOSED_WIRE", "repair_dose_applied": False,
        "candidate": {
            "assignment_file_sha256": ASSIGNMENT_SHA, "assignment_map_sha256": MAP_SHA,
            "exact_wire_bytes": WIRE_BYTES, "changed_cells": 1411, "unchanged_copythrough_cells": 20605,
            "vq_changed_cells": 1005, "qtip2_exact_copy_cells": 406,
            "qtip2_rep16_policy": "exact selected packed artifact bytes; no re-encode",
        },
        "instrument": {
            **contract, "pinned_p632_scorer_sha256": P632_SCORER_SHA,
            "baseline_parity_receipt": str(BASELINE_RECEIPT), "baseline_parity_receipt_sha256": sha256(BASELINE_RECEIPT),
            "p653_exact_assembly_manifest_sha256": P653_MANIFEST_SHA,
            "sparse_overlay_index_sha256": INDEX_SHA,
            "base_wire_manifest_sha256": WIRE_SHA,
            "qtip_tlut_file_sha256": TLUT_FILE_SHA, "qtip_tlut_tensor_sha256": TLUT_TENSOR_SHA,
            "qtip_decoder_sha256": QTIP_DECODER_SHA,
            "layer_overlap": int(os.environ["TWOBIN_LAYER_OVERLAP"]), "microbatch": mb,
            "window_forward_batch": len(wins), "window_count": len(wins),
        },
        "measured": {
            "global_weighted_kld": global_weighted,
            "class_weighted_kld": class_weighted,
            "paired_vs_p632_doped_baseline_diagnostic_only": paired,
            "pre_repair_undosed_measured_global": PRE_REPAIR_GLOBAL,
            "matched_delta_vs_measured_pre_repair_global": measured_delta,
            "matched_relative_change_vs_measured_pre_repair_global": measured_delta / PRE_REPAIR_GLOBAL,
        },
        "prediction_vs_measurement": {
            "prediction_basis": "P637 final fixed-count full512 raw delta applied to exact measured pre-repair baseline",
            "raw_without_solver_objective": RAW_WITHOUT, "raw_with_solver_objective": RAW_WITH,
            "predicted_delta": PREDICTED_DELTA,
            "predicted_measured_global": PREDICTED_MEASURED_GLOBAL,
            "measured_global": global_weighted, "measurement_minus_prediction": prediction_error,
            "measured_delta_minus_predicted_delta": measured_delta - PREDICTED_DELTA,
        },
        "rows": current_rows,
        "transfer": {
            "source_host": "compute-node-6", "source_qsfp": index["source_qsfp"],
            "layers": len(transfer_docs), "files": sum(int(x["files"]) for x in transfer_docs),
            "bytes": sum(int(x["bytes"]) for x in transfer_docs),
            "minimum_streams_when_payload_warrants": min(streams_for_payload),
            "all_received_files_sha256_verified": all(str(x["status"]).startswith("PASS") for x in transfer_docs),
            "bounded_layer_scratch_retired": not any(source.transfer.root.glob("L*")),
        },
        "coverage": {
            "layers": 43, "transfer_receipts": len(transfer_receipts),
            "overlay_apply_receipts": len(apply_receipts), "direct_base_receipts": len(direct_receipts),
            "changed_cells": sum(json.loads(x.read_text())["changed_cells"] for x in apply_receipts),
            "unchanged_copythrough_cells": sum(json.loads(x.read_text())["unchanged_copythrough_cells"] for x in apply_receipts),
        },
        "timing": {"started_unix": start_unix, "completed_unix": time.time(), "wall_seconds": time.time() - start_unix},
        "claim_start_sha256": hashlib.sha256(claim_raw).hexdigest(),
    }
    if receipt["coverage"]["changed_cells"] != 1411 or receipt["coverage"]["unchanged_copythrough_cells"] != 20605:
        raise RuntimeError("terminal rail coverage totals drift")
    if not receipt["transfer"]["bounded_layer_scratch_retired"]:
        raise RuntimeError("terminal overlay scratch leak")
    if exact_claim() != claim_raw:
        raise RuntimeError("claim bytes changed during candidate rail")
    atomic_json(receipt_path, receipt)
    seal = {"schema": "p651-p640-rail-seal-v1", "status": status, "task_id": TASK, "mode": mode,
            "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path), "created_unix": time.time()}
    atomic_json(sentinel, seal)
    print(json.dumps({"status": status, "mode": mode, "global": global_weighted, "pre_repair_delta": measured_delta,
                      "receipt": str(receipt_path), "receipt_sha256": sha256(receipt_path)}, sort_keys=True), flush=True)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["early8", "interim64", "full512"], required=True)
    args = parser.parse_args()
    try:
        evaluate(args.mode)
        return 0
    except Exception as exc:
        failure = MISSION / "receipts" / f"P640_{args.mode.upper()}_RAIL_FAILURE_{int(time.time())}.json"
        atomic_json(failure, {"schema": "p651-p640-sparse-overlay-rail-failure-v1", "status": "FAIL_CLOSED",
                              "task_id": TASK, "mode": args.mode, "error": repr(exc), "created_unix": time.time()})
        print(f"FAIL_CLOSED {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
