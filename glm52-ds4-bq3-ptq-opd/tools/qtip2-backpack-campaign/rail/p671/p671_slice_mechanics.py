#!/usr/bin/env python3
"""P651 exact sparse-stream consumer over the immutable P640/P602 physical wire.

This adapter deliberately reuses the pinned P632 reducer, P625 canonical reader,
and P602 canonical physical package.  Only P640's final 1,411 changed-cell
artifacts and their exact codebooks are staged, four-way over the direct QSFP
link, one layer at a time.  The layer scratch is retired after decode.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping

import numpy as np
import torch
import triton

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P651_STREAM_CONSUMER_PUBLIC_TASK_s7")
CLAIM = Path("$HOME/HOST_CLAIM.json")
P632_ROOT = Path("$HOME/run-bundles/P632_DIRECTIONAL_PUBLIC_TASK_s7")
P632_SCORE = P632_ROOT / "code/p632_score.py"
P640_ROOT = Path("$HOME/run-bundles/P640_BANANA_SMASHER_QTIP2_WIRE_PUBLIC_TASK_s6")
P640_HOST = "203.0.113.6"
META = ROOT / "inputs/P640_FINAL_META"
ASSIGNMENT = META / "inputs/ASSIGNMENT_RESPENT.json"
BASE_ASSIGNMENT = META / "inputs/CURRENT_BANANA_SMASHER_ASSIGNMENT.json"
BASELINE_PARITY = ROOT / "receipts/BASELINE_PARITY_CODE76.json"
QTIP_SOURCE = ROOT / "code/run_qtip2_anchor_pinned.py"
QTIP_KERNEL = ROOT / "code/qtip2_kernel_decompress_pinned.py"
QTIP_TLUT = META / "qtip2/PINNED_TLUT.pt"

FINAL_ASSIGNMENT_SHA = "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65"
FINAL_MAP_SHA = "36d0841986d5781186f766b3815e4b3c6332eece2090d3e6d73e7e3ffa33dc07"
BASE_ASSIGNMENT_SHA = "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
EXACT_WIRE_BYTES = 101_346_521_679
BASE_WIRE_MANIFEST_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
P632_SCORE_SHA = "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f"
QTIP_SOURCE_SHA = "b2a9b6c60e95aa387129246fd0f30354f356d9b89409d71089d5fddffb7eea4a"
QTIP_KERNEL_SHA = "01b6520c8f39982ac5f35de58364f31f79f00555350c8c1776e4fd9b1ca0a63f"
QTIP_TLUT_TENSOR_SHA = "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19"
CANONICAL_BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
PRE_REPAIR_GLOBAL = 0.1283743972596208
RAW_WITHOUT = 0.06708283585873699
RAW_WITH = 0.05541288213586761
PREDICTED_MEASURED_GLOBAL = 0.11670444353675142
FROZEN_CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
MODE_COUNTS = {"early8": 8, "interim64": 64, "full512": 512, "slice_w064_127": 64}
SLICE_MODE = "slice_w064_127"
SLICE_WINDOWS = list(range(64, 128))
SLICE_RUN_ID = "P640_SLICE_W064_127"

MANIFESTS = (
    {
        "name": "L00_11",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L00_11/OVERLAY_PUBLIC_TASK/receipts/SHARD_MANIFEST.json",
        "sha256": "20051b54730d9e46237c5cc84a94c6c16f24896f157293d758a83e49742568ea",
        "rows_key": "rows",
        "layers": range(0, 12),
        "source_prefix": "WIRE_STREAM_IN/P647_RESPENT/L00_11/OVERLAY_PUBLIC_TASK",
    },
    {
        "name": "L12_22",
        "path": META / "WIRE_STREAM_IN/P645_VALID_S8/OVERLAY/receipts/SHARD_MANIFEST.json",
        "sha256": "7262403f9fc5567e77026ebb0bb9f9ead660c82b3d19ebc702296a93e878e2db",
        "rows_key": "rows",
        "layers": range(12, 23),
        "source_prefix": "WIRE_STREAM_IN/P647_RESPENT/L12_22/OVERLAY",
    },
    {
        "name": "L23_32",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L23_32/overlay/receipts/SHARD_MANIFEST.json",
        "sha256": "a60e2efea128516a2148b747128f69d78a5ef18c3a62a09cf67503c1f1c6b3d5",
        "rows_key": "rows",
        "layers": range(23, 33),
        "source_prefix": "WIRE_STREAM_IN/P647_RESPENT/L23_32/overlay",
    },
    {
        "name": "L33_42",
        "path": META / "WIRE_STREAM_IN/P647_RESPENT/L33_42/P647_RESPENT_OVERLAY_L33_42_FINAL.json",
        "sha256": "1abbcf65837c573b365216f595cf87bca9658e70c9c23998740139b4825e95c8",
        "rows_key": "changed_artifact_validation_rows",
        "layers": range(33, 43),
        "source_prefix": "WIRE_STREAM_IN/P647_RESPENT/L33_42",
    },
)
P652_PINS = {
    META / "P652_FINAL/P652_EXACT_FANIN_RECEIPT.json": "132278bad3a38ee03f3a4d02728898881e163937da3ec2bc8497c7ac038005d5",
    META / "P652_FINAL/P652_EXACT_FANIN_LEDGER.json": "fc0e04457947ea470007ef35858e51a362e449b61c91dd08e306a2a49fd3c26c",
    META / "P652_FINAL/receipts/P647_RESPENT_L12_L22_FINAL_MANIFEST.json": "13300f002a0cbd0a012097ccf92001b80dc07c499d79c930bfda9b6aa1fc57c9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temp.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_claim() -> tuple[bytes, dict[str, Any]]:
    raw = CLAIM.read_bytes()
    payload = json.loads(raw)
    exact = {
        "host": "compute-node-7", "owner": TASK, "task": TASK, "task_id": TASK,
        "mission": str(ROOT),
    }
    drift = {key: (payload.get(key), value) for key, value in exact.items() if payload.get(key) != value}
    if drift:
        raise RuntimeError(f"compute-node-7 P651 claim drift: {drift}")
    if payload.get("status") not in ("CLAIMED", "ACTIVE"):
        raise RuntimeError(f"compute-node-7 P651 claim status drift: {payload.get('status')}")
    return raw, payload


def flatten_assignment(path: Path) -> dict[tuple[int, int, str], str]:
    payload = json.loads(path.read_text())
    rows = payload.get("assignment")
    if not isinstance(rows, dict) or set(map(int, rows)) != set(range(43)):
        raise RuntimeError(f"assignment shape drift: {path}")
    out: dict[tuple[int, int, str], str] = {}
    for layer in range(43):
        row = rows[str(layer)]
        if not isinstance(row, dict) or set(map(int, row)) != set(range(256)):
            raise RuntimeError(f"assignment expert surface drift L{layer}")
        for expert in range(256):
            cell = row[str(expert)]
            if not isinstance(cell, dict) or set(cell) != {"fused13", "down"}:
                raise RuntimeError(f"assignment cell drift L{layer}/E{expert}")
            for projection in ("fused13", "down"):
                out[(layer, expert, projection)] = str(cell[projection])
    return out


def normalize_row(row: Mapping[str, Any], group: Mapping[str, Any]) -> dict[str, Any]:
    identity_value = row.get("identity")
    if isinstance(identity_value, dict):
        identity = identity_value
    elif isinstance(identity_value, list) and len(identity_value) == 3:
        identity = {"layer": identity_value[0], "expert": identity_value[1], "projection": identity_value[2]}
    else:
        identity = {}
    layer = int(row.get("layer", identity.get("layer", -1)))
    expert = int(row.get("expert", identity.get("expert", -1)))
    projection = str(row.get("projection", identity.get("projection")))
    artifact = Path(str(row.get("artifact") or row.get("artifact_name"))).name
    artifact_sha = str(row.get("artifact_sha256"))
    artifact_bytes = int(row.get("artifact_physical_bytes") or row.get("artifact_bytes") or row.get("bytes") or 0)
    raw_kind = str(row.get("kind"))
    kind = "qtip2_exact" if raw_kind == "qtip2_exact_copy" else "banana_smasher_vq" if raw_kind == "banana_smasher_vq_rebuilt_cell" else raw_kind
    old_tier = str(row.get("old_tier", row.get("old")))
    new_tier = str(row.get("new_tier", row.get("new")))
    if projection not in ("fused13", "down") or layer not in group["layers"]:
        raise RuntimeError(f"row identity/range drift: {row}")
    if not artifact or artifact == "None" or len(artifact_sha) != 64 or artifact_bytes <= 0:
        raise RuntimeError(f"row artifact surface drift: {row}")
    if kind == "qtip2_exact":
        if new_tier != "qtip2_2.0117" or (row.get("source_artifact_sha256") is not None and row.get("source_artifact_sha256") != artifact_sha):
            raise RuntimeError(f"QTIP2 selection drift: {row}")
        codebook = codebook_sha = None
        codebook_bytes = 0
    elif kind == "banana_smasher_vq":
        codebook = Path(str(row.get("codebook") or row.get("codebook_name"))).name
        codebook_sha = str(row.get("codebook_sha256"))
        try:
            d_text, k_text = new_tier.split("_")
            derived_codebook_bytes = 2 * int(d_text.removeprefix("d")) * int(k_text.removeprefix("k"))
        except Exception as exc:
            raise RuntimeError(f"cannot derive VQ codebook geometry: {row}") from exc
        codebook_bytes = int(row.get("codebook_physical_bytes") or row.get("codebook_bytes") or derived_codebook_bytes)
        if not codebook or codebook == "None" or len(codebook_sha) != 64 or codebook_bytes <= 0:
            raise RuntimeError(f"VQ codebook surface drift: {row}")
    else:
        raise RuntimeError(f"unsupported changed-cell kind: {kind}")
    prefix = str(group["source_prefix"])
    if group["name"] == "L33_42":
        prefix = f"{prefix}/L{layer:03d}/overlay"
    return {
        "layer": layer, "expert": expert, "projection": projection,
        "old_tier": old_tier, "new_tier": new_tier, "kind": kind,
        "artifact": artifact, "artifact_sha256": artifact_sha,
        "artifact_physical_bytes": artifact_bytes,
        "artifact_source_rel": f"{prefix}/cells/{artifact}",
        "codebook": codebook, "codebook_sha256": codebook_sha,
        "codebook_physical_bytes": codebook_bytes,
        "codebook_source_rel": f"{prefix}/codebooks/{codebook}" if codebook else None,
        "group": group["name"],
    }


def preflight_manifests() -> dict[str, Any]:
    if sha256_file(ASSIGNMENT) != FINAL_ASSIGNMENT_SHA or sha256_file(BASE_ASSIGNMENT) != BASE_ASSIGNMENT_SHA:
        raise RuntimeError("assignment byte identity drift")
    final_payload = json.loads(ASSIGNMENT.read_text())
    final_text = json.dumps(final_payload, sort_keys=True)
    if FINAL_MAP_SHA not in final_text:
        raise RuntimeError("final embedded assignment-map authority drift")
    for path, expected in P652_PINS.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"P652 final fan-in control drift: {path}")
    fanin = json.loads((META / "P652_FINAL/P652_EXACT_FANIN_RECEIPT.json").read_text())
    fanin_text = json.dumps(fanin, sort_keys=True)
    if fanin.get("status") != "PASS_EXACT_FANIN" or FINAL_ASSIGNMENT_SHA not in fanin_text or FINAL_MAP_SHA not in fanin_text or "714" not in fanin_text:
        raise RuntimeError("P652 final fan-in receipt semantic drift")

    rows: list[dict[str, Any]] = []
    manifest_rows = []
    for group in MANIFESTS:
        path = group["path"]
        if sha256_file(path) != group["sha256"]:
            raise RuntimeError(f"overlay manifest SHA drift: {path}")
        payload = json.loads(path.read_text())
        text = json.dumps(payload, sort_keys=True)
        if FINAL_ASSIGNMENT_SHA not in text or FINAL_MAP_SHA not in text or str(EXACT_WIRE_BYTES) not in text:
            raise RuntimeError(f"overlay manifest authority drift: {path}")
        raw_rows = payload.get(group["rows_key"])
        if not isinstance(raw_rows, list):
            raise RuntimeError(f"overlay row surface missing: {path}")
        normalized = [normalize_row(row, group) for row in raw_rows]
        rows.extend(normalized)
        manifest_rows.append({
            "name": group["name"], "path": str(path), "sha256": group["sha256"],
            "rows": len(normalized), "bytes": path.stat().st_size,
        })

    identity = [(r["layer"], r["expert"], r["projection"]) for r in rows]
    if len(rows) != 1411 or len(set(identity)) != 1411:
        raise RuntimeError(f"final changed-cell coverage drift rows={len(rows)} unique={len(set(identity))}")
    base = flatten_assignment(BASE_ASSIGNMENT)
    final = flatten_assignment(ASSIGNMENT)
    expected = {key for key in final if final[key] != base[key]}
    if set(identity) != expected:
        raise RuntimeError(f"assignment/overlay identity mismatch missing={len(expected-set(identity))} extra={len(set(identity)-expected)}")
    for row in rows:
        key = (row["layer"], row["expert"], row["projection"])
        if row["old_tier"] != base[key] or row["new_tier"] != final[key]:
            raise RuntimeError(f"overlay tier transition drift: {row}")
    qtip_counts = Counter(r["layer"] for r in rows if r["kind"] == "qtip2_exact")
    if qtip_counts != Counter({0: 7, 6: 129, 16: 268, 22: 2}):
        raise RuntimeError(f"QTIP2 changed-cell coverage drift: {qtip_counts}")
    counts = Counter(r["group"] for r in rows)
    if counts != Counter({"L00_11": 542, "L12_22": 714, "L23_32": 86, "L33_42": 69}):
        raise RuntimeError(f"shard changed-cell coverage drift: {counts}")
    by_layer = {layer: [r for r in rows if r["layer"] == layer] for layer in range(43)}
    return {
        "rows": rows,
        "by_layer": by_layer,
        "manifest_rows": manifest_rows,
        "changed_cells": len(rows),
        "unchanged_cells": len(final) - len(rows),
        "qtip2_cells": sum(1 for r in rows if r["kind"] == "qtip2_exact"),
        "vq_cells": sum(1 for r in rows if r["kind"] == "banana_smasher_vq"),
        "identity_set_sha256": canonical_json_sha256(sorted(identity)),
    }


def stage_specs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidates = [
            {"rel": row["artifact_source_rel"], "sha256": row["artifact_sha256"], "bytes": row["artifact_physical_bytes"], "type": "cell"},
        ]
        if row["codebook"]:
            candidates.append({"rel": row["codebook_source_rel"], "sha256": row["codebook_sha256"], "bytes": row["codebook_physical_bytes"], "type": "codebook"})
        for spec in candidates:
            prior = specs.setdefault(spec["rel"], spec)
            if prior != spec:
                raise RuntimeError(f"conflicting source artifact identity: {spec['rel']}")
    return sorted(specs.values(), key=lambda item: item["rel"])


def stage_layer(layer: int, rows: list[dict[str, Any]], cache_root: Path) -> tuple[Path | None, dict[str, Any]]:
    if not rows:
        return None, {"layer": layer, "streams": 0, "files": 0, "bytes": 0, "elapsed_seconds": 0.0, "bytes_per_second": None}
    specs = stage_specs(rows)
    stage = cache_root / f"overlay_layer_{layer:03d}"
    partial = cache_root / f".overlay_layer_{layer:03d}.{uuid.uuid4().hex}.partial"
    if stage.exists() or partial.exists():
        raise RuntimeError(f"once-only overlay stage exists L{layer}")
    partial.mkdir(parents=True)
    streams = min(4, len(specs))
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(streams)]
    totals = [0] * streams
    for spec in sorted(specs, key=lambda item: int(item["bytes"]), reverse=True):
        idx = min(range(streams), key=lambda i: totals[i])
        buckets[idx].append(spec)
        totals[idx] += int(spec["bytes"])
    lists = []
    for idx, bucket in enumerate(buckets):
        path = partial / f"files_{idx}.txt"
        path.write_text("".join(f"{item['rel']}\n" for item in bucket))
        lists.append(path)
    started = time.time()

    def run_one(idx: int) -> dict[str, Any]:
        command = [
            "rsync", "-a", "--relative", f"--files-from={lists[idx]}",
            "-e", "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes",
            f"{P640_HOST}:{P640_ROOT}/", f"{partial}/",
        ]
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode:
            raise RuntimeError(f"rsync worker {idx} rc={proc.returncode}: {proc.stderr[-2000:]}")
        return {"worker": idx, "files": len(buckets[idx]), "bytes": totals[idx]}

    with ThreadPoolExecutor(max_workers=streams) as pool:
        workers = list(pool.map(run_one, range(streams)))
    for spec in specs:
        path = partial / spec["rel"]
        if not path.is_file() or path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
            raise RuntimeError(f"staged artifact identity drift L{layer}: {spec['rel']}")
    for path in lists:
        path.unlink()
    os.replace(partial, stage)
    elapsed = time.time() - started
    total_bytes = sum(int(spec["bytes"]) for spec in specs)
    return stage, {
        "layer": layer, "streams": streams, "workers": workers,
        "files": len(specs), "cells": len(rows), "bytes": total_bytes,
        "elapsed_seconds": elapsed, "bytes_per_second": total_bytes / elapsed,
        "source_host": P640_HOST, "source_root": str(P640_ROOT),
    }


class Qtip2Decoder:
    def __init__(self) -> None:
        if sha256_file(QTIP_SOURCE) != QTIP_SOURCE_SHA or sha256_file(QTIP_KERNEL) != QTIP_KERNEL_SHA:
            raise RuntimeError("pinned QTIP2 decoder source drift")
        qsource = load_module("p651_qtip2_anchor_pinned", QTIP_SOURCE)
        self.kernel = load_module("p651_qtip2_kernel_pinned", QTIP_KERNEL)
        self.fwht = qsource.QtipResolver.fwht
        tlut_payload = torch.load(QTIP_TLUT, map_location="cpu", mmap=True, weights_only=True)
        tlut = tlut_payload.get("tlut")
        if not isinstance(tlut, torch.Tensor) or tuple(tlut.shape) != (65536, 9):
            raise RuntimeError("QTIP2 TLUT tensor surface drift")
        tlut_sha = hashlib.sha256(tlut.contiguous().numpy().tobytes()).hexdigest()
        if tlut_sha != QTIP_TLUT_TENSOR_SHA:
            raise RuntimeError("QTIP2 TLUT tensor identity drift")
        n = 1 << 16
        torch.manual_seed(2)
        table = torch.randn(n, 16, dtype=torch.float32)
        table /= torch.linalg.norm(table, axis=1, keepdim=True)
        cb = table[tlut.to(torch.long)].reshape(n, -1).clone()
        signs = (torch.randint(0, 2, (n, 1), dtype=torch.int64) * 2 - 1).to(torch.float32)
        self.expanded = torch.cat([signs * cb, -signs * cb], dim=1)
        self.tlut_sha = tlut_sha

    def decode(self, path: Path, row: Mapping[str, Any], destination: torch.Tensor) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        identity = payload.get("identity")
        geometry = payload.get("geometry")
        if (
            payload.get("schema") not in ("qtip-rate-rung-unit-v1", "qtip-hyb-wire-unit-v1")
            or not isinstance(identity, dict)
            or int(identity.get("layer", -1)) != row["layer"]
            or int(identity.get("expert", -1)) != row["expert"]
            or identity.get("projection") != row["projection"]
            or geometry != {"L": 16, "K": 2, "V": 2, "tlut_bits": 9, "td_x": 16, "td_y": 16, "decode_mode": "quantlut_sym"}
            or payload.get("tlut_sha256") != QTIP_TLUT_TENSOR_SHA
        ):
            raise RuntimeError(f"QTIP2 payload identity/geometry drift: {path}")
        shape = tuple(map(int, payload["shape"]))
        if shape != tuple(destination.shape):
            raise RuntimeError(f"QTIP2 destination shape drift {path}: payload={shape} dest={tuple(destination.shape)}")
        trellis = payload["trellis"].contiguous().view(-1)
        raw = self.kernel.decode_compressed(16, 9, 2, 1, shape[0], shape[1], trellis, self.expanded)
        raw = raw * payload["Wscale"].float()
        raw = self.fwht(raw.T).T
        raw = raw * payload["SV"].float()[None, :]
        raw = self.fwht(raw)
        raw = raw * payload["SU"].float()[:, None]
        raw16 = raw.to(torch.float16)
        if tuple(raw16.shape) != shape or not bool(torch.isfinite(raw16).all()):
            raise RuntimeError(f"QTIP2 decoded surface invalid: {path}")
        destination.copy_(raw16.to(device=destination.device, dtype=destination.dtype))
        out = {"layer": row["layer"], "expert": row["expert"], "projection": row["projection"], "shape": list(shape), "finite": True}
        del payload, trellis, raw, raw16
        return out


def validate_base_layer(base: Any, source: Any, layer: int, row: Mapping[str, Any]) -> Path:
    source._cleanup_stage()
    stage = Path(source.PHYSICAL_PACKAGE) / f"layer_{layer:03d}"
    receipt_path = stage / "LAYER_RECEIPT.json"
    if not receipt_path.is_file() or sha256_file(receipt_path) != row["receipt_sha256"]:
        raise RuntimeError(f"immutable P602 L{layer} receipt identity drift")
    receipt = json.loads(receipt_path.read_text())
    required = int(row["physical_wire_bytes"])
    if (
        receipt.get("schema") != "banana_smasher-materialized-layer-v1"
        or receipt.get("status") != "PASS"
        or int(receipt.get("layer", -1)) != layer
        or receipt.get("assignment_sha256") != BASE_ASSIGNMENT_SHA
        or receipt.get("builder_sha256") != base.BUILD_BUILDER_SHA
        or int(receipt.get("physical_wire_bytes", -1)) != required
    ):
        raise RuntimeError(f"immutable P602 L{layer} receipt semantic drift")
    accounting = base.validate_staged_layer(
        stage, receipt, required_bytes=required,
        free_bytes_after=shutil.disk_usage(ROOT).free, floor=0,
    )
    atomic_json(ROOT / f"run/BASE_LAYER_{layer:03d}.json", {
        "schema": "p651-immutable-base-layer-proof-v1", "status": "PASS",
        "task_id": TASK, "layer": layer, "receipt_sha256": row["receipt_sha256"],
        "wire_manifest_sha256": BASE_WIRE_MANIFEST_SHA, **accounting,
    })
    return stage


def install_stream_source(base: Any, manifest: Mapping[str, Any], cache: Path, mode: str):
    by_layer = manifest["by_layer"]
    stage_rows: list[dict[str, Any]] = []
    applied: dict[tuple[int, int, str], dict[str, Any]] = {}
    decoder_holder: dict[str, Qtip2Decoder] = {}

    class P640StreamSource(base.BananaSmasherTierSource):
        def _stage_remote(self, layer: int, row: dict) -> Path:
            return validate_base_layer(base, self, layer, row)

        def fill_layer(self, layer: int, gate_up: torch.Tensor, down: torch.Tensor, hc: int, intermediate: int) -> None:
            super().fill_layer(layer, gate_up, down, hc, intermediate)
            rows = by_layer[layer]
            if not rows:
                stage_rows.append({"layer": layer, "changed_cells": 0, "streams": 0, "files": 0, "bytes": 0, "elapsed_seconds": 0.0})
                return
            stage, receipt = stage_layer(layer, rows, cache)
            assert stage is not None
            qtip_rows = [row for row in rows if row["kind"] == "qtip2_exact"]
            if qtip_rows and "decoder" not in decoder_holder:
                decoder_holder["decoder"] = Qtip2Decoder()
            try:
                for row in rows:
                    destination = gate_up[row["expert"]] if row["projection"] == "fused13" else down[row["expert"]]
                    artifact = stage / row["artifact_source_rel"]
                    if row["kind"] == "qtip2_exact":
                        decode_info = decoder_holder["decoder"].decode(artifact, row, destination)
                    else:
                        payload = torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
                        meta = payload.get("meta")
                        if (
                            not isinstance(meta, dict)
                            or meta.get("schema") != "p640-banana_smasher-vq-overlay-cell-v1"
                            or meta.get("assignment_sha256") != FINAL_ASSIGNMENT_SHA
                            or meta.get("canonical_builder_sha256") != CANONICAL_BUILDER_SHA
                            or int(meta.get("layer", -1)) != row["layer"]
                            or int(meta.get("expert", -1)) != row["expert"]
                            or meta.get("projection") != row["projection"]
                            or meta.get("tier") != row["new_tier"]
                            or meta.get("codebook_sha256") != row["codebook_sha256"]
                            or meta.get("fp16_codebook_replay_exact") is not True
                        ):
                            raise RuntimeError(f"VQ overlay payload metadata drift: {artifact}")
                        d, k = int(meta["d"]), int(meta["k"])
                        codebook_path = stage / row["codebook_source_rel"]
                        codebook = torch.from_file(str(codebook_path), dtype=torch.float16, size=k * d).reshape(k, d).clone()
                        codes = payload["codes"].unsqueeze(0)
                        scales = payload["scales"].unsqueeze(0)
                        if not bool(torch.isfinite(codebook).all()) or int(codes.min()) < 0 or int(codes.max()) >= k:
                            raise RuntimeError(f"VQ overlay numerical surface drift: {artifact}")
                        base.BananaSmasherTierSource._launch_vq(codes, scales, codebook, [row["expert"]], gate_up if row["projection"] == "fused13" else down, d)
                        decode_info = {"d": d, "k": k, "finite_codebook": True, "fp16_codebook_replay_exact": True}
                        del payload, codebook, codes, scales
                    key = (row["layer"], row["expert"], row["projection"])
                    record = {
                        "layer": row["layer"], "expert": row["expert"], "projection": row["projection"],
                        "old_tier": row["old_tier"], "new_tier": row["new_tier"], "kind": row["kind"],
                        "artifact_sha256": row["artifact_sha256"], "artifact_physical_bytes": row["artifact_physical_bytes"],
                        "codebook_sha256": row["codebook_sha256"], "decode": decode_info,
                    }
                    prior = applied.setdefault(key, record)
                    if prior != record:
                        raise RuntimeError(f"non-deterministic changed-cell application: {key}")
                torch.cuda.synchronize()
            finally:
                gc.collect()
                shutil.rmtree(stage)
            receipt.update({
                "schema": "p651-layer-overlay-consumption-v1", "status": "PASS",
                "task_id": TASK, "mode": mode, "changed_cells": len(rows),
                "qtip2_cells": len(qtip_rows), "vq_cells": len(rows) - len(qtip_rows),
                "stage_retired": not stage.exists(),
            })
            stage_rows.append(receipt)
            atomic_json(ROOT / f"run/{mode}/LAYER_{layer:03d}_OVERLAY.json", receipt)

    return P640StreamSource, {"stage_rows": stage_rows, "applied": applied, "decoder_holder": decoder_holder}


def six_class_summary(reduced: Mapping[str, Any], wins: list[int], classes: Mapping[int, str], baseline: Mapping[str, Any], p632: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    summaries = {}
    deltas = {}
    for label in FROZEN_CLASSES:
        selected = [win for win in wins if classes[win] == label]
        if selected:
            summaries[label] = reduced["by_class"][label]
            deltas[label] = p632.paired_delta(reduced["per_window"], baseline["per_window"], selected, label)
        else:
            summaries[label] = {"source_class": label, "mean": None, "n_windows": 0, "n_positions": 0, "window_mean_se": None, "window_mean_ci95": None}
            deltas[label] = {"label": label, "candidate_minus_pre_repair_mean": None, "n_windows": 0, "window_ids": [], "window_mean_se": None, "window_mean_ci95": None}
    return summaries, deltas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=tuple(MODE_COUNTS), default="early8")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    started = time.time()
    if sha256_file(P632_SCORE) != P632_SCORE_SHA:
        raise RuntimeError("P632 exact scorer source drift")
    manifest = preflight_manifests()
    preflight_public = {key: value for key, value in manifest.items() if key not in ("rows", "by_layer")}
    if args.preflight_only:
        receipt = {
            "schema": "p651-final-overlay-preflight-v1", "status": "PASS",
            "task_id": TASK, "host": "compute-node-8", "source_host": P640_HOST,
            "assignment_sha256": FINAL_ASSIGNMENT_SHA, "assignment_map_sha256": FINAL_MAP_SHA,
            "base_assignment_sha256": BASE_ASSIGNMENT_SHA, "exact_wire_bytes": EXACT_WIRE_BYTES,
            "base_wire_manifest_sha256": BASE_WIRE_MANIFEST_SHA, **preflight_public,
            "source_payload_policy": "read-only exact final P640 receipts; no immutable pack or full overlay duplication",
            "completed_unix": time.time(),
        }
        atomic_json(ROOT / "receipts/FINAL_OVERLAY_PREFLIGHT.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if not BASELINE_PARITY.is_file():
        raise RuntimeError("baseline parity receipt missing")
    parity = json.loads(BASELINE_PARITY.read_text())
    if parity.get("status") != "PASS_EXACT_SAME_HOST_INSTRUMENT" or float(parity.get("maximum_absolute_window_mean_delta", math.inf)) > 1e-12:
        raise RuntimeError("baseline parity gate failed")

    mode = args.mode
    count = MODE_COUNTS[mode]
    run_id = SLICE_RUN_ID if mode == SLICE_MODE else f"P640_FINAL_{mode}"
    out = ROOT / f"out/{run_id}"
    receipt_path = ROOT / ("receipts/RAIL_SLICE_W064_127.json" if mode == SLICE_MODE else f"receipts/RAIL_{mode.upper()}.json")
    cache = ROOT / f"scratch/{run_id}"
    progress = ROOT / f"run/{mode}/CANONICAL_PROGRESS.json"
    sentinel = ROOT / f"run/{mode}/LOADER_SENTINEL.json"
    if receipt_path.exists() or out.exists() or cache.exists() or progress.exists() or sentinel.exists():
        raise RuntimeError(f"once-only rail target already exists: {run_id}")
    out.mkdir(parents=True)
    cache.mkdir(parents=True)

    p632 = load_module(f"p651_p632_{mode}", P632_SCORE)
    p632.current_claim = current_claim
    # The corpus/input authority is still the frozen FULL512 contract; only the
    # task-owned refs/window selector below changes for this 64-window shard.
    sealed = p632.preflight_contract("full512" if mode == SLICE_MODE else mode)
    claim_raw_before, claim = p632.current_claim()
    claim_sha = hashlib.sha256(claim_raw_before).hexdigest()
    gpu_before = p632.gpu_snapshot(require_zero_util=True)
    base = p632.load_module(f"p651_canonical_reader_{mode}", p632.CANONICAL_READER)
    p632.configure_parent_module(base, cache=cache, progress=progress, sentinel=sentinel)
    base.PHYSICAL_PACKAGE = Path(p632.REMOTE_PACKAGE)
    env_contract = p632.install_environment()
    env_contract["TWOBIN_LAYER_OVERLAP"] = "1"
    os.environ["TWOBIN_LAYER_OVERLAP"] = "1"
    sys.path.insert(0, str(p632.PACKAGE))
    import t8192_ds4_build_v3 as builder
    P640StreamSource, overlay = install_stream_source(base, manifest, cache, mode)
    builder.PlaneSource = P640StreamSource
    labels_payload = json.loads(p632.LABELS.read_text())
    classes = {int(row["win"]): str(row["source_class"]) for row in labels_payload["per_window"]}
    counts = Counter(classes.values())
    if set(classes) != set(range(512)) or counts != Counter(sealed["window_contract"]["full512_class_counts"]):
        raise RuntimeError("class label surface drift")
    wins = SLICE_WINDOWS if mode == SLICE_MODE else list(range(count))
    original_argv = sys.argv
    original_cwd = Path.cwd()
    rc = -1
    try:
        sys.argv = [
            "t8192_ds4_build_v3.py", "--mode", "planes",
            "--planes-dir", str(p632.COMPACT_MANIFEST),
            "--ref-dir", str(p632.TEACHER / "t8192_eval"),
            "--corpus", str(p632.CORPUS),
            "--meta-dir", str(p632.MODEL), "--local-dir", str(p632.MODEL),
            "--out", str(out), "--cand-pos-limit", "1024",
            "--count", str(count), "--chunk", str(count), "--mb", "2",
            "--windows", ",".join(map(str, wins)),
            "--tag", f"PRE_REPAIR_UNDOSED_WIRE_{run_id}",
        ]
        os.chdir(p632.TEACHER)
        with p632.force_weights_only_torch_loads() as weights_only_stats:
            rc = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        base.retire_scratch(cache)
    if rc:
        raise RuntimeError(f"canonical builder rc={rc}")
    if any(cache.iterdir()):
        raise RuntimeError("task-local rail scratch not retired")
    if set(overlay["applied"]) != {(r["layer"], r["expert"], r["projection"]) for r in manifest["rows"]}:
        raise RuntimeError("changed-cell application coverage drift")
    expected_visits = list(range(43))
    progress_payload = json.loads(progress.read_text())
    if (
        progress_payload.get("completed_layers") != expected_visits
        or progress_payload.get("mmap_completed_layers") != expected_visits
        or progress_payload.get("completed_chunks") != 1
        or progress_payload.get("local_stage_retired") is not True
        or progress_payload.get("mmap_loader_mode") != "torch-mmap"
    ):
        raise RuntimeError("canonical layer/mmap coverage drift")
    reduced = p632.reduce_outputs(out, wins, classes)
    baseline = json.loads(p632.BASELINE_FULL512.read_text())
    summaries, deltas = six_class_summary(reduced, wins, classes, baseline, p632)
    global_delta = p632.paired_delta(reduced["per_window"], baseline["per_window"], wins, "global")
    measured = float(reduced["global"]["mean"])
    if mode == SLICE_MODE:
        prediction = {
            "solver_raw_without_qtip2": RAW_WITHOUT,
            "solver_raw_with_qtip2": RAW_WITH,
            "solver_raw_delta": RAW_WITH - RAW_WITHOUT,
            "measured_shard_global": measured,
            "measurement_scope": "frozen FULL512 shard windows 64-127",
            "standalone_full512_projection": None,
            "merge_required_for_full512_decision": True,
        }
    else:
        prediction = {
            "solver_raw_without_qtip2": RAW_WITHOUT,
            "solver_raw_with_qtip2": RAW_WITH,
            "solver_raw_delta": RAW_WITH - RAW_WITHOUT,
            "measured_pre_repair_baseline_global": PRE_REPAIR_GLOBAL,
            "predicted_measured_global": PREDICTED_MEASURED_GLOBAL,
            "predicted_fractional_improvement": (PRE_REPAIR_GLOBAL - PREDICTED_MEASURED_GLOBAL) / PRE_REPAIR_GLOBAL,
            "measured_global": measured,
            "measured_minus_predicted": measured - PREDICTED_MEASURED_GLOBAL,
            "prediction_absolute_error": abs(measured - PREDICTED_MEASURED_GLOBAL),
            "prediction_label": "full512-undosed forecast; raw solver objective is not measured KLD",
        }
    stage_rows = overlay["stage_rows"]
    transferred = sum(int(row.get("bytes", 0)) for row in stage_rows)
    transfer_seconds = sum(float(row.get("elapsed_seconds", 0.0)) for row in stage_rows)
    completed = time.time()
    claim_raw_after, _ = p632.current_claim()
    if claim_raw_after != claim_raw_before:
        raise RuntimeError("compute-node-7 claim changed during P651 rail")
    instrument = {
        "p632_exact_scorer_sha256": P632_SCORE_SHA,
        "canonical_reader_sha256": p632.CANONICAL_SHA256[p632.CANONICAL_READER],
        "canonical_builder_sha256": p632.CANONICAL_SHA256[p632.CANONICAL_BUILDER],
        "canonical_loader_sha256": p632.CANONICAL_SHA256[p632.LOADER_SOURCE],
        "adapter_sha256": sha256_file(Path(__file__)),
        "qtip2_resolver_sha256": QTIP_SOURCE_SHA,
        "qtip2_kernel_sha256": QTIP_KERNEL_SHA,
        "qtip2_tlut_tensor_sha256": QTIP_TLUT_TENSOR_SHA,
        "base_wire_manifest_sha256": BASE_WIRE_MANIFEST_SHA,
        "final_assignment_sha256": FINAL_ASSIGNMENT_SHA,
        "final_assignment_map_sha256": FINAL_MAP_SHA,
        "exact_wire_bytes": EXACT_WIRE_BYTES,
        "overlay_manifest_rows": manifest["manifest_rows"],
        "overlay_identity_set_sha256": manifest["identity_set_sha256"],
        "environment_contract": env_contract,
        "attention": "eager", "microbatch": 2, "chunk_size": count,
        "layer_overlap": True, "torch_load_safety": weights_only_stats,
    }
    receipt = {
        "schema": "p651-p640-final-sparse-stream-rail-v1",
        "status": "PASS_VALIDATED_RECEIPT",
        "measurement_label": "PRE_REPAIR_UNDOSED_WIRE / FULL512_SHARD_W064_127 / DECISION_GRADE_CANDIDATE_SHARD" if mode == SLICE_MODE else "PRE_REPAIR_UNDOSED_WIRE",
        "task_id": TASK, "host": "compute-node-8", "source_host": "sealed current-BANANA_SMASHER base local on compute-node-8; exact P653 sparse overlays from compute-node-6 direct QSFP 203.0.113.6",
        "mode": mode, "direction": "KL(teacher||candidate)", "support": 8192, "cutoff": 1024,
        "windows": count, "window_ids": wins,
        "global": reduced["global"], "six_classes": summaries,
        "matched_delta_vs_measured_pre_repair": {"global": global_delta, "six_classes": deltas},
        "prediction_vs_measurement": prediction,
        "instrument": instrument, "instrument_id_sha256": canonical_json_sha256(instrument),
        "baseline_parity_receipt": str(BASELINE_PARITY), "baseline_parity_receipt_sha256": sha256_file(BASELINE_PARITY),
        "pre_repair_baseline_receipt": str(p632.BASELINE_FULL512), "pre_repair_baseline_receipt_sha256": p632.EXPECTED_INPUT_SHA256[p632.BASELINE_FULL512],
        "coverage": {
            "changed_cells_expected": 1411, "changed_cells_applied": len(overlay["applied"]),
            "unchanged_cells_bound_to_immutable_base": manifest["unchanged_cells"],
            "qtip2_cells": manifest["qtip2_cells"], "vq_cells": manifest["vq_cells"],
            "completed_layers": progress_payload["completed_layers"],
            "mmap_completed_layers": progress_payload["mmap_completed_layers"],
            "overlay_stage_rows": stage_rows,
            "overlay_stage_retired": all(bool(row.get("stage_retired", True)) for row in stage_rows),
            "immutable_p640_pack_mutated": False, "persistent_p602_base_mutated": False,
        },
        "throughput": {
            "elapsed_seconds": completed - started,
            "windows_per_second": count / (completed - started),
            "windows_per_minute": count * 60 / (completed - started),
            "overlay_transferred_bytes": transferred,
            "overlay_transfer_seconds_sum": transfer_seconds,
            "overlay_transfer_bytes_per_second": transferred / transfer_seconds if transfer_seconds else None,
        },
        "outputs": {
            "directory": str(out), "window_output_set_sha256": reduced["window_output_set_sha256"],
            "per_window": reduced["per_window"],
        },
        "loader_proof": {
            "progress": str(progress), "progress_sha256": sha256_file(progress),
            "sentinel": str(sentinel), "sentinel_sha256": sha256_file(sentinel),
            "mode": "torch-mmap", "double_buffer": True,
        },
        "claim_sha256": claim_sha, "gpu_snapshot_before": gpu_before,
        "gpu_snapshot_before_child_exit": p632.gpu_snapshot(own_pid=os.getpid(), require_zero_util=False),
        "started_unix": started, "completed_unix": completed,
    }
    atomic_json(receipt_path, receipt, exclusive=True)
    print(json.dumps({
        "status": receipt["status"], "label": receipt["measurement_label"], "mode": mode,
        "global": measured, "delta_vs_pre_repair": global_delta["candidate_minus_pre_repair_mean"],
        "predicted_global": PREDICTED_MEASURED_GLOBAL,
        "receipt": str(receipt_path), "receipt_sha256": sha256_file(receipt_path),
        "elapsed_seconds": completed - started,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
