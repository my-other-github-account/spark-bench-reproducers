#!/usr/bin/env python3
"""Bounded 36-unit QTIP HYB validation for DS4.

Exactly 3 layers x 6 preselected held-out experts x fused13/down are built.
The runner is resume-safe at one-unit granularity and fails closed on claim,
source, population, baseline receipt, byte, and packed-decode mismatches.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import types
from typing import Any

import torch
from safetensors import safe_open

TASK = "PUBLIC_TASK"
MISSION = Path.home() / "missions" / "QTIP_VALIDATE_PUBLIC_TASK_s7"
MANIFEST_PATH = MISSION / "SELECTION_MANIFEST.json"
QTIP = MISSION / "qtip-official"
PARENT_CANDIDATE = (
    Path.home() / "missions" / "QTIP_PILOT_PUBLIC_TASK_s7" / "artifacts"
    / "L017_E005_fused13_QTIP_HYB_L16_K3_V2.pt"
)
MODEL = Path.home() / "models" / "hf" / "DeepSeek-V4-Flash"
FIT_ROOT = Path.home() / "missions" / "LEG_C_TOP50_PUBLIC_TASK_s7" / "capture_missing"
VAL_ROOT = MISSION / "capture_val"
OVERLAY = Path.home() / "missions" / "VQ_GPTQ_OVERLAY_PUBLIC_TASK" / "overlay"
PLANE_ROOT = Path.home() / "missions" / "BINREPAIR_PUBLIC_TASK" / "planes"

OUT = MISSION / "artifacts"
UNITS = OUT / "units"
RESULTS = OUT / "unit_results"
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
SWIGLU_LIMIT = 10.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@lru_cache(maxsize=None)
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=None)
def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def tensor_md5(tensor: torch.Tensor) -> str:
    return hashlib.md5(
        tensor.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def canonical_sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def require_claim() -> tuple[str, dict[str, Any]]:
    path = Path.home() / "HOST_CLAIM.json"
    raw = path.read_bytes()
    state = json.loads(raw)
    if state.get("owner") != TASK or state.get("task_id") != TASK:
        raise RuntimeError(
            f"compute-node-7 claim not owned by {TASK}: "
            f"{state.get('owner')} {state.get('task_id')}"
        )
    if state.get("mission") != str(MISSION):
        raise RuntimeError(f"claim mission mismatch: {state.get('mission')}")
    return hashlib.sha256(raw).hexdigest(), state


def require_host() -> None:
    if socket.gethostname() != "compute-node-7" or not torch.cuda.is_available():
        raise RuntimeError(
            f"wrong host/cuda: {socket.gethostname()} {torch.cuda.is_available()}"
        )


def fwht(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError(f"FWHT requires power-of-two last dimension, got {n}")
    y = x.contiguous()
    width = 1
    while width < n:
        z = y.reshape(*y.shape[:-1], n // (2 * width), 2, width)
        a, b = z[..., 0, :], z[..., 1, :]
        y = torch.cat((a + b, a - b), dim=-1).reshape(*y.shape[:-1], n)
        width *= 2
    return y / math.sqrt(n)


def load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_official_qtip():
    """Load exact pinned source, stubbing only unused compiled-runtime imports."""
    lib = types.ModuleType("lib")
    lib.__path__ = []
    codebook_pkg = types.ModuleType("lib.codebook")
    codebook_pkg.__path__ = []
    codebook_pkg.kdict = {}
    utils_pkg = types.ModuleType("lib.utils")
    utils_pkg.__path__ = []
    utils_pkg.clean = lambda: torch.cuda.empty_cache()
    kc = types.ModuleType("lib.utils.kernel_check")
    kc.has_kernel = lambda *args, **kwargs: False
    kd_stub = types.ModuleType("lib.utils.kernel_decompress")
    kd_stub.decode_compressed = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("runtime decode not used by builder import")
    )
    mh = types.ModuleType("lib.utils.matmul_had")
    mh.matmul_hadU_cuda = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("runtime hadamard not used by builder")
    )
    mh.matmul_hadUt_cuda = mh.matmul_hadU_cuda
    glog = types.ModuleType("glog")
    glog.info = print
    lib.codebook = codebook_pkg
    lib.utils = utils_pkg
    for name, module in {
        "lib": lib,
        "lib.codebook": codebook_pkg,
        "lib.utils": utils_pkg,
        "lib.utils.kernel_check": kc,
        "lib.utils.kernel_decompress": kd_stub,
        "lib.utils.matmul_had": mh,
        "glog": glog,
    }.items():
        sys.modules[name] = module
    bitshift = load_source_module(
        "qtip_validation_bitshift", QTIP / "lib/codebook/bitshift.py"
    )
    ldlq = load_source_module(
        "qtip_validation_ldlq", QTIP / "lib/algo/ldlq.py"
    )
    math_utils = load_source_module(
        "qtip_validation_math_utils", QTIP / "lib/utils/math_utils.py"
    )
    kernel_decode = load_source_module(
        "qtip_validation_kernel_decode", QTIP / "lib/utils/kernel_decompress.py"
    )
    return bitshift, ldlq, math_utils, kernel_decode


def verify_pins(manifest: dict[str, Any]) -> dict[str, Any]:
    pins = manifest["pins"]
    parent = Path(pins["parent_mission"])
    checks = {
        "parent_final_json": {
            "path": str(parent / "artifacts/FULL_PACKAGE_QTIP_L17.json"),
            "expected": pins["parent_final_json_sha256"],
        },
        "parent_candidate": {
            "path": str(parent / "artifacts/L017_E005_fused13_QTIP_HYB_L16_K3_V2.pt"),
            "expected": pins["parent_candidate_sha256"],
        },
        "parent_evidence": {
            "path": str(parent / "artifacts/QTIP_PILOT_EVIDENCE.json"),
            "expected": pins["parent_evidence_sha256"],
        },
    }
    for row in checks.values():
        path = Path(row["path"])
        row["actual"] = sha256(path)
        if row["actual"] != row["expected"]:
            raise RuntimeError(f"parent pin mismatch: {path}")
    parent_commit = subprocess.check_output(
        ["git", "-C", str(parent), "rev-parse", "HEAD"], text=True
    ).strip()
    if parent_commit != pins["parent_mission_git_commit"]:
        raise RuntimeError(f"parent git commit mismatch: {parent_commit}")
    qtip_sources = {
        "bitshift": "lib/codebook/bitshift.py",
        "ldlq": "lib/algo/ldlq.py",
        "math_utils": "lib/utils/math_utils.py",
        "kernel_decompress": "lib/utils/kernel_decompress.py",
    }
    checks["parent_git_commit"] = parent_commit
    checks["qtip_declared_upstream_commit"] = pins["qtip_commit"]
    checks["source_files"] = {}
    for name, relative in qtip_sources.items():
        local_path = QTIP / relative
        parent_path = parent / "qtip-official" / relative
        local_sha = sha256(local_path)
        parent_sha = sha256(parent_path)
        if local_sha != parent_sha:
            raise RuntimeError(
                f"QTIP source differs from sealed parent mission: {relative}"
            )
        checks["source_files"][name] = {
            "path": str(local_path),
            "sha256": local_sha,
            "sealed_parent_sha256": parent_sha,
        }
    return checks


class ModelReader:
    def __init__(self) -> None:
        self.index_path = MODEL / "model.safetensors.index.json"
        self.mapping = json.loads(self.index_path.read_text())["weight_map"]
        self.lut = torch.tensor(E2M1, dtype=torch.float32)

    def projection(self, layer: int, expert: int, projection: str) -> tuple[torch.Tensor, dict[str, Any]]:
        names = ("w1", "w3") if projection == "fused13" else ("w2",)
        matrices = []
        source = []
        for name in names:
            weight_key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
            scale_key = f"layers.{layer}.ffn.experts.{expert}.{name}.scale"
            shard = MODEL / self.mapping[weight_key]
            if self.mapping[scale_key] != self.mapping[weight_key]:
                raise RuntimeError(f"weight/scale split: {weight_key}")
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                packed = handle.get_tensor(weight_key).view(torch.uint8)
                scales = handle.get_tensor(scale_key).view(torch.uint8)
            nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
            matrix = self.lut[nibbles.long()] * torch.exp2(
                scales.float() - 127.0
            ).repeat_interleave(32, dim=1)
            matrices.append(matrix.contiguous())
            source.append({
                "weight_key": weight_key,
                "scale_key": scale_key,
                "shard": str(shard),
                "shard_sha256": sha256(shard),
            })
        result = torch.cat(matrices, dim=0) if len(matrices) == 2 else matrices[0]
        expected = (4096, 4096) if projection == "fused13" else (4096, 2048)
        if tuple(result.shape) != expected:
            raise RuntimeError(f"source shape mismatch {layer}/{expert}/{projection}: {result.shape}")
        return result.contiguous(), {
            "index_path": str(self.index_path),
            "index_sha256": sha256(self.index_path),
            "shards": source,
        }


def scale_columns(scale_bytes: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale_bytes.float() - 127.0).repeat_interleave(32, dim=1)


def decode_vq(codes: torch.Tensor, scales: torch.Tensor, cb: torch.Tensor) -> torch.Tensor:
    return (
        cb.float()[codes.long()].reshape(codes.shape[0], -1)
        * scale_columns(scales)
    ).contiguous()


def swiglu(inputs: torch.Tensor, fused13: torch.Tensor) -> torch.Tensor:
    gate = (inputs @ fused13[:2048].T).clamp(max=SWIGLU_LIMIT)
    up = (inputs @ fused13[2048:].T).clamp(
        min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT
    )
    return torch.nn.functional.silu(gate) * up


def current_ledger_row(layer: int, expert: int, projection: str) -> dict[str, Any]:
    projection_id = "13" if projection == "fused13" else "2"
    path = OVERLAY / "unit_ledgers" / f"LAYER_{layer:03d}_UNITS.jsonl"
    hits = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (
            int(row.get("layer", -1)) == layer
            and int(row.get("expert", -1)) == expert
            and str(row.get("projection")) == projection_id
            and str(row.get("tier")) == "d4_k4096"
        ):
            hits.append(row)
    if len(hits) != 1:
        raise RuntimeError(
            f"expected one current ledger row L{layer} E{expert} {projection}: {len(hits)}"
        )
    return hits[0]


def load_current(
    layer: int, expert: int, projection: str
) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
    projection_id = "13" if projection == "fused13" else "2"
    plane_path = PLANE_ROOT / f"vq3u_layer_{layer:03d}.pt"
    overlay_path = OVERLAY / "d4_k4096" / f"layer_{layer:03d}.pt"
    plane = torch.load(plane_path, map_location="cpu", mmap=True, weights_only=True)
    overlay = torch.load(overlay_path, map_location="cpu", mmap=True, weights_only=True)
    ids = [int(x) for x in overlay[f"expert_ids{projection_id}"].tolist()]
    if ids.count(expert) != 1:
        raise RuntimeError(
            f"refined overlay missing/duplicate L{layer} E{expert} {projection}: {ids.count(expert)}"
        )
    oi = ids.index(expert)
    refined_codes = overlay[f"codes{projection_id}"][oi].contiguous()
    source_codes = plane[f"codes{projection_id}"][expert].contiguous()
    scales = plane[f"sc{projection_id}"][expert].contiguous()
    cb = plane[f"cb{projection_id}"].contiguous()
    baseline = decode_vq(refined_codes, scales, cb)
    target_fused13 = None
    down_input_plane_path = None
    down_input_plane_sha256 = None
    if projection == "down":
        target_fused13 = decode_vq(
            plane["codes13"][expert].contiguous(),
            plane["sc13"][expert].contiguous(),
            plane["cb13"].contiguous(),
        )
        down_input_plane_path = str(plane_path)
        down_input_plane_sha256 = sha256(plane_path)
    ledger = current_ledger_row(layer, expert, projection)
    actual_hashes = {
        "source_codebook_md5": tensor_md5(cb),
        "source_scale_md5": tensor_md5(scales),
        "source_codes_md5": tensor_md5(source_codes),
        "refined_codes_md5": tensor_md5(refined_codes),
    }
    required_hashes = (
        "source_codebook_md5",
        "source_scale_md5",
        "refined_codes_md5",
    ) + (("source_codes_md5",) if projection == "fused13" else ())
    for key in required_hashes:
        actual = actual_hashes[key]
        if ledger.get(key) != actual:
            raise RuntimeError(
                f"current receipt hash mismatch L{layer} E{expert} {projection} {key}: "
                f"{ledger.get(key)} != {actual}"
            )
    source_codes_provenance = {
        "expected_md5": ledger.get("source_codes_md5"),
        "actual_md5": actual_hashes["source_codes_md5"],
        "match": ledger.get("source_codes_md5") == actual_hashes["source_codes_md5"],
        "required_gate": projection == "fused13",
        "consumed_by_scored_baseline": False,
        "waiver": (
            None if projection == "fused13" else
            "down source_codes2 are unused: baseline consumes refined overlay codes2; "
            "down inputs consume this plane's independently pinned fused13 codes/scales/codebook"
        ),
    }
    meta = overlay.get("meta", {})
    if not all(meta.get(key) is True for key in (
        "same_codebooks", "same_scales", "same_wire_format"
    )):
        raise RuntimeError(f"overlay invariants missing L{layer}: {meta}")
    pins = {
        "source_plane_path": str(plane_path),
        "source_plane_sha256": sha256(plane_path),
        "down_input_plane_path": down_input_plane_path,
        "down_input_plane_sha256": down_input_plane_sha256,
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256(overlay_path),
        "overlay_meta": meta,
        "ledger": ledger,
        **actual_hashes,
        "hash_gate": {
            "required_keys": list(required_hashes),
            "all_required_match": True,
            "source_codes_provenance": source_codes_provenance,
        },
        "logical_code_bytes": refined_codes.numel() * 12 // 8,
        "scale_bytes": scales.numel(),
        "layer_shared_codebook_bytes": cb.numel() * cb.element_size(),
        "layer_shared_codebook_amortized_256_experts_bytes": math.ceil(
            cb.numel() * cb.element_size() / 256
        ),
    }
    return baseline, target_fused13, pins


def capture_path(root: Path, layer: int, window: int) -> Path:
    return root / f"xmoe_L{layer:03d}_win{window:04d}.pt"


def load_layer_captures(
    layer: int, manifest: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], torch.Tensor]:
    corpus_md5 = manifest["population"]["corpus_md5"]
    splits = {
        "fit": (FIT_ROOT, range(0, 128)),
        "heldout": (VAL_ROOT, range(128, 152)),
    }
    loaded: dict[str, list[dict[str, Any]]] = {}
    receipt_seals: dict[str, Any] = {}
    route_mass = torch.zeros(256, dtype=torch.float64)
    for split, (root, windows) in splits.items():
        entries = []
        receipt_rows = []
        for window in windows:
            path = capture_path(root, layer, window)
            done_path = Path(str(path) + ".DONE.json")
            receipt = json.loads(done_path.read_text())
            actual_md5 = md5(path)
            if receipt.get("md5") != actual_md5:
                raise RuntimeError(f"capture MD5 mismatch: {path}")
            if receipt.get("corpus_md5") != corpus_md5:
                raise RuntimeError(f"capture receipt corpus mismatch: {path}")
            data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            if (
                int(data["layer"]) != layer
                or int(data["win"]) != window
                or str(data["corpus_md5"]) != corpus_md5
            ):
                raise RuntimeError(f"capture tensor identity mismatch: {path}")
            x = data["x"].to(torch.bfloat16).contiguous()
            topk = data["topk"].to(torch.int64).contiguous()
            route = data["w"].float().contiguous()
            entries.append({
                "window": window,
                "x": x,
                "topk": topk,
                "route": route,
            })
            receipt_rows.append({
                "file": path.name,
                "md5": actual_md5,
                "bytes": path.stat().st_size,
                "real_len": int(data["RL"]),
                "source_builder_md5": receipt.get("source_builder_md5"),
            })
            if split == "fit":
                for column in range(topk.shape[1]):
                    route_mass.scatter_add_(0, topk[:, column], route[:, column].double())
        loaded[split] = entries
        receipt_seals[split] = {
            "root": str(root),
            "layer": layer,
            "windows": [windows.start, windows.stop - 1],
            "count": len(receipt_rows),
            "corpus_md5": corpus_md5,
            "receipt_list_sha256": canonical_sha(receipt_rows),
            "receipts": receipt_rows,
        }
    return loaded, receipt_seals, route_mass


def audit_selection(
    layer: int,
    selected: list[int],
    route_mass: torch.Tensor,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    overlay_path = OVERLAY / "d4_k4096" / f"layer_{layer:03d}.pt"
    overlay = torch.load(overlay_path, map_location="cpu", mmap=True, weights_only=True)
    ids13 = set(int(x) for x in overlay["expert_ids13"].tolist())
    ids2 = set(int(x) for x in overlay["expert_ids2"].tolist())
    excluded = set(manifest["selection"]["source_codebook_fit_experts"])
    excluded.add(int(manifest["selection"]["pilot_expert_excluded"]))
    eligible = sorted((ids13 & ids2) - excluded)
    ranked = sorted(eligible, key=lambda expert: (float(route_mass[expert]), expert))
    seed = manifest["selection"]["seed_material"]
    bins = []
    picks = []
    for bin_id in range(6):
        lo = bin_id * len(ranked) // 6
        hi = (bin_id + 1) * len(ranked) // 6
        bucket = ranked[lo:hi]
        if not bucket:
            raise RuntimeError(f"empty selection bin L{layer} B{bin_id}")
        def selection_hash(expert: int) -> str:
            return hashlib.sha256(
                f"{seed}|L{layer}|B{bin_id}|E{expert}".encode()
            ).hexdigest()
        pick = min(bucket, key=selection_hash)
        picks.append(pick)
        bins.append({
            "bin": bin_id,
            "eligible_experts": bucket,
            "mass_min": float(route_mass[bucket[0]]),
            "mass_max": float(route_mass[bucket[-1]]),
            "selected": pick,
            "selected_mass": float(route_mass[pick]),
            "selection_hash": selection_hash(pick),
        })
    if picks != selected:
        raise RuntimeError(f"selection mismatch L{layer}: recomputed={picks} manifest={selected}")
    if excluded & set(selected):
        raise RuntimeError(f"held-out expert exclusion violated L{layer}: {excluded & set(selected)}")
    return {
        "layer": layer,
        "overlay_path": str(overlay_path),
        "overlay_sha256": sha256(overlay_path),
        "eligible_intersection_count": len(eligible),
        "eligible_experts": eligible,
        "excluded_experts": sorted(excluded),
        "bins": bins,
        "selected": picks,
        "status": "PASS",
    }


def expert_windows(
    entries: list[dict[str, Any]], expert: int
) -> list[dict[str, Any]]:
    result = []
    for entry in entries:
        hit = entry["topk"].eq(expert)
        weight = (entry["route"] * hit).sum(dim=1)
        keep = weight > 0
        result.append({
            "window": entry["window"],
            "x": entry["x"][keep].contiguous(),
            "weight": weight[keep].contiguous(),
        })
    return result


def down_windows(
    windows: list[dict[str, Any]], target_fused13: torch.Tensor, device: torch.device
) -> list[dict[str, Any]]:
    fused = target_fused13.to(device=device, dtype=torch.float32)
    result = []
    with torch.no_grad():
        for row in windows:
            pieces = []
            x = row["x"]
            for start in range(0, len(x), 128):
                xb = x[start:start + 128].to(device=device, dtype=torch.float32)
                pieces.append(swiglu(xb, fused).cpu())
            act = torch.cat(pieces) if pieces else torch.empty((0, 2048), dtype=torch.float32)
            result.append({
                "window": row["window"],
                "x": act.contiguous(),
                "weight": row["weight"],
            })
    del fused
    torch.cuda.empty_cache()
    return result


def build_hessian(
    windows: list[dict[str, Any]], signs: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, int, float]:
    width = signs.numel()
    hessian = torch.zeros((width, width), dtype=torch.float32, device=device)
    rows = 0
    mass = 0.0
    for row in windows:
        x = row["x"]
        weights = row["weight"]
        rows += len(x)
        mass += float(weights.double().sum())
        for start in range(0, len(x), 256):
            xb = x[start:start + 256].to(device=device, dtype=torch.float32)
            wb = weights[start:start + 256].to(device=device, dtype=torch.float32)
            z = fwht(xb * signs)
            hessian.addmm_(z.T, z * wb[:, None])
    if rows <= 0 or mass <= 0:
        raise RuntimeError(f"empty routed fit population rows={rows} mass={mass}")
    return hessian / mass, rows, mass


def pack_kernel_layout(
    cb, states: torch.Tensor, m: int, k: int
) -> tuple[torch.Tensor, dict[str, Any]]:
    tiled = states.reshape(m // 16, 16, k // 16, 16 // 2).transpose(1, 2).reshape(-1, 16 * 16 // 2)
    packed = cb.pack_trellis(tiled).contiguous()
    expected_shape = ((m // 16) * (k // 16), 48)
    if tuple(packed.shape) != expected_shape or packed.dtype != torch.uint16:
        raise RuntimeError(
            f"canonical packed shape/dtype mismatch {m}x{k}: "
            f"{tuple(packed.shape)} {packed.dtype} != {expected_shape} uint16"
        )
    unpacked = cb.unpack_trellis(packed, 256)
    roundtrip = unpacked.to(tiled.dtype).eq(tiled)
    # The prefix-compressed accelerator can choose a different floating-point
    # tie path than the sealed full-state kernel.  Canonical packing is the
    # wire authority; retain the measured roundtrip fraction and reconstruct
    # from the packed bytes below rather than rejecting a sane candidate.
    kernel = (
        packed.view(torch.uint8)
        .view(-1, 2)
        .flip((-1,))
        .reshape(m // 32, 2, k // 32, 2, 32, 3)
        .permute(0, 2, 4, 3, 1, 5)
        .flip((-1,))
        .contiguous()
        .flatten()
        .view(torch.int16)
        .reshape(packed.shape)
    )
    expected = 3 * m * k // 8
    actual = kernel.numel() * kernel.element_size()
    if actual != expected:
        raise RuntimeError(f"packed byte mismatch {m}x{k}: {actual} != {expected}")
    receipt = {
        "tile_states_shape": list(tiled.shape),
        "canonical_packed_shape": list(packed.shape),
        "canonical_packed_dtype": str(packed.dtype),
        "canonical_packed_sha256": tensor_sha256(packed),
        "canonical_unpack_state_sha256": tensor_sha256(unpacked),
        "input_state_sha256": tensor_sha256(tiled),
        "canonical_pack_roundtrip_fraction": float(roundtrip.float().mean()),
        "canonical_pack_roundtrip_exact": bool(roundtrip.all()),
        "kernel_swizzle": "reshape(m//32,2,k//32,2,32,K).permute(0,2,4,3,1,5)",
        "kernel_packed_shape": list(kernel.shape),
        "kernel_packed_sha256": tensor_sha256(kernel),
        "kernel_packed_bytes": actual,
    }
    return kernel, receipt


def decode_packed(
    candidate: dict[str, Any], kernel_decode, device: torch.device
) -> tuple[torch.Tensor, dict[str, Any]]:
    geometry = candidate["geometry"]
    m, k = [int(x) for x in candidate["shape"]]
    tlut = candidate["tlut"].float().to(device)
    index = torch.arange(1 << 16, device=device)
    quadratic = (index + 1) * index
    sign_flip = 1 - ((quadratic >> 15) & 1) * 2
    lut_index = (quadratic >> (16 - 9 - 1)) & ((1 << 9) - 1)
    expanded = tlut[lut_index]
    expanded[:, 0] *= sign_flip
    packed = candidate["trellis"].to(device)
    raw = kernel_decode.decode_compressed(
        16, 9, 3, 1, m, k, packed.reshape(-1), expanded
    )
    q = raw * candidate["Wscale"].to(device)
    q = fwht(q.T).T * candidate["SV"].float().to(device)[:, None]
    q = fwht(q) * candidate["SU"].float().to(device)
    stored = candidate["reconstructed_weight"]
    decoded_fp16 = q.half().cpu()
    equal = decoded_fp16.view(torch.int16).eq(stored.view(torch.int16))
    receipt = {
        "path": "pinned QTIP Python/CUDA tensor decompressor from kernel_decompress.py",
        "source_sha256": sha256(QTIP / "lib/utils/kernel_decompress.py"),
        "shape": [m, k],
        "geometry": geometry,
        "fp16_bit_equal_fraction": float(equal.float().mean()),
        "fp16_bit_exact": bool(equal.all()),
        "max_abs_fp32_vs_stored_fp16": float(
            (q - stored.to(device).float()).abs().max()
        ),
        "decoded_fp16_sha256": tensor_sha256(decoded_fp16),
        "stored_fp16_sha256": tensor_sha256(stored),
        "packed_sha256": tensor_sha256(candidate["trellis"]),
    }
    return q, receipt


def build_qtip(
    source_weight: torch.Tensor,
    fit_windows: list[dict[str, Any]],
    cb,
    ldlq,
    math_utils,
    kernel_decode,
    device: torch.device,
    rht_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    m, k = source_weight.shape
    torch.manual_seed(rht_seed)
    su = (torch.randn(k, device=device).sign() + 1e-5).sign().float()
    sv = (torch.randn(m, device=device).sign() + 1e-5).sign().float()
    hessian, fit_rows, fit_mass = build_hessian(fit_windows, su, device)
    hessian = math_utils.regularize_H(hessian, 1e-2)
    weight = source_weight.to(device=device, dtype=torch.float32)
    transformed = fwht(fwht(weight.T * sv).T * su)
    wscale = transformed.square().mean().sqrt() / (
        cb.lut.double().square().mean().sqrt().float() * 0.9
    )
    transformed = transformed / wscale
    lower, _ = math_utils.block_LDL(hessian, 16)
    diagonal = torch.arange(lower.shape[0], device=device)
    lower[diagonal, diagonal] = 0
    args = types.SimpleNamespace(td_x=16, td_y=16, V=2)
    started = time.time()
    quantized, states = ldlq.LDLQ(
        transformed, lower, cb, args, buf_cols=128, for_kernel=True
    )
    quant_seconds = time.time() - started
    packed, pack_conformance = pack_kernel_layout(cb, states, m, k)
    # Reconstruct from the canonical packed wire, not the pre-pack state
    # tensor.  This keeps the stored reconstruction exactly aligned with the
    # bytes consumed by the production decompressor even when pack/unpack
    # canonicalizes a tiny fraction of tie-sensitive states.
    index = torch.arange(1 << 16, device=device)
    quadratic = (index + 1) * index
    sign_flip = 1 - ((quadratic >> 15) & 1) * 2
    lut_index = (quadratic >> (16 - 9 - 1)) & ((1 << 9) - 1)
    expanded = cb.tlut.float().to(device)[lut_index]
    expanded[:, 0] *= sign_flip
    quantized = kernel_decode.decode_compressed(
        16, 9, 3, 1, m, k, packed.reshape(-1), expanded
    ) * wscale
    reconstructed = fwht(quantized.T).T * sv[:, None]
    reconstructed = fwht(reconstructed) * su
    reconstructed_fp16 = reconstructed.half().cpu()
    candidate = {
        "schema": "ds4-qtip-hyb-bounded36-unit-v1",
        "shape": [m, k],
        "trellis": packed.cpu(),
        "SU": su.half().cpu(),
        "SV": sv.half().cpu(),
        "Wscale": wscale.cpu(),
        "tlut": cb.tlut.cpu(),
        "reconstructed_weight": reconstructed_fp16,
        "geometry": {
            "L": 16, "K": 3, "V": 2, "tlut_bits": 9,
            "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16,
        },
    }
    del hessian, lower, transformed, quantized, states, reconstructed, weight
    torch.cuda.empty_cache()
    decoded, conformance = decode_packed(candidate, kernel_decode, device)
    if not conformance["fp16_bit_exact"]:
        raise RuntimeError(f"packed decode conformance failed {m}x{k}: {conformance}")
    del decoded
    torch.cuda.empty_cache()
    return candidate, {
        "rht_seed": rht_seed,
        "quant_seconds": quant_seconds,
        "fit_rows": fit_rows,
        "fit_route_mass": fit_mass,
        "canonical_pack": pack_conformance,
        "packed_decode": conformance,
    }


def split_metrics(
    windows: list[dict[str, Any]],
    source_weight: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    source = source_weight.to(device=device, dtype=torch.float32)
    base = baseline.to(device=device, dtype=torch.float32)
    cand = candidate.to(device=device, dtype=torch.float32)
    totals = defaultdict(float)
    rows = 0
    per_window = []
    with torch.no_grad():
        for row in windows:
            values = {
                "mass": 0.0,
                "ref": 0.0,
                "base": 0.0,
                "qtip": 0.0,
                "ref_unweighted": 0.0,
                "base_unweighted": 0.0,
                "qtip_unweighted": 0.0,
            }
            x = row["x"]
            route = row["weight"]
            rows += len(x)
            values["mass"] = float(route.double().sum())
            for start in range(0, len(x), 64):
                xb = x[start:start + 64].to(device=device, dtype=torch.float32)
                wb = route[start:start + 64].to(device=device, dtype=torch.float32)
                reference = xb @ source.T
                err_base = reference - xb @ base.T
                err_qtip = reference - xb @ cand.T
                values["ref"] += float((reference.square() * wb[:, None]).sum().double())
                values["base"] += float((err_base.square() * wb[:, None]).sum().double())
                values["qtip"] += float((err_qtip.square() * wb[:, None]).sum().double())
                values["ref_unweighted"] += float(reference.square().sum().double())
                values["base_unweighted"] += float(err_base.square().sum().double())
                values["qtip_unweighted"] += float(err_qtip.square().sum().double())
            per_window.append({
                "window": int(row["window"]),
                "rows": len(x),
                **dict(values),
            })
            for key, value in values.items():
                totals[key] += value
    del source, base, cand
    torch.cuda.empty_cache()
    if totals["base"] <= 0 or totals["ref"] <= 0 or totals["mass"] <= 0:
        raise RuntimeError(f"invalid metric totals: {dict(totals)}")
    out_dim = source_weight.shape[0]
    return {
        "rows": rows,
        "route_mass": totals["mass"],
        "windows": per_window,
        "true_vq_current": {
            "weighted_output_sse": totals["base"],
            "activation_mse": totals["base"] / (totals["mass"] * out_dim),
            "relative_output_rms": math.sqrt(totals["base"] / totals["ref"]),
            "unweighted_proxy_rms": math.sqrt(
                totals["base_unweighted"] / totals["ref_unweighted"]
            ),
        },
        "qtip_hyb": {
            "weighted_output_sse": totals["qtip"],
            "activation_mse": totals["qtip"] / (totals["mass"] * out_dim),
            "relative_output_rms": math.sqrt(totals["qtip"] / totals["ref"]),
            "unweighted_proxy_rms": math.sqrt(
                totals["qtip_unweighted"] / totals["ref_unweighted"]
            ),
            "sse_ratio_vs_true_vq": totals["qtip"] / totals["base"],
            "sse_improvement_fraction": 1.0 - totals["qtip"] / totals["base"],
        },
    }


def bootstrap_windows(
    windows: list[dict[str, Any]], samples: int = 20000, seed: int = 20260723
) -> dict[str, Any]:
    base = torch.tensor([row["base"] for row in windows], dtype=torch.float64)
    qtip = torch.tensor([row["qtip"] for row in windows], dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    values = []
    made = 0
    while made < samples:
        count = min(1000, samples - made)
        index = torch.randint(0, len(base), (count, len(base)), generator=generator)
        denominator = base[index].sum(dim=1)
        if bool((denominator <= 0).any()):
            raise RuntimeError("zero bootstrap denominator")
        values.append(1.0 - qtip[index].sum(dim=1) / denominator)
        made += count
    vector = torch.cat(values)
    lo, hi = torch.quantile(
        vector, torch.tensor([0.025, 0.975], dtype=torch.float64)
    ).tolist()
    return {
        "method": "window bootstrap with replacement, ratio of aggregate weighted SSE",
        "samples": samples,
        "seed": seed,
        "improvement_fraction_ci95": [lo, hi],
        "median": float(vector.median()),
    }


def storage_receipt(candidate: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    weights = math.prod(candidate["shape"])
    packed = candidate["trellis"].numel() * candidate["trellis"].element_size()
    signs = (candidate["SU"].numel() + candidate["SV"].numel()) * 2
    tlut = candidate["tlut"].numel() * candidate["tlut"].element_size()
    qtip_total = packed + signs + tlut + 4
    current_total = (
        current["logical_code_bytes"]
        + current["scale_bytes"]
        + current["layer_shared_codebook_amortized_256_experts_bytes"]
    )
    receipt = {
        "num_weights": weights,
        "qtip_packed_trellis_bytes": packed,
        "qtip_su_sv_fp16_bytes": signs,
        "qtip_tlut_bytes": tlut,
        "qtip_matrix_scale_bytes": 4,
        "qtip_total_logical_bytes": qtip_total,
        "qtip_effective_bpw": qtip_total * 8 / weights,
        "true_vq_logical_code_bytes": current["logical_code_bytes"],
        "true_vq_scale_bytes": current["scale_bytes"],
        "true_vq_layer_shared_codebook_amortized_256_experts_bytes": current[
            "layer_shared_codebook_amortized_256_experts_bytes"
        ],
        "true_vq_total_logical_bytes": current_total,
        "true_vq_effective_bpw": current_total * 8 / weights,
        "delta_qtip_minus_true_vq_bytes": qtip_total - current_total,
        "within_current_true_vq_budget": qtip_total <= current_total,
    }
    if not receipt["within_current_true_vq_budget"]:
        raise RuntimeError(f"QTIP exceeds true-VQ byte budget: {receipt}")
    return receipt


def unit_name(layer: int, expert: int, projection: str) -> str:
    return f"L{layer:03d}_E{expert:03d}_{projection}"


def verified_resume_result(
    result_path: Path,
    build_path: Path,
    artifact_path: Path,
    manifest_sha: str,
    kernel_decode,
    device: torch.device,
) -> dict[str, Any] | None:
    if not result_path.exists() or not build_path.exists() or not artifact_path.exists():
        return None
    result = json.loads(result_path.read_text())
    build = json.loads(build_path.read_text())
    artifact_sha = sha256(artifact_path)
    if (
        result.get("status") != "PASS"
        or result.get("manifest_sha256") != manifest_sha
        or result.get("artifact", {}).get("sha256") != artifact_sha
        or result.get("build_seal", {}).get("sha256") != sha256(build_path)
        or build.get("status") != "PASS"
        or build.get("manifest_sha256") != manifest_sha
        or build.get("artifact", {}).get("sha256") != artifact_sha
    ):
        return None
    candidate = torch.load(
        artifact_path, map_location="cpu", mmap=True, weights_only=True
    )
    if candidate.get("manifest_sha256") != manifest_sha:
        return None
    decoded, conformance = decode_packed(candidate, kernel_decode, device)
    del decoded, candidate
    torch.cuda.empty_cache()
    expected = build["build"]["packed_decode"]
    if (
        not conformance["fp16_bit_exact"]
        or conformance["decoded_fp16_sha256"] != expected["decoded_fp16_sha256"]
        or conformance["packed_sha256"] != expected["packed_sha256"]
    ):
        return None
    return result


def run_unit(
    *,
    layer: int,
    expert: int,
    projection: str,
    captures: dict[str, list[dict[str, Any]]],
    receipt_seals: dict[str, Any],
    model: ModelReader,
    cb,
    ldlq,
    math_utils,
    kernel_decode,
    manifest: dict[str, Any],
    manifest_sha: str,
    claim_sha: str,
    device: torch.device,
) -> dict[str, Any]:
    name = unit_name(layer, expert, projection)
    unit_dir = UNITS / name
    artifact_path = unit_dir / "candidate.pt"
    build_path = unit_dir / "BUILD.json"
    result_path = unit_dir / "RESULT.json"
    resumed = verified_resume_result(
        result_path, build_path, artifact_path, manifest_sha, kernel_decode, device
    )
    if resumed is not None:
        log(f"RESUME {name} verified")
        return resumed
    log(f"START {name}")
    raw_fit = expert_windows(captures["fit"], expert)
    raw_val = expert_windows(captures["heldout"], expert)
    baseline, target_fused13, current = load_current(layer, expert, projection)
    exact_fit_md5s = [row["md5"] for row in receipt_seals["fit"]["receipts"]]
    exact_val_md5s = [row["md5"] for row in receipt_seals["heldout"]["receipts"]]
    if exact_fit_md5s != current["overlay_meta"].get("fit_capture_md5s"):
        raise RuntimeError(f"exact fit population differs from current receipt: {name}")
    if exact_val_md5s != current["overlay_meta"].get("val_capture_md5s"):
        raise RuntimeError(f"exact heldout population differs from current receipt: {name}")
    if projection == "down":
        if target_fused13 is None:
            raise AssertionError("down target fused13 absent")
        fit_windows = down_windows(raw_fit, target_fused13, device)
        val_windows = down_windows(raw_val, target_fused13, device)
    else:
        fit_windows = raw_fit
        val_windows = raw_val
    source_weight, weight_source = model.projection(layer, expert, projection)
    rht_seed = int(manifest["qtip_package"]["rht_seed_map"][name])
    candidate, build = build_qtip(
        source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode, device,
        rht_seed
    )
    candidate.update({
        "task": TASK,
        "identity": {"layer": layer, "expert": expert, "projection": projection},
        "manifest_sha256": manifest_sha,
    })
    storage = storage_receipt(candidate, current)
    heldout = split_metrics(
        val_windows, source_weight, baseline,
        candidate["reconstructed_weight"].float(), device
    )
    fit = split_metrics(
        fit_windows, source_weight, baseline,
        candidate["reconstructed_weight"].float(), device
    )
    heldout["bootstrap_ci"] = bootstrap_windows(
        heldout["windows"],
        samples=int(manifest["metrics"]["bootstrap"]["samples"]),
        seed=int(manifest["metrics"]["bootstrap"]["seed"]),
    )
    ledger = current["ledger"]
    tolerance = float(manifest["current_true_vq"]["receipt_reproduction_abs_tolerance"])
    reproduction = {
        "expected_fit_proxy": ledger["proxy_fit"]["vq_gptq"],
        "measured_fit_proxy": fit["true_vq_current"]["unweighted_proxy_rms"],
        "expected_val_proxy": ledger["proxy_val"]["vq_gptq"],
        "measured_val_proxy": heldout["true_vq_current"]["unweighted_proxy_rms"],
    }
    reproduction["fit_abs_delta"] = abs(
        reproduction["expected_fit_proxy"] - reproduction["measured_fit_proxy"]
    )
    reproduction["val_abs_delta"] = abs(
        reproduction["expected_val_proxy"] - reproduction["measured_val_proxy"]
    )
    reproduction["tolerance"] = tolerance
    reproduction["pass"] = (
        reproduction["fit_abs_delta"] <= tolerance
        and reproduction["val_abs_delta"] <= tolerance
    )
    if not reproduction["pass"]:
        raise RuntimeError(f"current receipt reproduction failed {name}: {reproduction}")
    atomic_torch(artifact_path, candidate)
    artifact = {
        "path": str(artifact_path),
        "sha256": sha256(artifact_path),
        "bytes": artifact_path.stat().st_size,
    }
    build_seal = {
        "schema": "qtip-bounded36-unit-build-v1",
        "status": "PASS",
        "task": TASK,
        "identity": {"layer": layer, "expert": expert, "projection": projection},
        "manifest_sha256": manifest_sha,
        "rht_seed": rht_seed,
        "artifact": artifact,
        "source": {
            "qtip_commit": manifest["pins"]["qtip_commit"],
            "bitshift_sha256": sha256(QTIP / "lib/codebook/bitshift.py"),
            "ldlq_sha256": sha256(QTIP / "lib/algo/ldlq.py"),
            "math_utils_sha256": sha256(QTIP / "lib/utils/math_utils.py"),
            "runner_sha256": sha256(Path(__file__)),
            "weight": weight_source,
        },
        "inputs": {
            "population_fit_receipt_list_sha256": receipt_seals["fit"][
                "receipt_list_sha256"
            ],
            "population_heldout_receipt_list_sha256": receipt_seals["heldout"][
                "receipt_list_sha256"
            ],
            "current_source_plane_sha256": current["source_plane_sha256"],
            "down_input_plane_sha256": current["down_input_plane_sha256"],
            "current_overlay_sha256": current["overlay_sha256"],
            "refined_codes_md5": current["refined_codes_md5"],
            "current_hash_gate": current["hash_gate"],
        },
        "build": build,
        "storage": storage,
        "created_unix": time.time(),
    }
    atomic_json(build_path, build_seal)
    build_seal_ref = {
        "path": str(build_path),
        "sha256": sha256(build_path),
        "bytes": build_path.stat().st_size,
    }
    result = {
        "schema": "qtip-bounded36-unit-result-v1",
        "status": "PASS",
        "task": TASK,
        "identity": {"layer": layer, "expert": expert, "projection": projection},
        "shape": candidate["shape"],
        "manifest_sha256": manifest_sha,
        "claim_sha256_at_run": claim_sha,
        "population_receipts": receipt_seals,
        "current_true_vq": current,
        "current_receipt_reproduction": reproduction,
        "qtip_package": manifest["qtip_package"],
        "source": {
            "qtip_commit": manifest["pins"]["qtip_commit"],
            "bitshift_sha256": sha256(QTIP / "lib/codebook/bitshift.py"),
            "ldlq_sha256": sha256(QTIP / "lib/algo/ldlq.py"),
            "math_utils_sha256": sha256(QTIP / "lib/utils/math_utils.py"),
            "runner_sha256": sha256(Path(__file__)),
            "weight": weight_source,
        },
        "build": build,
        "storage": storage,
        "metrics": {
            "heldout_primary": heldout,
            "calibration_single_robustness_check": fit,
        },
        "artifact": artifact,
        "build_seal": build_seal_ref,
        "created_unix": time.time(),
    }
    atomic_json(result_path, result)
    log(
        f"PASS {name} improvement={heldout['qtip_hyb']['sse_improvement_fraction']:.6%} "
        f"CI={heldout['bootstrap_ci']['improvement_fraction_ci95']}"
    )
    del source_weight, baseline, target_fused13, candidate
    del raw_fit, raw_val, fit_windows, val_windows
    gc.collect()
    torch.cuda.empty_cache()
    return result


def aggregate_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    by_window: dict[int, dict[str, float]] = defaultdict(lambda: {"base": 0.0, "qtip": 0.0})
    base = 0.0
    qtip = 0.0
    route_mass = 0.0
    unit_improvements = []
    for row in rows:
        primary = row["metrics"]["heldout_primary"]
        b = float(primary["true_vq_current"]["weighted_output_sse"])
        q = float(primary["qtip_hyb"]["weighted_output_sse"])
        base += b
        qtip += q
        route_mass += float(primary["route_mass"])
        unit_improvements.append(1.0 - q / b)
        for window in primary["windows"]:
            slot = by_window[int(window["window"])]
            slot["base"] += float(window["base"])
            slot["qtip"] += float(window["qtip"])
    windows = [
        {"window": window, **by_window[window]}
        for window in sorted(by_window)
    ]
    if [row["window"] for row in windows] != list(range(128, 152)):
        raise RuntimeError(f"aggregate window mismatch: {[row['window'] for row in windows]}")
    return {
        "unit_count": len(rows),
        "route_mass_sum_across_units": route_mass,
        "true_vq_weighted_output_sse": base,
        "qtip_weighted_output_sse": qtip,
        "sse_ratio_vs_true_vq": qtip / base,
        "sse_improvement_fraction": 1.0 - qtip / base,
        "positive_unit_count": sum(value > 0 for value in unit_improvements),
        "unit_improvement_min": min(unit_improvements),
        "unit_improvement_max": max(unit_improvements),
        "unit_improvement_median": float(torch.tensor(unit_improvements).median()),
        "bootstrap_ci": bootstrap_windows(windows, 20000, seed),
        "windows": windows,
    }


def summarize(results: list[dict[str, Any]], manifest: dict[str, Any], manifest_sha: str) -> dict[str, Any]:
    expected = {
        unit_name(layer, expert, projection)
        for layer in manifest["scope"]["layers"]
        for expert in manifest["scope"]["experts_by_layer"][str(layer)]
        for projection in manifest["scope"]["projections"]
    }
    actual = {
        unit_name(
            int(row["identity"]["layer"]),
            int(row["identity"]["expert"]),
            str(row["identity"]["projection"]),
        )
        for row in results
    }
    if len(results) != 36 or actual != expected:
        raise RuntimeError(
            f"unit identity closure failed count={len(results)} missing={expected-actual} extra={actual-expected}"
        )
    seed = int(manifest["metrics"]["bootstrap"]["seed"])
    pooled = aggregate_rows(results, seed)
    groups: dict[str, Any] = {"by_projection": {}, "by_layer": {}, "by_layer_projection": {}, "by_expert": {}}
    for projection in manifest["scope"]["projections"]:
        subset = [row for row in results if row["identity"]["projection"] == projection]
        groups["by_projection"][projection] = aggregate_rows(subset, seed)
    for layer in manifest["scope"]["layers"]:
        subset = [row for row in results if int(row["identity"]["layer"]) == layer]
        groups["by_layer"][str(layer)] = aggregate_rows(subset, seed)
        for projection in manifest["scope"]["projections"]:
            subset_lp = [
                row for row in subset if row["identity"]["projection"] == projection
            ]
            groups["by_layer_projection"][f"L{layer:03d}_{projection}"] = aggregate_rows(
                subset_lp, seed
            )
    for layer in manifest["scope"]["layers"]:
        for expert in manifest["scope"]["experts_by_layer"][str(layer)]:
            subset = [
                row for row in results
                if int(row["identity"]["layer"]) == layer
                and int(row["identity"]["expert"]) == expert
            ]
            groups["by_expert"][f"L{layer:03d}_E{expert:03d}"] = aggregate_rows(subset, seed)
    all_lp_ci_positive = all(
        row["bootstrap_ci"]["improvement_fraction_ci95"][0] > 0
        for row in groups["by_layer_projection"].values()
    )
    threshold = float(manifest["classification"]["parent_step_change_threshold_fraction"])
    generalized = (
        pooled["sse_improvement_fraction"] >= threshold
        and pooled["bootstrap_ci"]["improvement_fraction_ci95"][0] > 0
    )
    if generalized:
        verdict = "STEP_CHANGE_GENERALIZES"
        subclass = "UNIFORM" if all_lp_ci_positive else "HETEROGENEOUS"
    else:
        verdict = "NOT_GENERALIZED"
        subclass = "HETEROGENEOUS" if not all_lp_ci_positive else "UNIFORM_BUT_SUBTHRESHOLD"
    unit_table = []
    for row in sorted(
        results,
        key=lambda item: (
            int(item["identity"]["layer"]),
            int(item["identity"]["expert"]),
            str(item["identity"]["projection"]),
        ),
    ):
        primary = row["metrics"]["heldout_primary"]
        unit_table.append({
            **row["identity"],
            "shape": row["shape"],
            "heldout_rows": primary["rows"],
            "heldout_route_mass": primary["route_mass"],
            "true_vq_sse": primary["true_vq_current"]["weighted_output_sse"],
            "qtip_sse": primary["qtip_hyb"]["weighted_output_sse"],
            "improvement_fraction": primary["qtip_hyb"]["sse_improvement_fraction"],
            "bootstrap_ci95": primary["bootstrap_ci"]["improvement_fraction_ci95"],
            "fit_improvement_fraction": row["metrics"]["calibration_single_robustness_check"]["qtip_hyb"]["sse_improvement_fraction"],
            "qtip_logical_bytes": row["storage"]["qtip_total_logical_bytes"],
            "true_vq_logical_bytes": row["storage"]["true_vq_total_logical_bytes"],
            "packed_decode_fp16_bit_exact": row["build"]["packed_decode"]["fp16_bit_exact"],
            "receipt_fit_abs_delta": row["current_receipt_reproduction"]["fit_abs_delta"],
            "receipt_val_abs_delta": row["current_receipt_reproduction"]["val_abs_delta"],
            "artifact_sha256": row["artifact"]["sha256"],
        })
    return {
        "schema": "qtip-bounded36-final-v1",
        "status": "PASS",
        "task": TASK,
        "manifest_sha256": manifest_sha,
        "unit_count": len(results),
        "identities": sorted(actual),
        "primary_metric": "expert-route-weighted projection-output SSE on exact 24 heldout windows",
        "single_robustness_check": "same metric on exact 128 fit windows",
        "pooled": pooled,
        "aggregates": groups,
        "classification": {
            "verdict": verdict,
            "subclass": subclass,
            "parent_step_change_threshold_fraction": threshold,
            "all_six_layer_projection_ci95_lower_positive": all_lp_ci_positive,
            "positive_unit_count": pooled["positive_unit_count"],
            "criteria": manifest["classification"],
        },
        "gates": {
            "exact_36_identity_closure": True,
            "all_current_receipts_reproduced": all(
                row["current_receipt_reproduction"]["pass"] for row in results
            ),
            "all_within_true_vq_bytes": all(
                row["storage"]["within_current_true_vq_budget"] for row in results
            ),
            "all_packed_decode_fp16_bit_exact": all(
                row["build"]["packed_decode"]["fp16_bit_exact"] for row in results
            ),
            "no_repair_training": True,
            "no_broad_sweep": True,
        },
        "unit_table": unit_table,
        "runtime_next_gate": {
            "smallest_gate": "Add and register only decompress_matvec<16,9,3,1,4096,1,2048>, build the existing 4096x4096 symbol plus the new 4096x2048 down symbol for SM121/CUDA 13, then run one sealed artifact GEMV per shape against fp32-TLUT Python dequant with relL2 <= 1e-3.",
            "scope_boundary": "This is only two loaded-artifact decompressor GEMVs; GGUF tensor traits, routed-expert dispatch, and serving remain later integration work.",
            "serve_claim": False,
        },
        "created_unix": time.time(),
    }


def write_report(final: dict[str, Any]) -> None:
    lines = [
        "# BOUNDED_36_QTIP_VALIDATION",
        "",
        f"Verdict: **{final['classification']['verdict']} / {final['classification']['subclass']}**",
        "",
        "Primary metric: expert-route-weighted projection-output SSE on exact held-out windows 128–151. The only robustness check is the same metric on fit windows 0–127.",
        "",
        "## Pooled held-out result",
        "",
        f"- units: {final['unit_count']}",
        f"- true-VQ SSE: {final['pooled']['true_vq_weighted_output_sse']:.9f}",
        f"- QTIP SSE: {final['pooled']['qtip_weighted_output_sse']:.9f}",
        f"- improvement: {100*final['pooled']['sse_improvement_fraction']:.6f}%",
        f"- 20k shared-window bootstrap 95% CI: {100*final['pooled']['bootstrap_ci']['improvement_fraction_ci95'][0]:.6f}% .. {100*final['pooled']['bootstrap_ci']['improvement_fraction_ci95'][1]:.6f}%",
        f"- positive units: {final['pooled']['positive_unit_count']}/36",
        "",
        "## Layer × projection aggregates",
        "",
        "| group | units | improvement | CI95 | positive units |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, row in final["aggregates"]["by_layer_projection"].items():
        lo, hi = row["bootstrap_ci"]["improvement_fraction_ci95"]
        lines.append(
            f"| {key} | {row['unit_count']} | {100*row['sse_improvement_fraction']:.6f}% | "
            f"{100*lo:.6f}%..{100*hi:.6f}% | {row['positive_unit_count']} |"
        )
    lines += [
        "",
        "## Per-unit table",
        "",
        "| layer | expert | projection | heldout rows | improvement | CI95 | QTIP bytes | true-VQ bytes | decode |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in final["unit_table"]:
        lo, hi = row["bootstrap_ci95"]
        lines.append(
            f"| {row['layer']} | {row['expert']} | {row['projection']} | {row['heldout_rows']} | "
            f"{100*row['improvement_fraction']:.6f}% | {100*lo:.6f}%..{100*hi:.6f}% | "
            f"{row['qtip_logical_bytes']} | {row['true_vq_logical_bytes']} | "
            f"{'PASS' if row['packed_decode_fp16_bit_exact'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Runtime next gate",
        "",
        final["runtime_next_gate"]["smallest_gate"],
        "",
        final["runtime_next_gate"]["scope_boundary"],
        "",
        "No model-wide build, repair training, GGUF integration, or serve pass is claimed.",
    ]
    path = OUT / "BOUNDED_36_QTIP_VALIDATION.md"
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    require_host()
    claim_sha, claim = require_claim()
    manifest = json.loads(MANIFEST_PATH.read_text())
    manifest_sha = sha256(MANIFEST_PATH)
    if manifest.get("task") != TASK or manifest["scope"]["unit_count"] != 36:
        raise RuntimeError("manifest task/unit mismatch")
    expected_seed_keys = {
        unit_name(layer, expert, projection)
        for layer in manifest["scope"]["layers"]
        for expert in manifest["scope"]["experts_by_layer"][str(layer)]
        for projection in manifest["scope"]["projections"]
    }
    actual_seed_keys = set(manifest["qtip_package"]["rht_seed_map"])
    if actual_seed_keys != expected_seed_keys:
        raise RuntimeError(
            f"RHT seed map identity mismatch: missing={expected_seed_keys-actual_seed_keys} "
            f"extra={actual_seed_keys-expected_seed_keys}"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    UNITS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pins = verify_pins(manifest)
    git_commit = subprocess.check_output(
        ["git", "-C", str(MISSION), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(OUT / "START_RECEIPT.json", {
        "schema": "qtip-bounded36-start-v1",
        "task": TASK,
        "host": socket.gethostname(),
        "claim_sha256": claim_sha,
        "claim_nonce": claim["claim_nonce"],
        "manifest_sha256": manifest_sha,
        "mission_git_commit": git_commit,
        "pins": pins,
        "device": torch.cuda.get_device_name(0),
        "created_unix": time.time(),
    })
    if args.preflight_only:
        rows = []
        for layer in manifest["scope"]["layers"]:
            for expert in manifest["scope"]["experts_by_layer"][str(layer)]:
                for projection in manifest["scope"]["projections"]:
                    _, _, current = load_current(layer, expert, projection)
                    rows.append({
                        "identity": {
                            "layer": layer,
                            "expert": expert,
                            "projection": projection,
                        },
                        "source_plane_path": current["source_plane_path"],
                        "source_plane_sha256": current["source_plane_sha256"],
                        "overlay_sha256": current["overlay_sha256"],
                        "hash_gate": current["hash_gate"],
                    })
        if len(rows) != 36:
            raise RuntimeError(f"hash preflight unit closure failed: {len(rows)}")
        receipt = {
            "schema": "qtip-bounded36-source-hash-preflight-v1",
            "status": "PASS",
            "task": TASK,
            "manifest_sha256": manifest_sha,
            "unit_count": len(rows),
            "strict_required_hash_units": sum(
                row["hash_gate"]["all_required_match"] for row in rows
            ),
            "down_source_codes_unused_waivers": sum(
                row["identity"]["projection"] == "down"
                and not row["hash_gate"]["source_codes_provenance"]["match"]
                for row in rows
            ),
            "rows": rows,
            "created_unix": time.time(),
        }
        atomic_json(OUT / "SOURCE_HASH_PREFLIGHT.json", receipt)
        print(json.dumps({
            "status": receipt["status"],
            "unit_count": receipt["unit_count"],
            "strict_required_hash_units": receipt["strict_required_hash_units"],
            "down_source_codes_unused_waivers": receipt[
                "down_source_codes_unused_waivers"
            ],
        }, indent=2, sort_keys=True))
        return 0
    if args.summary_only:
        results = [
            json.loads(path.read_text())
            for path in sorted(UNITS.glob("L*/RESULT.json"))
        ]
        final = summarize(results, manifest, manifest_sha)
        atomic_json(OUT / "BOUNDED_36_QTIP_VALIDATION.json", final)
        write_report(final)
        return 0
    device = torch.device("cuda")
    bitshift, ldlq, math_utils, kernel_decode = load_official_qtip()
    parent_candidate = torch.load(
        PARENT_CANDIDATE, map_location="cpu", mmap=True, weights_only=True
    )
    pinned_tlut = parent_candidate["tlut"].float().contiguous()
    expected_tlut_sha = manifest["qtip_package"]["tlut_tensor_sha256"]
    actual_tlut_sha = tensor_sha256(pinned_tlut)
    if actual_tlut_sha != expected_tlut_sha:
        raise RuntimeError(
            f"pinned parent TLUT mismatch: {actual_tlut_sha} != {expected_tlut_sha}"
        )
    cb = bitshift.bitshift_codebook(
        L=16, K=3, V=2, tlut_bits=9, decode_mode="quantlut_sym",
        tlut=pinned_tlut.to(device),
    ).to(device)
    model = ModelReader()
    all_results = []
    selection_audits = []
    population_audits = []
    completed = 0
    for layer in manifest["scope"]["layers"]:
        captures, receipt_seals, route_mass = load_layer_captures(layer, manifest)
        selected = [int(x) for x in manifest["scope"]["experts_by_layer"][str(layer)]]
        selection_audits.append(
            audit_selection(layer, selected, route_mass, manifest)
        )
        population_audits.append({"layer": layer, **receipt_seals})
        atomic_json(OUT / "SELECTION_AUDIT.json", {
            "schema": "qtip-bounded36-selection-audit-v1",
            "status": "PASS",
            "manifest_sha256": manifest_sha,
            "layers": selection_audits,
        })
        atomic_json(OUT / "POPULATION_RECEIPTS.json", {
            "schema": "qtip-bounded36-population-receipts-v1",
            "status": "PASS",
            "manifest_sha256": manifest_sha,
            "layers": population_audits,
        })
        for expert in selected:
            for projection in manifest["scope"]["projections"]:
                result = run_unit(
                    layer=layer,
                    expert=expert,
                    projection=projection,
                    captures=captures,
                    receipt_seals=receipt_seals,
                    model=model,
                    cb=cb,
                    ldlq=ldlq,
                    math_utils=math_utils,
                    kernel_decode=kernel_decode,
                    manifest=manifest,
                    manifest_sha=manifest_sha,
                    claim_sha=claim_sha,
                    device=device,
                )
                all_results.append(result)
                completed += 1
                atomic_json(MISSION / "STATUS.json", {
                    "status": "RUNNING",
                    "stage": "unit_validation",
                    "completed_units": completed,
                    "total_units": 36,
                    "last_unit": result["identity"],
                    "epoch": time.time(),
                })
        del captures, route_mass
        gc.collect()
        torch.cuda.empty_cache()
    final = summarize(all_results, manifest, manifest_sha)
    atomic_json(OUT / "BOUNDED_36_QTIP_VALIDATION.json", final)
    write_report(final)
    atomic_json(OUT / "DONE.json", {
        "schema": "qtip-bounded36-done-v1",
        "status": "PASS",
        "task": TASK,
        "manifest_sha256": manifest_sha,
        "mission_git_commit": git_commit,
        "final_json_sha256": sha256(OUT / "BOUNDED_36_QTIP_VALIDATION.json"),
        "report_sha256": sha256(OUT / "BOUNDED_36_QTIP_VALIDATION.md"),
        "unit_count": 36,
        "verdict": final["classification"]["verdict"],
        "created_unix": time.time(),
    })
    atomic_json(MISSION / "STATUS.json", {
        "status": "PASS",
        "stage": "complete",
        "completed_units": 36,
        "total_units": 36,
        "verdict": final["classification"]["verdict"],
        "epoch": time.time(),
    })
    (MISSION / "DONE").write_text(sha256(OUT / "DONE.json") + "\n")
    print(json.dumps(final["classification"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
