#!/usr/bin/env python3
"""Exact frozen-code per-cell prices for the eight sealed measured VQ tiers.

Pricing law mirrors compose: plane codes/scales are frozen; repaired variants only
substitute the exact deployed codebook. H is the 32-window GPTQ capture-bank
second moment. Down-projection H is built from teacher fused13 activations.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any
import uuid

import torch
import torch.nn.functional as F
from safetensors import safe_open

DEVICE = "cuda"

_E2M1_VALUES = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)
_W2_VALUES = torch.tensor(
    [1.0] * 5 + [4.0] * 3 + [-1.0] * 5 + [-4.0] * 3,
    dtype=torch.float32,
)
_BYTE_LUT: dict[tuple[str, torch.device], torch.Tensor] = {}


def deq_fp4_block32(
    packed: torch.Tensor,
    scales: torch.Tensor,
    kind: str,
) -> torch.Tensor:
    """Dequantize packed low-nibble-first FP4 blocks without private imports."""
    if kind not in {"e2m1", "w2"}:
        raise ValueError(f"unsupported FP4 kind: {kind}")
    key = (kind, packed.device)
    if key not in _BYTE_LUT:
        values = _E2M1_VALUES if kind == "e2m1" else _W2_VALUES
        byte = torch.arange(256)
        _BYTE_LUT[key] = torch.stack(
            [values[byte & 0xF], values[byte >> 4]], dim=-1
        ).to(packed.device)
    values = _BYTE_LUT[key][packed.long()].flatten(-2)
    scale = torch.exp2(scales.view(torch.uint8).float() - 127.0).repeat_interleave(
        32, dim=-1
    )
    return (values * scale).to(torch.bfloat16)

TIERS = (
    "d4_k1024", "d4_k2048", "d4_k4096", "d4_k8192",
    "d8_k256", "d8_k1024", "d8_k4096", "vqa",
)
REPAIRED = {"d4_k1024", "d8_k4096", "vqa"}
SOLVER_INPUTS_SCHEMA = "banana-smasher-solver-inputs-v1"
STAGED_INPUT_SCHEMA = "banana-smasher-staged-input-v1"


def _solver_inputs(root: Path) -> dict[str, Any]:
    path = root.resolve() / "SOLVER_INPUTS.json"
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"explicit solver input manifest is required: {path}"
        ) from exc
    if value.get("schema") != SOLVER_INPUTS_SCHEMA or value.get("status") != "PASS":
        raise ValueError(f"invalid solver input manifest: {path}")
    return value


def _declared_layer_source(
    root: Path,
    section: str,
    layer: int,
    *,
    tier: str | None = None,
) -> str:
    value: Any = _solver_inputs(root).get(section)
    if tier is not None:
        if not isinstance(value, dict):
            raise ValueError(f"solver input manifest lacks {section} mapping")
        value = value.get(tier)
    if isinstance(value, dict):
        value = value.get(f"L{layer:03d}")
    if not isinstance(value, str) or not value:
        label = f"{section}.{tier}" if tier is not None else section
        raise ValueError(f"solver input manifest lacks {label} for L{layer:03d}")
    return value.format(L=layer)


def tensor_md5(t: torch.Tensor) -> str:
    return hashlib.md5(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a staged input without materializing it in Python memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def staged_input_receipt_path(path: Path) -> Path:
    return path.with_name(path.name + ".STAGED.json")


def _fsync_parent(path: Path) -> None:
    dfd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _fsync_parent(path)


def seal_staged_input(path: Path, source: str, *, min_size: int) -> dict[str, Any]:
    """Hash-bind an immutable staged file and publish its completion atomically."""
    stat = path.stat()
    if not path.is_file() or stat.st_size < min_size:
        raise RuntimeError(f"staged input is missing/truncated: {path} ({stat.st_size})")
    digest = file_sha256(path)
    path.chmod(0o444)
    stat = path.stat()
    receipt = {
        "schema": STAGED_INPUT_SCHEMA,
        "status": "COMPLETE",
        "source": source,
        "destination": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode & 0o777,
        "sha256": digest,
        "completed_epoch": time.time(),
    }
    _atomic_json(staged_input_receipt_path(path), receipt)
    return receipt


def validate_staged_input(path: Path, source: str, *, min_size: int) -> dict[str, Any]:
    """Fail closed unless data, source identity, hash, and completion all match."""
    receipt_path = staged_input_receipt_path(path)
    try:
        receipt = json.loads(receipt_path.read_text())
        stat = path.stat()
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"staged input has no valid completion receipt: {path}") from exc
    expected = {
        "schema": STAGED_INPUT_SCHEMA,
        "status": "COMPLETE",
        "source": source,
        "destination": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "mode": stat.st_mode & 0o777,
    }
    mismatches = {
        key: (receipt.get(key), value)
        for key, value in expected.items()
        if receipt.get(key) != value
    }
    if stat.st_size < min_size:
        mismatches["min_size"] = (stat.st_size, min_size)
    if stat.st_mode & 0o222:
        mismatches["writable_mode"] = (stat.st_mode & 0o777, 0o444)
    digest = receipt.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        mismatches["sha256"] = (digest, "64 lowercase hex characters")
    else:
        actual_digest = file_sha256(path)
        if actual_digest != digest:
            mismatches["sha256"] = (actual_digest, digest)
    if mismatches:
        raise RuntimeError(f"staged input validation failed for {path}: {mismatches}")
    return receipt


def ensure_staged_remote(
    destination: Path,
    source: str,
    *,
    min_size: int,
) -> Path:
    """Reuse only validated inputs; serialize and atomically publish first fill."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = staged_input_receipt_path(destination)
    lock_path = destination.with_name(destination.name + ".STAGED.lock")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            # Recheck under the cross-process lock. A receipt is authoritative
            # only after source identity, metadata, and bytes all validate.
            if destination.is_file() and receipt_path.is_file():
                validate_staged_input(destination, source, min_size=min_size)
                return destination
            if destination.exists():
                raise RuntimeError(
                    f"staged input exists without a completion receipt: {destination}"
                )
            if receipt_path.exists():
                raise RuntimeError(
                    f"staged input completion receipt exists without data: {destination}"
                )

            # First fill starts only from a clean absent state. Fetch into a
            # process-unique partial and publish the completion receipt last.
            partial = destination.with_name(
                destination.name + f".partial.{uuid.uuid4().hex}"
            )
            try:
                subprocess.run([
                    "rsync", "-a", "--partial", "--timeout=120", "--bwlimit=200000",
                    "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10",
                    source,
                    str(partial),
                ], check=True)
                stat = partial.stat()
                if not partial.is_file() or stat.st_size < min_size:
                    raise RuntimeError(
                        f"staged source is missing/truncated: {partial} ({stat.st_size})"
                    )
                with partial.open("rb") as handle:
                    os.fsync(handle.fileno())
                partial.chmod(0o444)
                os.replace(partial, destination)
                _fsync_parent(destination)
                seal_staged_input(destination, source, min_size=min_size)
                validate_staged_input(destination, source, min_size=min_size)
                return destination
            finally:
                if partial.exists():
                    partial.chmod(0o600)
                    partial.unlink()
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def checkpoint_index(model_root: Path) -> dict[str, str]:
    return json.loads((model_root / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]


def weight_shard_map(index: dict[str, str], layer: int) -> dict[str, str]:
    base = f"layers.{layer}.ffn.experts."
    out: dict[str, str] = {}
    for expert in range(256):
        for proj in ("w1", "w3", "w2"):
            for suffix in ("weight", "scale"):
                key = f"{base}{expert}.{proj}.{suffix}"
                out[key] = index[key]
    return out


def open_shards(
    root: Path,
    layer: int,
    layer_map: dict[str, str],
    *,
    model_root: Path,
    staging_root: Path | None = None,
) -> dict[str, Any]:
    """Open the routed-expert shard from a reusable resident staging root.

    ``root`` continues to own solver outputs. Only immutable plane/weight input
    bytes may live under ``staging_root``; no objective or assignment state is
    loaded from it.
    """
    scratch = (staging_root or root) / "weight_shards" / f"L{layer:03d}"
    scratch.mkdir(parents=True, exist_ok=True)
    handles = {}
    for name in sorted(set(layer_map.values())):
        dst = scratch / name
        source = str((model_root / name).resolve())
        ensure_staged_remote(dst, source, min_size=1_000_000_000)
        handles[name] = safe_open(str(dst), framework="pt", device="cpu")
    return handles


def get_tensor(handles: dict[str, Any], layer_map: dict[str, str], key: str) -> torch.Tensor:
    return handles[layer_map[key]].get_tensor(key)


def load_weights(handles: dict[str, Any], layer_map: dict[str, str], layer: int, expert: int) -> tuple[torch.Tensor, torch.Tensor]:
    prefix = f"layers.{layer}.ffn.experts.{expert}"
    def deq(proj: str) -> torch.Tensor:
        w = get_tensor(handles, layer_map, f"{prefix}.{proj}.weight").view(torch.uint8).to(DEVICE)
        s = get_tensor(handles, layer_map, f"{prefix}.{proj}.scale").view(torch.uint8).to(DEVICE)
        return deq_fp4_block32(w, s, "e2m1").to(torch.bfloat16)
    fused = torch.cat((deq("w1"), deq("w3")), dim=0)
    down = deq("w2")
    assert fused.shape == (4096, 4096) and down.shape == (4096, 2048)
    return fused, down


def load_weights_batch(
    handles: dict[str, Any],
    layer_map: dict[str, str],
    layer: int,
    experts: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bulk materialize an expert slab with one H2D/dequant launch per projection."""
    def deq(proj: str) -> torch.Tensor:
        weights = []
        scales = []
        for expert in experts:
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            weights.append(
                get_tensor(handles, layer_map, f"{prefix}.{proj}.weight").view(torch.uint8)
            )
            scales.append(
                get_tensor(handles, layer_map, f"{prefix}.{proj}.scale").view(torch.uint8)
            )
        packed = torch.stack(weights, dim=0).to(DEVICE)
        packed_scales = torch.stack(scales, dim=0).to(DEVICE)
        return deq_fp4_block32(packed, packed_scales, "e2m1").to(torch.bfloat16)

    w1 = deq("w1")
    w3 = deq("w3")
    fused = torch.cat((w1, w3), dim=1)
    down = deq("w2")
    assert fused.shape == (len(experts), 4096, 4096)
    assert down.shape == (len(experts), 4096, 2048)
    return fused, down


def stage_weights_batch_cpu(
    handles: dict[str, Any],
    layer_map: dict[str, str],
    layer: int,
    experts: list[int],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Bulk-pack one expert batch into contiguous resident host buffers."""
    staged: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for proj in ("w1", "w3", "w2"):
        weights = []
        scales = []
        for expert in experts:
            prefix = f"layers.{layer}.ffn.experts.{expert}"
            weights.append(
                get_tensor(handles, layer_map, f"{prefix}.{proj}.weight").view(torch.uint8)
            )
            scales.append(
                get_tensor(handles, layer_map, f"{prefix}.{proj}.scale").view(torch.uint8)
            )
        # Grace/GB10 uses coherent unified memory; a second page-locking copy is
        # slower than stacking directly into the resident host slab. The custom
        # CUDA stream still overlaps H2D/dequant with current-batch scoring.
        packed = torch.stack(weights, dim=0).contiguous()
        packed_scales = torch.stack(scales, dim=0).contiguous()
        staged[proj] = packed, packed_scales
    return staged


def materialize_staged_weights(
    staged: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Asynchronously H2D/dequant a bulk CPU slab on the active CUDA stream."""
    def deq(proj: str) -> torch.Tensor:
        packed, packed_scales = staged[proj]
        packed = packed.to(DEVICE, non_blocking=True)
        packed_scales = packed_scales.to(DEVICE, non_blocking=True)
        return deq_fp4_block32(packed, packed_scales, "e2m1").to(torch.bfloat16)

    w1 = deq("w1")
    w3 = deq("w3")
    fused = torch.cat((w1, w3), dim=1)
    down = deq("w2")
    experts = staged["w1"][0].shape[0]
    assert fused.shape == (experts, 4096, 4096)
    assert down.shape == (experts, 4096, 2048)
    return fused, down


def prefetch_staged_weights(
    staged: dict[str, tuple[torch.Tensor, torch.Tensor]],
    stream: torch.cuda.Stream,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.cuda.Event,
    torch.cuda.Event,
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    float,
]:
    """Queue one resident batch and retain timing plus host buffers to completion."""
    queued_mono = time.perf_counter()
    with torch.cuda.stream(stream):
        started = torch.cuda.Event(enable_timing=True)
        started.record(stream)
        fused, down = materialize_staged_weights(staged)
        ready = torch.cuda.Event(enable_timing=True)
        ready.record(stream)
    return fused, down, started, ready, staged, queued_mono


def wait_prefetched_weights(
    prefetched: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.cuda.Event,
        torch.cuda.Event,
        dict[str, tuple[torch.Tensor, torch.Tensor]],
        float,
    ],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Wait for the tail and report complete async H2D/dequant device work."""
    fused, down, started, ready, _host_lifetime, queued_mono = prefetched
    ready.synchronize()
    completed_mono = time.perf_counter()
    return fused, down, {
        "device_seconds": max(0.0, float(started.elapsed_time(ready)) / 1000.0),
        "queued_mono": queued_mono,
        "completed_mono": completed_mono,
        "queue_to_complete_wall_s": max(0.0, completed_mono - queued_mono),
    }


def _capture_source(root: Path, layer: int) -> str:
    return _declared_layer_source(root, "captures", layer)


def _capture_bank_complete(path: Path, layer: int, nwin: int) -> bool:
    return all(
        (path / f"xmoe_L{layer:03d}_win{win:04d}.pt").is_file()
        and (path / f"xmoe_L{layer:03d}_win{win:04d}.pt.DONE.json").is_file()
        for win in range(nwin)
    )


def capture_dir(
    root: Path,
    layer: int,
    nwin: int,
    *,
    staging_root: Path | None = None,
) -> Path:
    candidates = [
        # Canonical task-local backfills are preferred when present (held-out
        # pilot layers and rare-expert 64-window expansions).
        root / "captures/pilot",
        root / "captures/staged_from_s6/VQ_GPTQ_SHARD_t_f90571d5/capture_train",
        root / "captures/staged_from_s6/VQ_GPTQ_FULLBIN_t_07dd5170/capture_train",
    ]
    for path in candidates:
        if _capture_bank_complete(path, layer, nwin):
            return path

    source_root = _capture_source(root, layer)
    source_name = Path(source_root.rstrip("/")).name
    destination_root = (
        (staging_root or root) / "capture_scratch" / source_name
    )
    for win in range(nwin):
        filename = f"xmoe_L{layer:03d}_win{win:04d}.pt"
        ensure_staged_remote(
            destination_root / filename,
            f"{source_root.rstrip('/')}/{filename}",
            min_size=1,
        )
        ensure_staged_remote(
            destination_root / f"{filename}.DONE.json",
            f"{source_root.rstrip('/')}/{filename}.DONE.json",
            min_size=1,
        )
    if not _capture_bank_complete(destination_root, layer, nwin):
        raise RuntimeError(f"capture staging incomplete for L{layer:03d}")
    return destination_root


def load_captures(
    root: Path,
    layer: int,
    nwin: int,
    *,
    staging_root: Path | None = None,
    capture_root: Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    if capture_root is None:
        cdir = capture_dir(root, layer, nwin, staging_root=staging_root)
    else:
        cdir = capture_root.resolve()
        if not _capture_bank_complete(cdir, layer, nwin):
            raise RuntimeError(
                f"manifest-bound capture bank incomplete for L{layer:03d}: {cdir}"
            )
    out = []
    expected_builder: str | None = None
    expected_corpus: str | None = None
    for win in range(nwin):
        p = cdir / f"xmoe_L{layer:03d}_win{win:04d}.pt"
        d = Path(str(p) + ".DONE.json")
        receipt = json.loads(d.read_text())
        builder = receipt.get("source_builder_md5")
        if not isinstance(builder, str) or not builder:
            raise RuntimeError(f"capture builder identity missing: {d}")
        if expected_builder is None:
            expected_builder = builder
        elif builder != expected_builder:
            raise RuntimeError(f"capture builder mismatch: {d}")
        obj = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        corpus = obj.get("corpus_md5")
        if not isinstance(corpus, str) or not corpus:
            raise RuntimeError(f"capture corpus identity missing: {p}")
        if expected_corpus is None:
            expected_corpus = corpus
        elif corpus != expected_corpus:
            raise RuntimeError(f"capture corpus mismatch: {p}")
        out.append({"x": obj["x"], "topk": obj["topk"], "w": obj["w"]})
    return out


def hdiag_fused(captures: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    h = torch.zeros((256, 4096), dtype=torch.float64)
    counts = torch.zeros(256, dtype=torch.int64)
    for obj in captures:
        x2 = obj["x"].float().square().to(torch.float64)
        topk = obj["topk"].long()
        one = torch.ones(topk.shape[0], dtype=torch.int64)
        for slot in range(6):
            ids = topk[:, slot]
            h.index_add_(0, ids, x2)
            counts.index_add_(0, ids, one)
    if (counts == 0).any():
        raise RuntimeError(f"unrouted experts: {(counts == 0).nonzero().flatten().tolist()}")
    return h.float(), counts


def expert_x(captures: list[dict[str, torch.Tensor]], expert: int) -> torch.Tensor:
    pieces = []
    for obj in captures:
        mask = (obj["topk"] == expert).any(dim=1)
        if mask.any():
            pieces.append(obj["x"][mask].to(torch.bfloat16))
    if not pieces:
        raise RuntimeError(f"expert {expert} has zero rows")
    return torch.cat(pieces, dim=0)


def down_hdiag(x_cpu: torch.Tensor, fused: torch.Tensor, batch: int = 128) -> torch.Tensor:
    gate, up = fused.chunk(2, dim=0)
    h = torch.zeros(2048, device=DEVICE, dtype=torch.float32)
    for start in range(0, x_cpu.shape[0], batch):
        x = x_cpu[start:start + batch].to(DEVICE)
        z = F.silu(x @ gate.T) * (x @ up.T)
        h.add_(z.float().square().sum(dim=0))
    return h


def stage_declared_plane(
    root: Path,
    tier: str,
    layer: int,
    *,
    staging_root: Path | None = None,
) -> Path:
    scratch = (staging_root or root) / "planes_scratch" / f"L{layer:03d}"
    scratch.mkdir(parents=True, exist_ok=True)
    destination = scratch / f"{tier}.pt"
    source = _declared_layer_source(root, "planes", layer, tier=tier)
    return ensure_staged_remote(destination, source, min_size=1)


def plane_paths(
    root: Path,
    layer: int,
    *,
    selected_tiers: tuple[str, ...],
    staging_root: Path | None = None,
) -> dict[str, Path]:
    # Stage only immutable planes needed by the selected fixed tiers. Exact d4
    # search uses d4/k4096 only as the byte-identical scale geometry source.
    needed = set(selected_tiers)
    if needed.intersection({"d4_k2048", "d4_k4096"}):
        needed.add("d4_k4096")
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            tier: pool.submit(
                stage_declared_plane,
                root,
                tier,
                layer,
                staging_root=staging_root,
            )
            for tier in sorted(needed - {"d4_k2048"})
        }
        paths = {tier: future.result() for tier, future in futures.items()}
    if "d4_k2048" in needed:
        paths["d4_k2048"] = paths["d4_k4096"]
    return paths


def catalog_path(root: Path, variant: str, tier: str, layer: int) -> Path:
    bases = [root / "codebooks/core", root / "codebooks/k8192", root / "codebooks/vqa"]
    for base in bases:
        p = base / variant / tier / f"layer_{layer:03d}.pt"
        if p.is_file():
            return p
    raise FileNotFoundError((variant, tier, layer))


def stage_codebooks(
    root: Path,
    source_root: Path,
    tiers: tuple[str, ...],
    layer: int,
) -> None:
    """Materialize only selected fixed-tier catalogs into the task-owned root."""
    for tier in tiers:
        for variant in variants(tier):
            source_variant = "repaired" if variant == "deployed" else "base"
            source = catalog_path(source_root, source_variant, tier, layer)
            relative = source.resolve().relative_to(source_root.resolve())
            destination = root / relative
            ensure_staged_remote(destination, str(source), min_size=1)


def dequant_row(data: dict[str, Any], cb: torch.Tensor, expert: int, projection: str) -> torch.Tensor:
    if data.get("meta", {}).get("tier") == "vqa" or "vqa_codes13" in data:
        codes = data[f"vqa_codes{projection}"][expert].to(DEVICE)
        scales = data[f"vqa_sc{projection}"][expert].to(DEVICE)
    else:
        codes = data[f"codes{projection}"][expert].to(DEVICE)
        scales = data[f"sc{projection}"][expert].to(DEVICE)
    columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    q = cb[codes.long()].reshape(codes.shape[0], -1) * columns
    return q.to(torch.bfloat16)


def frozen_weighted_errors_batched(
    w: torch.Tensor,
    h: torch.Tensor,
    plane_data: dict[str, dict[str, Any]],
    codebooks: dict[tuple[str, str, str], torch.Tensor],
    expert: int,
    projection: str,
    specs: list[tuple[str, str]],
    *,
    row_chunk: int = 512,
) -> dict[tuple[str, str], float]:
    """Fuse frozen-plane dequant and weighted-SSE scoring across tier variants.

    Tiers with the same vector width and plane geometry share one codes/scales
    transfer, one concatenated resident codebook gather, and one weighted-error
    reduction per row chunk. The per-tier BF16 dequantization and FP32 weighted
    SSE expression are unchanged; grouping only removes serial launches and
    repeated synchronization.
    """
    if row_chunk < 1:
        raise ValueError("row_chunk must be positive")
    if not specs:
        return {}

    device = w.device
    hf = h.to(device).float()
    grouped: dict[
        tuple[int, tuple[int, ...], tuple[int, ...]], list[tuple[str, str]]
    ] = {}
    for tier, variant in specs:
        data = plane_data[tier]
        if data.get("meta", {}).get("tier") == "vqa" or "vqa_codes13" in data:
            codes = data[f"vqa_codes{projection}"][expert]
            scales = data[f"vqa_sc{projection}"][expert]
        else:
            codes = data[f"codes{projection}"][expert]
            scales = data[f"sc{projection}"][expert]
        vector_width = int(codebooks[(tier, variant, projection)].shape[1])
        key = (vector_width, tuple(codes.shape), tuple(scales.shape))
        grouped.setdefault(key, []).append((tier, variant))

    errors: dict[tuple[str, str], float] = {}
    for (vector_width, _codes_shape, _scales_shape), members in grouped.items():
        code_parts = []
        scale_parts = []
        cb_parts = []
        offsets = []
        offset = 0
        for tier, variant in members:
            data = plane_data[tier]
            if data.get("meta", {}).get("tier") == "vqa" or "vqa_codes13" in data:
                codes = data[f"vqa_codes{projection}"][expert]
                scales = data[f"vqa_sc{projection}"][expert]
            else:
                codes = data[f"codes{projection}"][expert]
                scales = data[f"sc{projection}"][expert]
            code_parts.append(codes)
            scale_parts.append(scales)
            cb_part = codebooks[(tier, variant, projection)].to(device).float()
            cb_parts.append(cb_part)
            offsets.append(offset)
            offset += int(cb_part.shape[0])

        # Keep compact plane dtypes resident and widen only the active row chunk.
        codes = torch.stack(code_parts, dim=0).to(device)
        scales = torch.stack(scale_parts, dim=0).to(device)
        flat_codebook = torch.cat(cb_parts, dim=0)
        code_offsets = torch.tensor(offsets, device=device, dtype=torch.int64)
        if codes.is_cuda:
            from .frozen_score import fused_frozen_weighted_errors

            totals = fused_frozen_weighted_errors(
                w,
                hf,
                codes,
                scales,
                flat_codebook,
                code_offsets,
                vector_width=vector_width,
            ).detach().cpu().tolist()
        else:
            code_offsets_3d = code_offsets.view(-1, 1, 1)
            columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=2)
            totals = [0.0 for _ in members]
            for start in range(0, w.shape[0], row_chunk):
                stop = min(int(w.shape[0]), start + row_chunk)
                active_codes = codes[:, start:stop].long() + code_offsets_3d
                q = flat_codebook[active_codes].reshape(
                    len(members), stop - start, -1
                )
                q = (q * columns[:, start:stop]).to(torch.bfloat16)
                wf = w[start:stop].float().unsqueeze(0)
                chunk_errors = (
                    (wf - q.float()).square() * hf.unsqueeze(0).unsqueeze(0)
                ).sum(dim=(1, 2))
                # Preserve existing chunked Python accumulation on the portable path.
                for index, value in enumerate(chunk_errors.detach().cpu().tolist()):
                    totals[index] += float(value)

        errors.update(
            {member: float(totals[index]) for index, member in enumerate(members)}
        )
    return errors


def build_certified_shortlist(
    coarse_cb: torch.Tensor,
    fine_cb: torch.Tensor,
    n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return N candidates/coarse center and first-excluded squared radius."""
    coarse_cb = coarse_cb.float()
    fine_cb = fine_cb.float()
    dist = (
        (coarse_cb * coarse_cb).sum(1, keepdim=True)
        - 2.0 * (coarse_cb @ fine_cb.t())
        + (fine_cb * fine_cb).sum(1).unsqueeze(0)
    )
    values, ids = dist.topk(n + 1, largest=False, sorted=True)
    return ids[:, :n], values[:, n].clamp_min_(0.0)


def build_resident_candidate_slab(
    coarse_cb: torch.Tensor,
    fine_cb: torch.Tensor,
    n: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build quality-safe projection-resident coarse→fine candidate vectors."""
    coarse = coarse_cb.float()
    fine = fine_cb.float()
    dist = (
        (coarse * coarse).sum(1, keepdim=True)
        - 2.0 * (coarse @ fine.t())
        + (fine * fine).sum(1).unsqueeze(0)
    )
    ids = dist.topk(n, largest=False, sorted=True).indices.contiguous()
    slab = fine[ids].to(torch.bfloat16).contiguous()
    return ids, slab


def encode_dequant_row_resident(
    w: torch.Tensor,
    scale_source: dict[str, Any],
    expert: int,
    projection: str,
    coarse_codes: torch.Tensor,
    candidate_slab: torch.Tensor,
    d: int = 4,
    chunk: int = 1_048_576,
) -> tuple[torch.Tensor, dict[str, int | str]]:
    """Quality-tolerant NN over a resident top-64 candidate slab.

    This is the accelerated public solver path: cells are flattened into large
    slabs, both projection codebooks remain resident, and no assignment or
    optimizer state is reused. The exact serial path remains available for A/B.
    """
    scales = scale_source[f"sc{projection}"][expert].to(DEVICE)
    columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    x = (w.float() / columns).reshape(-1, d).to(torch.bfloat16)
    coarse = coarse_codes.to(DEVICE).long().reshape(-1)
    qparts = []
    for start in range(0, x.shape[0], chunk):
        part = x[start:start + chunk]
        candidates = candidate_slab[coarse[start:start + chunk]]
        distance = (part.unsqueeze(1) - candidates).square().sum(2)
        best_local = distance.argmin(1)
        qparts.append(
            candidates.gather(
                1,
                best_local[:, None, None].expand(-1, 1, d),
            ).squeeze(1)
        )
    q = torch.cat(qparts, dim=0).reshape_as(w) * columns
    return q.to(torch.bfloat16), {
        "certified_vectors": 0,
        "fallback_vectors": 0,
        "approximate_vectors": int(x.shape[0]),
        "candidate_count": int(candidate_slab.shape[1]),
        "distance_dtype": "bfloat16",
    }


def encode_dequant_row_exact_gemm(
    w: torch.Tensor,
    scale_source: dict[str, Any],
    cb: torch.Tensor,
    expert: int,
    projection: str,
    d: int = 4,
    row_chunk: int = 65_536,
    plan: Any | None = None,
    audit_assignments: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Full-codebook TF32x3 top-2 with bound-certified IEEE-FP32 verification.

    Every vector evaluates every codebook row. Rows whose tensor-core top-2
    margin does not dominate the conservative error envelope are recomputed by
    the fused IEEE-FP32 full-codebook verifier before codewords are gathered.
    """
    from .exact_codebook import exact_codebook_winners

    device = w.device
    scales = scale_source[f"sc{projection}"][expert].to(device)
    columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    x = (w.float() / columns).reshape(-1, d)
    codebook = cb.to(device).float()
    winners, meta = exact_codebook_winners(
        x, codebook, row_chunk=row_chunk, plan=plan
    )
    q = codebook[winners].reshape_as(w.float()) * columns
    meta = {
        **meta,
        "certified_vectors": int(meta["certified_rows"]),
        "fallback_vectors": int(meta["verified_rows"]),
        "exact_vectors": int(meta["rows"]),
        "encoder": "full-codebook TF32x3 tensor-core top2 + bound-gated fused IEEE-FP32 verification",
    }
    if audit_assignments:
        assignment_bytes = (
            winners.to(torch.int16).detach().cpu().contiguous().numpy().tobytes()
        )
        meta["codeword_assignment_sha256"] = hashlib.sha256(
            assignment_bytes
        ).hexdigest()
        meta["codeword_assignment_count"] = int(winners.numel())
        meta["codeword_assignment_dtype"] = "int16-le"
    return q.to(torch.bfloat16), meta


def prepare_exact_codebook_plan(cb: torch.Tensor) -> Any:
    """Build resident FP32/TF32x3 and BF16 codebook slabs plus bounds once."""
    from .exact_codebook import prepare_exact_codebook

    return prepare_exact_codebook(cb)


def encode_dequant_row(
    w: torch.Tensor,
    scale_source: dict[str, Any],
    cb: torch.Tensor,
    expert: int,
    projection: str,
    coarse_codes: torch.Tensor,
    coarse_cb: torch.Tensor,
    shortlist: torch.Tensor,
    first_excluded_sq: torch.Tensor,
    d: int = 4,
    chunk: int = 65536,
    audit_assignments: bool = False,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Exact k2048 NN via k4096 shortlist + triangle-bound certification.

    Any row not certified by ||c-f_excluded|| >= ||x-c||+||x-f_best||
    falls back to exhaustive search, so the result is exact despite the fast
    shortlist path.
    """
    scales = scale_source[f"sc{projection}"][expert].to(DEVICE)
    columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
    x = (w.float() / columns).reshape(-1, d)
    cb = cb.float()
    coarse_cb = coarse_cb.float()
    coarse_codes = coarse_codes.to(DEVICE).long().reshape(-1)
    cb_norm = (cb * cb).sum(1)
    qparts = []
    assignment_hasher = hashlib.sha256() if audit_assignments else None
    assignment_count = 0
    certified_total = 0
    fallback_total = 0
    for start in range(0, x.shape[0], chunk):
        part = x[start:start + chunk]
        cids = coarse_codes[start:start + chunk]
        candidate_ids = shortlist[cids]
        candidate_cb = cb[candidate_ids]
        distance = (part.unsqueeze(1) - candidate_cb).square().sum(2)
        best_sq, best_local = distance.min(1)
        best_ids = candidate_ids.gather(1, best_local.unsqueeze(1)).squeeze(1)
        x_to_coarse = (part - coarse_cb[cids]).square().sum(1).clamp_min_(0.0).sqrt_()
        excluded_radius = first_excluded_sq[cids].sqrt()
        best_radius = best_sq.clamp_min_(0.0).sqrt_()
        certified = (excluded_radius - x_to_coarse) > (best_radius + 1e-6)
        certified_total += int(certified.sum())
        fallback = ~certified
        if fallback.any():
            rows = part[fallback]
            exact = rows @ cb.t()
            exact.mul_(-2).add_(cb_norm.unsqueeze(0))
            best_ids[fallback] = exact.argmin(1)
            fallback_total += int(fallback.sum())
            del rows, exact
        if assignment_hasher is not None:
            assignment_hasher.update(
                best_ids.to(torch.int16).detach().cpu().contiguous().numpy().tobytes()
            )
            assignment_count += int(best_ids.numel())
        qparts.append(cb[best_ids])
        del candidate_ids, candidate_cb, distance, best_sq, best_local, best_ids
    q = torch.cat(qparts, dim=0).reshape_as(w.float()) * columns
    meta: dict[str, Any] = {
        "certified_vectors": certified_total,
        "fallback_vectors": fallback_total,
    }
    if assignment_hasher is not None:
        meta["codeword_assignment_sha256"] = assignment_hasher.hexdigest()
        meta["codeword_assignment_count"] = assignment_count
        meta["codeword_assignment_dtype"] = "int16-le"
    return q.to(torch.bfloat16), meta


def weighted_energy(w: torch.Tensor, h: torch.Tensor, chunk: int = 512) -> float:
    energy = 0.0
    h = h.to(DEVICE).float()
    for start in range(0, w.shape[0], chunk):
        wf = w[start:start + chunk].float()
        energy += float((wf.square() * h.unsqueeze(0)).sum().item())
    return energy


def weighted_error(w: torch.Tensor, q: torch.Tensor, h: torch.Tensor, chunk: int = 512) -> float:
    err = 0.0
    h = h.to(DEVICE).float()
    for start in range(0, w.shape[0], chunk):
        wf = w[start:start + chunk].float()
        qf = q[start:start + chunk].float()
        err += float(((wf - qf).square() * h.unsqueeze(0)).sum().item())
    return err


def variants(tier: str) -> tuple[str, ...]:
    return ("deployed", "base") if tier in REPAIRED else ("base",)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--staging-root", type=Path)
    ap.add_argument("--keep-planes", action="store_true")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",") if x]
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    index = checkpoint_index(args.model_root)

    for layer in layers:
        started = time.time()
        out_dir = args.root / "prices" / f"L{layer:03d}"
        complete = out_dir / "COMPLETE.json"
        if complete.is_file():
            print(f"L{layer:03d} resume COMPLETE", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        captures = load_captures(
            args.root, layer, args.windows, staging_root=args.staging_root
        )
        try:
            hfused, counts = hdiag_fused(captures)
        except RuntimeError as exc:
            # Cold experts can be absent from the canonical 32-window panel.
            # Fall back only to the sealed 64-window continuation; never invent H.
            if "unrouted experts" not in str(exc):
                raise
            captures = load_captures(
                args.root, layer, 64, staging_root=args.staging_root
            )
            hfused, counts = hdiag_fused(captures)
        actual_windows = len(captures)
        paths = plane_paths(
            args.root,
            layer,
            selected_tiers=TIERS,
            staging_root=args.staging_root,
        )
        plane_data = {tier: torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                      for tier, path in paths.items()}
        cb = {}
        cb_md5 = {}
        for tier in TIERS:
            base_obj = torch.load(catalog_path(args.root, "base", tier, layer), map_location="cpu", weights_only=True)
            for variant in variants(tier):
                source_variant = "repaired" if variant == "deployed" else "base"
                obj = (torch.load(catalog_path(args.root, source_variant, tier, layer), map_location="cpu", weights_only=True)
                       if source_variant == "repaired" else base_obj)
                for proj in ("13", "2"):
                    tensor = obj[f"cb{proj}"].to(DEVICE).float()
                    cb[(tier, variant, proj)] = tensor
                    cb_md5[(tier, variant, proj)] = tensor_md5(obj[f"cb{proj}"])
            # Source planes contribute frozen codes/scales only. Their embedded
            # codebooks can be stale historical anchors; pricing deliberately
            # substitutes the hash-bound deployed/base catalog codebooks above,
            # exactly like the composer. k2048 is replayed separately.

        certified_shortlists = {}
        for proj in ("13", "2"):
            certified_shortlists[proj] = build_certified_shortlist(
                cb[("d4_k4096", "base", proj)],
                cb[("d4_k2048", "base", proj)],
            )
        layer_map = weight_shard_map(index, layer)
        handles = open_shards(
            args.root,
            layer,
            layer_map,
            model_root=args.model_root,
            staging_root=args.staging_root,
        )
        rows = []
        jsonl_tmp = out_dir / "prices.jsonl.tmp"
        with jsonl_tmp.open("w") as sink:
            for expert in range(256):
                fused, down = load_weights(handles, layer_map, layer, expert)
                x = expert_x(captures, expert)
                hdown = down_hdiag(x, fused)
                for projection, w, h in (("13", fused, hfused[expert].to(DEVICE)), ("2", down, hdown)):
                    energy = weighted_energy(w, h)
                    for tier in TIERS:
                        for variant in variants(tier):
                            if tier == "d4_k2048":
                                shortlist, first_excluded_sq = certified_shortlists[projection]
                                q, encode_meta = encode_dequant_row(
                                    w,
                                    plane_data["d4_k4096"],
                                    cb[(tier, variant, projection)],
                                    expert,
                                    projection,
                                    plane_data["d4_k4096"][f"codes{projection}"][expert],
                                    cb[("d4_k4096", "base", projection)],
                                    shortlist,
                                    first_excluded_sq,
                                )
                                encoder = "exact NN: k4096 shortlist + triangle certification + exhaustive fallback"
                            else:
                                q = dequant_row(plane_data[tier], cb[(tier, variant, projection)], expert, projection)
                                encode_meta = {"certified_vectors": 0, "fallback_vectors": 0}
                                encoder = "frozen uniform-plane codes/scales"
                            err = weighted_error(w, q, h)
                            rel = err / energy if energy else math.inf
                            row = {
                                "schema": "solver-pricing-v2-cell-tier-v1",
                                "layer": layer,
                                "expert": expert,
                                "projection": projection,
                                "cell": f"L{layer:03d}.E{expert:03d}.P{projection}",
                                "tier": tier,
                                "variant": variant,
                                "weighted_sse": err,
                                "teacher_energy": energy,
                                "relative_weighted_error": rel,
                                "routed_rows": int(counts[expert]),
                                "n_windows": actual_windows,
                                "hessian": "diag(X^T X), teacher fused13 activation for down",
                                "encoder": encoder,
                                "encode_certified_vectors": encode_meta["certified_vectors"],
                                "encode_fallback_vectors": encode_meta["fallback_vectors"],
                                "codebook_md5": cb_md5[(tier, variant, projection)],
                                "plane": str(paths[tier]),
                            }
                            sink.write(json.dumps(row, sort_keys=True) + "\n")
                            rows.append(row)
                            del q
                del fused, down, x, hdown
                if expert % 8 == 7:
                    print(f"L{layer:03d} E{expert:03d}/255 rows={len(rows)}", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
        final_jsonl = out_dir / "prices.jsonl"
        os.replace(jsonl_tmp, final_jsonl)
        # Compact tensors make the table cheap for the solver to load.
        torch.save(rows, out_dir / "prices.pt")
        manifest = {
            "schema": "solver-pricing-v2-layer-complete-v1",
            "layer": layer,
            "cells": 512,
            "deployed_tiers": list(TIERS),
            "rows": len(rows),
            "expected_rows": 512 * (len(TIERS) + len(REPAIRED)),
            "windows": actual_windows,
            "capture_dir": str(
                capture_dir(
                    args.root,
                    layer,
                    actual_windows,
                    staging_root=args.staging_root,
                )
            ),
            "plane_paths": {k: str(v) for k, v in paths.items()},
            "elapsed_s": time.time() - started,
            "jsonl_bytes": final_jsonl.stat().st_size,
        }
        if manifest["rows"] != manifest["expected_rows"]:
            raise RuntimeError(manifest)
        complete_tmp = complete.with_suffix(".tmp")
        complete_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(complete_tmp, complete)
        print(json.dumps({"status": "PASS", **manifest}, sort_keys=True), flush=True)
        del plane_data, cb, handles, captures, hfused
        gc.collect()
        torch.cuda.empty_cache()
        shard_scratch = args.root / "weight_shards" / f"L{layer:03d}"
        for p in shard_scratch.glob("*.safetensors"):
            p.unlink()
        try:
            shard_scratch.rmdir()
        except OSError:
            pass
        if not args.keep_planes:
            scratch = args.root / "planes_scratch" / f"L{layer:03d}"
            for p in scratch.glob("*.pt"):
                p.unlink()
            try:
                scratch.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
