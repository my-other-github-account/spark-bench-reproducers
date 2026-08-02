#!/usr/bin/env python3
"""Fresh exact QTIP unit profiler used by the public ``smash profile`` verb.

This is a profiling adapter around the sealed QTIP builder and exact-prefix
Viterbi kernels.  It does not change the objective, prune states, or reuse a
prior assignment.  The reference unit is read only after the fresh solve and
is used solely for an exact assignment/trellis digest gate.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import types
from typing import Any

import torch
from safetensors import safe_open


QTIP_RHT_DOMAIN = "qtip-rht-bounded36-v1"
QTIP_RHT_SEED_MATERIAL = (
    "4fa7b1213db1d6a4670b534785edb1681d1538bb6d12a90222e33c30251c2462"
    "|t_782dc70e|heldout-experts-v1"
)

# A config-directory solve is one public process. These caches remove repeated
# import, capture-bank, manifest, TLUT, and index staging between independent
# exact units. Candidate state, objectives, codebooks, weights, and assignments
# remain unit-local.
_MODULE_CACHE: dict[Path, Any] = {}
_CAPTURE_CACHE: dict[tuple[Path, int, int], list[dict[str, Any]]] = {}
_HESSIAN_BINDING_CACHE: dict[
    tuple[Path, str, Path, int, int], tuple[Path, int, dict[str, Any]]
] = {}
_TLUT_CACHE: dict[Path, torch.Tensor] = {}
_MODEL_INDEX_CACHE: dict[Path, dict[str, str]] = {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _basis_sha(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("index_sha256", "source_model_index_sha256", "sha256"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
    return None


def _verify_basis(config: dict[str, Any], run_root: Path) -> dict[str, Any]:
    model_root = Path(config["model_root"]).resolve()
    index_path = model_root / "model.safetensors.index.json"
    actual = _sha256(index_path)
    identity = config.get("input_identity")
    configured = (
        _basis_sha(identity.get("model_index"))
        if isinstance(identity, dict)
        else None
    )
    if configured is None:
        configured = _basis_sha(config.get("model_index"))
    if configured is None:
        raise ValueError("QTIP config lacks a SHA-bound model index identity")
    if configured != actual:
        raise ValueError(f"QTIP config model-index mismatch: {actual} != {configured}")

    shards_path = run_root.resolve() / "SHARDS.json"
    shards = json.loads(shards_path.read_text())
    intended = _basis_sha(shards.get("intended_basis"))
    if intended is None:
        raise ValueError(f"SHARDS.json lacks intended_basis: {shards_path}")
    if intended != actual:
        raise ValueError(f"QTIP basis mismatch: {actual} != {intended}")
    return {
        "schema": "banana-smasher-qtip-basis-gate-v1",
        "status": "PASS",
        "index_path": str(index_path),
        "index_sha256": actual,
        "intended_basis": intended,
        "shards_manifest": str(shards_path),
        "shards_manifest_sha256": _sha256(shards_path),
    }


def _canonical_rht_seed(layer: int, expert: int, projection: str) -> int:
    identity = f"L{layer:03d}_E{expert:03d}_{projection}"
    digest = hashlib.sha256(
        f"{QTIP_RHT_DOMAIN}|{QTIP_RHT_SEED_MATERIAL}|{identity}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _resolve_rht_seed(
    config: dict[str, Any],
    reference: dict[str, Any],
    *,
    layer: int,
    expert: int,
    projection: str,
) -> tuple[int, str]:
    policy = str(config.get("rht_seed_policy", "reference-unit-v1"))
    if policy == QTIP_RHT_DOMAIN:
        expected = _canonical_rht_seed(layer, expert, projection)
        configured = config.get("rht_seed")
        if not isinstance(configured, int) or configured != expected:
            raise ValueError(
                "canonical RHT seed mismatch: "
                f"configured={configured!r} expected={expected} "
                f"identity=L{layer:03d}_E{expert:03d}_{projection}"
            )
        return expected, policy
    if policy != "reference-unit-v1":
        raise ValueError(f"unsupported RHT seed policy: {policy}")
    reference_seed = reference.get("rht_seed")
    if not isinstance(reference_seed, int):
        raise ValueError("reference unit is missing an integer rht_seed")
    configured = config.get("rht_seed", reference_seed)
    if not isinstance(configured, int) or configured != reference_seed:
        raise ValueError(
            f"reference-unit RHT seed mismatch: {configured!r} != {reference_seed}"
        )
    return reference_seed, policy


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


_QTIP_UNIT_PAYLOAD_SCHEMA = "ds4-qtip-hyb-bounded36-unit-v1"
_QTIP_SOLVE_RECEIPT_SCHEMA = "banana-smasher-qtip-solve-v1"
_QTIP_UNIT_REQUIRED_TENSORS = ("trellis", "SU", "SV", "Wscale", "tlut")


def _is_sha256_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_intended_basis(root: Path) -> str:
    """Read the run's intended model basis so existing units bind to THIS run."""
    shards_path = root.resolve() / "SHARDS.json"
    try:
        shards = json.loads(shards_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit cannot bind run basis: {shards_path}"
        ) from exc
    intended = _basis_sha(shards.get("intended_basis"))
    if not _is_sha256_digest(intended):
        raise RuntimeError(
            f"existing QTIP unit cannot bind run basis: {shards_path}"
        )
    assert isinstance(intended, str)
    return intended


def _validated_existing_unit(
    config_path: Path,
    root: Path,
    layer: int,
    *,
    profile_mode: bool,
) -> dict[str, Any] | None:
    """Return the receipt of an immutable, hash-valid existing PASS unit.

    Returns ``None`` when the unit has never been solved (nothing durable
    exists), so the caller computes it fresh.  Any partial, divergent,
    corrupt, or internally inconsistent existing state raises instead of
    silently rerunning or overwriting sealed bytes.  Profiling never
    resumes: a profile receipt is a measurement, not a solve artifact.
    """
    if profile_mode:
        return None
    try:
        config = json.loads(config_path.read_text())
        configured_layer = config["layer"]
        expert = config["expert"]
        projection = config["projection"]
        geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
        sealed_geometry = (geometry["L"], geometry["K"], geometry["V"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit has invalid config: {config_path}"
        ) from exc
    if (
        isinstance(configured_layer, bool)
        or not isinstance(configured_layer, int)
        or configured_layer != layer
        or isinstance(expert, bool)
        or not isinstance(expert, int)
        or not 0 <= expert < 256
        or not isinstance(projection, str)
        or projection not in {"fused13", "down"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in sealed_geometry
        )
        or sealed_geometry not in {(16, 3, 2), (16, 2, 2)}
    ):
        raise RuntimeError(
            f"existing QTIP unit has invalid config identity: {config_path}"
        )
    out = root / "solve" / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
    artifact_path = out / "QTIP_UNIT.pt"
    receipt_path = out / "QTIP_SOLVE_RECEIPT.json"
    artifact_exists = artifact_path.is_file()
    receipt_exists = receipt_path.is_file()
    if not artifact_exists and not receipt_exists:
        return None
    if artifact_exists != receipt_exists:
        raise RuntimeError(
            "existing QTIP unit is partial: "
            f"payload={artifact_exists} receipt={receipt_exists} unit={out}"
        )
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"existing QTIP unit receipt is corrupt: {receipt_path}"
        ) from exc
    if not isinstance(receipt, dict):
        raise RuntimeError(
            f"existing QTIP unit receipt is corrupt: {receipt_path}"
        )
    expected_identity = {
        "schema": _QTIP_SOLVE_RECEIPT_SCHEMA,
        "status": "PASS",
        "layer": layer,
        "expert": expert,
        "projection": projection,
    }
    drift = {
        key: (receipt.get(key), expected)
        for key, expected in expected_identity.items()
        if receipt.get(key) != expected
    }
    if drift:
        raise RuntimeError(
            f"existing QTIP unit identity drift at {receipt_path}: {drift}"
        )
    if receipt.get("config_sha256") != _sha256(config_path):
        raise RuntimeError(
            f"existing QTIP unit config hash drift: {config_path}"
        )
    run_basis = _run_intended_basis(root)
    configured_basis = None
    identity = config.get("input_identity")
    if isinstance(identity, dict):
        configured_basis = _basis_sha(identity.get("model_index"))
    if configured_basis is None:
        configured_basis = _basis_sha(config.get("model_index"))
    try:
        model_index = (
            Path(str(config["model_root"])).resolve()
            / "model.safetensors.index.json"
        )
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            f"existing QTIP unit lacks model root: {config_path}"
        ) from exc
    if not model_index.is_file() or _sha256(model_index) != run_basis:
        raise RuntimeError(
            f"existing QTIP unit live model basis drift: {model_index}"
        )
    gate = receipt.get("basis_gate")
    if (
        not isinstance(gate, dict)
        or gate.get("schema") != "banana-smasher-qtip-basis-gate-v1"
        or gate.get("status") != "PASS"
        or gate.get("index_sha256") != run_basis
        or gate.get("intended_basis") != run_basis
        or configured_basis != run_basis
    ):
        raise RuntimeError(f"existing QTIP unit basis drift: {receipt_path}")
    try:
        recorded_artifact = Path(str(receipt["artifact"])).resolve()
    except (KeyError, OSError) as exc:
        raise RuntimeError(
            f"existing QTIP unit lacks an artifact path: {receipt_path}"
        ) from exc
    if recorded_artifact != artifact_path.resolve():
        raise RuntimeError(
            "existing QTIP unit artifact path drift: "
            f"{recorded_artifact} != {artifact_path.resolve()}"
        )
    if not _is_sha256_digest(receipt.get("artifact_sha256")):
        raise RuntimeError(
            f"existing QTIP unit lacks a payload hash: {receipt_path}"
        )
    if receipt["artifact_sha256"] != _sha256(artifact_path):
        raise RuntimeError(
            f"existing QTIP unit payload hash drift: {artifact_path}"
        )
    try:
        artifact = torch.load(
            artifact_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"existing QTIP unit payload is unreadable: {artifact_path}"
        ) from exc
    expected_payload_geometry = {
        "L": sealed_geometry[0],
        "K": sealed_geometry[1],
        "V": sealed_geometry[2],
        "tlut_bits": 9,
        "decode_mode": "quantlut_sym",
        "td_x": 16,
        "td_y": 16,
    }
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != _QTIP_UNIT_PAYLOAD_SCHEMA
        or artifact.get("geometry") != expected_payload_geometry
        or any(
            not isinstance(artifact.get(key), torch.Tensor)
            for key in _QTIP_UNIT_REQUIRED_TENSORS
        )
    ):
        raise RuntimeError(
            f"existing QTIP unit payload schema is invalid: {artifact_path}"
        )
    if not _is_sha256_digest(receipt.get("assignment_sha256")):
        raise RuntimeError(
            f"existing QTIP unit lacks an assignment digest: {receipt_path}"
        )
    if receipt["assignment_sha256"] != _tensor_sha256(artifact["trellis"]):
        raise RuntimeError(
            f"existing QTIP unit assignment digest drift: {artifact_path}"
        )
    total_wall_seconds = receipt.get("total_wall_seconds")
    if (
        isinstance(total_wall_seconds, bool)
        or not isinstance(total_wall_seconds, (int, float))
        or not math.isfinite(total_wall_seconds)
        or total_wall_seconds < 0
    ):
        raise RuntimeError(
            f"existing QTIP unit timing is invalid: {receipt_path}"
        )
    return receipt


def _process_receipt() -> dict[str, int]:
    stat = Path("/proc/self/stat")
    return {
        "pid": os.getpid(),
        "startticks": int(stat.read_text().split()[21]) if stat.is_file() else 0,
    }


def _load_module(name: str, path: Path):
    path = path.resolve()
    cached = _MODULE_CACHE.get(path)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODULE_CACHE[path] = module
    return module


def _load_captures(
    root: Path,
    layer: int,
    windows: int,
) -> list[dict[str, Any]]:
    root = root.resolve()
    cache_key = (root, layer, windows)
    cached = _CAPTURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    rows = []
    for window in range(windows):
        path = root / f"xmoe_L{layer:03d}_win{window:04d}.pt"
        done_path = path.with_suffix(path.suffix + ".DONE.json")
        if not path.is_file() or not done_path.is_file():
            raise FileNotFoundError(f"missing fit capture or receipt: {path}")
        done = json.loads(done_path.read_text())
        data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        if int(data["layer"]) != layer or int(data["win"]) != window:
            raise RuntimeError(f"capture identity mismatch: {path}")
        rows.append({
            "window": window,
            "x": data["x"].to(torch.bfloat16).contiguous(),
            "topk": data["topk"].to(torch.int64).contiguous(),
            "route": data["w"].float().contiguous(),
            "receipt_md5": done.get("md5"),
        })
    _CAPTURE_CACHE[cache_key] = rows
    return rows


def _bind_hessian_layer_manifest(
    config: dict[str, Any],
    *,
    layer: int,
) -> tuple[Path, int, dict[str, Any]]:
    manifest_path = Path(config["hessian_layer_manifest"]).resolve()
    windows = int(config["fit_windows"])
    expected_sha = str(config["hessian_layer_manifest_sha256"])
    configured_root = Path(config["fit_capture_root"]).resolve()
    cache_key = (manifest_path, expected_sha, configured_root, layer, windows)
    cached = _HESSIAN_BINDING_CACHE.get(cache_key)
    if cached is not None:
        return cached
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    expected = {
        "schema": "banana-smasher-hessian-layer-manifest-v1",
        "status": "PASS",
        "layer": layer,
        "windows": windows,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"QTIP Hessian layer-manifest binding mismatch: {manifest_path}")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError(
            f"QTIP Hessian layer-manifest hash drift: {actual_sha} != {expected_sha}"
        )
    capture_root = Path(str(manifest["capture_root"])).resolve()
    if capture_root != configured_root:
        raise ValueError(
            f"QTIP capture root differs from Hessian manifest: {configured_root} != {capture_root}"
        )
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != windows:
        raise ValueError(f"QTIP Hessian member population mismatch: {manifest_path}")
    for window, member in enumerate(members):
        if member.get("window") != window:
            raise ValueError(f"QTIP Hessian member order mismatch at window {window}")
        expected_capture = capture_root / f"xmoe_L{layer:03d}_win{window:04d}.pt"
        expected_done = expected_capture.with_suffix(expected_capture.suffix + ".DONE.json")
        for key, expected_path in (("capture", expected_capture), ("capture_done", expected_done)):
            artifact = member.get(key)
            if not isinstance(artifact, dict):
                raise ValueError(f"QTIP Hessian member lacks {key}: window {window}")
            path = Path(str(artifact.get("path", ""))).resolve()
            if path != expected_path.resolve() or not path.is_file():
                raise ValueError(f"QTIP Hessian {key} path mismatch: {path}")
            if path.stat().st_size != artifact.get("bytes"):
                raise ValueError(f"QTIP Hessian {key} size drift: {path}")
    binding = {
        "path": str(manifest_path),
        "bytes": len(raw),
        "sha256": actual_sha,
        "windows": windows,
        "capture_root": str(capture_root),
    }
    value = (capture_root, windows, binding)
    _HESSIAN_BINDING_CACHE[cache_key] = value
    return value


def _load_tlut(path: Path) -> torch.Tensor:
    path = path.resolve()
    cached = _TLUT_CACHE.get(path)
    if cached is None:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        cached = payload["tlut"].float().contiguous()
        _TLUT_CACHE[path] = cached
    return cached


_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _load_weight(model_root: Path, layer: int, expert: int, projection: str) -> tuple[torch.Tensor, dict[str, Any]]:
    index_path = model_root / "model.safetensors.index.json"
    resolved_index = index_path.resolve()
    mapping = _MODEL_INDEX_CACHE.get(resolved_index)
    if mapping is None:
        mapping = json.loads(index_path.read_text())["weight_map"]
        _MODEL_INDEX_CACHE[resolved_index] = mapping
    names = ("w1", "w3") if projection == "fused13" else ("w2",)
    matrices = []
    source = []
    for name in names:
        weight_key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
        scale_key = f"layers.{layer}.ffn.experts.{expert}.{name}.scale"
        shard = model_root / mapping[weight_key]
        if mapping[scale_key] != mapping[weight_key]:
            raise RuntimeError(f"weight/scale shard split: {weight_key}")
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            packed = handle.get_tensor(weight_key).view(torch.uint8)
            scales = handle.get_tensor(scale_key).view(torch.uint8)
        nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
        matrices.append(
            (_E2M1[nibbles.long()] * torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)).contiguous()
        )
        source.append({"path": str(shard), "bytes": shard.stat().st_size, "weight_key": weight_key})
    value = torch.cat(matrices, dim=0) if len(matrices) == 2 else matrices[0]
    expected = (4096, 4096) if projection == "fused13" else (4096, 2048)
    if tuple(value.shape) != expected:
        raise RuntimeError(f"source shape mismatch: {tuple(value.shape)} != {expected}")
    return value.contiguous(), {
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "shards": source,
    }


def _prepare_fit_windows(
    runner: Any,
    captures: list[Any],
    *,
    model_root: Path,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device,
) -> tuple[list[Any], dict[str, Any]]:
    routed = runner.expert_windows(captures, expert)
    if projection != "down":
        return routed, {"mode": "routed-source-activation"}
    source_fused13, source_ref = _load_weight(
        model_root,
        layer,
        expert,
        "fused13",
    )
    try:
        windows = runner.down_windows(routed, source_fused13, device)
    finally:
        del source_fused13
    return windows, {
        "mode": "source-fused13",
        "source_weight": source_ref,
    }


class _ExactTimers:
    def __init__(self) -> None:
        self.codebook_distance_seconds = 0.0
        self.transition_seconds = 0.0
        self.calls = 0
        self.sequences = 0


def _install_profiled_exact_viterbi(
    cb: Any,
    exact: Any,
    timers: _ExactTimers,
    *,
    profile_mode: bool,
) -> dict[str, Any]:
    """Install exact Viterbi; instrumentation is profile-only, never solve overhead."""
    def solve(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None) -> torch.Tensor:
        if not x.is_cuda or x.ndim != 2 or x.shape[0] != 256:
            raise ValueError(f"exact prefix Viterbi expects CUDA [256,B], got {tuple(x.shape)}")
        batch = int(x.shape[1])
        if not 1 <= batch <= 8192:
            raise ValueError(f"batch outside 1..8192: {batch}")
        if profile_mode:
            with torch.profiler.record_function("qtip.viterbi_transition_scoring"):
                states = exact.exact_prefix_viterbi(self, x, overlap)
        else:
            states = exact.exact_prefix_viterbi(self, x, overlap)
        timers.calls += 1
        timers.sequences += batch
        return states

    def quantize_seq(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None, **_: Any):
        return solve(self, x, overlap)

    cb.viterbi = types.MethodType(solve, cb)
    cb.quantize_seq = types.MethodType(quantize_seq, cb)
    return {
        "implementation": "persistent-soa-exact-prefix-dp-v2",
        "full_states": 65536,
        "retained_prefix_costs": 1024,
        "branches_per_prefix": 64,
        "branch_sampling": "full",
        "steps": 128,
        "warps_per_sequence": 16,
        "lut_layout": "soa-two-contiguous-state-planes",
        "ordering": "one persistent launch per independent sequence batch",
        "production_default": True,
    }


def _install_configured_viterbi(
    cb: Any,
    exact: Any,
    timers: _ExactTimers,
    config: dict[str, Any],
    *,
    profile_mode: bool,
) -> dict[str, Any]:
    geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
    sealed = (int(geometry["L"]), int(geometry["K"]), int(geometry["V"]))
    if sealed == (16, 3, 2):
        return _install_profiled_exact_viterbi(
            cb, exact, timers, profile_mode=profile_mode
        )
    if sealed != (16, 2, 2):
        raise ValueError(f"unsupported QTIP geometry: {sealed}")
    from .trellis_v2 import install_trellis_v2

    metadata = install_trellis_v2(cb)
    cb._trellis_v2_collect_stats = False
    base = cb.quantize_seq

    def solve(self: Any, x: torch.Tensor, overlap: torch.Tensor | None = None) -> torch.Tensor:
        if profile_mode:
            with torch.profiler.record_function("qtip.viterbi_transition_scoring"):
                states = base(x, overlap)
        else:
            states = base(x, overlap)
        timers.calls += 1
        timers.sequences += int(x.shape[1])
        return states

    cb.viterbi = types.MethodType(solve, cb)
    cb.quantize_seq = types.MethodType(solve, cb)
    return {
        **metadata,
        "stats_collection_during_timing": False,
    }


def _top_ops(profile: Any) -> list[dict[str, Any]]:
    rows = []
    for event in profile.key_averages():
        cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
        device_us = float(
            getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))
        )
        rows.append({
            "op": str(event.key),
            "calls": int(event.count),
            "self_cpu_seconds": cpu_us / 1e6,
            "self_device_seconds": device_us / 1e6,
            "rank_seconds": (cpu_us + device_us) / 1e6,
        })
    return sorted(rows, key=lambda row: row["rank_seconds"], reverse=True)[:50]


def main(
    config_path: Path,
    root: Path,
    layer: int,
    *,
    profile_mode: bool = True,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text())
    if int(config["layer"]) != layer:
        raise ValueError("QTIP config layer differs from --layer")
    expert = int(config["expert"])
    projection = str(config["projection"])
    if projection not in {"fused13", "down"}:
        raise ValueError(projection)
    basis_gate = _verify_basis(config, root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    mode = "profile" if profile_mode else "solve"
    out = root / mode / f"L{layer:03d}" / f"E{expert:03d}_{projection}"
    out.mkdir(parents=True, exist_ok=True)
    outer_started = time.perf_counter()
    epoch_started = time.time()

    qv_path = Path(config["qtip_runner"])
    qv = _load_module("banana_smasher_qtip_runner", qv_path)
    qv.QTIP = Path(config["qtip_root"])
    bitshift, ldlq, math_utils, kernel_decode = qv.load_official_qtip()
    from . import qtip_viterbi as exact

    reference_path = Path(config["reference_unit"])
    reference = torch.load(reference_path, map_location="cpu", mmap=True, weights_only=True)
    seed, seed_policy = _resolve_rht_seed(
        config,
        reference,
        layer=layer,
        expert=expert,
        projection=projection,
    )
    pinned_tlut = _load_tlut(Path(config["tlut_source"]))
    if _tensor_sha256(pinned_tlut) != str(reference["tlut_sha256"]):
        raise RuntimeError("TLUT digest differs from sealed reference unit")
    geometry = config.get("geometry", {"L": 16, "K": 3, "V": 2})
    cb = bitshift.bitshift_codebook(
        L=int(geometry["L"]),
        K=int(geometry["K"]),
        V=int(geometry["V"]),
        tlut_bits=9,
        decode_mode="quantlut_sym",
        tlut=pinned_tlut.to("cuda"),
    ).to("cuda")
    timers = _ExactTimers()
    solver = _install_configured_viterbi(
        cb,
        exact,
        timers,
        config,
        profile_mode=profile_mode,
    )

    model_root = Path(config["model_root"])
    capture_root, fit_window_count, hessian_binding = _bind_hessian_layer_manifest(
        config,
        layer=layer,
    )
    captures = _load_captures(capture_root, layer, fit_window_count)
    fit_windows, fit_source = _prepare_fit_windows(
        qv,
        captures,
        model_root=model_root,
        layer=layer,
        expert=expert,
        projection=projection,
        device=torch.device("cuda"),
    )
    source_weight, source_ref = _load_weight(model_root, layer, expert, projection)
    staging_seconds = time.perf_counter() - outer_started

    dequant_seconds = 0.0
    original_decode = qv.decode_packed
    def timed_decode(*args: Any, **kwargs: Any):
        nonlocal dequant_seconds
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.profiler.record_function("qtip.dequant_and_reread"):
            value = original_decode(*args, **kwargs)
        torch.cuda.synchronize()
        dequant_seconds += time.perf_counter() - started
        return value
    qv.decode_packed = timed_decode

    first_gpu_phase = {
        "schema": "banana-smasher-qtip-first-gpu-phase-v1",
        "phase": "fresh_exact_qtip_build",
        "pid": os.getpid(),
        "startticks": int(Path("/proc/self/stat").read_text().split()[21]),
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "staging_seconds": staging_seconds,
        "epoch": time.time(),
    }
    _atomic_json(out / "FIRST_GPU_PHASE.json", first_gpu_phase)
    print("FIRST_GPU_PHASE " + json.dumps(first_gpu_phase, sort_keys=True), flush=True)

    if not profile_mode:
        qv.decode_packed = original_decode
        torch.cuda.reset_peak_memory_stats()
        build_started = time.perf_counter()
        candidate, build = qv.build_qtip(
            source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode,
            torch.device("cuda"), seed,
        )
        torch.cuda.synchronize()
        build_seconds = time.perf_counter() - build_started
        reconstructed = candidate.pop("reconstructed_weight", None)
        if reconstructed is None:
            raise RuntimeError(
                "QTIP builder omitted reconstructed_weight before public wire seal"
            )
        artifact_path = out / "QTIP_UNIT.pt"
        _atomic_torch(artifact_path, candidate)
        total_seconds = time.perf_counter() - outer_started
        receipt = {
            "schema": "banana-smasher-qtip-solve-v1",
            "status": "PASS",
            "host": os.uname().nodename,
            "layer": layer,
            "expert": expert,
            "projection": projection,
            "fresh_no_warm_start": True,
            "public_command_config": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "basis_gate": basis_gate,
            "epoch_started": epoch_started,
            "epoch_ended": time.time(),
            "total_wall_seconds": total_seconds,
            "staging_seconds": staging_seconds,
            "solve_seconds": build_seconds,
            "assignment_sha256": _tensor_sha256(candidate["trellis"]),
            "artifact": str(artifact_path),
            "artifact_sha256": _sha256(artifact_path),
            "viterbi_launches": timers.calls,
            "viterbi_sequences": timers.sequences,
            "transition_decisions": timers.sequences * 127 * 64,
            "solver": solver,
            "build": build,
            "source_weight": source_ref,
            "fit_source": fit_source,
            "fit_windows": fit_window_count,
            "hessian_layer_manifest": hessian_binding,
            "rht_seed": seed,
            "rht_seed_policy": seed_policy,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        _atomic_json(out / "QTIP_SOLVE_RECEIPT.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return receipt

    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=False) as profile:
        with torch.profiler.record_function("qtip.fresh_exact_build"):
            candidate, build = qv.build_qtip(
                source_weight, fit_windows, cb, ldlq, math_utils, kernel_decode,
                torch.device("cuda"), seed,
            )
    torch.cuda.synchronize()
    transition_keys = {
        "_persistent_prefix_viterbi",
        "_persistent_k2_viterbi",
        "aten::argmin",
    }
    timers.transition_seconds = sum(
        float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0)))
        for event in profile.key_averages()
        if (
            str(event.key) in transition_keys
            or "pair_kernel" in str(event.key)
            or "backtrack_kernel" in str(event.key)
        )
    ) / 1e6
    outer_seconds = time.perf_counter() - outer_started

    assignment_sha = _tensor_sha256(candidate["trellis"])
    bucket_seconds = {
        "trellis_viterbi_transition_scoring": timers.transition_seconds,
        "codebook_distances": timers.codebook_distance_seconds,
        "staging": staging_seconds,
        "dequant": dequant_seconds,
    }
    bucket_seconds["remainder"] = max(0.0, outer_seconds - sum(bucket_seconds.values()))
    census = {key: int(value) for key, value in config["layer_census"].items()}
    pack_counts = {key: int(value) for key, value in config["pack_counts"].items()}
    qtip_fraction = (pack_counts["qtip3"] + pack_counts["qtip2"]) / sum(pack_counts.values())
    receipt = {
        "schema": "banana-smasher-qtip-profile-v1",
        "status": "PASS",
        "host": os.uname().nodename,
        "layer": layer,
        "expert": expert,
        "projection": projection,
        "fresh_no_warm_start": True,
        "public_command_config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "basis_gate": basis_gate,
        "epoch_started": epoch_started,
        "epoch_ended": time.time(),
        "outer_wall_seconds": outer_seconds,
        "bucket_seconds": bucket_seconds,
        "bucket_percent": {key: 100.0 * value / outer_seconds for key, value in bucket_seconds.items()},
        "bucket_definition": {
            "trellis_viterbi_transition_scoring": "all 127 exact prefix-DP advance steps plus final argmin/backtrack; each advance kernel fuses predecessor transition minimization with that step's codebook-distance term",
            "codebook_distances": "exclusive initial-state exact distance kernel; later distance terms are fused into the transition bucket",
            "staging": "capture receipt identity reads, local model/TLUT/reference reads, QTIP import, and source dequant; full payload hashing remains in tests/CI",
            "dequant": "canonical packed-wire decode plus inverse FWHT reread conformance",
            "remainder": "Hessian/FWHT/LDLQ matrix work, packing, Python dispatch, compile overhead, profiler overhead, serialization, and closed residual",
        },
        "viterbi_calls": timers.calls,
        "solver": solver,
        "build": build,
        "top_10_ops": _top_ops(profile),
        "assignment_sha256": assignment_sha,
        "acceptance_provenance": {
            "source_commit": "48dd3443d86eae585c2c1b41e49f47912c50170f",
            "receipt": "P2C_QTIP3_PUBLIC_FIRST64_PASS",
            "ordered_assignment_sha256": (
                "96e0fd6c689cb1af387dce9843dc96ca52a086f85cc7e0caf7101d6ad92dfb26"
            ),
            "mean_public_outer_seconds": 1.9163911582144217,
        },
        "reference_unit": {"path": str(reference_path), "file_sha256": _sha256(reference_path)},
        "source_weight": source_ref,
        "fit_source": fit_source,
        "rht_seed": seed,
        "rht_seed_policy": seed_policy,
        "layer_census": census,
        "layer_qtip3_fraction": census["qtip3"] / sum(census.values()),
        "pack_projection": {
            "counts": pack_counts,
            "qtip_count_fraction": qtip_fraction,
            "qtip_dominated": qtip_fraction > 0.5,
            "profiled_geometry": {key: int(value) for key, value in geometry.items()},
            "profiled_unit_wall_seconds": outer_seconds,
            "projected_matching_geometry_pack_wall_seconds": (
                pack_counts["qtip2"]
                if int(geometry["K"]) == 2
                else pack_counts["qtip3"]
            ) * outer_seconds,
            "method": "physical fresh unit wall multiplied only by the sealed count for the matching QTIP geometry",
        },
        "next_kernel_recommendation": "The measured exact DP floor is now the persistent kernel itself; the next legal throughput lever is one resident layer solve that batches independent projection units and amortizes capture/model/TLUT staging without sharing assignment or objective state.",
    }
    receipt_path = out / "QTIP_PROFILE_RECEIPT.json"
    _atomic_json(receipt_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def _ordered_qtip_configs(
    config_root: Path,
    layer: int,
    *,
    tier: str | None = None,
    all_cells: bool = False,
) -> list[Path]:
    """Order one manifest-declared tier without a package-global tier menu."""
    projection_order = {"fused13": 0, "down": 1}
    rows: list[tuple[int, int, Path]] = []
    identities: set[tuple[int, str]] = set()
    selected_geometry: tuple[int, int, int] | None = None
    for path in config_root.rglob("E*_*.json"):
        config = json.loads(path.read_text())
        if int(config["layer"]) != layer:
            continue
        configured_tier = config.get("tier")
        if configured_tier is not None:
            if not isinstance(configured_tier, str) or not configured_tier:
                raise ValueError(f"invalid QTIP tier in {path}: {configured_tier!r}")
        if tier is not None and configured_tier is not None and configured_tier != tier:
            continue
        geometry = config.get("geometry")
        if not isinstance(geometry, dict) or set(geometry) != {"L", "K", "V"}:
            raise ValueError(f"QTIP config lacks exact L/K/V geometry: {path}")
        sealed_values = tuple(geometry[key] for key in ("L", "K", "V"))
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in sealed_values
        ):
            raise ValueError(f"invalid QTIP geometry in {path}: {sealed_values}")
        sealed = (
            int(sealed_values[0]),
            int(sealed_values[1]),
            int(sealed_values[2]),
        )
        if selected_geometry is None:
            selected_geometry = sealed
        elif sealed != selected_geometry:
            raise ValueError(
                f"manifest tier {tier or configured_tier or 'QTIP'} has mixed geometries: "
                f"{selected_geometry} != {sealed} in {path}"
            )
        expert = int(config["expert"])
        projection = str(config["projection"])
        if not 0 <= expert < 256:
            raise ValueError(f"QTIP expert outside 0..255 in {path}: {expert}")
        if projection not in projection_order:
            raise ValueError(f"unsupported QTIP projection in {path}: {projection}")
        identity = (expert, projection)
        if identity in identities:
            raise ValueError(
                f"duplicate resident QTIP config for E{expert:03d}_{projection}"
            )
        identities.add(identity)
        rows.append((expert, projection_order[projection], path))
    if not rows:
        label = tier or "QTIP"
        raise ValueError(
            f"no L{layer:03d} {label} configs under {config_root}; "
            "run public producer `smash qtip-configs`"
        )
    ordered = [path for _expert, _projection, path in sorted(rows)]
    if all_cells and len(ordered) != 512:
        raise ValueError(
            f"public {tier} --all-cells requires exactly 512 ordered configs "
            f"for L{layer:03d}, got {len(ordered)}"
        )
    return ordered


def main_many(
    config_root: Path,
    root: Path,
    layer: int,
    *,
    limit: int | None = None,
    tier: str | None = None,
    all_cells: bool = False,
    profile_mode: bool = False,
) -> dict[str, Any]:
    """Solve an ordered config directory in one resident public process."""
    if limit is not None and limit < 1:
        raise ValueError("--qtip-units must be positive")
    if all_cells and limit is not None:
        raise ValueError("--all-cells refuses a QTIP unit limit")
    paths = _ordered_qtip_configs(
        config_root,
        layer,
        tier=tier,
        all_cells=all_cells,
    )
    if limit is not None:
        paths = paths[:limit]
    batch_started = time.perf_counter()
    epoch_started = time.time()
    ordered_assignments = []
    unit_receipts = []
    resumed_units = 0
    computed_units = 0
    # Idempotent resume preflight: every pre-existing unit is hash-validated
    # BEFORE any new compute so a divergent/corrupt/partial unit fails loudly
    # instead of being rerun or overwritten.  Valid PASS units are skipped
    # byte-for-byte (no content, metadata, or mtime rewrite); execution then
    # continues at the first missing unit.
    existing_units = [
        _validated_existing_unit(
            path,
            root,
            layer,
            profile_mode=profile_mode,
        )
        for path in paths
    ]
    for path, existing in zip(paths, existing_units, strict=True):
        if existing is None:
            receipt = main(path, root, layer, profile_mode=profile_mode)
            computed_units += 1
        else:
            receipt = existing
            resumed_units += 1
        if not str(receipt.get("status", "")).startswith("PASS"):
            raise RuntimeError(f"resident QTIP unit failed: {path}")
        ordered_assignments.append(
            {
                "layer": int(receipt["layer"]),
                "expert": int(receipt["expert"]),
                "projection": str(receipt["projection"]),
                "assignment_sha256": str(receipt["assignment_sha256"]),
            }
        )
        unit_receipts.append(receipt)
    batch_wall_seconds = time.perf_counter() - batch_started
    assignment_payload = json.dumps(
        ordered_assignments, separators=(",", ":"), sort_keys=True
    ).encode()
    unit_wall_key = "outer_wall_seconds" if profile_mode else "total_wall_seconds"
    unit_wall_seconds = [float(receipt[unit_wall_key]) for receipt in unit_receipts]
    process = _process_receipt()
    batch = {
        "schema": "banana-smasher-qtip-resident-batch-v1",
        "status": "PASS",
        "host": os.uname().nodename,
        "layer": layer,
        "tier": tier,
        "all_cells": all_cells,
        "mode": "profile" if profile_mode else "solve",
        "fresh_no_warm_start": True,
        "unit_state_isolation": "independent objectives/codebooks/weights/assignments",
        "shared_staging": [
            "python modules",
            "capture bank",
            "hessian manifest binding",
            "TLUT",
            "model index",
        ],
        "process": process,
        "epoch_started": epoch_started,
        "epoch_ended": time.time(),
        "units": len(unit_receipts),
        "resumed_units": resumed_units,
        "computed_units": computed_units,
        "batch_wall_seconds": batch_wall_seconds,
        "mean_public_outer_seconds": batch_wall_seconds / len(unit_receipts),
        "mean_unit_receipt_outer_seconds": sum(unit_wall_seconds) / len(unit_wall_seconds),
        "min_unit_receipt_outer_seconds": min(unit_wall_seconds),
        "max_unit_receipt_outer_seconds": max(unit_wall_seconds),
        "ordered_assignment_sha256": hashlib.sha256(assignment_payload).hexdigest(),
        "ordered_assignment_encoding": "canonical-json-sort-keys-compact-v1",
        "ordered_assignments": ordered_assignments,
        "config_root": str(config_root.resolve()),
        "config_paths": [str(path.resolve()) for path in paths],
    }
    receipt_path = root / "solve" / f"L{layer:03d}" / "QTIP_BATCH_RECEIPT.json"
    _atomic_json(receipt_path, batch)
    print(json.dumps(batch, sort_keys=True), flush=True)
    return batch
