#!/usr/bin/env python3
"""Canonical bounded BASIC repair for the sealed GENESIS physical wire.

Recipe: immutable assignment/codes/scales/native rows; train all live VQ
codebooks plus all RMSNorm masters and 43 attention output log-gains. The
optimizer schedule is the sealed COMBO schedule (Adam, b4, codebook/output
1e-2, norms 1e-4, cosine to 0.1, 64 updates). Checkpoints are durable at every
update and natural clean-72 gates are updates 0,8,...,64.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn.utils import parametrize

from repair_accel import (
    PhaseTimer,
    adam_kwargs,
    configure_determinism,
    cuda_sync,
    ordered_backward,
)

CONTAMINATED_CODE_WINDOWS = frozenset({2, 5, 6, 10})
NATURAL_UPDATES = tuple(range(0, 65, 8))
FORMAT = "genesis-basic-repair-v1"
MECHANISM = "physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains"
GAIN_CLAMP = 0.25


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evict_file_cache(root: str | os.PathLike[str]) -> dict[str, int]:
    """Advise Linux to drop clean package pages after launch identity hashing."""
    advised_files = 0
    advised_bytes = 0
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return {"files": 0, "bytes": 0}
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        size = path.stat().st_size
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)
        advised_files += 1
        advised_bytes += size
    return {"files": advised_files, "bytes": advised_bytes}


def atomic_json(path: str | os.PathLike[str], value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable")


def clean72(code76: Iterable[int]) -> list[int]:
    values = list(code76)
    if len(values) != 76 or len(set(values)) != 76:
        raise ValueError("code76 must contain 76 unique window IDs")
    if not CONTAMINATED_CODE_WINDOWS.issubset(values):
        raise ValueError(
            "code76 is missing one or more known contaminated windows "
            f"{sorted(CONTAMINATED_CODE_WINDOWS)}"
        )
    clean = sorted(set(values) - CONTAMINATED_CODE_WINDOWS)
    if len(clean) != 72:
        raise AssertionError(f"clean-72 cardinality drift: {len(clean)}")
    return clean


def cosine_multiplier(step: int, total_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0,1]")
    clamped = min(max(step, 0), total_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * clamped / total_steps))
    return min_ratio + (1.0 - min_ratio) * cosine


def build_combined_assets(
    *,
    train_corpus: str | os.PathLike[str],
    eval_corpus: str | os.PathLike[str],
    train_refs: str | os.PathLike[str],
    eval_refs: str | os.PathLike[str],
    out_corpus: str | os.PathLike[str],
    out_refs: str | os.PathLike[str],
    required_train: Iterable[int],
    required_eval: Iterable[int],
) -> dict[str, object]:
    train_corpus = Path(train_corpus).resolve()
    eval_corpus = Path(eval_corpus).resolve()
    train_refs = Path(train_refs).resolve()
    eval_refs = Path(eval_refs).resolve()
    out_corpus = Path(out_corpus)
    out_refs = Path(out_refs)
    train_rows = json.loads(train_corpus.read_text())
    eval_rows = json.loads(eval_corpus.read_text())
    if not isinstance(train_rows, list) or not isinstance(eval_rows, list):
        raise ValueError("corpora must be JSON arrays")
    required_train = sorted(set(map(int, required_train)))
    required_eval = sorted(set(map(int, required_eval)))
    if any(index < 0 or index >= len(train_rows) for index in required_train):
        raise ValueError("required train index outside corpus")
    if any(index < 0 or index >= len(eval_rows) for index in required_eval):
        raise ValueError("required eval index outside corpus")

    out_corpus.parent.mkdir(parents=True, exist_ok=True)
    out_corpus.write_text(json.dumps(train_rows + eval_rows, separators=(",", ":")) + "\n")
    out_refs.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, object]] = []
    for source_index, combined_index, root in [
        *[(index, index, train_refs) for index in required_train],
        *[(index, len(train_rows) + index, eval_refs) for index in required_eval],
    ]:
        source = root / f"t8192_win{source_index}.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = out_refs / f"t8192_win{combined_index}.pt"
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"combined teacher link drift: {destination}")
        else:
            destination.symlink_to(source)
        links.append(
            {
                "combined_win": combined_index,
                "source_win": source_index,
                "source": str(source),
                "source_bytes": source.stat().st_size,
                "source_sha256": sha256_file(source),
            }
        )
    return {
        "schema": "genesis-basic-combined-assets-v1",
        "train_count": len(train_rows),
        "eval_count": len(eval_rows),
        "combined_count": len(train_rows) + len(eval_rows),
        "train_corpus": str(train_corpus),
        "train_corpus_sha256": sha256_file(train_corpus),
        "eval_corpus": str(eval_corpus),
        "eval_corpus_sha256": sha256_file(eval_corpus),
        "combined_corpus": str(out_corpus.resolve()),
        "combined_corpus_sha256": sha256_file(out_corpus),
        "required_ref_links": len(links),
        "links": links,
    }


def select_terminal_checkpoint(
    rows: Iterable[dict[str, object]], *, natural_updates: Iterable[int] = NATURAL_UPDATES[1:]
) -> dict[str, object]:
    expected = list(map(int, natural_updates))
    by_update = {int(row["update"]): dict(row) for row in rows}
    if sorted(by_update) != sorted(expected):
        raise ValueError(
            f"clean72 gate updates mismatch: got={sorted(by_update)} expected={sorted(expected)}"
        )
    for row in by_update.values():
        value = float(row["clean72_kld"])
        if not math.isfinite(value):
            raise ValueError("non-finite clean72 KLD")
    selected = min(by_update.values(), key=lambda row: (float(row["clean72_kld"]), int(row["update"])))
    selected["selection_metric"] = "minimum_clean72_kld_over_predeclared_natural_updates"
    selected["natural_updates"] = expected
    return selected


class WireBf16(nn.Module):
    def forward(self, master: torch.Tensor) -> torch.Tensor:
        return master.to(torch.bfloat16)


def _output_gain_hook(module, _inputs, output):
    gain = torch.exp(
        module._genesis_output_log_gain.clamp(-GAIN_CLAMP, GAIN_CLAMP)
    ).to(output.dtype)
    return output * gain


def attach_output_gain(module: nn.Module) -> nn.Parameter:
    wire = module._parameters.get("weight")
    if wire is None:
        raise RuntimeError("o_b_proj has no weight parameter")
    if hasattr(module, "_genesis_output_log_gain"):
        raise RuntimeError("GENESIS output gain already attached")
    gain = nn.Parameter(torch.zeros((), dtype=torch.float32, device=wire.device))
    module.register_parameter("_genesis_output_log_gain", gain)
    module.register_forward_hook(_output_gain_hook)
    return gain


def expose_dense_parameters(student):
    norms = []
    outputs = []
    modules = list(student.model.named_modules())
    for name, module in modules:
        leaf = name.rsplit(".", 1)[-1].lower()
        wire = module._parameters.get("weight")
        if "norm" not in leaf or wire is None or wire.ndim != 1:
            continue
        before = wire.detach().clone()
        parametrize.register_parametrization(module, "weight", WireBf16(), unsafe=True)
        master = module.parametrizations.weight.original
        master.data = master.data.float()
        master.requires_grad_(True)
        if module.weight.dtype != torch.bfloat16 or not torch.equal(module.weight.detach(), before):
            raise AssertionError(f"RMSNorm identity changed at {name}")
        norms.append((name, module, master))
    for name, module in modules:
        if name.endswith(".self_attn.o_b_proj"):
            outputs.append((name + ".output_log_gain", module, attach_output_gain(module)))
    norm_count = sum(parameter.numel() for _name, _module, parameter in norms)
    if norm_count != 446080 or len(outputs) != 43:
        raise RuntimeError(f"canonical dense surface drift: norms={norm_count} outputs={len(outputs)}")
    return norms, outputs


def dense_state(norms, outputs) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "norms": {name: parameter.detach().cpu().clone() for name, _module, parameter in norms},
        "outputs": {name: parameter.detach().cpu().clone() for name, _module, parameter in outputs},
    }


def load_dense_state(norms, outputs, state, device="cuda") -> None:
    norm_map = {name: parameter for name, _module, parameter in norms}
    output_map = {name: parameter for name, _module, parameter in outputs}
    if set(norm_map) != set(state["norms"]) or set(output_map) != set(state["outputs"]):
        raise RuntimeError("GENESIS dense state key drift")
    for name, parameter in norm_map.items():
        parameter.data.copy_(state["norms"][name].to(device))
    for name, parameter in output_map.items():
        parameter.data.copy_(state["outputs"][name].to(device))


def _stat_binding(path: Path) -> dict[str, object]:
    stat = path.stat()
    if not path.is_file():
        raise RuntimeError(f"frozen payload is not a regular file: {path}")
    return {
        "bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _stat_fingerprint(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def physical_identity(
    package: Path,
    assignment: Path,
    *,
    expected_assignment_sha256: str | None = None,
) -> dict[str, object]:
    package = package.resolve()
    assignment = assignment.resolve()
    assignment_sha = sha256_file(assignment)
    if expected_assignment_sha256 is not None and assignment_sha != expected_assignment_sha256:
        raise RuntimeError(
            f"assignment SHA drift: {assignment_sha} != {expected_assignment_sha256}"
        )
    rows: list[dict[str, object]] = []
    stat_rows: list[dict[str, object]] = []
    payload_bytes = 0
    for layer in range(43):
        layer_dir = package / f"layer_{layer:03d}"
        receipt = layer_dir / "LAYER_RECEIPT.json"
        if not receipt.is_file():
            raise FileNotFoundError(receipt)
        receipt_value = json.loads(receipt.read_text())
        expected_receipt = {
            "schema": "genesis-materialized-layer-v1",
            "status": "PASS",
            "layer": layer,
            "assignment_sha256": assignment_sha,
        }
        receipt_drift = {
            key: (receipt_value.get(key), expected)
            for key, expected in expected_receipt.items()
            if receipt_value.get(key) != expected
        }
        if receipt_drift:
            raise RuntimeError(f"L{layer:03d} receipt identity drift: {receipt_drift}")
        listed = receipt_value.get("files")
        if not isinstance(listed, list) or not listed:
            raise RuntimeError(f"L{layer:03d} receipt has no payload file list")
        files: list[dict[str, object]] = []
        file_stats: list[dict[str, object]] = []
        seen: set[str] = set()
        for binding in listed:
            if not isinstance(binding, dict):
                raise RuntimeError(f"L{layer:03d} malformed payload binding")
            relative = str(binding.get("path", ""))
            if not relative or relative in seen:
                raise RuntimeError(f"L{layer:03d} duplicate/empty payload path: {relative!r}")
            seen.add(relative)
            path = (layer_dir / relative).resolve()
            if path.parent != layer_dir.resolve():
                raise RuntimeError(f"L{layer:03d} payload escapes layer directory: {relative}")
            stat_binding = _stat_binding(path)
            actual_sha = sha256_file(path)
            if (
                stat_binding["bytes"] != binding.get("bytes")
                or actual_sha != binding.get("sha256")
            ):
                raise RuntimeError(
                    f"L{layer:03d} payload binding drift {relative}: "
                    f"bytes={stat_binding['bytes']}/{binding.get('bytes')} "
                    f"sha={actual_sha}/{binding.get('sha256')}"
                )
            payload_bytes += int(stat_binding["bytes"])
            files.append(
                {
                    "path": relative,
                    "bytes": stat_binding["bytes"],
                    "sha256": actual_sha,
                }
            )
            file_stats.append({"path": relative, **stat_binding})
        rows.append(
            {
                "layer": layer,
                "receipt_sha256": sha256_file(receipt),
                "files": files,
            }
        )
        stat_rows.append(
            {
                "layer": layer,
                "receipt": _stat_binding(receipt),
                "files": file_stats,
            }
        )
    identity = {
        "assignment": str(assignment),
        "assignment_sha256": assignment_sha,
        "assignment_stat": _stat_binding(assignment),
        "physical_package": str(package),
        "layers": rows,
        "payload_bytes": payload_bytes,
        "payload_surface_sha256": _stat_fingerprint(rows),
        "stat_fingerprint_sha256": _stat_fingerprint(
            {"assignment": _stat_binding(assignment), "layers": stat_rows}
        ),
    }
    return identity


def assert_physical_stat_identity(
    package: Path, assignment: Path, identity: dict[str, object]
) -> None:
    package = package.resolve()
    assignment = assignment.resolve()
    base_assignment = Path(str(identity["base_assignment"])).resolve()
    if str(package) != identity.get("physical_package") or str(assignment) != identity.get("assignment"):
        raise RuntimeError("physical stat identity path drift")
    assignment_doc = json.loads(assignment.read_text())
    assignment_map_sha = _stat_fingerprint(assignment_doc.get("assignment"))
    if (
        sha256_file(assignment) != identity.get("assignment_file_sha256")
        or assignment_doc.get("assignment_sha256") != identity.get("assignment_sha256")
        or assignment_doc.get("assignment_map_sha256") != identity.get("assignment_map_sha256")
        or assignment_map_sha != identity.get("assignment_map_sha256")
    ):
        raise RuntimeError("F521 target assignment SHA drift")
    if _stat_binding(assignment) != identity.get("assignment_stat"):
        raise RuntimeError("F521 target assignment stat drift")
    stat_rows: list[dict[str, object]] = []
    for layer_row in identity["layers"]:
        layer = int(layer_row["layer"])
        layer_dir = package / f"layer_{layer:03d}"
        receipt = layer_dir / "LAYER_RECEIPT.json"
        file_stats = [
            {"path": binding["path"], **_stat_binding(layer_dir / str(binding["path"]))}
            for binding in layer_row["files"]
        ]
        stat_rows.append(
            {"layer": layer, "receipt": _stat_binding(receipt), "files": file_stats}
        )
    current = _stat_fingerprint(
        {"assignment": _stat_binding(base_assignment), "layers": stat_rows}
    )
    if current != identity.get("base_stat_fingerprint_sha256"):
        raise RuntimeError(
            "immutable base-package stat identity drift: "
            f"{current} != {identity.get('base_stat_fingerprint_sha256')}"
        )


def validate_claim(
    path: Path,
    *,
    expected_sha256: str,
    expected_task: str,
    expected_host: str,
    expected_mission: str,
    now: float | None = None,
) -> dict[str, object]:
    """Validate the fleet's canonical spark-host-claim-v2 lease exactly."""
    now = time.time() if now is None else float(now)
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    claim = json.loads(raw)
    expected = {
        "schema": "spark-host-claim-v2",
        "owner": expected_task,
        "task": expected_task,
        "task_id": expected_task,
        "host": expected_host,
        "mission": expected_mission,
        "mission_root": expected_mission,
        "state": "CLAIMED",
        "status": "CLAIMED",
    }
    drift = {
        key: (claim.get(key), value)
        for key, value in expected.items()
        if claim.get(key) != value
    }
    if actual_sha != expected_sha256:
        drift["sha256"] = (actual_sha, expected_sha256)
    for key in ("claimed_unix", "lease_until_unix"):
        value = claim.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            drift[key] = (value, "finite epoch")
    if isinstance(claim.get("claimed_unix"), (int, float)) and float(claim["claimed_unix"]) > now:
        drift["claimed_unix"] = (claim["claimed_unix"], f"<= {now}")
    if isinstance(claim.get("lease_until_unix"), (int, float)) and float(claim["lease_until_unix"]) <= now:
        drift["lease_until_unix"] = (claim["lease_until_unix"], f"> {now}")
    for key in ("exact_cas_from_sha256", "previous_claim_sha256"):
        value = claim.get(key)
        if not isinstance(value, str) or len(value) != 64:
            drift[key] = (value, "64-char SHA256")
    if drift:
        raise RuntimeError(f"claim identity drift: {drift}")
    return claim


def _load_base(path: Path):
    spec = importlib.util.spec_from_file_location("genesis_binrepair_base", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_payload(
    *, B, surface, student, norms, outputs, optimizer, scheduler, next_update: int,
    identity: dict[str, object], config: dict[str, object], microprobes: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "next_update": next_update,
        "identity": identity,
        "config": config,
        "state": {
            "codebooks": surface.surface_state(student),
            **dense_state(norms, outputs),
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "microprobes": microprobes,
        "saved_unix": time.time(),
        "host": os.uname().nodename,
    }


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary = Path(str(path) + f".{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_latest_link(latest: Path, checkpoint: Path) -> None:
    """Atomically bind LATEST to an already-durable immutable checkpoint."""
    if checkpoint.parent != latest.parent:
        raise RuntimeError("LATEST and immutable checkpoint must share a directory")
    temporary = latest.with_name(f".{latest.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(checkpoint, temporary)
        os.replace(temporary, latest)
    finally:
        temporary.unlink(missing_ok=True)
    checkpoint_stat = checkpoint.stat()
    latest_stat = latest.stat()
    if (
        checkpoint_stat.st_dev != latest_stat.st_dev
        or checkpoint_stat.st_ino != latest_stat.st_ino
        or checkpoint_stat.st_size != latest_stat.st_size
    ):
        raise RuntimeError("LATEST hard-link identity verification failed")
    directory_fd = os.open(latest.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def recover_latest_checkpoint(
    checkpoints: Path,
    *,
    identity: dict[str, object],
    config: dict[str, object],
) -> Path | None:
    """Promote the highest checkpoint with a durable, hash-valid sidecar.

    An immutable .pt without its atomic .json marker is incomplete and is ignored;
    a marker that exists but fails validation is corruption and fails closed.
    """
    complete: list[tuple[int, Path]] = []
    for sidecar_path in sorted(checkpoints.glob("UPDATE_[0-9][0-9][0-9].json")):
        try:
            sidecar = json.loads(sidecar_path.read_text())
        except Exception as exc:
            raise RuntimeError(f"invalid checkpoint sidecar {sidecar_path}: {exc}") from exc
        try:
            filename_update = int(sidecar_path.stem.split("_")[1])
            update = int(sidecar["update"])
        except Exception as exc:
            raise RuntimeError(f"checkpoint sidecar update malformed: {sidecar_path}") from exc
        checkpoint = checkpoints / f"UPDATE_{update:03d}.pt"
        expected = {
            "schema": "genesis-basic-checkpoint-v1",
            "update": filename_update,
            "checkpoint": str(checkpoint.resolve()),
            "identity": identity,
        }
        drift = {
            key: (sidecar.get(key), value)
            for key, value in expected.items()
            if sidecar.get(key) != value
        }
        if update != filename_update:
            drift["filename_update"] = (filename_update, update)
        if not checkpoint.is_file():
            drift["checkpoint_exists"] = (False, True)
        else:
            actual_sha = sha256_file(checkpoint)
            if actual_sha != sidecar.get("checkpoint_sha256"):
                drift["checkpoint_sha256"] = (
                    actual_sha, sidecar.get("checkpoint_sha256")
                )
        if drift:
            raise RuntimeError(f"incomplete/corrupt checkpoint sidecar {sidecar_path}: {drift}")
        complete.append((update, checkpoint))
    if not complete:
        return None
    update, checkpoint = max(complete)
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    expected_payload = {
        "format": FORMAT,
        "mechanism": MECHANISM,
        "next_update": update,
        "identity": identity,
        "config": config,
    }
    drift = {
        key: (payload.get(key), value)
        for key, value in expected_payload.items()
        if payload.get(key) != value
    }
    if drift:
        raise RuntimeError(f"checkpoint payload identity drift {checkpoint}: {list(drift)}")
    _atomic_latest_link(checkpoints / "LATEST.pt", checkpoint)
    return checkpoint


def main() -> int:
    from genesis_physical_surface import (
        GenesisPhysicalExperts,
        load_surface_state,
        surface_parameters,
    )
    import genesis_physical_surface as surface
    import f521_repair_overlay as f521_overlay
    import f521_rolling_stage as f521_stage

    acceleration = {
        "determinism": configure_determinism(),
        "fused_adam": bool(adam_kwargs()["fused"]),
        "dequant_chunk": int(os.environ.get("GENESIS_REPAIR_DEQ_CHUNK", "4")),
        "native_chunk": int(os.environ.get("GENESIS_REPAIR_NATIVE_CHUNK", "2")),
        "evict_after_use": os.environ.get("GENESIS_REPAIR_EVICT", "1") == "1",
        "activation_checkpoint": os.environ.get("GENESIS_REPAIR_CHECKPOINT", "1") == "1",
        "packed_code_decode": "device-gather-v1",
        "compiled_vq_dequant": os.environ.get("GENESIS_REPAIR_COMPILE_VQ", "0") == "1",
        "compiled_vq_scope": "disabled-dense-dequant-replaced-by-fused-expert-linear",
        "fused_packed_expert_linear": True,
        "fused_packed_expert_scope": "tile-SRAM VQ/MXFP4 dequantize-plus-linear with custom grad-input",
        "dense_expert_weight_materialized_in_forward": False,
        "microbatch": int(os.environ.get("GENESIS_REPAIR_MICROBATCH", "1")),
        "expert_resident_scope": int(
            os.environ.get("GENESIS_REPAIR_EXPERT_RESIDENT_SCOPE", "4")
        ),
        "mem_floor_bytes": int(
            os.environ.get("GENESIS_REPAIR_MEM_FLOOR_BYTES", str(8 * 1024**3))
        ),
        "gradient_accumulation": "sequential-weighted-microbatch-roots",
        "deterministic_codebook_reduction_required": (
            os.environ.get("GENESIS_REPAIR_REQUIRE_DETERMINISTIC_REDUCTION", "0") == "1"
        ),
        "backward": "ordered-separate-calls-single-incumbent-root",
        "teacher_targets": "existing-banked-targets",
        "update0_smoke_only": os.environ.get("GENESIS_REPAIR_UPDATE0_SMOKE_ONLY", "0") == "1",
    }
    root = Path(os.environ["GENESIS_REPAIR_ROOT"]).resolve()
    package = Path(os.environ["GENESIS_PHYSICAL_PACKAGE"]).resolve()
    assignment = Path(os.environ["GENESIS_ASSIGNMENT"]).resolve()
    base_assignment = Path(os.environ["GENESIS_BASE_ASSIGNMENT"]).resolve()
    base_path = Path(os.environ["COMBO_BINREPAIR_BASE"]).resolve()
    config_path = Path(os.environ["GENESIS_REPAIR_CONFIG"]).resolve()
    config = json.loads(config_path.read_text())
    if config.get("format") != FORMAT:
        raise RuntimeError("GENESIS BASIC config format drift")
    steps = int(config["steps"])
    stop_after_update = int(os.environ.get("GENESIS_REPAIR_STOP_AFTER_UPDATE", str(steps)))
    batch = int(config["batch"])
    microbatch = int(acceleration["microbatch"])
    probe_every = int(config["probe_every"])
    if steps != 64 or batch != 4 or probe_every != 8:
        raise RuntimeError("noncanonical BASIC schedule")
    if microbatch <= 0 or microbatch > batch or batch % microbatch:
        raise RuntimeError(
            f"GENESIS_REPAIR_MICROBATCH must divide canonical batch {batch}: {microbatch}"
        )
    if not acceleration["activation_checkpoint"]:
        raise RuntimeError("reduced-resident arm requires activation checkpointing")
    if not 1 <= int(acceleration["expert_resident_scope"]) <= 16:
        raise RuntimeError("expert resident scope must be in [1,16]")
    if not 0 < stop_after_update <= steps:
        raise RuntimeError(f"invalid GENESIS_REPAIR_STOP_AFTER_UPDATE={stop_after_update}")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)

    physical_receipt_path = Path(config["physical_code76_receipt"]).resolve()
    physical_receipt_sha = sha256_file(physical_receipt_path)
    if physical_receipt_sha != config["physical_code76_receipt_sha256"]:
        raise RuntimeError("physical code76 receipt SHA drift")
    physical_receipt = json.loads(physical_receipt_path.read_text())
    expected_claim_sha = os.environ.get(
        "GENESIS_REPAIR_EXPECTED_CLAIM_SHA256",
        physical_receipt.get("claim_sha256", ""),
    )
    if not isinstance(expected_claim_sha, str) or len(expected_claim_sha) != 64:
        raise RuntimeError("no exact claim SHA binding")
    claim_path = Path(os.environ.get("GENESIS_HOST_CLAIM", "${SPARK_HOME}/HOST_CLAIM.json")).resolve()
    task_id = os.environ.get("GENESIS_TASK_ID", "task-redacted")
    claim_mission = os.environ.get("GENESIS_REPAIR_CLAIM_MISSION", str(root))
    validate_claim(
        claim_path,
        expected_sha256=expected_claim_sha,
        expected_task=task_id,
        expected_host=str(config["host"]),
        expected_mission=claim_mission,
    )
    # Full immutable-package hashes are paid once at launch. F521 payload hashes
    # are checked on first use by the dedicated adapter/rolling stage. During the
    # run, the package/assignment stat fingerprint, sealed inputs, and exact claim
    # are checked before and after every optimizer/checkpoint boundary.
    bound_paths = {
        "teacher_all40": Path(os.environ["GENESIS_F521_TEACHERS_ALL40"]).resolve(),
        "teacher9_closure": Path(os.environ["GENESIS_F521_TEACHER9_RECEIPT"]).resolve(),
        "codebook_receipt": Path(os.environ["GENESIS_F521_CODEBOOK_RECEIPT"]).resolve(),
        "native_receipt": Path(os.environ["GENESIS_F521_NATIVE_MEMBERS_RECEIPT"]).resolve(),
        "approval": Path(os.environ["GENESIS_F521_APPROVAL"]).resolve(),
        "approval_validation": Path(os.environ["GENESIS_F521_APPROVAL_VALIDATION"]).resolve(),
        "update_plan": Path(os.environ["GENESIS_F521_UPDATE_PLAN"]).resolve(),
    }
    expected_bound_shas = {
        "teacher_all40": "3420daec1828ae03f9872b0bdfffc4819fc248ced25254aae86326a815e540fc",
        "teacher9_closure": "722d8a0a2dc442ab0ea172b6a1c7d02d35a8ae190a1ddb44b2491ab14dc76d70",
        "codebook_receipt": "4dd17e2acacaba14ff104d4a05e65a8c3f6303e51684c06a3c5962212023acca",
        "native_receipt": "b9e38bced59c45a8545488217ddd6321a3e64c709b459f76e317b01e667e6a9f",
        "approval": "6cdd2baa22ad51e2b79297933ef4b28d528c8e36480b4405860c961d6e2fe4bb",
        "approval_validation": "2b3da4a99f02666f523c6be9e42656f1afdf9b45e0b267f04a09f23e61a0386b",
        "update_plan": "357dc1d57c81ec5eac0ea7e80258b6a9dada9a42df3400b68d7c40ddb234a196",
    }
    observed_bound_shas = {name: sha256_file(path) for name, path in bound_paths.items()}
    drift = {name: (observed_bound_shas[name], wanted) for name, wanted in expected_bound_shas.items() if observed_bound_shas[name] != wanted}
    if drift:
        raise RuntimeError(f"F521 sealed prerequisite drift: {drift}")
    identity = physical_identity(
        package,
        base_assignment,
        expected_assignment_sha256=(
            "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d"
        ),
    )
    assignment_doc = json.loads(assignment.read_text())
    target_assignment_sha = str(assignment_doc.get("assignment_sha256", ""))
    target_assignment_map_sha = str(assignment_doc.get("assignment_map_sha256", ""))
    target_assignment_file_sha = sha256_file(assignment)
    canonical_assignment_map_sha = _stat_fingerprint(assignment_doc.get("assignment"))
    expected_f521 = {
        "changed_cells": 21_472,
        "qtip3_cells": 14_979,
        "qtip2_cells": 2_266,
        "vq_cells": 4_173,
        "native_cells": 54,
        "unique_vq_codebooks": 184,
        "cache_max_bytes": 20 * 1024**3,
        "streams": 8,
        "resource_floor_bytes": 8 * 1024**3,
        "pack_fraction": 1.0,
        "restored_manifest_sha256": f521_overlay.F521_MANIFEST_SHA,
        "source_plan_sha256": f521_overlay.SOURCE_PLAN_SHA,
        "base_assignment_sha256": f521_overlay.BASE_ASSIGNMENT_SHA,
        "assignment_map_sha256": f521_overlay.ASSIGNMENT_MAP_SHA,
    }
    f521_config = config.get("f521_repair")
    f521_drift = {
        key: (f521_config.get(key) if isinstance(f521_config, dict) else None, value)
        for key, value in expected_f521.items()
        if not isinstance(f521_config, dict) or f521_config.get(key) != value
    }
    assignment_drift = {
        "schema": (assignment_doc.get("schema"), "wire-c-v2-global-reconstructed-assignment-v1"),
        "status": (assignment_doc.get("status"), "PASS"),
        "assignment_sha256": (target_assignment_sha, str(config["assignment_sha256"])),
        "assignment_map_sha256": (target_assignment_map_sha, f521_overlay.ASSIGNMENT_MAP_SHA),
        "canonical_assignment_map_sha256": (canonical_assignment_map_sha, f521_overlay.ASSIGNMENT_MAP_SHA),
        "base_assignment_sha256": (assignment_doc.get("base_assignment_sha256"), f521_overlay.BASE_ASSIGNMENT_SHA),
        "changed_cells_applied": (assignment_doc.get("changed_cells_applied"), 21_472),
        "total_cells": (assignment_doc.get("total_cells"), 22_016),
        "unchanged_cells_inherited": (assignment_doc.get("unchanged_cells_inherited"), 544),
    }
    assignment_drift = {key: value for key, value in assignment_drift.items() if value[0] != value[1]}
    if f521_drift or assignment_drift:
        raise RuntimeError(f"F521 config/assignment drift: config={f521_drift} assignment={assignment_drift}")
    identity["base_assignment"] = identity.pop("assignment")
    identity["base_assignment_sha256"] = identity.pop("assignment_sha256")
    identity["base_assignment_stat"] = identity.pop("assignment_stat")
    identity["base_stat_fingerprint_sha256"] = identity.pop(
        "stat_fingerprint_sha256"
    )
    identity["assignment"] = str(assignment)
    identity["assignment_sha256"] = target_assignment_sha
    identity["assignment_map_sha256"] = target_assignment_map_sha
    identity["assignment_file_sha256"] = target_assignment_file_sha
    identity["assignment_stat"] = _stat_binding(assignment)
    identity["f521_source_index"] = surface.source_index_identity()
    identity["exact_logical_wire_bytes"] = 105_971_968_682
    identity["overlay_changed_cells"] = 21_472
    identity["overlay_qtip3_cells"] = 14_979
    identity["overlay_qtip2_cells"] = 2_266
    identity["overlay_vq_cells"] = 4_173
    identity["overlay_native_cells"] = 54
    identity["pack_fraction"] = 1.0
    identity["zero_substitution"] = True
    identity["zero_quarantine"] = True
    identity["claim_sha256"] = expected_claim_sha
    identity["claim_path"] = str(claim_path)
    identity["sealed_prerequisites"] = {
        name: {"path": str(bound_paths[name]), "sha256": observed_bound_shas[name]}
        for name in sorted(bound_paths)
    }
    identity["physical_pre_repair_receipt"] = str(physical_receipt_path)
    identity["physical_pre_repair_receipt_sha256"] = physical_receipt_sha
    identity["config"] = str(config_path)
    identity["config_sha256"] = sha256_file(config_path)
    identity["base_harness"] = str(base_path)
    identity["base_harness_sha256"] = sha256_file(base_path)
    identity["trainer_code"] = str(Path(__file__).resolve())
    identity["trainer_code_sha256"] = sha256_file(Path(__file__))
    identity["surface_code"] = str(Path(surface.__file__).resolve())
    identity["surface_sha256"] = sha256_file(Path(surface.__file__))
    identity["f521_overlay_code"] = str(Path(f521_overlay.__file__).resolve())
    identity["f521_overlay_code_sha256"] = sha256_file(Path(f521_overlay.__file__))
    identity["f521_stage_code"] = str(Path(f521_stage.__file__).resolve())
    identity["f521_stage_code_sha256"] = sha256_file(Path(f521_stage.__file__))
    def assert_runtime_guard() -> None:
        validate_claim(
            claim_path,
            expected_sha256=expected_claim_sha,
            expected_task=task_id,
            expected_host=str(config["host"]),
            expected_mission=claim_mission,
        )
        assert_physical_stat_identity(package, assignment, identity)
        if sha256_file(physical_receipt_path) != physical_receipt_sha:
            raise RuntimeError("F521 physical pre-repair receipt byte drift")
        for name, wanted in expected_bound_shas.items():
            if sha256_file(bound_paths[name]) != wanted:
                raise RuntimeError(f"F521 sealed prerequisite byte drift: {name}")
        for path_key, sha_key in (
            ("config", "config_sha256"),
            ("base_harness", "base_harness_sha256"),
            ("trainer_code", "trainer_code_sha256"),
            ("surface_code", "surface_sha256"),
            ("f521_overlay_code", "f521_overlay_code_sha256"),
            ("f521_stage_code", "f521_stage_code_sha256"),
        ):
            current = sha256_file(Path(str(identity[path_key])))
            if current != identity[sha_key]:
                raise RuntimeError(f"F521 runtime code/config byte drift: {path_key}")
        surface.assert_source_index_stat_identity(identity["f521_source_index"])
        floor = int(config["f521_repair"]["resource_floor_bytes"])
        root_free = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize
        mem_free = mem_available_bytes()
        if root_free < floor or mem_free < floor:
            raise RuntimeError(f"F521 runtime resource floor root={root_free} mem={mem_free} floor={floor}")
    acceleration["launch_cache_evict"] = evict_file_cache(package)
    B = _load_base(base_path)
    B.T.TrainableExperts = GenesisPhysicalExperts
    B.T.PILOT = tuple(range(43))
    student = B.T.Student()
    acceleration["compiled_vq_warmup"] = surface.warm_compiled_vq_dequant()
    from fused_expert_linear import warm_fused_expert_linear
    acceleration["fused_expert_warmup"] = warm_fused_expert_linear()
    norms, outputs = expose_dense_parameters(student)
    codebooks = surface_parameters(student)
    norm_params = [parameter for _name, _module, parameter in norms]
    output_params = [parameter for _name, _module, parameter in outputs]
    optimizer = torch.optim.Adam(
        [
            {"params": codebooks, "lr": 1e-2, "group_name": "codebooks"},
            {"params": norm_params, "lr": 1e-4, "group_name": "norms"},
            {"params": output_params, "lr": 1e-2, "group_name": "outputs"},
        ],
        **adam_kwargs(),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[lambda update: cosine_multiplier(update, steps, 0.1)] * 3,
    )
    all_params = codebooks + norm_params + output_params
    identity["n_codebook_params"] = sum(parameter.numel() for parameter in codebooks)
    identity["n_norm_params"] = sum(parameter.numel() for parameter in norm_params)
    identity["n_output_params"] = sum(parameter.numel() for parameter in output_params)
    identity["n_trainable_params"] = sum(parameter.numel() for parameter in all_params)
    assert_runtime_guard()

    checkpoints = root / "checkpoints"
    logs = root / "logs"
    receipts = root / "receipts"
    for path in (checkpoints, logs, receipts):
        path.mkdir(parents=True, exist_ok=True)
    jlog = logs / "BASIC_REPAIR.jsonl"
    phase_log = logs / "UPDATE_PHASES.jsonl"
    status_path = root / "run/BASIC_REPAIR_STATUS.json"
    latest = checkpoints / "LATEST.pt"
    microprobes: list[dict[str, object]] = []

    def emit(**row):
        row.setdefault("unix", time.time())
        with jlog.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def emit_phase(*, update: int, phase: str, seconds: float, **fields) -> None:
        fields.setdefault("mem_available_after_bytes", mem_available_bytes())
        fields.setdefault(
            "mem_available_after_gib",
            fields["mem_available_after_bytes"] / 1024**3,
        )
        row = {
            "event": "update_phase",
            "update": int(update),
            "phase": str(phase),
            "seconds": float(seconds),
            "unix": time.time(),
            **fields,
        }
        with phase_log.open("a") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    mem_floor = int(acceleration["mem_floor_bytes"])

    def guard_memory(where: str) -> int:
        before = mem_available_bytes()
        if before >= mem_floor:
            return before
        torch.cuda.empty_cache()
        cuda_sync()
        after = mem_available_bytes()
        receipt = {
            "schema": "repair-reduced-resident-memory-floor-stop-v1",
            "task_id": os.environ.get("GENESIS_TASK_ID"),
            "where": where,
            "mem_floor_bytes": mem_floor,
            "mem_available_before_empty_cache_bytes": before,
            "mem_available_after_empty_cache_bytes": after,
            "preserved_checkpoint": os.environ.get("GENESIS_REPAIR_CANARY_SEED"),
            "stopped_unix": time.time(),
        }
        atomic_json(receipts / "MEMORY_FLOOR_STOP.json", receipt)
        raise RuntimeError(
            f"MEMORY_FLOOR_STOP at {where}: MemAvailable={after} floor={mem_floor}"
        )

    def write_status(**fields):
        current = json.loads(status_path.read_text()) if status_path.is_file() else {}
        current.update(fields)
        current["updated_unix"] = time.time()
        atomic_json(status_path, current)

    corpus = B.T.load_corpus()
    acache = B.ActCache(student)
    start_update = 0
    recovered = recover_latest_checkpoint(checkpoints, identity=identity, config=config)
    canary_seed_path = os.environ.get("GENESIS_REPAIR_CANARY_SEED")
    state_seed_path = os.environ.get("GENESIS_REPAIR_STATE_SEED")
    if recovered is not None:
        checkpoint = torch.load(recovered, map_location="cpu", mmap=True, weights_only=False)
        load_surface_state(student, checkpoint["state"]["codebooks"])
        load_dense_state(norms, outputs, checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_update = int(checkpoint["next_update"])
        microprobes = list(checkpoint.get("microprobes", []))
        assert_runtime_guard()
        emit(
            event="resume", next_update=start_update, checkpoint=str(recovered),
            acceleration=acceleration,
        )
    elif state_seed_path:
        # P600 is a new dose: inherit only UPDATE_030 model parameters while
        # deliberately resetting optimizer/scheduler/update numbering.
        seed_path = Path(state_seed_path).expanduser().resolve()
        seed = torch.load(seed_path, map_location="cpu", mmap=True, weights_only=False)
        seed_drift = {}
        for key, expected in (("format", FORMAT), ("mechanism", MECHANISM)):
            if seed.get(key) != expected:
                seed_drift[key] = (seed.get(key), expected)
        for key in ("assignment_sha256", "payload_surface_sha256"):
            if seed.get("identity", {}).get(key) != identity.get(key):
                seed_drift[f"identity.{key}"] = (
                    seed.get("identity", {}).get(key), identity.get(key)
                )
        if seed_drift:
            raise RuntimeError(f"state seed identity drift: {seed_drift}")
        saved_codebooks = seed["state"]["codebooks"]
        terminal_seed_extension = {}
        missing_saved_codebooks = {}
        for layer in range(43):
            label = f"L{layer}"
            live_keys = {name for name, _parameter in student.experts[layer].named_codebooks()}
            saved_keys = set(saved_codebooks[label])
            extra = sorted(live_keys - saved_keys)
            missing = sorted(saved_keys - live_keys)
            if extra:
                terminal_seed_extension[label] = extra
            if missing:
                missing_saved_codebooks[label] = missing
        allow_terminal_extension = (
            os.environ.get("GENESIS_REPAIR_ALLOW_TERMINAL_SEED_EXTENSION", "0") == "1"
        )
        if missing_saved_codebooks:
            raise RuntimeError(
                f"canonical seed codebooks absent from terminal surface: {missing_saved_codebooks}"
            )
        if terminal_seed_extension and not allow_terminal_extension:
            raise RuntimeError(
                f"terminal seed extension requires explicit gate: {terminal_seed_extension}"
            )
        load_surface_state(student, saved_codebooks)
        load_dense_state(norms, outputs, seed["state"])
        start_update = 0
        microprobes = []
        seed_receipt = {
            "schema": "genesis-basic-state-only-dose-seed-v1",
            "seed": str(seed_path),
            "seed_sha256": sha256_file(seed_path),
            "source_next_update": int(seed["next_update"]),
            "new_next_update": 0,
            "state_loaded": True,
            "optimizer_loaded": False,
            "scheduler_loaded": False,
            "terminal_seed_extension_enabled": allow_terminal_extension,
            "terminal_seed_extension_policy": "load every canonical UPDATE_000 key; retain exact terminal-wire initialization only for newly introduced live keys",
            "terminal_seed_extension": terminal_seed_extension,
            "terminal_seed_extension_layers": len(terminal_seed_extension),
            "terminal_seed_extension_keys": sum(map(len, terminal_seed_extension.values())),
            "missing_saved_codebooks": missing_saved_codebooks,
            "speculative_seed_used": False,
            "source_config_sha256": seed.get("identity", {}).get("config_sha256"),
            "new_config_sha256": identity.get("config_sha256"),
            "acceleration": acceleration,
        }
        atomic_json(receipts / "STATE_ONLY_SEED.json", seed_receipt)
        payload = _checkpoint_payload(
            B=B, surface=surface, student=student, norms=norms, outputs=outputs,
            optimizer=optimizer, scheduler=scheduler, next_update=0,
            identity=identity, config=config, microprobes=microprobes,
        )
        step0 = checkpoints / "UPDATE_000.pt"
        _atomic_torch_save(step0, payload)
        step0_sha = sha256_file(step0)
        atomic_json(checkpoints / "UPDATE_000.json", {
            "schema": "genesis-basic-checkpoint-v1",
            "update": 0,
            "checkpoint": str(step0.resolve()),
            "checkpoint_sha256": step0_sha,
            "identity": identity,
            "clean72_gate_required": True,
            "clean72_gate_status": "INHERITED_TERMINAL_UPDATE_030",
            "state_seed": seed_receipt,
        })
        _atomic_latest_link(latest, step0)
        emit(event="state_only_seed", checkpoint=str(step0), sha256=step0_sha,
             **seed_receipt)
        assert_runtime_guard()
    elif canary_seed_path:
        seed_path = Path(canary_seed_path).expanduser().resolve()
        seed = torch.load(seed_path, map_location="cpu", mmap=True, weights_only=False)
        expected_seed = {
            "format": FORMAT,
            "mechanism": MECHANISM,
            "config": config,
        }
        seed_drift = {
            key: (seed.get(key), value)
            for key, value in expected_seed.items()
            if seed.get(key) != value
        }
        for key in ("assignment_sha256", "payload_surface_sha256"):
            if seed.get("identity", {}).get(key) != identity.get(key):
                seed_drift[f"identity.{key}"] = (
                    seed.get("identity", {}).get(key), identity.get(key)
                )
        if seed_drift:
            raise RuntimeError(f"canary seed identity drift: {list(seed_drift)}")
        load_surface_state(student, seed["state"]["codebooks"])
        load_dense_state(norms, outputs, seed["state"])
        optimizer.load_state_dict(seed["optimizer"])
        scheduler.load_state_dict(seed["scheduler"])
        start_update = int(seed["next_update"])
        microprobes = list(seed.get("microprobes", []))
        seed_receipt = {
            "schema": "genesis-basic-canary-seed-migration-v1",
            "seed": str(seed_path),
            "seed_sha256": sha256_file(seed_path),
            "next_update": start_update,
            "old_surface_sha256": seed["identity"].get("surface_sha256"),
            "new_surface_sha256": identity.get("surface_sha256"),
            "old_base_harness_sha256": seed["identity"].get("base_harness_sha256"),
            "new_base_harness_sha256": identity.get("base_harness_sha256"),
            "state_loaded": True,
            "optimizer_loaded": True,
            "scheduler_loaded": True,
            "acceleration": acceleration,
        }
        atomic_json(receipts / "CANARY_SEED_MIGRATION.json", seed_receipt)
        emit(event="canary_seed_migration", **seed_receipt)
        assert_runtime_guard()
    else:
        # P959 terminal cutover: rebuild UPDATE_000 directly from the sealed
        # TRUE-C live surface.  The predecessor BASELINE_R seed uses a
        # different key-space and must never be aliased or loaded here.
        terminal_state = surface.surface_state(student)
        terminal_inventory = []
        for layer in range(43):
            label = f"L{layer}"
            for name, tensor in sorted(terminal_state[label].items()):
                tensor_sha = hashlib.sha256(
                    tensor.detach().cpu().contiguous().numpy().tobytes()
                ).hexdigest()
                name_sha = name.rsplit("_", 1)[-1]
                if tensor_sha != name_sha:
                    raise RuntimeError(
                        f"terminal codebook name/byte SHA mismatch {label}/{name}: "
                        f"{tensor_sha} != {name_sha}"
                    )
                terminal_inventory.append({
                    "layer": layer,
                    "name": name,
                    "tensor_sha256": tensor_sha,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "numel": int(tensor.numel()),
                })
        if len(terminal_inventory) != 80:
            raise RuntimeError(
                f"P943 terminal live codebook cardinality drift: {len(terminal_inventory)} != 80"
            )
        terminal_seal_path = root / "inputs/P943_TRUE_C_TERMINAL_SEAL.json"
        terminal_seal_sha = sha256_file(terminal_seal_path)
        expected_terminal_seal_sha = (
            "90e6d6b131d14b353be2976848dc90e947cb6fc1cda376e03b760a63dce8d31c"
        )
        if terminal_seal_sha != expected_terminal_seal_sha:
            raise RuntimeError(
                f"P943 terminal seal SHA drift: {terminal_seal_sha} != {expected_terminal_seal_sha}"
            )
        terminal_seal = json.loads(terminal_seal_path.read_text())
        expected_overlay_sha = (
            "9a4b709851c62c32f59b17556ef14d53e89cbbfc0fcc93686fc51530e4cf4d62"
        )
        terminal_drift = {
            key: (terminal_seal.get(key), expected)
            for key, expected in {
                "codebooks": 80,
                "target_rows": 2860,
                "pack_fraction": 1.0,
                "zero_substitution": True,
                "zero_quarantine": True,
                "active_overlay_sha256": expected_overlay_sha,
            }.items()
            if terminal_seal.get(key) != expected
        }
        if terminal_drift:
            raise RuntimeError(f"P943 terminal seal semantic drift: {terminal_drift}")
        payload = _checkpoint_payload(
            B=B, surface=surface, student=student, norms=norms, outputs=outputs,
            optimizer=optimizer, scheduler=scheduler, next_update=0,
            identity=identity, config=config, microprobes=microprobes,
        )
        step0 = checkpoints / "UPDATE_000.pt"
        _atomic_torch_save(step0, payload)
        step0_sha = sha256_file(step0)
        atomic_json(checkpoints / "UPDATE_000.json", {
            "schema": "genesis-basic-checkpoint-v1",
            "update": 0,
            "checkpoint": str(step0.resolve()),
            "checkpoint_sha256": step0_sha,
            "identity": identity,
            "clean72_gate_required": True,
            "clean72_gate_status": "F521_PRE_REPAIR_PHYSICAL_RECEIPT_BOUND",
        })
        _atomic_latest_link(latest, step0)
        atomic_json(receipts / "STEP0_IDENTITY.json", {
            "schema": "genesis-basic-step0-v1",
            "status": "PASS_EXACT_INITIALIZATION",
            "checkpoint": str(step0.resolve()),
            "checkpoint_sha256": step0_sha,
            "identity": identity,
            "config": config,
            "codebook_wire_roundtrip": True,
            "rmsnorm_wire_identity": True,
            "output_gains_zero": True,
            "parent_physical_code76_receipt": config["physical_code76_receipt"],
            "parent_physical_code76_receipt_sha256": config["physical_code76_receipt_sha256"],
        })
        atomic_json(receipts / "P959_TERMINAL_UPDATE_000_REBUILT.json", {
            "schema": "p959-terminal-update-000-rebuilt-v1",
            "status": "PASS_EXACT_TRUE_C_TERMINAL_SEED_REBUILD",
            "task_id": os.environ.get("GENESIS_TASK_ID"),
            "checkpoint": str(step0.resolve()),
            "checkpoint_sha256": step0_sha,
            "next_update": 0,
            "p943_terminal_seal": str(terminal_seal_path.resolve()),
            "p943_terminal_seal_sha256": terminal_seal_sha,
            "active_overlay_sha256": expected_overlay_sha,
            "terminal_codebook_count": len(terminal_inventory),
            "terminal_codebooks": terminal_inventory,
            "terminal_codebook_inventory_sha256": _stat_fingerprint(terminal_inventory),
            "old_baseline_r_seed_loaded": False,
            "old_to_new_key_aliasing": False,
            "optimizer_loaded": False,
            "optimizer_state_entries": len(optimizer.state),
            "scheduler_loaded": False,
            "speculative_seed_used": False,
            "sealed_unix": time.time(),
        })
        emit(event="step0_checkpoint", checkpoint=str(step0), sha256=step0_sha)

    # Two fixed, clean code windows are only a liveness microprobe. The
    # authoritative clean-72 gates are produced by the independent consumer.
    probe_wins = list(map(int, config["microprobe_combined_wins"]))
    train_wins = list(map(int, config["train_combined_wins"]))
    groups = (len(train_wins) + batch - 1) // batch
    if train_wins != B.TRAIN_WINS or probe_wins != B.PROBE_WINS:
        raise RuntimeError("base-harness train/probe binding drift")

    directional_spec_path = Path(
        os.environ["GENESIS_REPAIR_DIRECTIONAL_SPEC"]
    ).resolve()
    directional_spec = json.loads(directional_spec_path.read_text())
    class_wins = directional_spec.get("class_wins", {})
    expected_classes = ("multilingual", "prose", "reasoning", "chat", "code")
    if directional_spec.get("format") != "p600-train8-directional-v1":
        raise RuntimeError("directional spec format drift")
    if tuple(class_wins) != expected_classes:
        raise RuntimeError("directional class order/surface drift")
    for class_name, wins in class_wins.items():
        if len(wins) != 8 or len(set(map(int, wins))) != 8:
            raise RuntimeError(f"{class_name} must bind exactly eight unique TRAIN wins")
        if not set(map(int, wins)).issubset(set(train_wins)):
            raise RuntimeError(f"{class_name} TRAIN-8 is not contained in training schedule")
        if any(int(win) < 0 or int(win) >= 256 for win in wins):
            raise RuntimeError(f"{class_name} references non-TRAIN combined window")
    directional_spec_sha = sha256_file(directional_spec_path)
    # Keep scorer batches at the sealed optimizer batch shape. Larger batches
    # choose different dense-GEMM reductions and move decision-scale KLD rows.
    directional_batch = int(os.environ.get("GENESIS_REPAIR_DIRECTIONAL_BATCH", "4"))
    if directional_batch <= 0 or directional_batch > batch:
        raise RuntimeError(f"directional batch must be in [1,{batch}]")

    def bootstrap_ci95(values: list[float], *, update: int, class_index: int) -> list[float]:
        rng = random.Random(seed + 1009 * int(update) + 7919 * int(class_index))
        draws = []
        n = len(values)
        for _ in range(20000):
            draws.append(sum(values[rng.randrange(n)] for _j in range(n)) / n)
        draws.sort()
        return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]

    def directional_measure(update: int) -> dict[str, object]:
        destination = receipts / f"TRAIN8_DIRECTIONAL_UPDATE_{update:03d}.json"
        if destination.is_file():
            existing = json.loads(destination.read_text())
            if existing.get("directional_spec_sha256") != directional_spec_sha:
                raise RuntimeError(f"directional receipt spec drift: {destination}")
            return existing
        assert_runtime_guard()
        measured_started = time.time()
        measured_classes = {}
        ordered_wins = [
            int(win)
            for class_name in expected_classes
            for win in class_wins[class_name]
        ]
        if len(ordered_wins) != 40 or len(set(ordered_wins)) != 40:
            raise RuntimeError("directional TRAIN-40 must be globally unique")
        segments = [
            ordered_wins[offset:offset + directional_batch]
            for offset in range(0, len(ordered_wins), directional_batch)
        ]
        value_by_win = {}
        for segment_index, segment in enumerate(segments):
            next_segment = segments[segment_index + 1] if segment_index + 1 < len(segments) else ()
            segment_values = B.batch_kld_values(
                student, corpus, acache, segment, prefetch_wins=next_segment
            )
            value_by_win.update(zip(segment, segment_values))
            for win in segment:
                cached = acache.mem.pop(win, None)
                if cached is not None:
                    del cached
            torch.cuda.empty_cache()
        for class_name in expected_classes:
            wins = list(map(int, class_wins[class_name]))
            values = [value_by_win[win] for win in wins]
            measured_classes[class_name] = {
                "wins": wins,
                "kld": values,
                "mean_kld": sum(values) / len(values),
            }
        baseline_path = receipts / "TRAIN8_DIRECTIONAL_UPDATE_000.json"
        if update == 0:
            comparison = "fresh_exact_f521_pre_repair_baseline"
        else:
            if not baseline_path.is_file():
                raise RuntimeError("missing fresh exact F521 TRAIN-8 baseline")
            baseline = json.loads(baseline_path.read_text())
            comparison = "paired_vs_fresh_exact_f521_pre_repair"
            for class_index, class_name in enumerate(expected_classes):
                current = measured_classes[class_name]
                parent = baseline["classes"][class_name]
                if current["wins"] != parent["wins"]:
                    raise RuntimeError(f"paired TRAIN-8 drift for {class_name}")
                deltas = [float(after) - float(before) for after, before in
                          zip(current["kld"], parent["kld"])]
                current["baseline_mean_kld"] = float(parent["mean_kld"])
                current["paired_delta"] = deltas
                current["mean_delta"] = sum(deltas) / len(deltas)
                current["bootstrap_ci95_mean_delta"] = bootstrap_ci95(
                    deltas, update=update, class_index=class_index
                )
                current["improved_count"] = sum(delta < 0.0 for delta in deltas)
                current["sign_consistency_improved"] = (
                    current["improved_count"] / len(deltas)
                )
        receipt = {
            "schema": "p600-train8-paired-directional-receipt-v1",
            "task_id": os.environ.get("GENESIS_TASK_ID"),
            "update": int(update),
            "comparison": comparison,
            "classes": measured_classes,
            "bootstrap_resamples": 20000,
            "directional_batch": directional_batch,
            "directional_spec": str(directional_spec_path),
            "directional_spec_sha256": directional_spec_sha,
            "train_manifest_sha256": directional_spec["train_manifest_sha256"],
            "checkpoint": str(latest.resolve()),
            "checkpoint_sha256": sha256_file(latest),
            "seconds": time.time() - measured_started,
            "measured_unix": time.time(),
        }
        atomic_json(destination, receipt)
        emit(event="train8_directional", update=update,
             seconds=receipt["seconds"], classes={
                 name: {
                     "mean_kld": row["mean_kld"],
                     "mean_delta": row.get("mean_delta"),
                     "sign_consistency_improved": row.get("sign_consistency_improved"),
                 } for name, row in measured_classes.items()
             })
        return receipt

    if start_update in (0, 12, 24):
        directional_measure(start_update)

    def microprobe(update: int) -> None:
        rows = []
        for win in probe_wins:
            started = time.time()
            value = B.kld_window(student, corpus, acache, win)
            rows.append({"win": win, "kld": value})
            emit(event="microprobe_window", update=update, win=win, kld=value, seconds=time.time()-started)
            torch.cuda.empty_cache()
        record = {
            "update": update,
            "mean_kld": sum(row["kld"] for row in rows) / len(rows),
            "rows": rows,
            "scope": "liveness_only_not_clean72_selection",
        }
        microprobes.append(record)
        emit(event="microprobe", **record)

    if start_update == 0 and not microprobes:
        microprobe(0)

    if acceleration["update0_smoke_only"]:
        if start_update != 0:
            raise RuntimeError(
                f"UPDATE_000 smoke requires fresh next_update=0, observed {start_update}"
            )
        forbidden = [
            path for path in (
                checkpoints / "UPDATE_001.pt",
                checkpoints / "UPDATE_001.json",
                receipts / "BASIC_CANARY_DONE.json",
                receipts / "BASIC_TRAINING_DONE.json",
            ) if path.exists()
        ]
        if forbidden:
            raise RuntimeError(f"UPDATE_000 smoke found post-update artifacts: {forbidden}")
        if optimizer.state or any(parameter.grad is not None for parameter in all_params):
            raise RuntimeError("UPDATE_000 smoke detected optimizer/gradient mutation")
        step0 = checkpoints / "UPDATE_000.pt"
        step0_sidecar = checkpoints / "UPDATE_000.json"
        directional0 = receipts / "TRAIN8_DIRECTIONAL_UPDATE_000.json"
        if not step0.is_file() or not step0_sidecar.is_file() or not directional0.is_file():
            raise RuntimeError("UPDATE_000 smoke evidence surface incomplete")
        assert_runtime_guard()
        smoke_receipt = {
            "schema": "p680-update0-parity-smoke-v1",
            "status": "PASS_UPDATE0_SMOKE_NO_OPTIMIZER_STEP",
            "task_id": os.environ.get("GENESIS_TASK_ID"),
            "next_update": 0,
            "checkpoint": str(step0.resolve()),
            "checkpoint_sha256": sha256_file(step0),
            "checkpoint_sidecar": str(step0_sidecar.resolve()),
            "checkpoint_sidecar_sha256": sha256_file(step0_sidecar),
            "directional_receipt": str(directional0.resolve()),
            "directional_receipt_sha256": sha256_file(directional0),
            "microprobes": microprobes,
            "optimizer_state_entries": len(optimizer.state),
            "all_gradients_none": all(parameter.grad is None for parameter in all_params),
            "source_index": identity["f521_source_index"],
            "identity": identity,
            "completed_unix": time.time(),
        }
        atomic_json(receipts / "UPDATE0_PARITY_SMOKE.json", smoke_receipt)
        write_status(state="UPDATE0_SMOKE_COMPLETE", next_update=0, smoke=smoke_receipt)
        emit(event="update0_smoke_complete", **smoke_receipt)
        return 0

    write_status(
        state="RUNNING", next_update=start_update, pid=os.getpid(),
        config=config, identity=identity,
    )
    started_wall = time.time()
    for update_index in range(start_update, stop_after_update):
        assert_runtime_guard()
        group = update_index % groups
        wins = train_wins[group * batch : (group + 1) * batch]
        started = time.time()
        phases = PhaseTimer()
        optimizer.zero_grad(set_to_none=True)
        micro_wins = [
            wins[start : start + microbatch]
            for start in range(0, len(wins), microbatch)
        ]
        loss_before = 0.0
        forward_seconds = 0.0
        backward_total_seconds = 0.0
        for segment_index, segment_wins in enumerate(micro_wins):
            forward_name = f"forward_microbatch_{segment_index}"
            forward_mem_before = guard_memory(f"{forward_name}:before")
            phases.start(forward_name)
            if segment_index + 1 < len(micro_wins):
                prefetch_wins = micro_wins[segment_index + 1]
            elif update_index + 1 < stop_after_update:
                next_group = (update_index + 1) % groups
                prefetch_wins = train_wins[next_group * batch:(next_group + 1) * batch]
            else:
                prefetch_wins = ()
            segment_loss = B.batch_loss(
                student, corpus, acache, segment_wins, True,
                prefetch_wins=prefetch_wins,
            )
            segment_forward_seconds = phases.stop(forward_name)
            forward_seconds += segment_forward_seconds
            segment_weight = len(segment_wins) / len(wins)
            segment_loss_before = float(segment_loss.detach())
            loss_before += segment_loss_before * segment_weight
            forward_mem_after = guard_memory(f"{forward_name}:after")
            emit_phase(
                update=update_index + 1,
                phase="forward_microbatch",
                segment_index=segment_index,
                segment_count=len(micro_wins),
                seconds=segment_forward_seconds,
                wins=segment_wins,
                loss_before_update=segment_loss_before,
                mem_available_before_bytes=forward_mem_before,
                mem_available_before_gib=forward_mem_before / 1024**3,
                mem_available_after_bytes=forward_mem_after,
                mem_available_after_gib=forward_mem_after / 1024**3,
            )

            backward_name = f"backward_segment_{segment_index}"
            backward_mem_before = guard_memory(f"{backward_name}:before")
            phases.start(backward_name)
            ordered_backward([segment_loss * segment_weight])
            backward_seconds = phases.stop(backward_name)
            backward_total_seconds += backward_seconds
            backward_mem_after = guard_memory(f"{backward_name}:after")
            emit_phase(
                update=update_index + 1,
                phase="backward_segment",
                segment_index=segment_index,
                segment_count=len(micro_wins),
                seconds=backward_seconds,
                grouped=False,
                microbatch=microbatch,
                gradient_weight=segment_weight,
                mem_available_before_bytes=backward_mem_before,
                mem_available_before_gib=backward_mem_before / 1024**3,
                mem_available_after_bytes=backward_mem_after,
                mem_available_after_gib=backward_mem_after / 1024**3,
            )
            del segment_loss
            for win in segment_wins:
                cached = acache.mem.pop(win, None)
                if cached is not None:
                    del cached
            torch.cuda.empty_cache()
        emit_phase(
            update=update_index + 1,
            phase="forward_teacher_target",
            seconds=forward_seconds,
            wins=wins,
            microbatch=microbatch,
            segment_count=len(micro_wins),
            loss_before_update=loss_before,
        )
        emit_phase(
            update=update_index + 1,
            phase="backward_total",
            seconds=backward_total_seconds,
            microbatch=microbatch,
            segment_count=len(micro_wins),
        )
        phases.start("grad_norm")
        grad_norms = {}
        for name, parameters in (
            ("codebooks", codebooks), ("norms", norm_params), ("outputs", output_params)
        ):
            squared = sum(
                parameter.grad.float().norm() ** 2
                for parameter in parameters if parameter.grad is not None
            )
            value = float(squared ** 0.5)
            if not math.isfinite(value) or value <= 0.0:
                raise RuntimeError(f"non-finite/zero {name} gradient at update {update_index+1}: {value}")
            grad_norms[name] = value
        phases.stop("grad_norm")
        optimizer_mem_before = guard_memory("optimizer:before")
        phases.start("optimizer")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_seconds = phases.stop("optimizer")
        optimizer_mem_after = guard_memory("optimizer:after")
        emit_phase(
            update=update_index + 1,
            phase="optimizer",
            seconds=optimizer_seconds,
            mem_available_before_bytes=optimizer_mem_before,
            mem_available_before_gib=optimizer_mem_before / 1024**3,
            mem_available_after_bytes=optimizer_mem_after,
            mem_available_after_gib=optimizer_mem_after / 1024**3,
        )
        torch.cuda.empty_cache()
        next_update = update_index + 1
        if next_update % probe_every == 0 or next_update == steps:
            microprobe(next_update)
        payload = _checkpoint_payload(
            B=B, surface=surface, student=student, norms=norms, outputs=outputs,
            optimizer=optimizer, scheduler=scheduler, next_update=next_update,
            identity=identity, config=config, microprobes=microprobes,
        )
        checkpoint_path = checkpoints / f"UPDATE_{next_update:03d}.pt"
        assert_runtime_guard()
        checkpoint_mem_before = guard_memory("checkpoint_io:before")
        phases.start("checkpoint_io")
        _atomic_torch_save(checkpoint_path, payload)
        checkpoint_sha = sha256_file(checkpoint_path)
        checkpoint_seconds = phases.stop("checkpoint_io")
        checkpoint_mem_after = guard_memory("checkpoint_io:after")
        emit_phase(
            update=next_update,
            phase="checkpoint_io",
            seconds=checkpoint_seconds,
            checkpoint=str(checkpoint_path.resolve()),
            checkpoint_sha256=checkpoint_sha,
            mem_available_before_bytes=checkpoint_mem_before,
            mem_available_before_gib=checkpoint_mem_before / 1024**3,
            mem_available_after_bytes=checkpoint_mem_after,
            mem_available_after_gib=checkpoint_mem_after / 1024**3,
        )
        total_seconds = time.time() - started
        emit_phase(
            update=next_update,
            phase="total",
            seconds=total_seconds,
            loss_before_update=loss_before,
            phase_seconds=phases.snapshot(),
        )
        sidecar = {
            "schema": "genesis-basic-checkpoint-v1",
            "update": next_update,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "loss_before_update": loss_before,
            "grad_norms": grad_norms,
            "learning_rates": {
                group["group_name"]: group["lr"] for group in optimizer.param_groups
            },
            "train_wins": wins,
            "seconds": time.time() - started,
            "phase_seconds": phases.snapshot(),
            "acceleration": acceleration,
            "identity": identity,
            "clean72_gate_required": next_update in NATURAL_UPDATES,
            "clean72_gate_status": "PENDING_CONSUMER" if next_update in NATURAL_UPDATES else "NOT_NATURAL_UPDATE",
        }
        atomic_json(checkpoints / f"UPDATE_{next_update:03d}.json", sidecar)
        _atomic_latest_link(latest, checkpoint_path)
        emit(event="update", **sidecar)
        write_status(
            state="RUNNING", next_update=next_update,
            last_checkpoint=str(checkpoint_path), last_checkpoint_sha256=checkpoint_sha,
            last_loss=loss_before, last_grad_norms=grad_norms,
        )
        if next_update in (12, 24):
            directional_measure(next_update)
        # Production-resume safety gate: only the first newly executed update is
        # bounded here.  It is checked after the immutable checkpoint + sidecar
        # are sealed, but before the loop may begin another update.
        first_resume_gate = os.environ.get("GENESIS_REPAIR_FIRST_RESUME_GATE_SECONDS")
        if next_update == start_update + 1 and first_resume_gate is not None:
            gate_seconds = float(first_resume_gate)
            gate_receipt = {
                "schema": "genesis-basic-first-resume-gate-v1",
                "task_id": os.environ.get("GENESIS_TASK_ID"),
                "resume_seed_update": start_update,
                "completed_update": next_update,
                "total_seconds": total_seconds,
                "gate_seconds": gate_seconds,
                "passed": total_seconds <= gate_seconds,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha,
                "loss": loss_before,
                "grad_norms": grad_norms,
                "sealed_unix": time.time(),
            }
            atomic_json(receipts / "FIRST_RESUME_GATE.json", gate_receipt)
            if total_seconds > gate_seconds:
                write_status(
                    state="FAILED_FIRST_RESUME_GATE",
                    next_update=next_update,
                    first_resume_gate=gate_receipt,
                )
                raise RuntimeError(
                    f"FIRST_RESUME_GATE exceeded at update {next_update}: "
                    f"{total_seconds:.9f}s > {gate_seconds:.9f}s"
                )

    if stop_after_update < steps:
        done = {
            "schema": "genesis-basic-canary-complete-v1",
            "state": "CANARY_COMPLETE",
            "next_update": stop_after_update,
            "start_update": start_update,
            "updates_run": stop_after_update - start_update,
            "identity": identity,
            "config": config,
            "acceleration": acceleration,
            "elapsed_seconds": time.time() - started_wall,
            "completed_unix": time.time(),
        }
        atomic_json(receipts / "BASIC_CANARY_DONE.json", done)
        write_status(**done)
        emit(event="canary_complete", **done)
        return 0

    done = {
        "schema": "genesis-basic-training-done-v1",
        "state": "TRAINING_COMPLETE_AWAITING_CLEAN72_SELECTION",
        "updates": steps,
        "natural_updates": list(NATURAL_UPDATES),
        "identity": identity,
        "config": config,
        "elapsed_seconds": time.time() - started_wall,
        "completed_unix": time.time(),
    }
    atomic_json(receipts / "BASIC_TRAINING_DONE.json", done)
    write_status(**done)
    emit(event="training_complete", **done)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        try:
            root = Path(os.environ.get("GENESIS_REPAIR_ROOT", "."))
            atomic_json(root / "run/BASIC_REPAIR_STATUS.json", {
                "state": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                "failed_unix": time.time(),
            })
        except Exception:
            pass
        raise
