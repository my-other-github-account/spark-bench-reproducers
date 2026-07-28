"""Banana Smasher's stdlib-only durable stage engine."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.request import Request, urlopen


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS_PATH = PACKAGE_ROOT / "contracts" / "STAGE_CONTRACTS.json"
VENDOR_INDEX_PATH = PACKAGE_ROOT / "vendor" / "VENDOR_INDEX.json"
MANIFEST_PATH = PACKAGE_ROOT / "TOOLS_MANIFEST.json"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def atomic_write(path: Path, payload: bytes, mode: int = 0o644) -> None:
    """Write, fsync, and rename a file without exposing a partial receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        try:
            directory = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError:
            pass
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def contracts() -> Mapping[str, Any]:
    return load_json(CONTRACTS_PATH)["stages"]


def public_args(namespace: Any) -> Dict[str, Any]:
    omitted = {"handler", "workspace", "command", "dry_run"}
    return {
        key: value
        for key, value in sorted(vars(namespace).items())
        if key not in omitted and value is not None
    }


def stage_plan(stage: str, namespace: Any) -> Dict[str, Any]:
    contract = contracts()[stage]
    return {
        "schema": "banana-smasher-stage-plan-v1",
        "stage": stage,
        "mode": "dry-run" if namespace.dry_run else "live",
        "offline": bool(namespace.dry_run),
        "workspace": "workspace",
        "stage_directory": "workspace/" + stage,
        "receipt": contract["receipt"],
        "prerequisites": contract["prerequisites"],
        "inputs": contract["inputs"],
        "outputs": contract["outputs"],
        "vendor_capabilities": contract["vendor_capabilities"],
        "arguments": public_args(namespace),
        "write_policy": "current-stage-only; atomic receipt is written last",
    }


def _pick(config: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in config and config[name] is not None:
            return config[name]
    return default


def _load_model_config(model: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        config_path = candidate / "config.json"
        payload = config_path.read_bytes()
        source: Dict[str, Any] = {"kind": "local-directory", "model": model, "config_file": "config.json"}
    elif candidate.is_file():
        payload = candidate.read_bytes()
        source = {"kind": "local-config", "model": model, "config_file": candidate.name}
    else:
        url = "https://huggingface.co/{}/resolve/main/config.json".format(model)
        headers = {"User-Agent": "banana-smasher/0.1"}
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if token:
            headers["Authorization"] = "Bearer " + token
        request = Request(url, headers=headers)
        with urlopen(request, timeout=30) as response:
            payload = response.read()
        source = {"kind": "huggingface", "model": model, "config_url": url}
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        raise RuntimeError("model config must be a JSON object")
    source["config_sha256"] = sha256_bytes(payload)
    source["config_bytes"] = len(payload)
    return parsed, source


def _architecture_profile(config: Mapping[str, Any]) -> Dict[str, Any]:
    layers = int(_pick(config, "num_hidden_layers", "n_layer", "num_layers", default=0) or 0)
    experts = int(_pick(config, "n_routed_experts", "num_local_experts", "num_experts", default=0) or 0)
    top_k = int(_pick(config, "num_experts_per_tok", "num_experts_per_token", "moe_top_k", default=0) or 0)
    hidden = int(_pick(config, "hidden_size", "n_embd", "d_model", default=0) or 0)
    intermediate = int(_pick(config, "moe_intermediate_size", "intermediate_size", "ffn_hidden_size", default=0) or 0)
    architectures = _pick(config, "architectures", default=[])
    if isinstance(architectures, str):
        architectures = [architectures]
    projections: Dict[str, List[int]] = {}
    if hidden and intermediate:
        projections = {
            "fused13_out_in": [2 * intermediate, hidden],
            "down_out_in": [hidden, intermediate],
        }
    return {
        "architectures": list(architectures or []),
        "model_type": str(_pick(config, "model_type", default="unknown")),
        "layers": layers,
        "experts": experts,
        "experts_per_token": top_k,
        "hidden_size": hidden,
        "expert_intermediate_size": intermediate,
        "projection_shapes": projections,
        "cells": layers * experts * len(projections),
    }


def _init_documents(namespace: Any) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    if not namespace.model:
        raise RuntimeError("init requires --model outside dry-run")
    if namespace.budget_bytes is None or namespace.budget_bytes <= 0:
        raise RuntimeError("init requires positive --budget-bytes outside dry-run")
    if namespace.node_ram is None or namespace.node_ram <= 0:
        raise RuntimeError("init requires positive --node-ram outside dry-run")
    config, source = _load_model_config(namespace.model)
    architecture = _architecture_profile(config)
    profile = {
        "schema": "banana-smasher-model-profile-v1",
        "model": namespace.model,
        "source": source,
        "architecture": architecture,
        "budget": {
            "bytes": int(namespace.budget_bytes),
            "node_ram_gb": float(namespace.node_ram),
            "node_ram_bytes_decimal": int(float(namespace.node_ram) * 1_000_000_000),
        },
        "config_projection": {
            key: config[key]
            for key in sorted(config)
            if key in {
                "architectures", "model_type", "num_hidden_layers", "n_layer", "num_layers",
                "hidden_size", "n_embd", "d_model", "moe_intermediate_size", "intermediate_size",
                "ffn_hidden_size", "n_routed_experts", "num_local_experts", "num_experts",
                "num_experts_per_tok", "num_experts_per_token", "moe_top_k",
            }
        },
    }
    menu = {
        "schema": "banana-smasher-menu-template-v1",
        "model_profile_sha256": canonical_sha256(profile),
        "tiers": [
            {"name": "qtip3", "family": "qtip-rep16", "enabled": True},
            {"name": "qtip2", "family": "qtip-rep16", "enabled": True},
            {"name": "d4_k1024", "family": "true-vq-d4", "enabled": True},
            {"name": "d4_k2048", "family": "true-vq-d4", "enabled": True},
            {"name": "d4_k4096", "family": "true-vq-d4", "enabled": True},
            {"name": "mxfp4", "family": "native-mxfp4", "enabled": True},
        ],
        "classes": ["agentic", "chat", "code", "multilingual", "prose", "reasoning"],
        "budget_bytes": int(namespace.budget_bytes),
        "assignment_unit": "layer-expert-projection",
    }
    invocation = {
        "stage": "init",
        "arguments": public_args(namespace),
        "config_sha256": source["config_sha256"],
    }
    return profile, menu, invocation


def run_init(namespace: Any) -> Dict[str, Any]:
    if namespace.dry_run:
        return stage_plan("init", namespace)
    workspace = Path(namespace.workspace).expanduser().resolve()
    stage_dir = workspace / "init"
    profile, menu, invocation = _init_documents(namespace)
    invocation_sha = canonical_sha256(invocation)
    receipt_path = stage_dir / "RECEIPT.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt.get("invocation_sha256") != invocation_sha:
            raise RuntimeError("sealed init receipt exists for different inputs; use a new workspace")
        return {
            "stage": "init",
            "status": "ALREADY_COMPLETE",
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
    profile_path = stage_dir / "MODEL_PROFILE.json"
    menu_path = stage_dir / "MENU_TEMPLATE.json"
    atomic_json(profile_path, profile)
    atomic_json(menu_path, menu)
    receipt = {
        "schema": "banana-smasher-stage-receipt-v1",
        "stage": "init",
        "status": "PASS",
        "validity": "PROFILED",
        "invocation_sha256": invocation_sha,
        "inputs": {"config_sha256": profile["source"]["config_sha256"]},
        "outputs": {
            "MODEL_PROFILE.json": sha256_file(profile_path),
            "MENU_TEMPLATE.json": sha256_file(menu_path),
        },
        "atomic": True,
        "resumable": True,
    }
    atomic_json(receipt_path, receipt)
    return {
        "stage": "init",
        "status": "PASS",
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _prerequisite_receipts(workspace: Path, stage: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    missing: List[str] = []
    for prerequisite in contracts()[stage]["prerequisites"]:
        path = workspace / prerequisite / "RECEIPT.json"
        if not path.is_file():
            missing.append(str(path))
        else:
            result[prerequisite] = sha256_file(path)
    if missing:
        raise RuntimeError("missing prerequisite receipt: " + ", ".join(missing))
    return result


def run_generic_stage(stage: str, namespace: Any) -> Dict[str, Any]:
    if namespace.dry_run:
        return stage_plan(stage, namespace)
    workspace = Path(namespace.workspace).expanduser().resolve()
    upstream = _prerequisite_receipts(workspace, stage)
    stage_dir = workspace / stage
    plan = stage_plan(stage, namespace)
    plan["mode"] = "prototype-contract"
    plan["offline"] = False
    invocation = {
        "stage": stage,
        "arguments": public_args(namespace),
        "prerequisite_receipts": upstream,
    }
    invocation_sha = canonical_sha256(invocation)
    receipt_path = stage_dir / "RECEIPT.json"
    if receipt_path.exists():
        receipt = load_json(receipt_path)
        if receipt.get("invocation_sha256") != invocation_sha:
            raise RuntimeError("sealed {} receipt exists for different inputs; use a new workspace".format(stage))
        return {
            "stage": stage,
            "status": "ALREADY_COMPLETE",
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
        }
    plan_path = stage_dir / "EXECUTION_PLAN.json"
    atomic_json(plan_path, plan)
    receipt = {
        "schema": "banana-smasher-stage-receipt-v1",
        "stage": stage,
        "status": "PASS_PROTOTYPE_CONTRACT",
        "validity": "UNMEASURED" if stage in {"capture", "anchors", "anchor-mix", "measure", "serve", "eval"} else "PROJECTED",
        "physical_execution": False,
        "invocation_sha256": invocation_sha,
        "inputs": upstream,
        "outputs": {"EXECUTION_PLAN.json": sha256_file(plan_path)},
        "atomic": True,
        "resumable": True,
        "warning": "Prototype mode seals wiring only; it never relabels a plan as a physical measurement.",
    }
    atomic_json(receipt_path, receipt)
    return {
        "stage": stage,
        "status": receipt["status"],
        "validity": receipt["validity"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
    }


def status(namespace: Any) -> Dict[str, Any]:
    if namespace.dry_run:
        return stage_plan("status", namespace)
    workspace = Path(namespace.workspace).expanduser().resolve()
    rows = []
    for stage in contracts():
        if stage == "status":
            continue
        receipt = workspace / stage / "RECEIPT.json"
        if receipt.is_file():
            payload = load_json(receipt)
            rows.append({
                "stage": stage,
                "state": payload.get("status", "UNKNOWN"),
                "validity": payload.get("validity"),
                "receipt_sha256": sha256_file(receipt),
            })
        else:
            rows.append({"stage": stage, "state": "PENDING", "validity": None, "receipt_sha256": None})
    stage_dir = workspace / "status"
    ledger_path = stage_dir / "LEDGER.json"
    ledger = {"schema": "banana-smasher-ledger-v1", "stages": rows}
    atomic_json(ledger_path, ledger)
    receipt_path = stage_dir / "RECEIPT.json"
    receipt = {
        "schema": "banana-smasher-stage-receipt-v1",
        "stage": "status",
        "status": "PASS",
        "validity": "LEDGER",
        "outputs": {"LEDGER.json": sha256_file(ledger_path)},
        "atomic": True,
        "resumable": True,
    }
    atomic_json(receipt_path, receipt)
    return ledger


def iter_package_files() -> Iterable[Path]:
    for path in sorted(PACKAGE_ROOT.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(PACKAGE_ROOT)
        if relative.parts and relative.parts[0] == "workspace":
            continue
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if relative.as_posix() in {"TOOLS_MANIFEST.json", "PACKAGE_MANIFEST.json"}:
            continue
        yield path


def verify_manifest() -> Dict[str, Any]:
    manifest = load_json(MANIFEST_PATH)
    expected = {row["path"]: row for row in manifest["files"]}
    actual_paths = {path.relative_to(PACKAGE_ROOT).as_posix(): path for path in iter_package_files()}
    failures: List[Dict[str, Any]] = []
    if set(expected) != set(actual_paths):
        failures.append({
            "inventory_mismatch": {
                "missing": sorted(set(expected) - set(actual_paths)),
                "unmanifested": sorted(set(actual_paths) - set(expected)),
            }
        })
    for relative in sorted(set(expected).intersection(actual_paths)):
        path = actual_paths[relative]
        row = expected[relative]
        observed = sha256_file(path)
        if observed != row["sha256"] or path.stat().st_size != row["bytes"]:
            failures.append({"path": relative, "expected": row, "observed_sha256": observed, "observed_bytes": path.stat().st_size})
        if row.get("shipped_sha256") != observed or not row.get("source_sha256"):
            failures.append({"path": relative, "provenance": "missing or mismatched source/shipped SHA"})
    package_path = PACKAGE_ROOT / "PACKAGE_MANIFEST.json"
    if not package_path.is_file():
        failures.append({"package_manifest": "missing"})
    else:
        package = load_json(package_path)
        if package.get("tools_manifest_sha256") != sha256_file(MANIFEST_PATH):
            failures.append({"package_manifest": "TOOLS_MANIFEST SHA mismatch"})
        if package.get("tools_aggregate_sha256") != manifest.get("aggregate_sha256"):
            failures.append({"package_manifest": "TOOLS aggregate mismatch"})
    return {
        "schema": "banana-smasher-manifest-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "files": len(actual_paths),
        "failures": failures,
    }


def verify_self_contained() -> Dict[str, Any]:
    from .self_containment import scan_package
    return scan_package(PACKAGE_ROOT)
