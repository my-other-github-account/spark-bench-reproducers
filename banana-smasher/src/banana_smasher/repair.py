from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

REPAIR_FORMAT = "bs-basic-repair-v1"
_LEGACY_REPAIR_FORMAT_SHA256 = (
    "cac903bdf1efe7a916865de1bb0648387087ce0d8810f7eb3abac6dd16e1ff0e"
)
REPAIR_MECHANISM = (
    "physical-vq-codebooks-plus-all-rmsnorms-plus-attention-output-gains"
)
REPAIR_MANIFEST_PATH = Path("repair/REPAIR_MANIFEST.json")
REPAIR_STATE_PATH = Path("repair/repair_state.safetensors")
PRODUCTION_COUNTS = (196, 235, 43)
_SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
_OUTPUT_RE = re.compile(
    r"model\.layers\.(\d+)\.self_attn\.o_b_proj\.output_log_gain\Z"
)


def _normalize_checkpoint_format(value: object) -> str:
    """Accept sealed pre-rename checkpoints while emitting the current format."""
    if not isinstance(value, str):
        raise ValueError("repair checkpoint format must be a string")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    allowed = {
        hashlib.sha256(REPAIR_FORMAT.encode("utf-8")).hexdigest(),
        _LEGACY_REPAIR_FORMAT_SHA256,
    }
    if digest not in allowed:
        raise ValueError("repair checkpoint format is not an approved sealed format")
    return REPAIR_FORMAT


@dataclass(frozen=True)
class CodebookRepair:
    checkpoint_key: str
    source_wire_sha256: str
    array: np.ndarray


@dataclass(frozen=True)
class RepairBundle:
    checkpoint_path: Path
    checkpoint_sha256: str
    active_overlay_path: Path
    active_overlay_sha256: str
    assignment_path: Path
    assignment_sha256: str
    checkpoint_format: str
    mechanism: str
    update: int
    codebooks: dict[str, CodebookRepair]
    dense_tensors: dict[str, np.ndarray]
    norm_count: int
    output_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _wire_sha(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _require_sha256(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _as_numpy(name: str, value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - repair runtime always has torch
            raise ValueError(
                "repair checkpoint loading requires torch in the export environment"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"repair state {name} is not a tensor")
        if value.device.type != "cpu":
            raise ValueError(f"repair state {name} must be on CPU")
        array = value.detach().numpy()
    if array.dtype != np.dtype("float32"):
        raise ValueError(f"repair state {name} must be float32, got {array.dtype}")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"repair state {name} contains non-finite values")
    if array.ndim == 0:
        return np.asarray(array)
    return np.ascontiguousarray(array)


def validate_repair_state(
    state: Mapping[str, Any],
    *,
    expected_counts: tuple[int, int, int] = PRODUCTION_COUNTS,
) -> dict[str, Any]:
    if not isinstance(state, Mapping) or set(state) != {
        "codebooks",
        "norms",
        "outputs",
    }:
        raise ValueError("repair checkpoint state must contain codebooks/norms/outputs")
    codebook_layers = state["codebooks"]
    norms = state["norms"]
    outputs = state["outputs"]
    if not all(isinstance(value, Mapping) for value in (codebook_layers, norms, outputs)):
        raise ValueError("repair checkpoint state maps are malformed")

    codebooks: dict[str, CodebookRepair] = {}
    for layer_name, values in sorted(codebook_layers.items()):
        if not isinstance(layer_name, str) or not isinstance(values, Mapping):
            raise ValueError("repair codebook layer map is malformed")
        for leaf_name, value in sorted(values.items()):
            checkpoint_key = f"{layer_name}/{leaf_name}"
            if not isinstance(leaf_name, str):
                raise ValueError("repair codebook key is not a string")
            source_sha = leaf_name.rsplit("_", 1)[-1]
            _require_sha256(source_sha, f"repair codebook source SHA {checkpoint_key}")
            array = _as_numpy(checkpoint_key, value)
            if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
                raise ValueError(
                    f"repair codebook {checkpoint_key} must be a non-empty matrix"
                )
            if source_sha in codebooks:
                raise ValueError(f"duplicate repair codebook source SHA: {source_sha}")
            codebooks[source_sha] = CodebookRepair(
                checkpoint_key=checkpoint_key,
                source_wire_sha256=source_sha,
                array=np.ascontiguousarray(array, dtype=np.float16),
            )

    dense: dict[str, np.ndarray] = {}
    for name, value in sorted(norms.items()):
        if not isinstance(name, str) or not name:
            raise ValueError("repair norm key is malformed")
        dense[f"norms/{name}"] = _as_numpy(name, value)
    for name, value in sorted(outputs.items()):
        if not isinstance(name, str) or _OUTPUT_RE.fullmatch(name) is None:
            raise ValueError(f"repair output key is malformed: {name!r}")
        array = _as_numpy(name, value)
        if array.shape != ():
            raise ValueError(f"repair output gain {name} must be scalar")
        dense[f"outputs/{name}"] = array

    actual = (len(codebooks), len(norms), len(outputs))
    if actual != expected_counts:
        raise ValueError(
            f"repair state surface count drift: expected={expected_counts} actual={actual}"
        )
    return {
        "codebooks": codebooks,
        "dense_tensors": dense,
        "norm_count": len(norms),
        "output_count": len(outputs),
    }


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    actual = _sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain an object")
    return value


def load_repair_bundle(
    *,
    checkpoint: str | Path,
    checkpoint_sha256: str,
    active_overlay: str | Path,
    active_overlay_sha256: str,
    assignment: str | Path,
    assignment_sha256: str,
    update: int,
) -> RepairBundle:
    checkpoint_path = Path(checkpoint).resolve()
    overlay_path = Path(active_overlay).resolve()
    assignment_path = Path(assignment).resolve()
    for value, label in (
        (checkpoint_sha256, "repair checkpoint SHA-256"),
        (active_overlay_sha256, "active overlay SHA-256"),
        (assignment_sha256, "assignment SHA-256"),
    ):
        _require_sha256(value, label)
    actual_checkpoint_sha = _sha256_file(checkpoint_path)
    if actual_checkpoint_sha != checkpoint_sha256:
        raise ValueError(
            "repair checkpoint SHA-256 mismatch: "
            f"expected={checkpoint_sha256} actual={actual_checkpoint_sha}"
        )
    overlay_doc = _load_bound_json(
        overlay_path, active_overlay_sha256, "active overlay"
    )
    assignment_doc = _load_bound_json(
        assignment_path, assignment_sha256, "assignment"
    )
    if overlay_doc.get("status") != "PASS_EXACT_ACTIVE_LAYERS" or overlay_doc.get(
        "stale"
    ) is not False:
        raise ValueError("active overlay is not a non-stale PASS_EXACT_ACTIVE_LAYERS seal")
    for field in ("active_assignment_sha256", "final_assignment_sha256"):
        if overlay_doc.get(field) != assignment_sha256:
            raise ValueError(f"active overlay {field} does not bind the assignment")
    assignment_map = assignment_doc.get("assignment")
    if not isinstance(assignment_map, dict) or set(assignment_map) != {
        str(layer) for layer in range(43)
    }:
        raise ValueError("assignment must bind exactly layers 0..42")

    try:
        import torch

        payload = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError(f"cannot load weights-only repair checkpoint: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("repair checkpoint payload is not a mapping")
    normalized_format = _normalize_checkpoint_format(payload.get("format"))
    exact = {
        "mechanism": REPAIR_MECHANISM,
        "next_update": update,
    }
    drift = {
        key: (payload.get(key), expected)
        for key, expected in exact.items()
        if payload.get(key) != expected
    }
    if drift:
        raise ValueError(f"repair checkpoint header drift: {drift}")
    validated = validate_repair_state(payload.get("state"), expected_counts=PRODUCTION_COUNTS)
    return RepairBundle(
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        active_overlay_path=overlay_path,
        active_overlay_sha256=active_overlay_sha256,
        assignment_path=assignment_path,
        assignment_sha256=assignment_sha256,
        checkpoint_format=normalized_format,
        mechanism=REPAIR_MECHANISM,
        update=update,
        codebooks=validated["codebooks"],
        dense_tensors=validated["dense_tensors"],
        norm_count=validated["norm_count"],
        output_count=validated["output_count"],
    )


def materialize_codebook_plane(
    source: Path,
    destination: Path,
    repairs: Mapping[str, CodebookRepair],
) -> list[dict[str, Any]] | None:
    if source.name != "codebooks.npy" and not source.name.endswith(".codebooks.npy"):
        return None
    base = np.load(source, mmap_mode="r", allow_pickle=False)
    if base.dtype != np.dtype("float16"):
        raise ValueError(f"repair codebook source must be float16: {source}")
    if base.ndim == 2:
        slices = [(None, base)]
    elif base.ndim == 3:
        slices = [(index, base[index]) for index in range(base.shape[0])]
    else:
        raise ValueError(
            f"repair codebook source must be a matrix or matrix stack: {source}"
        )
    selected: list[tuple[int | None, str, CodebookRepair]] = []
    for index, array in slices:
        source_wire_sha = _wire_sha(array)
        repair = repairs.get(source_wire_sha)
        if repair is not None:
            selected.append((index, source_wire_sha, repair))
    if not selected:
        return None
    materialized = np.array(base, copy=True)
    rows: list[dict[str, Any]] = []
    for index, source_wire_sha, repair in selected:
        target = materialized if index is None else materialized[index]
        if tuple(repair.array.shape) != tuple(target.shape):
            raise ValueError(
                f"repair codebook shape drift for {source}: "
                f"checkpoint={repair.array.shape} target={target.shape} index={index}"
            )
        target[...] = repair.array
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, materialized, allow_pickle=False)
    readback = np.load(destination, mmap_mode="r", allow_pickle=False)
    source_npy_sha = _sha256_file(source)
    materialized_npy_sha = _sha256_file(destination)
    for index, source_wire_sha, repair in selected:
        actual = readback if index is None else readback[index]
        materialized_wire_sha = _wire_sha(actual)
        expected_wire_sha = _wire_sha(repair.array)
        if materialized_wire_sha != expected_wire_sha:
            raise ValueError(
                f"repair codebook readback drift: {destination} index={index}"
            )
        rows.append(
            {
                "checkpoint_key": repair.checkpoint_key,
                "codebook_index": index,
                "source_wire_sha256": source_wire_sha,
                "materialized_wire_sha256": materialized_wire_sha,
                "source_npy_sha256": source_npy_sha,
                "materialized_npy_sha256": materialized_npy_sha,
                "path": destination.as_posix(),
                "shape": list(repair.array.shape),
                "dtype": str(repair.array.dtype),
                "data_bytes": int(repair.array.nbytes),
            }
        )
    return rows


def write_repair_payload(
    root: Path,
    bundle: RepairBundle,
    codebook_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    covered = {row["checkpoint_key"] for row in codebook_rows}
    expected = {repair.checkpoint_key for repair in bundle.codebooks.values()}
    missing = sorted(expected - covered)
    if missing:
        raise ValueError(
            "checkpoint codebooks were not materialized into the plane source: "
            f"missing={missing[:8]} count={len(missing)}"
        )
    state_path = root / REPAIR_STATE_PATH
    state_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        bundle.dense_tensors,
        state_path,
        metadata={
            "format": bundle.checkpoint_format,
            "mechanism": bundle.mechanism,
            "update": str(bundle.update),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "active_overlay_sha256": bundle.active_overlay_sha256,
            "assignment_sha256": bundle.assignment_sha256,
        },
    )
    dense_rows = []
    for name, array in sorted(bundle.dense_tensors.items()):
        dense_rows.append(
            {
                "name": name,
                "kind": name.split("/", 1)[0],
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data_bytes": int(array.nbytes),
                "data_sha256": _wire_sha(array),
            }
        )
    normalized_codebooks = []
    for row in sorted(codebook_rows, key=lambda item: (item["checkpoint_key"], item["path"])):
        normalized = dict(row)
        normalized["path"] = Path(row["path"]).relative_to(root).as_posix()
        normalized_codebooks.append(normalized)
    document = {
        "schema": "bs-repair-materialization-v1",
        "status": "MATERIALIZED",
        "format": bundle.checkpoint_format,
        "mechanism": bundle.mechanism,
        "update": bundle.update,
        "identity": {
            "checkpoint": {
                "path": str(bundle.checkpoint_path),
                "sha256": bundle.checkpoint_sha256,
            },
            "active_overlay": {
                "path": str(bundle.active_overlay_path),
                "sha256": bundle.active_overlay_sha256,
            },
            "assignment": {
                "path": str(bundle.assignment_path),
                "sha256": bundle.assignment_sha256,
            },
        },
        "codebook_checkpoint_keys": len(bundle.codebooks),
        "codebook_target_files": len(normalized_codebooks),
        "codebooks": normalized_codebooks,
        "dense_state": {
            "path": REPAIR_STATE_PATH.as_posix(),
            "sha256": _sha256_file(state_path),
            "norms": bundle.norm_count,
            "outputs": bundle.output_count,
            "tensors": dense_rows,
        },
    }
    manifest_path = root / REPAIR_MANIFEST_PATH
    manifest_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "MATERIALIZED",
        "manifest": REPAIR_MANIFEST_PATH.as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "state": REPAIR_STATE_PATH.as_posix(),
        "state_sha256": _sha256_file(state_path),
        "format": bundle.checkpoint_format,
        "mechanism": bundle.mechanism,
        "update": bundle.update,
        "checkpoint_sha256": bundle.checkpoint_sha256,
        "active_overlay_sha256": bundle.active_overlay_sha256,
        "assignment_sha256": bundle.assignment_sha256,
        "codebook_checkpoint_keys": len(bundle.codebooks),
        "codebook_target_files": len(normalized_codebooks),
        "norms": bundle.norm_count,
        "outputs": bundle.output_count,
    }


def verify_repair_payload(root: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / str(summary.get("manifest"))
    state_path = root / str(summary.get("state"))
    if _sha256_file(manifest_path) != summary.get("manifest_sha256"):
        raise ValueError("repair manifest SHA-256 mismatch")
    if _sha256_file(state_path) != summary.get("state_sha256"):
        raise ValueError("repair state SHA-256 mismatch")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_summary = {
        "status": document.get("status"),
        "format": document.get("format"),
        "mechanism": document.get("mechanism"),
        "update": document.get("update"),
        "checkpoint_sha256": document.get("identity", {})
        .get("checkpoint", {})
        .get("sha256"),
        "active_overlay_sha256": document.get("identity", {})
        .get("active_overlay", {})
        .get("sha256"),
        "assignment_sha256": document.get("identity", {})
        .get("assignment", {})
        .get("sha256"),
        "codebook_checkpoint_keys": document.get("codebook_checkpoint_keys"),
        "codebook_target_files": document.get("codebook_target_files"),
        "norms": document.get("dense_state", {}).get("norms"),
        "outputs": document.get("dense_state", {}).get("outputs"),
    }
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise ValueError(f"repair summary drift for {key}")
    seen_keys: set[str] = set()
    for row in document.get("codebooks", []):
        path = root / row["path"]
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        index = row.get("codebook_index")
        target = array if index is None else array[index]
        if _wire_sha(target) != row.get("materialized_wire_sha256"):
            raise ValueError(f"repair codebook wire drift: {row.get('path')}")
        seen_keys.add(row["checkpoint_key"])
    if len(seen_keys) != document.get("codebook_checkpoint_keys"):
        raise ValueError("repair codebook checkpoint-key coverage drift")

    dense_rows = document.get("dense_state", {}).get("tensors")
    if not isinstance(dense_rows, list):
        raise ValueError("repair dense tensor manifest is missing")
    expected_names = {row["name"] for row in dense_rows}
    with safe_open(state_path, framework="np") as handle:
        if set(handle.keys()) != expected_names:
            raise ValueError("repair dense tensor key surface drift")
        for row in dense_rows:
            array = handle.get_tensor(row["name"])
            actual = {
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "data_bytes": int(array.nbytes),
                "data_sha256": _wire_sha(array),
            }
            for key, value in actual.items():
                if row.get(key) != value:
                    raise ValueError(
                        f"repair dense tensor drift {row['name']}.{key}"
                    )
    return {
        "status": "PASS",
        "format": document["format"],
        "update": document["update"],
        "codebook_checkpoint_keys": document["codebook_checkpoint_keys"],
        "codebook_target_files": document["codebook_target_files"],
        "norms": document["dense_state"]["norms"],
        "outputs": document["dense_state"]["outputs"],
    }
