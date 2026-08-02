from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class KnapsackValidationError(ValueError):
    """Raised when manifest-bound knapsack inputs are incomplete or inconsistent."""


@dataclass(frozen=True)
class _Source:
    path: Path
    relative_path: str
    sha256: str
    byte_count: int
    producer_command: str
    remedy_command: str
    payload: dict[str, Any]


def _canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise KnapsackValidationError(f"cannot read {label} {path}: {exc}") from exc
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnapsackValidationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KnapsackValidationError(f"{label} {path} must contain a JSON object")
    return value, payload


def _local_path(root: Path, raw_path: object, *, label: str) -> tuple[Path, str]:
    if not isinstance(raw_path, str) or not raw_path:
        raise KnapsackValidationError(f"{label} path must be a non-empty string")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise KnapsackValidationError(f"{label} path must be local to run root: {raw_path}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KnapsackValidationError(f"{label} path escapes run root: {raw_path}") from exc
    return resolved, relative.as_posix()


def _sha_field(value: object, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise KnapsackValidationError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _producer(descriptor: dict[str, Any], *, fallback: str) -> str:
    value = descriptor.get("producer_command", fallback)
    if not isinstance(value, str) or not value.strip():
        raise KnapsackValidationError("producer_command must be a non-empty string")
    return value


def _preflight_source(
    *,
    root: Path,
    descriptor: object,
    label: str,
    missing_message: str,
    fallback_producer: str,
) -> _Source:
    if not isinstance(descriptor, dict):
        raise KnapsackValidationError(f"{label} descriptor must be an object")
    producer_command = _producer(descriptor, fallback=fallback_producer)
    path, relative_path = _local_path(root, descriptor.get("path"), label=label)
    expected_sha = _sha_field(descriptor.get("sha256"), label=f"{label} sha256")
    if not path.is_file():
        raise KnapsackValidationError(
            f"{missing_message}: {path}; required producer: {fallback_producer}"
        )
    payload = path.read_bytes()
    actual_sha = _sha256(payload)
    if actual_sha != expected_sha:
        raise KnapsackValidationError(
            f"{label} SHA-256 mismatch: {path}; expected {expected_sha}, got {actual_sha}; "
            f"required producer: {fallback_producer}"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KnapsackValidationError(f"invalid JSON in {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise KnapsackValidationError(f"{label} {path} must contain a JSON object")
    return _Source(
        path=path,
        relative_path=relative_path,
        sha256=actual_sha,
        byte_count=len(payload),
        producer_command=producer_command,
        remedy_command=fallback_producer,
        payload=value,
    )


def _intended_tiers(manifest: dict[str, Any]) -> list[str]:
    value = manifest.get("intended_tiers")
    if not isinstance(value, list) or not value:
        raise KnapsackValidationError("run manifest intended_tiers must be a non-empty list")
    tiers: list[str] = []
    for index, tier in enumerate(value):
        if not isinstance(tier, str) or not tier:
            raise KnapsackValidationError(
                f"run manifest intended_tiers[{index}] must be a non-empty string"
            )
        if tier in tiers:
            raise KnapsackValidationError(f"duplicate intended tier {tier!r}")
        tiers.append(tier)
    return tiers


def _basis(manifest: dict[str, Any]) -> str:
    for field in ("intended_basis_sha256", "basis_sha256"):
        if field in manifest:
            return _sha_field(manifest[field], label=f"run manifest {field}")
    intended = manifest.get("intended_basis")
    if isinstance(intended, dict) and "model_index_sha256" in intended:
        return _sha_field(
            intended["model_index_sha256"], label="run manifest intended_basis.model_index_sha256"
        )
    raise KnapsackValidationError(
        "run manifest must declare intended_basis_sha256, basis_sha256, or "
        "intended_basis.model_index_sha256"
    )


def _source_basis(source: _Source, *, label: str, intended_basis: str) -> None:
    actual = source.payload.get("basis_sha256")
    if actual is None and isinstance(source.payload.get("intended_basis"), dict):
        actual = source.payload["intended_basis"].get("model_index_sha256")
    actual_sha = _sha_field(actual, label=f"{label} basis_sha256")
    if actual_sha != intended_basis:
        raise KnapsackValidationError(
            f"{label} basis mismatch: expected {intended_basis}, got {actual_sha} at {source.path}"
        )


def _anchor_cells(source: _Source, *, tier: str, intended_basis: str) -> dict[str, int]:
    value = source.payload
    if value.get("tier") != tier:
        raise KnapsackValidationError(
            f"anchor manifest tier mismatch at {source.path}: expected {tier!r}, "
            f"got {value.get('tier')!r}"
        )
    status = value.get("status")
    if not isinstance(status, str) or not (status == "SEALED" or status.startswith("PASS")):
        raise KnapsackValidationError(
            f"anchor manifest for tier {tier!r} is not sealed/PASS at {source.path}: {status!r}"
        )
    _source_basis(source, label=f"anchor manifest for tier {tier!r}", intended_basis=intended_basis)
    rows = value.get("cells")
    if not isinstance(rows, list) or not rows:
        raise KnapsackValidationError(
            f"anchor manifest for tier {tier!r} must contain a non-empty cells list"
        )
    cells: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise KnapsackValidationError(f"anchor {tier!r} cells[{index}] must be an object")
        cell_id = row.get("cell_id")
        byte_count = row.get("bytes")
        if not isinstance(cell_id, str) or not cell_id:
            raise KnapsackValidationError(
                f"anchor {tier!r} cells[{index}].cell_id must be a non-empty string"
            )
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise KnapsackValidationError(
                f"anchor {tier!r} cell {cell_id!r} bytes must be a non-negative integer"
            )
        if cell_id in cells:
            raise KnapsackValidationError(
                f"duplicate cell {cell_id!r} in anchor manifest for tier {tier!r}"
            )
        cells[cell_id] = byte_count
    return cells


def _damage_values(
    source: _Source,
    *,
    cells: list[str],
    tiers: list[str],
    intended_basis: str,
) -> dict[tuple[str, str], float]:
    _source_basis(source, label="damage rows", intended_basis=intended_basis)
    rows = source.payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise KnapsackValidationError("damage rows manifest must contain a non-empty rows list")
    allowed = {(cell, tier) for cell in cells for tier in tiers}
    values: dict[tuple[str, str], float] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise KnapsackValidationError(f"damage rows[{index}] must be an object")
        key = (row.get("cell_id"), row.get("tier"))
        if key not in allowed:
            raise KnapsackValidationError(
                f"damage rows[{index}] identifies undeclared cell/tier pair {key!r}"
            )
        raw_damage = row.get("damage")
        if isinstance(raw_damage, bool) or not isinstance(raw_damage, (int, float)):
            raise KnapsackValidationError(
                f"damage for cell {key[0]!r}, tier {key[1]!r} must be numeric"
            )
        damage = float(raw_damage)
        if not math.isfinite(damage):
            raise KnapsackValidationError(
                f"damage for cell {key[0]!r}, tier {key[1]!r} must be finite"
            )
        if key in values:
            raise KnapsackValidationError(f"duplicate damage row for {key!r}")
        values[key] = damage
    missing = sorted(allowed - values.keys())
    if missing:
        cell, tier = missing[0]
        raise KnapsackValidationError(
            f"missing damage row for cell {cell!r}, intended tier {tier!r}; "
            f"required producer: {source.remedy_command}"
        )
    return values


def _write_once(path: Path, value: object) -> tuple[str, int]:
    payload = _canonical_json(value)
    digest = _sha256(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_bytes()
        if existing != payload:
            raise FileExistsError(f"refusing to replace different sealed output: {path}")
        return digest, len(payload)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise FileExistsError(f"refusing to replace different sealed output: {path}")
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return digest, len(payload)


def _preflight_write_once(path: Path, value: object) -> None:
    """Refuse conflicting sealed outputs before publishing either paired output."""

    payload = _canonical_json(value)
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError(f"refusing to replace different sealed output: {path}")


def _output_path(root: Path, value: Path | None, *, default: str, label: str) -> Path:
    if value is None:
        value = Path(default)
    if value.is_absolute():
        resolved = value.resolve()
    else:
        resolved = (root / value).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise KnapsackValidationError(f"{label} must be local to run root: {resolved}") from exc
    return resolved


def _metadata_nodes(receipt: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [receipt]
    for field in ("verified_sources", "sealed_shards"):
        value = receipt.get(field, [])
        if not isinstance(value, list):
            raise KnapsackValidationError(f"receipt {field} must be a list")
        for index, node in enumerate(value):
            if not isinstance(node, dict):
                raise KnapsackValidationError(f"receipt {field}[{index}] must be an object")
            nodes.append(node)
    return nodes


def _node_tiers(node: dict[str, Any]) -> list[str]:
    values: object | None = None
    intended = node.get("intended_tiers")
    if isinstance(intended, list) and intended:
        values = intended
    else:
        identity = node.get("identity_coverage")
        if isinstance(identity, dict) and "tiers" in identity:
            values = identity["tiers"]
        elif "tiers" in node:
            values = node["tiers"]
        elif "tier" in node:
            values = [node["tier"]]
    if values is None:
        return []
    if not isinstance(values, list):
        raise KnapsackValidationError("receipt tiers metadata must be a list")
    result: list[str] = []
    for index, tier in enumerate(values):
        if not isinstance(tier, str) or not tier:
            raise KnapsackValidationError(
                f"receipt tiers metadata[{index}] must be a non-empty string"
            )
        result.append(tier)
    return result


def _receipt_basis(receipt: dict[str, Any], *, path: Path) -> str:
    candidates: list[str] = []
    for node in _metadata_nodes(receipt):
        for field in ("basis_sha256", "intended_basis_sha256"):
            if field in node:
                candidates.append(_sha_field(node[field], label=f"{path} {field}"))
        intended = node.get("intended_basis")
        if isinstance(intended, str):
            candidates.append(_sha_field(intended, label=f"{path} intended_basis"))
        elif isinstance(intended, dict) and "model_index_sha256" in intended:
            candidates.append(
                _sha_field(
                    intended["model_index_sha256"],
                    label=f"{path} intended_basis.model_index_sha256",
                )
            )
    if not candidates:
        raise KnapsackValidationError(f"receipt does not declare a basis SHA-256: {path}")
    if len(set(candidates)) != 1:
        raise KnapsackValidationError(f"receipt basis mismatch within {path}: {sorted(set(candidates))}")
    return candidates[0]


def _merge_descriptor(
    target: dict[str, Any], *, key: str, descriptor: object, label: str
) -> None:
    if not isinstance(descriptor, dict):
        raise KnapsackValidationError(f"{label} descriptor for {key!r} must be an object")
    existing = target.get(key)
    if existing is not None and existing != descriptor:
        raise KnapsackValidationError(f"conflicting {label} descriptors for {key!r}")
    target[key] = descriptor


def build_knapsack_input_index(
    *,
    receipts: list[str | Path],
    output: str | Path,
    selection_receipt: str | Path,
    envelope_bytes: int,
) -> dict[str, Any]:
    """Build a deterministic open-tier knapsack index from sealed receipt metadata."""

    if not receipts:
        raise KnapsackValidationError("at least one sealed receipt is required")
    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    output_path = Path(output).expanduser().resolve()
    selection_path = Path(selection_receipt).expanduser().resolve()
    if output_path == selection_path:
        raise KnapsackValidationError("output and selection receipt paths must differ")

    source_rows: list[dict[str, Any]] = []
    tier_names: set[str] = set()
    bases: set[str] = set()
    missing_inputs: list[dict[str, Any]] = []
    anchor_manifests: dict[str, Any] = {}
    damage_rows: dict[str, Any] | None = None
    for raw_path in receipts:
        path = Path(raw_path).expanduser().resolve()
        value, payload = _read_object(path, label="sealed anchor receipt")
        status = value.get("status")
        if not isinstance(status, str) or not any(
            marker in status.upper() for marker in ("PASS", "SEALED", "MERGEABLE")
        ):
            raise KnapsackValidationError(f"anchor receipt is not sealed/PASS at {path}: {status!r}")
        basis = _receipt_basis(value, path=path)
        bases.add(basis)
        declared_envelope = value.get("envelope_bytes")
        if declared_envelope is not None and declared_envelope != envelope_bytes:
            raise KnapsackValidationError(
                f"receipt envelope_bytes mismatch at {path}: "
                f"expected {envelope_bytes}, got {declared_envelope}"
            )
        for node in _metadata_nodes(value):
            tier_names.update(_node_tiers(node))
            descriptors = node.get("anchor_manifests")
            if descriptors is not None:
                if not isinstance(descriptors, dict):
                    raise KnapsackValidationError("receipt anchor_manifests must be an object")
                for tier, descriptor in descriptors.items():
                    if not isinstance(tier, str) or not tier:
                        raise KnapsackValidationError("anchor manifest tier keys must be non-empty strings")
                    tier_names.add(tier)
                    _merge_descriptor(
                        anchor_manifests,
                        key=tier,
                        descriptor=descriptor,
                        label="anchor manifest",
                    )
            descriptor = node.get("anchor_manifest")
            node_tiers = _node_tiers(node)
            if descriptor is not None:
                if len(node_tiers) != 1:
                    raise KnapsackValidationError(
                        "anchor_manifest metadata requires exactly one declared tier"
                    )
                _merge_descriptor(
                    anchor_manifests,
                    key=node_tiers[0],
                    descriptor=descriptor,
                    label="anchor manifest",
                )
            current_damage = node.get("damage_rows")
            if current_damage is not None:
                if not isinstance(current_damage, dict):
                    raise KnapsackValidationError("receipt damage_rows must be an object")
                if damage_rows is not None and damage_rows != current_damage:
                    raise KnapsackValidationError("conflicting damage_rows descriptors")
                damage_rows = current_damage
        for field in ("missing_set", "missing_inputs"):
            rows = value.get(field, [])
            if not isinstance(rows, list):
                raise KnapsackValidationError(f"receipt {field} must be a list")
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    raise KnapsackValidationError(f"receipt {field}[{index}] must be an object")
                missing_inputs.append(row)
        source_rows.append(
            {
                "path": str(path),
                "sha256": _sha256(payload),
                "bytes": len(payload),
                "schema": value.get("schema"),
                "status": status,
            }
        )

    if len(bases) != 1:
        raise KnapsackValidationError(f"receipt basis mismatch: {sorted(bases)}")
    if not tier_names:
        raise KnapsackValidationError("sealed receipt metadata declares no tiers")
    selected_tiers = sorted(tier_names)
    source_rows.sort(key=lambda row: (row["sha256"], row["path"]))
    missing_inputs.sort(key=lambda row: _canonical_json(row))
    status = "PRELIM_NOT_DECISION_GRADE" if missing_inputs else "PASS"
    basis = next(iter(bases))
    index_value: dict[str, Any] = {
        "schema": "banana-smasher-knapsack-input-index-v1",
        "status": status,
        "intended_basis_sha256": basis,
        "intended_tiers": selected_tiers,
        "envelope_bytes": envelope_bytes,
        "source_receipts": source_rows,
    }
    if anchor_manifests:
        index_value["anchor_manifests"] = {
            tier: anchor_manifests[tier] for tier in sorted(anchor_manifests)
        }
    if damage_rows is not None:
        index_value["damage_rows"] = damage_rows
    if missing_inputs:
        index_value["missing_inputs"] = missing_inputs

    index_payload = _canonical_json(index_value)
    selection_value = {
        "schema": "banana-smasher-knapsack-index-receipt-v1",
        "status": status,
        "basis_sha256": basis,
        "selected_tiers": selected_tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing_inputs,
        "source_receipts": source_rows,
        "input_index": {
            "path": str(output_path),
            "sha256": _sha256(index_payload),
            "bytes": len(index_payload),
        },
    }
    _preflight_write_once(output_path, index_value)
    _preflight_write_once(selection_path, selection_value)
    index_sha, index_bytes = _write_once(output_path, index_value)
    selection_sha, selection_bytes = _write_once(selection_path, selection_value)
    return {
        "status": status,
        "command": "knapsack-index",
        "basis_sha256": basis,
        "selected_tiers": selected_tiers,
        "byte_accounting": {"envelope_bytes": envelope_bytes},
        "missing_inputs": missing_inputs,
        "input_index": {
            "path": str(output_path),
            "sha256": index_sha,
            "bytes": index_bytes,
        },
        "receipt": {
            "path": str(selection_path),
            "sha256": selection_sha,
            "bytes": selection_bytes,
        },
    }


def run_knapsack(
    *,
    run_root: str | Path,
    envelope_bytes: int,
    output: str | Path | None = None,
    receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Solve a manifest-bound multiple-choice integer knapsack exactly with HiGHS."""

    if isinstance(envelope_bytes, bool) or not isinstance(envelope_bytes, int) or envelope_bytes < 0:
        raise KnapsackValidationError("envelope_bytes must be a non-negative integer")
    root = Path(run_root).expanduser().resolve()
    manifest_path = root / "MANIFEST.json"
    manifest, manifest_bytes = _read_object(manifest_path, label="run manifest")
    tiers = _intended_tiers(manifest)
    intended_basis = _basis(manifest)
    descriptors = manifest.get("anchor_manifests")
    if not isinstance(descriptors, dict):
        raise KnapsackValidationError("run manifest anchor_manifests must be an object")

    # Complete local+SHA preflight for every declared tier before parsing or solving any one tier.
    anchor_sources: dict[str, _Source] = {}
    anchor_producer = f"smash anchor --run-root {root}"
    for tier in tiers:
        descriptor = descriptors.get(tier)
        if descriptor is None:
            raise KnapsackValidationError(
                f"missing intended anchor manifest descriptor for tier {tier!r}; "
                f"required producer: {anchor_producer}"
            )
        anchor_sources[tier] = _preflight_source(
            root=root,
            descriptor=descriptor,
            label=f"anchor manifest for tier {tier!r}",
            missing_message=f"missing intended anchor manifest for tier {tier!r}",
            fallback_producer=anchor_producer,
        )

    damage_descriptor = manifest.get("damage_rows")
    if damage_descriptor is None:
        raise KnapsackValidationError(
            "missing damage rows manifest descriptor; "
            f"required producer: {anchor_producer}"
        )
    damage_source = _preflight_source(
        root=root,
        descriptor=damage_descriptor,
        label="damage rows manifest",
        missing_message="missing damage rows manifest",
        fallback_producer=anchor_producer,
    )

    costs: dict[str, dict[str, int]] = {}
    expected_cells: set[str] | None = None
    for tier in tiers:
        costs[tier] = _anchor_cells(
            anchor_sources[tier], tier=tier, intended_basis=intended_basis
        )
        current_cells = set(costs[tier])
        if expected_cells is None:
            expected_cells = current_cells
        elif current_cells != expected_cells:
            missing = sorted(expected_cells - current_cells)
            extra = sorted(current_cells - expected_cells)
            raise KnapsackValidationError(
                f"anchor cell-set mismatch for tier {tier!r}: missing={missing[:3]}, extra={extra[:3]}"
            )
    cells = sorted(expected_cells or ())
    damages = _damage_values(
        damage_source, cells=cells, tiers=tiers, intended_basis=intended_basis
    )
    minimum_required_bytes = sum(min(costs[tier][cell] for tier in tiers) for cell in cells)
    if minimum_required_bytes > envelope_bytes:
        raise KnapsackValidationError(
            f"envelope infeasible: minimum required {minimum_required_bytes} bytes exceeds "
            f"--envelope-bytes {envelope_bytes}"
        )

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:  # pragma: no cover - exercised by installation smoke tests
        raise RuntimeError(
            "exact knapsack solver unavailable; install banana-smasher[knapsack] "
            "(requires scipy)"
        ) from exc

    # HiGHS accepts only float64 coefficients. Subtract the mandatory per-cell
    # baseline in Python integers, divide the remaining byte deltas by their
    # exact GCD, and refuse any still-binding row outside exact float64 range.
    baseline_by_cell = {
        cell: min(costs[tier][cell] for tier in tiers) for cell in cells
    }
    remaining_envelope = envelope_bytes - minimum_required_bytes
    byte_deltas = {
        (cell, tier): costs[tier][cell] - baseline_by_cell[cell]
        for cell in cells
        for tier in tiers
    }
    feasible_positive_deltas = [
        delta for delta in byte_deltas.values() if 0 < delta <= remaining_envelope
    ]
    byte_divisor = math.gcd(*feasible_positive_deltas) if feasible_positive_deltas else 1
    scaled_capacity = remaining_envelope // byte_divisor
    scaled_deltas = {
        key: delta // byte_divisor
        for key, delta in byte_deltas.items()
        if delta <= remaining_envelope
    }
    maximum_scaled_use = sum(
        max(scaled_deltas.get((cell, tier), 0) for tier in tiers) for cell in cells
    )
    exact_float_integer_max = 2**53
    enforce_byte_constraint = scaled_capacity < maximum_scaled_use
    if enforce_byte_constraint and (
        scaled_capacity > exact_float_integer_max
        or any(delta > exact_float_integer_max for delta in scaled_deltas.values())
    ):
        raise KnapsackValidationError(
            "exact byte constraint remains outside float64 integer range after "
            f"baseline/GCD normalization: capacity={scaled_capacity}, "
            f"divisor={byte_divisor}"
        )

    variable_count = len(cells) * len(tiers)
    objective = np.empty(variable_count, dtype=np.float64)
    variable_upper = np.ones(variable_count, dtype=np.float64)
    row_indices: list[int] = []
    column_indices: list[int] = []
    coefficients: list[float] = []
    for cell_index, cell in enumerate(cells):
        for tier_index, tier in enumerate(tiers):
            variable_index = cell_index * len(tiers) + tier_index
            objective[variable_index] = damages[(cell, tier)]
            row_indices.append(cell_index)
            column_indices.append(variable_index)
            coefficients.append(1.0)
            delta = byte_deltas[(cell, tier)]
            if delta > remaining_envelope:
                variable_upper[variable_index] = 0.0
            elif enforce_byte_constraint:
                row_indices.append(len(cells))
                column_indices.append(variable_index)
                coefficients.append(float(scaled_deltas[(cell, tier)]))
    constraint_count = len(cells) + int(enforce_byte_constraint)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(constraint_count, variable_count),
    ).tocsr()
    lower = np.ones(constraint_count)
    upper = np.ones(constraint_count)
    if enforce_byte_constraint:
        lower[-1] = -np.inf
        upper[-1] = float(scaled_capacity)
    solution = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int8),
        bounds=Bounds(np.zeros(variable_count), variable_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"presolve": True, "mip_rel_gap": 0.0},
    )
    if not solution.success or solution.x is None or int(solution.status) != 0:
        raise RuntimeError(
            f"exact knapsack solve failed: status={solution.status}, message={solution.message}"
        )
    if float(getattr(solution, "mip_gap", math.inf)) != 0.0:
        raise RuntimeError(
            "exact knapsack solve returned a nonzero MIP gap: "
            f"{getattr(solution, 'mip_gap', None)}"
        )
    rounded = np.rint(solution.x).astype(np.int8)
    if not np.allclose(solution.x, rounded, rtol=0.0, atol=1e-6):
        raise RuntimeError("exact knapsack solver returned a non-integral assignment")

    assignments: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(cells):
        offset = cell_index * len(tiers)
        selected = np.flatnonzero(rounded[offset : offset + len(tiers)])
        if len(selected) != 1:
            raise RuntimeError(
                f"exact knapsack solver selected {len(selected)} tiers for cell {cell!r}"
            )
        tier = tiers[int(selected[0])]
        assignments.append(
            {
                "cell_id": cell,
                "tier": tier,
                "bytes": costs[tier][cell],
                "damage": damages[(cell, tier)],
            }
        )
    assigned_bytes = sum(item["bytes"] for item in assignments)
    if assigned_bytes > envelope_bytes:
        raise RuntimeError(
            f"exact knapsack solver violated envelope: {assigned_bytes} > {envelope_bytes}"
        )
    total_damage = math.fsum(item["damage"] for item in assignments)

    output_path = _output_path(
        root, Path(output) if output is not None else None, default="knapsack/ASSIGNMENT.json", label="output"
    )
    receipt_path = _output_path(
        root, Path(receipt) if receipt is not None else None, default="knapsack/RECEIPT.json", label="receipt"
    )
    if output_path == receipt_path:
        raise KnapsackValidationError("output and receipt paths must differ")
    assignment_value = {
        "schema": "banana-smasher-knapsack-assignment-v1",
        "status": "PASS",
        "basis_sha256": intended_basis,
        "tiers": tiers,
        "objective": {"name": "min_total_damage", "total_damage": total_damage},
        "byte_accounting": {
            "assigned_bytes": assigned_bytes,
            "envelope_bytes": envelope_bytes,
            "slack_bytes": envelope_bytes - assigned_bytes,
        },
        "assignments": assignments,
    }
    assignment_payload = _canonical_json(assignment_value)
    assignment_sha = _sha256(assignment_payload)
    assignment_bytes = len(assignment_payload)
    receipt_value = {
        "schema": "banana-smasher-knapsack-receipt-v1",
        "status": "PASS",
        "run_root": str(root),
        "basis_sha256": intended_basis,
        "tiers": tiers,
        "cell_count": len(cells),
        "objective": assignment_value["objective"],
        "byte_accounting": assignment_value["byte_accounting"],
        "run_manifest": {
            "path": "MANIFEST.json",
            "sha256": _sha256(manifest_bytes),
            "bytes": len(manifest_bytes),
        },
        "anchor_manifests": [
            {
                "tier": tier,
                "path": anchor_sources[tier].relative_path,
                "sha256": anchor_sources[tier].sha256,
                "bytes": anchor_sources[tier].byte_count,
                "producer_command": anchor_sources[tier].producer_command,
            }
            for tier in tiers
        ],
        "damage_rows": {
            "path": damage_source.relative_path,
            "sha256": damage_source.sha256,
            "bytes": damage_source.byte_count,
            "producer_command": damage_source.producer_command,
        },
        "assignment": {
            "path": output_path.relative_to(root).as_posix(),
            "sha256": assignment_sha,
            "bytes": assignment_bytes,
        },
        "solver": {
            "backend": "scipy.optimize.milp/HiGHS",
            "status": int(solution.status),
            "message": str(solution.message),
            "mip_gap": float(getattr(solution, "mip_gap", 0.0)),
            "byte_normalization": {
                "baseline_bytes": minimum_required_bytes,
                "remaining_envelope_bytes": remaining_envelope,
                "gcd_divisor": byte_divisor,
                "scaled_capacity": scaled_capacity,
                "constraint_required": enforce_byte_constraint,
            },
        },
    }
    # Validate both immutable destinations before publishing either half of the
    # assignment/receipt pair. The receipt is fully staged from validated input
    # bytes and the canonical assignment payload before any PASS file appears.
    _preflight_write_once(output_path, assignment_value)
    _preflight_write_once(receipt_path, receipt_value)
    written_assignment_sha, written_assignment_bytes = _write_once(
        output_path, assignment_value
    )
    if (written_assignment_sha, written_assignment_bytes) != (
        assignment_sha,
        assignment_bytes,
    ):
        raise RuntimeError("canonical assignment changed during pair publication")
    receipt_sha, receipt_bytes = _write_once(receipt_path, receipt_value)
    return {
        "status": "PASS",
        "command": "knapsack",
        "run_root": str(root),
        "tiers": tiers,
        "cell_count": len(cells),
        "objective": assignment_value["objective"],
        "byte_accounting": assignment_value["byte_accounting"],
        "assignment": {
            "path": str(output_path),
            "sha256": assignment_sha,
            "bytes": assignment_bytes,
        },
        "receipt": {
            "path": str(receipt_path),
            "sha256": receipt_sha,
            "bytes": receipt_bytes,
        },
    }
