from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
import uuid

PRODUCER_VERB = "smash qtip-configs"
RUN_MANIFEST_NAME = "QTIP_RUN_MANIFEST.json"
OUTPUT_MANIFEST_NAME = "QTIP_CONFIG_MANIFEST.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_path(value: object, *, label: str, base: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid {label} path")
    if "://" in value:
        raise ValueError(f"{label} must be disk-local, got {value!r}")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _artifact(record: object, *, label: str, base: Path) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} artifact record")
    path = _local_path(record.get("path"), label=label, base=base)
    if not path.is_file():
        raise ValueError(f"missing local {label}; run {PRODUCER_VERB}: {path}")
    size = record.get("bytes")
    sha = record.get("sha256")
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or not isinstance(sha, str)
        or len(sha) != 64
    ):
        raise ValueError(f"invalid {label} hash/size record")
    observed_size = path.stat().st_size
    observed_sha = _sha256(path)
    if observed_size != size or observed_sha != sha:
        raise ValueError(
            f"{label} hash/size drift: {path}; "
            f"expected {size}/{sha}, observed {observed_size}/{observed_sha}"
        )
    return path


def _directory_binding(record: object, *, label: str, base: Path) -> Path:
    if not isinstance(record, dict):
        raise ValueError(f"invalid {label} directory binding")
    root = _local_path(record.get("path"), label=label, base=base)
    if not root.is_dir():
        raise ValueError(f"missing local {label}; run {PRODUCER_VERB}: {root}")
    seal = _artifact(record.get("manifest"), label=f"{label} manifest", base=base)
    if not seal.is_relative_to(root):
        raise ValueError(f"{label} manifest is outside its local root: {seal}")
    return root


def _geometry(value: object, *, tier: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"L", "K", "V"}:
        raise ValueError(f"manifest tier {tier!r} requires exact L/K/V geometry")
    result: dict[str, int] = {}
    for key in ("L", "K", "V"):
        item = value[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"invalid manifest geometry {key}={item!r} for tier {tier!r}")
        result[key] = item
    return result


def _unique_row(rows: object, *, key: str, value: object, label: str) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise ValueError(f"manifest {label} must be a list")
    matches = [row for row in rows if isinstance(row, dict) and row.get(key) == value]
    if len(matches) != 1:
        raise ValueError(f"manifest requires exactly one {label} for {key}={value!r}, got {len(matches)}")
    return matches[0]


def _atomic_bytes(path: Path, raw: bytes) -> bool:
    """Write and fsync new bytes, or preserve an identical existing file byte-for-byte."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"materialized output is not a file: {path}")
        existing = path.read_bytes()
        if existing != raw:
            raise ValueError(f"refuse to rewrite divergent materialized config: {path}")
        return False
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _validate_basis(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"invalid {label} basis SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {label} basis SHA-256") from exc
    return value


def _iter_selected_layers(tier_row: dict[str, Any], selected: list[int]) -> Iterable[dict[str, Any]]:
    rows = tier_row.get("layers")
    for layer in selected:
        yield _unique_row(rows, key="layer", value=layer, label="layer row")


def materialize_qtip_configs(
    manifest_path: Path,
    *,
    tier: str,
    layers: list[int],
    output_root: Path,
) -> dict[str, Any]:
    """Materialize hash-bound QTIP configs from one open-tier run manifest.

    The manifest owns every tier name, geometry, layer, model, bank, and runtime path.
    This function contains no campaign tier menu, layer count, model default, or remote
    transport. All source bytes are validated before the first output is written.
    """
    manifest_path = manifest_path.resolve()
    output_root = output_root.resolve()
    if not tier:
        raise ValueError("QTIP tier name must be non-empty")
    if not layers or len(set(layers)) != len(layers) or any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in layers
    ):
        raise ValueError(f"invalid materialization layers: {layers!r}")
    if not manifest_path.is_file():
        raise ValueError(f"missing QTIP run manifest; run {PRODUCER_VERB}: {manifest_path}")
    manifest_raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid QTIP run manifest JSON: {manifest_path}") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "banana-smasher-qtip-run-manifest-v1"
        or manifest.get("status") != "PASS"
    ):
        raise ValueError(f"invalid QTIP run manifest: {manifest_path}")
    basis = _validate_basis(manifest.get("basis_sha256"), label="manifest")
    tier_row = _unique_row(manifest.get("tiers"), key="name", value=tier, label="tier row")
    geometry = _geometry(tier_row.get("geometry"), tier=tier)
    bindings = tier_row.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError(f"manifest tier {tier!r} requires bindings")

    model = bindings.get("model_root")
    if not isinstance(model, dict):
        raise ValueError(f"manifest tier {tier!r} requires model_root binding")
    model_root = _local_path(model.get("path"), label="model root", base=manifest_path.parent)
    if not model_root.is_dir():
        raise ValueError(f"missing local model root; run {PRODUCER_VERB}: {model_root}")
    model_index = _artifact(model.get("index"), label="model index", base=manifest_path.parent)
    if not model_index.is_relative_to(model_root):
        raise ValueError(f"model index is outside model root: {model_index}")
    if _sha256(model_index) != basis:
        raise ValueError(f"model basis mismatch: manifest={basis} model-index={_sha256(model_index)}")

    qtip_root = _directory_binding(
        bindings.get("qtip_root"), label="QTIP runtime root", base=manifest_path.parent
    )
    qtip_runner = _artifact(
        bindings.get("qtip_runner"), label="public QTIP runner", base=manifest_path.parent
    )
    reference = _artifact(
        bindings.get("reference_unit"), label="QTIP reference unit", base=manifest_path.parent
    )
    tlut = _artifact(bindings.get("tlut_source"), label="QTIP TLUT", base=manifest_path.parent)

    plan: list[tuple[Path, bytes, dict[str, Any]]] = []
    identities: set[tuple[int, int, str]] = set()
    for layer_row in _iter_selected_layers(tier_row, layers):
        layer = layer_row["layer"]
        capture_root = _directory_binding(
            layer_row.get("fit_capture_root"),
            label=f"L{layer:03d} capture bank",
            base=manifest_path.parent,
        )
        hessian = _artifact(
            layer_row.get("hessian_layer_manifest"),
            label=f"L{layer:03d} Hessian manifest",
            base=manifest_path.parent,
        )
        source_rows = layer_row.get("source_configs")
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"L{layer:03d} has no source configs; run {PRODUCER_VERB}")
        population = len(source_rows)
        for index, record in enumerate(source_rows):
            source = _artifact(
                record,
                label=f"L{layer:03d} source config {index}",
                base=manifest_path.parent,
            )
            try:
                config = json.loads(source.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid source config JSON: {source}") from exc
            if not isinstance(config, dict) or config.get("schema") != "banana-smasher-qtip-profile-config-v1":
                raise ValueError(f"invalid source QTIP config: {source}")
            if config.get("layer") != layer:
                raise ValueError(f"source config layer mismatch in {source}")
            source_basis = (
                config.get("input_identity", {}).get("model_index", {}).get("sha256")
                if isinstance(config.get("input_identity"), dict)
                else None
            )
            if source_basis != basis:
                raise ValueError(f"source config basis mismatch in {source}: {source_basis} != {basis}")
            expert = config.get("expert")
            projection = config.get("projection")
            if isinstance(expert, bool) or not isinstance(expert, int) or expert < 0:
                raise ValueError(f"invalid source config expert in {source}")
            if not isinstance(projection, str) or not projection:
                raise ValueError(f"invalid source config projection in {source}")
            identity = (layer, expert, projection)
            if identity in identities:
                raise ValueError(f"duplicate source config identity: {identity}")
            identities.add(identity)

            census = config.get("layer_census")
            if not isinstance(census, dict) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in census.values()
            ):
                raise ValueError(f"invalid source layer census in {source}")
            materialized_census = {str(name): 0 for name in census}
            materialized_census[tier] = population
            materialized = dict(config)
            materialized.update(
                {
                    "tier": tier,
                    "geometry": geometry,
                    "model_root": str(model_root),
                    "fit_capture_root": str(capture_root),
                    "hessian_layer_manifest": str(hessian),
                    "hessian_layer_manifest_sha256": _sha256(hessian),
                    "qtip_root": str(qtip_root),
                    "qtip_runner": str(qtip_runner),
                    "reference_unit": str(reference),
                    "tlut_source": str(tlut),
                    "layer_census": materialized_census,
                    "input_identity": {
                        "model_index": {"path": str(model_index), "sha256": basis}
                    },
                    "materialization": {
                        "schema": "banana-smasher-qtip-config-materialization-v1",
                        "run_manifest": str(manifest_path),
                        "run_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
                        "source_config": str(source),
                        "source_config_sha256": _sha256(source),
                    },
                }
            )
            output = output_root / f"L{layer:03d}" / f"E{expert:03d}_{projection}.json"
            raw = _json_bytes(materialized)
            plan.append(
                (
                    output,
                    raw,
                    {
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "path": str(output),
                        "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "source_sha256": _sha256(source),
                    },
                )
            )

    created = 0
    existing = 0
    member_rows: list[dict[str, Any]] = []
    for output, raw, row in sorted(plan, key=lambda item: item[0].as_posix()):
        if _atomic_bytes(output, raw):
            created += 1
        else:
            existing += 1
        member_rows.append(row)
    ordered_member_sha = hashlib.sha256(
        "".join(str(row["sha256"]) for row in member_rows).encode()
    ).hexdigest()
    sealed_receipt: dict[str, Any] = {
        "schema": "banana-smasher-qtip-config-manifest-v1",
        "status": "PASS",
        "producer": PRODUCER_VERB,
        "tier": tier,
        "geometry": geometry,
        "basis_sha256": basis,
        "run_manifest": str(manifest_path),
        "run_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "output_root": str(output_root),
        "layers": layers,
        "members": len(member_rows),
        "ordered_member_sha256": ordered_member_sha,
        "member_records": member_rows,
    }
    _atomic_bytes(output_root / OUTPUT_MANIFEST_NAME, _json_bytes(sealed_receipt))
    return {
        **sealed_receipt,
        "created_members": created,
        "existing_valid_members": existing,
    }


def ensure_qtip_configs(
    source_root: Path,
    *,
    tier: str,
    layers: list[int],
) -> dict[str, Any] | None:
    """Materialize only when the selected tier is absent; otherwise preserve inputs."""
    source_root = source_root.resolve()
    selected = set(layers)
    existing_layers: set[int] = set()
    if source_root.is_dir():
        for path in source_root.rglob("E*_*.json"):
            try:
                value = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(value, dict) and value.get("tier") == tier:
                layer = value.get("layer")
                if isinstance(layer, int) and not isinstance(layer, bool):
                    existing_layers.add(layer)
    missing = sorted(selected - existing_layers)
    if not missing:
        return None
    manifest_path = source_root / RUN_MANIFEST_NAME
    if not manifest_path.is_file():
        # Preserve dispatch compatibility. The resident config gate emits the
        # producer-specific failure after its exact population scan.
        return None
    return materialize_qtip_configs(
        manifest_path,
        tier=tier,
        layers=layers,
        output_root=source_root,
    )
