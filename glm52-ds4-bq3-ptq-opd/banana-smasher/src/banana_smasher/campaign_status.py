from __future__ import annotations

import hashlib
import json
import math
import stat
import time
from pathlib import Path
from typing import Any


STATUS_SCHEMA = "banana-smasher-anchor-campaign-status-v1"
CAMPAIGN_NAME = "flash-full-0731-anchor-campaign"
TIER_ORDER = ("qtip3", "qtip2", "d4_k2048", "d4_k4096", "mxfp4")
REFERENCE_TIERS = {"mxfp4"}
EXPECTED_LAYERS = tuple(range(43))
EXPECTED_MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash-0731"
MAX_ACTIVE_STALE_SECONDS = 3_600.0


class StatusContractError(ValueError):
    """A required status manifest or receipt is absent, malformed, or stale."""


def _anchor_verb(root: Path) -> str:
    return f"smash anchor --run-root {root}"


def _merge_verb(root: Path, tier: str) -> str:
    return f"smash merge --run-root {root} --tier {tier}"


def _solve_verb(root: Path, tier: str) -> str:
    if tier in {"qtip3", "qtip2"}:
        return (
            "smash solve --source-root <source-root> --root <qtip-output-root> "
            "--layer <layer> --qtip-profile-config <config-dir> --qtip-units 64"
        )
    return _anchor_verb(root)


def _fail(message: str, *, producer: str) -> StatusContractError:
    return StatusContractError(f"{message}; produce or refresh it with: {producer}")


def _json_object(path: Path, *, label: str, producer: str) -> dict[str, Any]:
    try:
        raw = path.read_text()
    except FileNotFoundError as exc:
        raise _fail(f"missing {label}: {path}", producer=producer) from exc
    except UnicodeDecodeError as exc:
        raise _fail(f"malformed {label} {path}: invalid UTF-8", producer=producer) from exc
    except OSError as exc:
        raise _fail(f"cannot read {label} {path}: {exc}", producer=producer) from exc
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _fail(f"malformed {label} {path}: {exc}", producer=producer) from exc
    if not isinstance(value, dict):
        raise _fail(f"malformed {label} {path}: expected a JSON object", producer=producer)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_without_symlinks(path: Path, *, label: str, producer: str) -> Path:
    """Resolve a path only after proving that no declared component is a symlink."""

    candidate = path.expanduser()
    if ".." in candidate.parts:
        raise _fail(f"unsafe {label} path contains '..': {path}", producer=producer)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise _fail(
                f"cannot inspect {label} path component {cursor}: {exc}",
                producer=producer,
            ) from exc
        if stat.S_ISLNK(mode):
            raise _fail(
                f"symlink component in {label} path is forbidden: {cursor}",
                producer=producer,
            )
    return candidate.resolve()


def _artifact_path(record: Any, *, base: Path, label: str, producer: str) -> Path:
    if not isinstance(record, dict):
        raise _fail(f"malformed {label} artifact record: expected an object", producer=producer)
    raw_path = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise _fail(f"malformed {label} artifact record: path is required", producer=producer)
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise _fail(
            f"malformed {label} artifact record for {raw_path}: sha256 is invalid",
            producer=producer,
        )
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    path = _resolve_without_symlinks(
        candidate,
        label=f"{label} artifact",
        producer=producer,
    )
    campaign_collection = base.resolve().parent
    if path != campaign_collection and campaign_collection not in path.parents:
        raise _fail(
            f"unsafe {label} artifact path is outside campaign collection "
            f"{campaign_collection}: {path}",
            producer=producer,
        )
    if not path.is_file():
        raise _fail(f"missing {label}: {path}", producer=producer)
    expected_bytes = record.get("bytes")
    if expected_bytes is not None:
        if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes < 0:
            raise _fail(f"malformed {label} byte count for {path}", producer=producer)
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise _fail(
                f"byte-count mismatch for {label} {path}: expected {expected_bytes}, got {actual_bytes}",
                producer=producer,
            )
    actual_sha = _sha256(path)
    if actual_sha != expected_sha:
        raise _fail(
            f"SHA256 mismatch for {label} {path}: expected {expected_sha}",
            producer=producer,
        )
    return path


def _require_schema_status(
    value: dict[str, Any],
    *,
    path: Path,
    schemas: set[str],
    statuses: set[str],
    producer: str,
) -> None:
    schema = value.get("schema")
    status = value.get("status")
    if schema not in schemas:
        raise _fail(
            f"malformed manifest {path}: unsupported schema {schema!r}; expected one of {sorted(schemas)}",
            producer=producer,
        )
    if status not in statuses:
        raise _fail(
            f"malformed manifest {path}: status {status!r} is not one of {sorted(statuses)}",
            producer=producer,
        )


def _int(value: Any, *, field: str, path: Path, minimum: int = 0, producer: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise _fail(f"malformed {field} in {path}: expected integer >= {minimum}", producer=producer)
    return value


def _number(value: Any, *, field: str, path: Path, minimum: float, producer: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"malformed {field} in {path}: expected a number", producer=producer)
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise _fail(
            f"malformed {field} in {path}: expected finite value >= {minimum}",
            producer=producer,
        )
    return result


def _receipt_created(value: dict[str, Any], *, path: Path, producer: str) -> float:
    candidates = [
        value.get("created_unix"),
        value.get("updated_unix"),
        value.get("sealed_epoch"),
    ]
    timing = value.get("timing")
    if isinstance(timing, dict):
        candidates.extend(
            [
                timing.get("ended_unix"),
                timing.get("stage_ended_unix"),
                timing.get("tier_ended_unix"),
            ]
        )
    for raw in candidates:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(float(raw)):
            return float(raw)
    raise _fail(f"malformed receipt {path}: no creation/update timestamp", producer=producer)


def _load_receipt(
    record: Any,
    *,
    base: Path,
    tier: str,
    layer: int,
    producer: str,
    units_required: bool = True,
) -> tuple[int, tuple[float, str]]:
    path = _artifact_path(record, base=base, label=f"{tier} L{layer:03d} receipt", producer=producer)
    value = _json_object(path, label=f"{tier} L{layer:03d} receipt", producer=producer)
    if value.get("schema") not in {
        "banana-smasher-fixed-anchor-layer-receipt-v1",
        "banana-smasher-anchor-layer-receipt-v1",
        f"banana-smasher-{tier}-anchor-layer-receipt-v1",
    }:
        raise _fail(
            f"malformed receipt {path}: unsupported schema {value.get('schema')!r}",
            producer=producer,
        )
    if value.get("status") != "PASS":
        raise _fail(f"malformed receipt {path}: status is not PASS", producer=producer)
    if value.get("tier") != tier or value.get("layer") != layer:
        raise _fail(
            f"malformed receipt {path}: tier/layer binding does not match {tier}/L{layer:03d}",
            producer=producer,
        )
    raw_units = value.get("units", value.get("selected_cells"))
    if raw_units is None and not units_required:
        units = 0
    else:
        units = _int(raw_units, field="receipt units", path=path, minimum=1, producer=producer)
    return units, (_receipt_created(value, path=path, producer=producer), str(path))


def _parse_direct_anchor(
    root: Path,
    tier: str,
    row: dict[str, Any],
) -> tuple[dict[int, int], list[tuple[float, str]], str]:
    producer = _anchor_verb(root)
    anchor_path = _artifact_path(row, base=root, label=f"{tier} anchor", producer=producer)
    anchor = _json_object(anchor_path, label=f"{tier} anchor", producer=producer)
    _require_schema_status(
        anchor,
        path=anchor_path,
        schemas={"banana-smasher-tier-anchor-v1"},
        statuses={"PASS"},
        producer=producer,
    )
    if anchor.get("tier") != tier:
        raise _fail(f"malformed anchor {anchor_path}: tier binding mismatch", producer=producer)
    manifest_record = row.get("fixed_anchor_manifest", row.get("manifest"))
    if not isinstance(manifest_record, dict):
        raise _fail(
            f"malformed anchor row for {tier}: aggregate manifest artifact is required",
            producer=producer,
        )
    manifest_path = _artifact_path(
        manifest_record,
        base=root,
        label=f"{tier} aggregate anchor manifest",
        producer=producer,
    )
    pointer = anchor.get("fixed_anchor_manifest")
    if not isinstance(pointer, dict):
        raise _fail(
            f"malformed anchor {anchor_path}: aggregate manifest pointer is required",
            producer=producer,
        )
    pointer_path = _artifact_path(
        pointer,
        base=root,
        label=f"{tier} anchor aggregate manifest pointer",
        producer=producer,
    )
    if pointer_path != manifest_path or pointer.get("sha256") != manifest_record.get("sha256"):
        raise _fail(
            f"stale anchor {anchor_path}: aggregate manifest pointer does not match anchors/MANIFEST.json",
            producer=producer,
        )
    manifest = _json_object(
        manifest_path, label=f"{tier} aggregate anchor manifest", producer=producer
    )
    allowed_schemas = {
        "banana-smasher-fixed-anchor-manifest-v1",
        "banana-smasher-qtip3-anchor-manifest-v1",
        "banana-smasher-qtip2-anchor-manifest-v1",
        "banana-smasher-mxfp4-reference-manifest-v1",
    }
    _require_schema_status(
        manifest,
        path=manifest_path,
        schemas=allowed_schemas,
        statuses={"PASS"},
        producer=producer,
    )
    if manifest.get("tier") != tier:
        raise _fail(f"malformed manifest {manifest_path}: tier binding mismatch", producer=producer)
    layers = manifest.get("layers")
    rows = manifest.get("layer_receipts")
    if not isinstance(layers, list) or not isinstance(rows, list):
        raise _fail(
            f"malformed manifest {manifest_path}: layers and layer_receipts arrays are required",
            producer=producer,
        )
    parsed_layers = [
        _int(layer, field="layer", path=manifest_path, producer=producer) for layer in layers
    ]
    if len(set(parsed_layers)) != len(parsed_layers):
        raise _fail(f"malformed manifest {manifest_path}: duplicate layers", producer=producer)
    by_layer: dict[int, dict[str, Any]] = {}
    for receipt_row in rows:
        if not isinstance(receipt_row, dict):
            raise _fail(f"malformed receipt row in {manifest_path}", producer=producer)
        layer = _int(
            receipt_row.get("layer"),
            field="layer_receipts.layer",
            path=manifest_path,
            producer=producer,
        )
        if layer in by_layer:
            raise _fail(f"malformed manifest {manifest_path}: duplicate L{layer:03d} receipt", producer=producer)
        by_layer[layer] = receipt_row
    if set(parsed_layers) != set(by_layer):
        raise _fail(
            f"malformed manifest {manifest_path}: layers and layer receipt population differ",
            producer=producer,
        )
    units: dict[int, int] = {}
    newest: list[tuple[float, str]] = []
    for layer in sorted(parsed_layers):
        count, receipt = _load_receipt(
            by_layer[layer],
            base=root,
            tier=tier,
            layer=layer,
            producer=producer,
        )
        units[layer] = count
        newest.append(receipt)
    selected = _int(
        manifest.get("selected_cells"),
        field="selected_cells",
        path=manifest_path,
        minimum=1,
        producer=producer,
    )
    if selected != sum(units.values()):
        raise _fail(
            f"malformed manifest {manifest_path}: selected_cells={selected} but receipts total {sum(units.values())}",
            producer=producer,
        )
    return units, newest, str(manifest_path)


def _parse_shards(
    root: Path,
    tier: str,
    *,
    expected_units: dict[int, int],
) -> tuple[dict[int, int], list[tuple[float, str]], str | None]:
    producer = _merge_verb(root, tier)
    path = _resolve_without_symlinks(
        root / "anchors" / tier / "SHARDS.json",
        label=f"{tier} shard index",
        producer=producer,
    )
    if not path.is_file():
        return {}, [], None
    index = _json_object(path, label=f"{tier} shard index", producer=producer)
    _require_schema_status(
        index,
        path=path,
        schemas={
            f"banana-smasher-{tier}-shard-index-v1",
            "banana-smasher-anchor-shard-index-v1",
        },
        statuses={"PASS"},
        producer=producer,
    )
    if index.get("tier") != tier:
        raise _fail(f"malformed shard index {path}: tier binding mismatch", producer=producer)
    shards = index.get("shards")
    if not isinstance(shards, list):
        raise _fail(f"malformed shard index {path}: shards array is required", producer=producer)
    completed: dict[int, int] = {}
    newest: list[tuple[float, str]] = []
    source_index_sha = index.get("source_index_sha256")
    if (
        not isinstance(source_index_sha, str)
        or len(source_index_sha) != 64
        or any(character not in "0123456789abcdef" for character in source_index_sha)
    ):
        raise _fail(
            f"malformed shard index {path}: source_index_sha256 is required",
            producer=producer,
        )
    for shard_number, row in enumerate(shards):
        if not isinstance(row, dict):
            raise _fail(f"malformed shard row {shard_number} in {path}", producer=producer)
        manifest_path = _artifact_path(
            row.get("manifest"),
            base=root,
            label=f"{tier} shard manifest",
            producer=producer,
        )
        manifest = _json_object(
            manifest_path, label=f"{tier} shard manifest", producer=producer
        )
        _require_schema_status(
            manifest,
            path=manifest_path,
            schemas={
                f"banana-smasher-{tier}-anchor-shard-manifest-v1",
                "banana-smasher-anchor-shard-manifest-v1",
            },
            statuses={"PASS"},
            producer=producer,
        )
        manifest_tier = manifest.get("tier", tier)
        if manifest_tier != tier:
            raise _fail(f"malformed shard manifest {manifest_path}: tier mismatch", producer=producer)
        manifest_source_index_sha = manifest.get("source_index_sha256")
        if not isinstance(manifest_source_index_sha, str):
            raise _fail(
                f"malformed shard manifest {manifest_path}: source_index_sha256 is required",
                producer=producer,
            )
        if manifest_source_index_sha != source_index_sha:
            raise _fail(
                f"stale shard manifest {manifest_path}: source index SHA mismatch",
                producer=producer,
            )
        layers = manifest.get("layers")
        receipt_rows = manifest.get("layer_receipts")
        if not isinstance(layers, list) or not isinstance(receipt_rows, list):
            raise _fail(
                f"malformed shard manifest {manifest_path}: layers and layer_receipts are required",
                producer=producer,
            )
        row_by_layer: dict[int, dict[str, Any]] = {}
        for receipt_row in receipt_rows:
            if not isinstance(receipt_row, dict):
                raise _fail(f"malformed receipt row in {manifest_path}", producer=producer)
            layer = _int(
                receipt_row.get("layer"),
                field="layer receipt layer",
                path=manifest_path,
                producer=producer,
            )
            if layer in row_by_layer:
                raise _fail(
                    f"malformed shard manifest {manifest_path}: duplicate L{layer:03d}",
                    producer=producer,
                )
            row_by_layer[layer] = receipt_row
        parsed_layers = [
            _int(layer, field="layer", path=manifest_path, producer=producer)
            for layer in layers
        ]
        if len(set(parsed_layers)) != len(parsed_layers) or set(parsed_layers) != set(row_by_layer):
            raise _fail(
                f"malformed shard manifest {manifest_path}: layer population mismatch",
                producer=producer,
            )
        shard_units = 0
        for layer in sorted(parsed_layers):
            if layer not in expected_units:
                raise _fail(
                    f"malformed shard manifest {manifest_path}: unexpected L{layer:03d}",
                    producer=producer,
                )
            if layer in completed:
                raise _fail(
                    f"overlapping sealed shard coverage for {tier} L{layer:03d}: {manifest_path}",
                    producer=producer,
                )
            count, receipt = _load_receipt(
                row_by_layer[layer],
                base=root,
                tier=tier,
                layer=layer,
                producer=producer,
            )
            if count != expected_units[layer]:
                raise _fail(
                    f"incomplete sealed shard {manifest_path}: L{layer:03d} has {count}/{expected_units[layer]} units",
                    producer=producer,
                )
            completed[layer] = count
            shard_units += count
            newest.append(receipt)
        for owner, label in ((manifest, "manifest"), (row, "index row")):
            raw_count = owner.get("cell_count")
            if raw_count is not None and _int(
                raw_count,
                field=f"shard {label} cell_count",
                path=manifest_path,
                producer=producer,
            ) != shard_units:
                raise _fail(
                    f"malformed shard {label} {manifest_path}: cell_count does not match receipts",
                    producer=producer,
                )
        exactness = row.get("exactness")
        if isinstance(exactness, dict):
            if exactness.get("status") != "PASS":
                raise _fail(f"malformed shard exactness row in {path}: status is not PASS", producer=producer)
            raw_units = exactness.get("unit_count")
            if raw_units is not None and raw_units != shard_units:
                raise _fail(
                    f"malformed shard exactness row in {path}: unit_count mismatch",
                    producer=producer,
                )
    return completed, newest, str(path)


def _parse_active_runs(
    root: Path,
    tier: str,
    *,
    expected_units: dict[int, int],
    sealed_units: dict[int, int],
    now: float,
) -> tuple[dict[int, tuple[int, int]], list[dict[str, Any]], list[tuple[float, str]]]:
    producer = _solve_verb(root, tier)
    path = _resolve_without_symlinks(
        root / "anchors" / tier / "RUNS.json",
        label=f"{tier} run index",
        producer=producer,
    )
    if not path.is_file():
        return {}, [], []
    index = _json_object(path, label=f"{tier} run index", producer=producer)
    _require_schema_status(
        index,
        path=path,
        schemas={"banana-smasher-anchor-run-index-v1"},
        statuses={"PASS"},
        producer=producer,
    )
    if index.get("tier") != tier or not isinstance(index.get("runs"), list):
        raise _fail(f"malformed run index {path}: tier and runs array are required", producer=producer)
    progress_by_layer: dict[int, tuple[int, int]] = {}
    current_rows: list[dict[str, Any]] = []
    newest: list[tuple[float, str]] = []
    for row_number, row in enumerate(index["runs"]):
        if not isinstance(row, dict):
            raise _fail(f"malformed run row {row_number} in {path}", producer=producer)
        manifest_path = _artifact_path(
            row.get("manifest"),
            base=root,
            label=f"{tier} active run manifest",
            producer=producer,
        )
        manifest = _json_object(
            manifest_path, label=f"{tier} active run manifest", producer=producer
        )
        _require_schema_status(
            manifest,
            path=manifest_path,
            schemas={"banana-smasher-anchor-run-v1"},
            statuses={"RUNNING"},
            producer=producer,
        )
        if manifest.get("tier") != tier:
            raise _fail(f"malformed active run {manifest_path}: tier mismatch", producer=producer)
        updated = _number(
            manifest.get("updated_unix"),
            field="updated_unix",
            path=manifest_path,
            minimum=0,
            producer=producer,
        )
        stale_after = _number(
            manifest.get("stale_after_seconds"),
            field="stale_after_seconds",
            path=manifest_path,
            minimum=0.001,
            producer=producer,
        )
        if stale_after > MAX_ACTIVE_STALE_SECONDS:
            raise _fail(
                f"malformed active run {manifest_path}: stale_after_seconds={stale_after:.3f} "
                f"exceeds maximum {MAX_ACTIVE_STALE_SECONDS:.3f}",
                producer=producer,
            )
        if updated > now:
            raise _fail(
                f"future-dated active run manifest {manifest_path}: updated_unix is "
                f"{updated - now:.3f}s ahead of status time",
                producer=producer,
            )
        if now - updated > stale_after:
            raise _fail(
                f"stale active run manifest {manifest_path}: age {now - updated:.3f}s exceeds {stale_after:.3f}s",
                producer=producer,
            )
        progress = manifest.get("progress")
        if not isinstance(progress, list) or not progress:
            raise _fail(f"malformed active run {manifest_path}: progress array is required", producer=producer)
        manifest_layers: set[int] = set()
        manifest_active_units = 0
        for progress_row in progress:
            if not isinstance(progress_row, dict):
                raise _fail(f"malformed progress row in {manifest_path}", producer=producer)
            layer = _int(
                progress_row.get("layer"),
                field="progress.layer",
                path=manifest_path,
                producer=producer,
            )
            if layer not in expected_units:
                raise _fail(f"malformed active run {manifest_path}: unexpected L{layer:03d}", producer=producer)
            if layer in sealed_units:
                raise _fail(
                    f"active run {manifest_path} overlaps sealed {tier} L{layer:03d}",
                    producer=producer,
                )
            if layer in progress_by_layer:
                raise _fail(
                    f"multiple active runs overlap {tier} L{layer:03d}",
                    producer=producer,
                )
            if layer in manifest_layers:
                raise _fail(
                    f"malformed active run {manifest_path}: duplicate L{layer:03d} progress",
                    producer=producer,
                )
            completed = _int(
                progress_row.get("completed_units"),
                field="progress.completed_units",
                path=manifest_path,
                producer=producer,
            )
            active = _int(
                progress_row.get("active_units"),
                field="progress.active_units",
                path=manifest_path,
                producer=producer,
            )
            if completed + active > expected_units[layer]:
                raise _fail(
                    f"malformed active run {manifest_path}: L{layer:03d} progress exceeds expected units",
                    producer=producer,
                )
            manifest_layers.add(layer)
            manifest_active_units += active
            progress_by_layer[layer] = (completed, active)
        if manifest_active_units == 0:
            raise _fail(
                f"malformed active run {manifest_path}: RUNNING manifest has zero active units",
                producer=producer,
            )
        receipt_path = _artifact_path(
            manifest.get("newest_receipt"),
            base=root,
            label=f"{tier} newest progress receipt",
            producer=producer,
        )
        receipt = _json_object(
            receipt_path, label=f"{tier} newest progress receipt", producer=producer
        )
        if receipt.get("schema") != "banana-smasher-anchor-progress-receipt-v1":
            raise _fail(
                f"malformed progress receipt {receipt_path}: unsupported schema {receipt.get('schema')!r}",
                producer=producer,
            )
        if receipt.get("status") not in {"PASS", "RUNNING"}:
            raise _fail(f"malformed progress receipt {receipt_path}: invalid status", producer=producer)
        receipt_created = _receipt_created(receipt, path=receipt_path, producer=producer)
        if receipt_created > now:
            raise _fail(
                f"future-dated active progress receipt {receipt_path}: timestamp is "
                f"{receipt_created - now:.3f}s ahead of status time",
                producer=producer,
            )
        if now - receipt_created > stale_after:
            raise _fail(
                f"stale active progress receipt {receipt_path}: age {now - receipt_created:.3f}s "
                f"exceeds {stale_after:.3f}s",
                producer=producer,
            )
        newest.append((receipt_created, str(receipt_path)))
        current = manifest.get("current")
        if not isinstance(current, dict):
            raise _fail(f"malformed active run {manifest_path}: current object is required", producer=producer)
        current_layer = _int(
            current.get("layer"), field="current.layer", path=manifest_path, producer=producer
        )
        if current_layer not in manifest_layers:
            raise _fail(
                f"malformed active run {manifest_path}: current layer is absent from progress",
                producer=producer,
            )
        current_rows.append(
            {
                "host": str(manifest.get("host", row.get("host", "unknown"))),
                "layer": f"L{current_layer:03d}",
                "batch": _int(
                    current.get("batch"),
                    field="current.batch",
                    path=manifest_path,
                    producer=producer,
                ),
                "unit": _int(
                    current.get("unit"),
                    field="current.unit",
                    path=manifest_path,
                    producer=producer,
                ),
                "manifest": str(manifest_path),
            }
        )
        expected_current = {
            "tier": tier,
            "layer": current_layer,
            "batch": current_rows[-1]["batch"],
            "unit": current_rows[-1]["unit"],
        }
        for field, expected in expected_current.items():
            if receipt.get(field) != expected:
                raise _fail(
                    f"stale progress receipt {receipt_path}: receipt {field}={receipt.get(field)!r}, "
                    f"active manifest declares {expected!r}",
                    producer=producer,
                )
    current_rows.sort(key=lambda row: (row["layer"], row["host"], row["manifest"]))
    return progress_by_layer, current_rows, newest


def _load_campaign_manifests(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    producer = _anchor_verb(root)
    anchor_path = _resolve_without_symlinks(
        root / "anchors" / "MANIFEST.json",
        label="anchor campaign manifest",
        producer=producer,
    )
    anchor = _json_object(anchor_path, label="anchor campaign manifest", producer=producer)
    _require_schema_status(
        anchor,
        path=anchor_path,
        schemas={"banana-smasher-anchor-manifest-v1"},
        statuses={"PASS"},
        producer=producer,
    )
    solve_path = _artifact_path(
        anchor.get("input_solve_manifest"),
        base=root,
        label="input solve manifest",
        producer=producer,
    )
    solve = _json_object(solve_path, label="input solve manifest", producer=producer)
    _require_schema_status(
        solve,
        path=solve_path,
        schemas={"banana-smasher-vq-solve-manifest-v1"},
        statuses={"PASS"},
        producer=producer,
    )
    declared_model_id = anchor.get("model_id")
    if declared_model_id is not None and declared_model_id != EXPECTED_MODEL_ID:
        raise _fail(
            f"malformed anchor manifest {anchor_path}: model_id must be {EXPECTED_MODEL_ID!r}, "
            f"got {declared_model_id!r}",
            producer=producer,
        )
    chain_path = _resolve_without_symlinks(
        root / "WORKFLOW_CHAIN.json",
        label="workflow chain manifest",
        producer=producer,
    )
    chain = _json_object(chain_path, label="workflow chain manifest", producer=producer)
    _require_schema_status(
        chain,
        path=chain_path,
        schemas={"banana-smasher-workflow-chain-v1"},
        statuses={"PASS"},
        producer=producer,
    )
    chained_anchor = _artifact_path(
        chain.get("anchor_manifest"),
        base=root,
        label="workflow-chain anchor manifest",
        producer=producer,
    )
    if chained_anchor != anchor_path:
        raise _fail(
            f"stale workflow chain {chain_path}: anchor manifest path is {chained_anchor}, expected {anchor_path}",
            producer=producer,
        )
    chained_solve = _artifact_path(
        chain.get("solve_manifest"),
        base=root,
        label="workflow-chain solve manifest",
        producer=producer,
    )
    if chained_solve != solve_path:
        raise _fail(
            f"stale workflow chain {chain_path}: solve manifest path is {chained_solve}, expected {solve_path}",
            producer=producer,
        )
    if chain.get("run_root") not in {None, str(root)}:
        raise _fail(
            f"stale workflow chain {chain_path}: run_root binding does not match {root}",
            producer=producer,
        )
    raw_rows = anchor.get("anchors")
    if not isinstance(raw_rows, list):
        raise _fail(f"malformed anchor campaign manifest {anchor_path}: anchors array is required", producer=producer)
    rows: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        if not isinstance(row, dict) or not isinstance(row.get("tier"), str):
            raise _fail(f"malformed anchor row in {anchor_path}", producer=producer)
        tier = row["tier"]
        if tier not in TIER_ORDER:
            raise _fail(f"malformed anchor row in {anchor_path}: unexpected tier {tier!r}", producer=producer)
        if tier in rows:
            raise _fail(f"malformed anchor campaign manifest {anchor_path}: duplicate tier {tier}", producer=producer)
        rows[tier] = row
    return anchor, rows


def _ranges(layers: list[str]) -> str:
    if not layers:
        return "-"
    values = [int(layer[1:]) for layer in layers]
    groups: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        groups.append(f"L{start:03d}" if start == previous else f"L{start:03d}-L{previous:03d}")
        start = previous = value
    groups.append(f"L{start:03d}" if start == previous else f"L{start:03d}-L{previous:03d}")
    return ",".join(groups)


def inspect_anchor_campaign(run_root: str | Path, *, now: float | None = None) -> dict[str, Any]:
    """Return manifest-only flash-full anchor coverage without scanning for artifacts."""

    requested_root = Path(run_root).expanduser()
    producer = _anchor_verb(requested_root)
    root = _resolve_without_symlinks(
        requested_root,
        label="run root",
        producer=producer,
    )
    if not root.is_dir():
        raise _fail(f"missing run root: {root}", producer=_anchor_verb(root))
    checked = time.time() if now is None else float(now)
    if not math.isfinite(checked) or checked < 0:
        raise ValueError("now must be a finite non-negative timestamp")
    anchor, direct_rows = _load_campaign_manifests(root)

    baseline_by_tier: dict[str, dict[int, int]] = {}
    direct_data: dict[str, tuple[dict[int, int], list[tuple[float, str]], str]] = {}
    for tier in TIER_ORDER:
        if tier in direct_rows:
            direct_data[tier] = _parse_direct_anchor(root, tier, direct_rows[tier])
            if tier in {"d4_k2048", "d4_k4096"}:
                baseline_by_tier[tier] = direct_data[tier][0]
    missing_baselines = sorted({"d4_k2048", "d4_k4096"} - set(baseline_by_tier))
    if missing_baselines:
        raise _fail(
            "anchor campaign lacks manifest-count baselines for " + ", ".join(missing_baselines),
            producer=_anchor_verb(root),
        )
    expected_units = baseline_by_tier["d4_k2048"]
    if not expected_units or baseline_by_tier["d4_k4096"] != expected_units:
        raise _fail(
            "d4 anchor manifests disagree on expected layer/unit population",
            producer=_anchor_verb(root),
        )
    expected_layers = list(EXPECTED_LAYERS)
    if sorted(expected_units) != expected_layers:
        baseline_paths = ", ".join(
            direct_data[tier][2] for tier in ("d4_k2048", "d4_k4096")
        )
        raise _fail(
            f"malformed d4 anchor manifests {baseline_paths}: flash-full must declare "
            "exactly L000-L042",
            producer=_anchor_verb(root),
        )

    tier_values: list[dict[str, Any]] = []
    all_blockers: list[str] = []
    for tier in TIER_ORDER:
        receipt_rows: list[tuple[float, str]] = []
        source_manifest: str | None = None
        source_kind: str | None = None
        if tier in direct_data:
            completed = dict(direct_data[tier][0])
            if set(completed) != set(expected_units) or any(
                completed[layer] != expected_units[layer] for layer in expected_layers
            ):
                raise _fail(
                    f"{tier} aggregate manifest population differs from the d4 campaign baseline",
                    producer=_anchor_verb(root),
                )
            receipt_rows.extend(direct_data[tier][1])
            source_manifest = direct_data[tier][2]
            source_kind = "aggregate"
        else:
            completed, sealed_receipts, shard_manifest = _parse_shards(
                root, tier, expected_units=expected_units
            )
            receipt_rows.extend(sealed_receipts)
            source_manifest = shard_manifest
            source_kind = "shard-index" if shard_manifest is not None else None

        active_progress, current, active_receipts = _parse_active_runs(
            root,
            tier,
            expected_units=expected_units,
            sealed_units=completed,
            now=checked,
        )
        receipt_rows.extend(active_receipts)

        layer_rows: list[dict[str, Any]] = []
        coverage_completed: list[str] = []
        coverage_active: list[str] = []
        coverage_missing: list[str] = []
        completed_units = 0
        active_units = 0
        for layer in expected_layers:
            sealed = completed.get(layer, 0)
            in_progress_completed, active = active_progress.get(layer, (0, 0))
            done = sealed + in_progress_completed
            missing = expected_units[layer] - done - active
            if missing < 0:
                raise _fail(
                    f"{tier} L{layer:03d} coverage exceeds manifest expectation",
                    producer=_merge_verb(root, tier),
                )
            label = f"L{layer:03d}"
            if done == expected_units[layer] and active == 0:
                state = "complete"
                coverage_completed.append(label)
            elif active > 0:
                state = "active"
                coverage_active.append(label)
            else:
                state = "missing"
                coverage_missing.append(label)
            completed_units += done
            active_units += active
            layer_rows.append(
                {
                    "layer": label,
                    "expected_units": expected_units[layer],
                    "completed_units": done,
                    "active_units": active,
                    "missing_units": missing,
                    "state": state,
                }
            )
        expected_total = sum(expected_units.values())
        missing_units = expected_total - completed_units - active_units
        percent = round(100.0 * completed_units / expected_total, 6)
        blockers: list[str] = []
        if source_manifest is None:
            blockers.append(
                f"missing anchors/{tier}/SHARDS.json; produce it with: {_merge_verb(root, tier)}"
            )
        if active_units:
            blockers.append(
                f"{active_units} active {tier} unit(s) are not sealed; continue with: {_solve_verb(root, tier)}"
            )
        if missing_units:
            blockers.append(
                f"{missing_units} {tier} unit(s) remain across {_ranges(coverage_missing + coverage_active)}; "
                f"produce with: {_solve_verb(root, tier)} then {_merge_verb(root, tier)}"
            )
        mergeable = source_manifest is not None
        ready = completed_units == expected_total and active_units == 0 and missing_units == 0
        newest_value: dict[str, Any] | None = None
        if receipt_rows:
            created, receipt_path = max(receipt_rows, key=lambda row: (row[0], row[1]))
            newest_value = {
                "path": receipt_path,
                "created_unix": created,
                "age_seconds": round(max(0.0, checked - created), 3),
            }
        value = {
            "tier": tier,
            "role": "reference" if tier in REFERENCE_TIERS else "anchor",
            "source": {"kind": source_kind, "manifest": source_manifest},
            "coverage": {
                "expected": [f"L{layer:03d}" for layer in expected_layers],
                "completed": coverage_completed,
                "active": coverage_active,
                "missing": coverage_missing,
                "counts": {
                    "expected": len(expected_layers),
                    "completed": len(coverage_completed),
                    "active": len(coverage_active),
                    "missing": len(coverage_missing),
                },
            },
            "units": {
                "expected": expected_total,
                "completed": completed_units,
                "active": active_units,
                "missing": missing_units,
                "percent_completed": percent,
            },
            "layers": layer_rows,
            "current": current,
            "newest_receipt": newest_value,
            "mergeable": mergeable,
            "ready": ready,
            "blockers": blockers,
        }
        tier_values.append(value)
        all_blockers.extend(f"{tier}: {blocker}" for blocker in blockers)

    campaign_expected = sum(tier["units"]["expected"] for tier in tier_values)
    campaign_completed = sum(tier["units"]["completed"] for tier in tier_values)
    campaign_active = sum(tier["units"]["active"] for tier in tier_values)
    campaign_missing = sum(tier["units"]["missing"] for tier in tier_values)
    campaign_ready = all(tier["ready"] for tier in tier_values)
    status = "READY" if campaign_ready else ("RUNNING" if campaign_active else "IN_PROGRESS")
    return {
        "schema": STATUS_SCHEMA,
        "status": status,
        "campaign_name": CAMPAIGN_NAME,
        "run_root": str(root),
        "model_id": anchor.get("model_id"),
        "checked_unix": checked,
        "campaign": {
            "tier_order": list(TIER_ORDER),
            "expected_layers": [f"L{layer:03d}" for layer in expected_layers],
            "expected_units_per_layer": {
                f"L{layer:03d}": expected_units[layer] for layer in expected_layers
            },
            "units": {
                "expected": campaign_expected,
                "completed": campaign_completed,
                "active": campaign_active,
                "missing": campaign_missing,
                "percent_completed": round(
                    100.0 * campaign_completed / campaign_expected, 6
                ),
            },
            "mergeable": all(tier["mergeable"] for tier in tier_values),
            "ready": campaign_ready,
        },
        "tiers": tier_values,
        "blockers": all_blockers,
    }


def _format_age(receipt: dict[str, Any] | None) -> str:
    if receipt is None:
        return "-"
    age = float(receipt["age_seconds"])
    if age < 60:
        return f"{age:.0f}s"
    if age < 3600:
        return f"{age / 60:.1f}m"
    return f"{age / 3600:.1f}h"


def render_anchor_campaign(status: dict[str, Any]) -> str:
    """Render every tier and every declared layer without hiding missing coverage."""

    campaign = status["campaign"]
    units = campaign["units"]
    lines = [
        f"RUN ROOT: {status['run_root']}",
        f"CAMPAIGN: {status['campaign_name']}   MODEL: {status.get('model_id') or '-'}",
        (
            f"STATUS: {status['status']}   UNITS: {units['completed']}/{units['expected']} "
            f"({units['percent_completed']:.2f}%)   ACTIVE: {units['active']}   "
            f"MISSING: {units['missing']}   MERGEABLE: {'yes' if campaign['mergeable'] else 'no'}   "
            f"READY: {'yes' if campaign['ready'] else 'no'}"
        ),
        "",
        "TIER          DONE/EXPECTED     ACTIVE    MISSING      %      LAYERS C/A/M   MERGEABLE READY CURRENT",
    ]
    for tier in status["tiers"]:
        coverage = tier["coverage"]["counts"]
        tier_units = tier["units"]
        current = ",".join(
            f"{row['layer']}/B{row['batch']:03d}/U{row['unit']:03d}@{row['host']}"
            for row in tier["current"]
        ) or "-"
        lines.append(
            f"{tier['tier']:<13} {tier_units['completed']:>6}/{tier_units['expected']:<8} "
            f"{tier_units['active']:>8} {tier_units['missing']:>10} "
            f"{tier_units['percent_completed']:>6.2f}%   "
            f"{coverage['completed']:>2}/{coverage['active']:>2}/{coverage['missing']:<2}       "
            f"{'yes' if tier['mergeable'] else 'no':<9} "
            f"{'yes' if tier['ready'] else 'no':<5} {current}"
        )
    for tier in status["tiers"]:
        lines.extend(
            [
                "",
                f"{tier['tier']} COVERAGE",
                f"  complete: {_ranges(tier['coverage']['completed'])}",
                f"  active:   {_ranges(tier['coverage']['active'])}",
                f"  missing:  {_ranges(tier['coverage']['missing'])}",
                (
                    "  newest:   -"
                    if tier["newest_receipt"] is None
                    else f"  newest:   {_format_age(tier['newest_receipt'])}  {tier['newest_receipt']['path']}"
                ),
                "  layers:",
            ]
        )
        tokens = [
            (
                f"{row['layer']}:{row['state'][0].upper()}"
                f"({row['completed_units']}/{row['active_units']}/{row['missing_units']})"
            )
            for row in tier["layers"]
        ]
        for offset in range(0, len(tokens), 6):
            lines.append("    " + "  ".join(tokens[offset : offset + 6]))
        if tier["blockers"]:
            lines.append("  blockers:")
            lines.extend(f"    - {blocker}" for blocker in tier["blockers"])
        else:
            lines.append("  blockers: -")
    lines.extend(
        [
            "",
            "LEGEND: layer token = LNNN:state(completed/active/missing units); C=complete A=active M=missing",
            "READINESS: mergeable means a validated aggregate/shard manifest exists; ready means every expected unit is sealed.",
        ]
    )
    return "\n".join(lines) + "\n"
