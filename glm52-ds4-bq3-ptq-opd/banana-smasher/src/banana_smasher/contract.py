from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Literal

import numpy as np

MANIFEST_NAME = "BANANA_PACK_MANIFEST.json"
COMPLETE_MARKER_NAME = "PACK_COMPLETE"
KERNEL_MANIFEST_NAME = "BS_KERNEL_CACHE_MANIFEST.json"
SCHEMA = "bs-pack"
SCHEMA_VERSION = 1
QUANT_METHOD = "bs-mixed-tier"
TIER_FAMILIES = (
    "qtip2",
    "qtip3",
    "truevq_d4",
    "truevq_d8",
    "native_mxfp4",
)
TIER_CODES = {name: code for code, name in enumerate(TIER_FAMILIES)}
REQUIRED_FAMILY_FIELDS = {
    "qtip2": {"codes", "scales", "codebooks", "expert_ids", "tensor_offsets"},
    "qtip3": {"codes", "scales", "codebooks", "expert_ids", "tensor_offsets"},
    "truevq_d4": {
        "codes",
        "scales",
        "codebooks",
        "expert_ids",
        "tensor_offsets",
    },
    "truevq_d8": {
        "codes",
        "scales",
        "codebooks",
        "expert_ids",
        "tensor_offsets",
    },
    "native_mxfp4": {"packed", "scales", "expert_ids", "tensor_offsets"},
}
LAYER_RE = re.compile(r"^layers/layer_(\d{3})/(.+)\.npy$")
TENSOR_RE = re.compile(
    r"^layers\.(\d+)\.(experts\.(?:tier_map|subtier_map)|"
    r"(?:qtip2|qtip3|truevq_d4|truevq_d8|native_mxfp4)\."
    r"((?:[a-z0-9_]+\.)*[a-z0-9_]+))$"
)
BANANA_SMASHER_LAYER_RE = re.compile(r"^layer_(\d{3})$")
BANANA_SMASHER_PLANE_RE = re.compile(
    r"^(d4_k(256|1024|2048|4096))\.(down|fused13)\."
    r"(codebook\.fp16|codes\.le(8|10|11|12)|expert_ids\.i16|scales\.e8m0)\.bin$"
)
BANANA_SMASHER_SUBTIERS = (256, 1024, 2048, 4096)
BANANA_SMASHER_PROJECTIONS = ("down", "fused13")
BANANA_SMASHER_ROLES = ("codebooks", "codes", "expert_ids", "scales")


class PackValidationError(ValueError):
    """Raised when a pack fails any fail-closed contract gate."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_npy_payload(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if not array.flags.c_contiguous:
        raise PackValidationError(f"Fortran/non-C npy arrays are forbidden: {path}")
    digest = hashlib.sha256()
    view = memoryview(array).cast("B")
    for start in range(0, len(view), chunk_bytes):
        digest.update(view[start : start + chunk_bytes])
    return digest.hexdigest()


def _tensor_name(relative: Path) -> str:
    normalized = relative.as_posix()
    match = LAYER_RE.fullmatch(normalized)
    if match is None:
        raise PackValidationError(
            "npy plane path must match layers/layer_NNN/<family>/<field>.npy: "
            f"{normalized}"
        )
    layer = int(match.group(1))
    suffix = match.group(2).replace("/", ".")
    name = f"layers.{layer}.{suffix}"
    if TENSOR_RE.fullmatch(name) is None:
        raise PackValidationError(f"unsupported bs-pack tensor name: {name}")
    return name


def _npy_metadata(path: Path) -> dict[str, Any]:
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise PackValidationError(f"invalid npy plane {path}: {exc}") from exc
    if array.dtype.hasobject:
        raise PackValidationError(f"object dtype is forbidden: {path}")
    if not array.flags.c_contiguous:
        raise PackValidationError(f"only C-contiguous arrays are allowed: {path}")
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data_bytes": int(array.nbytes),
        "data_sha256": _sha256_npy_payload(path),
    }


def _raw_metadata(
    path: Path, *, dtype: np.dtype[Any], shape: list[int]
) -> dict[str, Any]:
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise PackValidationError(
            f"raw tensor byte count mismatch for {path}: {actual_bytes} != {expected_bytes}"
        )
    return {
        "dtype": dtype.str,
        "shape": shape,
        "data_bytes": actual_bytes,
        "data_sha256": _sha256_file(path),
    }


def _banana_smasher_plane_descriptor(path: Path, *, layer: int) -> dict[str, Any]:
    match = BANANA_SMASHER_PLANE_RE.fullmatch(path.name)
    if match is None:
        raise PackValidationError(f"unsupported banana_smasher plane name: {path.name}")
    tier_name = match.group(1)
    subtier = int(match.group(2))
    projection = match.group(3)
    encoded_role = match.group(4)
    if encoded_role.startswith("codebook"):
        role = "codebooks"
        dtype = np.dtype("<f2")
        if path.stat().st_size % (4 * dtype.itemsize):
            raise PackValidationError(f"d4 codebook is not [K,4] fp16: {path}")
        shape = [path.stat().st_size // (4 * dtype.itemsize), 4]
    elif encoded_role.startswith("codes"):
        role = "codes"
        dtype = np.dtype("uint8")
        shape = [path.stat().st_size]
    elif encoded_role.startswith("expert_ids"):
        role = "expert_ids"
        dtype = np.dtype("<i2")
        if path.stat().st_size % dtype.itemsize:
            raise PackValidationError(f"unaligned int16 expert ids: {path}")
        shape = [path.stat().st_size // dtype.itemsize]
    else:
        role = "scales"
        dtype = np.dtype("uint8")
        shape = [path.stat().st_size]
    encoding = encoded_role.split(".", 1)[1]
    name = f"layers.{layer}.truevq_d4.{tier_name}.{projection}.{role}"
    return {
        "name": name,
        "tier": tier_name,
        "subtier": subtier,
        "projection": projection,
        "role": role,
        "encoding": encoding,
        "dtype": dtype,
        "shape": shape,
    }


def _verify_banana_smasher_source(source_root: Path) -> tuple[int, list[Path], str]:
    receipt_path = source_root / "LAYER_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackValidationError(
            f"cannot read banana_smasher LAYER_RECEIPT.json: {exc}"
        ) from exc
    if not isinstance(receipt, dict) or receipt.get("status") != "PASS":
        raise PackValidationError("banana_smasher LAYER_RECEIPT.json must be a PASS object")
    layer_match = BANANA_SMASHER_LAYER_RE.fullmatch(source_root.name)
    receipt_layer = receipt.get("layer")
    if layer_match is None or receipt_layer != int(layer_match.group(1)):
        raise PackValidationError(
            f"banana_smasher layer identity mismatch: directory={source_root.name!r}, receipt={receipt_layer!r}"
        )
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackValidationError("banana_smasher receipt files must be a non-empty list")
    expected: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise PackValidationError("malformed banana_smasher receipt file row")
        relative = Path(row["path"])
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise PackValidationError(f"unsafe banana_smasher receipt path: {relative}")
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise PackValidationError(
                f"missing/non-regular banana_smasher source file: {relative}"
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != row.get("bytes"):
            raise PackValidationError(
                f"banana_smasher source byte count mismatch for {relative}: "
                f"expected {row.get('bytes')}, got {actual_bytes}"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != row.get("sha256"):
            raise PackValidationError(
                f"banana_smasher source sha256 mismatch for {relative}: "
                f"expected {row.get('sha256')}, got {actual_sha}"
            )
        if BANANA_SMASHER_PLANE_RE.fullmatch(relative.name) is None:
            raise PackValidationError(f"unsupported banana_smasher receipt plane: {relative}")
        expected.add(relative.name)
    actual = {path.name for path in source_root.glob("*.bin") if path.is_file()}
    if actual != expected:
        raise PackValidationError(
            f"banana_smasher source file-set mismatch: extras={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )
    planes = [source_root / name for name in sorted(expected)]
    return int(receipt_layer), planes, _sha256_file(receipt_path)


def _banana_smasher_tier_maps(planes: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    tier_map = np.full(256, TIER_CODES["truevq_d4"], dtype=np.uint8)
    subtier_map = np.zeros(256, dtype=np.uint16)
    seen_by_projection: dict[str, set[int]] = {
        projection: set() for projection in BANANA_SMASHER_PROJECTIONS
    }
    ids_by_tier_projection: dict[tuple[int, str], np.ndarray] = {}
    for path in planes:
        descriptor = _banana_smasher_plane_descriptor(path, layer=0)
        if descriptor["role"] != "expert_ids":
            continue
        ids = np.fromfile(path, dtype="<i2")
        if np.any(ids < 0) or np.any(ids >= 256) or len(np.unique(ids)) != len(ids):
            raise PackValidationError(f"invalid/duplicate banana_smasher expert ids: {path}")
        projection = str(descriptor["projection"])
        overlap = seen_by_projection[projection].intersection(
            int(value) for value in ids
        )
        if overlap:
            raise PackValidationError(
                f"banana_smasher tier expert overlap for {projection}: {sorted(overlap)}"
            )
        seen_by_projection[projection].update(int(value) for value in ids)
        key = (int(descriptor["subtier"]), projection)
        ids_by_tier_projection[key] = ids
        subtier_map[ids] = int(descriptor["subtier"])
    expected_ids = set(range(256))
    for projection, seen in seen_by_projection.items():
        if seen != expected_ids:
            raise PackValidationError(
                f"banana_smasher {projection} expert partition is incomplete: "
                f"missing={sorted(expected_ids - seen)}, extras={sorted(seen - expected_ids)}"
            )
    for subtier in BANANA_SMASHER_SUBTIERS:
        down = ids_by_tier_projection.get((subtier, "down"))
        fused = ids_by_tier_projection.get((subtier, "fused13"))
        if down is None or fused is None or not np.array_equal(down, fused):
            raise PackValidationError(
                f"banana_smasher expert ids disagree across projections for d4_k{subtier}"
            )
    return tier_map, subtier_map


def _file_entry(root: Path, relative: Path, role: str) -> dict[str, Any]:
    path = root / relative
    if not path.is_file() or path.is_symlink():
        raise PackValidationError(
            f"pack file must be a regular non-symlink: {relative}"
        )
    return {
        "path": relative.as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _link_file(
    source: Path, destination: Path, mode: Literal["hardlink", "copy", "auto"]
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, destination)
        return "hardlink"
    if mode == "copy":
        shutil.copy2(source, destination)
        return "copy"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _layout_contract() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "tier_codes": TIER_CODES,
        "tier_map": {
            "dtype": "|u1",
            "shape": [256],
            "partitions": [[0, 64], [64, 128], [128, 192], [192, 256]],
            "semantics": "tier_map[e] selects one family for expert e; partitions are storage-only and never renumber experts",
        },
        "truevq_subtier_map": {
            "dtype": "<u2",
            "shape": [256],
            "allowed_values": list(BANANA_SMASHER_SUBTIERS),
            "semantics": "subtier_map[e] stores trueVQ d4 codebook cardinality K for expert e",
        },
        "banana_smasher_raw_tensor_name": (
            "layers.{layer}.truevq_d4.d4_k{K}.{projection}.{role}"
        ),
        "banana_smasher_raw_storage": "headerless little-endian source bytes, manifest-bound dtype/shape/encoding",
        "required_family_fields": {
            family: sorted(fields) for family, fields in REQUIRED_FAMILY_FIELDS.items()
        },
        "tensor_name": "layers.{layer}.{family}.{field}",
    }


def layout_sha256() -> str:
    return _sha256_bytes(_canonical_json(_layout_contract()))


def _layer_meta(layer: int, tensor_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    prefix = f"layers.{layer}."
    tensors = sorted(name for name in tensor_index if name.startswith(prefix))
    families = sorted(
        {
            name.split(".", 3)[2]
            for name in tensors
            if name.split(".", 3)[2] != "experts"
        }
    )
    return {
        "schema": "bs-pack-layer-meta",
        "schema_version": 1,
        "layer": layer,
        "experts_per_layer": 256,
        "expert_partitions": [64, 64, 64, 64],
        "tier_map": f"layers.{layer}.experts.tier_map",
        "dispatch_admission": {
            "scalar": {"predicate": "valid_m<4", "valid_m": [1, 2, 3]},
            "vector_m4": {"predicate": "valid_m==4", "valid_m": [4]},
        },
        "families": families,
        "tensors": tensors,
    }


def _complete_marker(instance_id: str, tensor_layout_sha256: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "instance_id": instance_id,
        "status": "COMPLETE",
        "tensor_layout_sha256": tensor_layout_sha256,
    }


def export_pack(
    *,
    source_root: str | Path,
    knapsack_manifest: str | Path | None = None,
    output: str | Path,
    model_id: str,
    instance_id: str,
    link_mode: Literal["hardlink", "copy", "auto"] = "hardlink",
) -> dict[str, Any]:
    """Export canonical npy planes or a sealed BANANA_SMASHER layer as bs-pack v1."""
    source_root_input = Path(source_root).expanduser()
    from .knapsack import preflight_export_manifest

    manifest_path = (
        Path(knapsack_manifest).expanduser().resolve()
        if knapsack_manifest is not None
        else source_root_input / "MANIFEST.json"
    )
    preflight_export_manifest(manifest_path)
    source_root = source_root_input.resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    config_source = source_root / "config.json"
    banana_smasher_receipt = source_root / "LAYER_RECEIPT.json"
    source_receipt_sha256: str | None = None
    if banana_smasher_receipt.is_file():
        layer, planes, source_receipt_sha256 = _verify_banana_smasher_source(source_root)
        tier_map, subtier_map = _banana_smasher_tier_maps(planes)
        source_format = "banana_smasher-materialized-layer-v1"
    else:
        if not config_source.is_file():
            raise PackValidationError(
                f"source config.json is required: {config_source}"
            )
        planes = sorted(path for path in source_root.rglob("*.npy") if path.is_file())
        if not planes:
            raise PackValidationError(f"source contains no .npy planes: {source_root}")
        source_format = "canonical-npy-v1"

    output.mkdir(parents=True)
    linked: list[dict[str, str]] = []
    tensor_index: dict[str, dict[str, Any]] = {}
    try:
        if source_format == "canonical-npy-v1":
            for source in planes:
                relative = source.relative_to(source_root)
                name = _tensor_name(relative)
                if name in tensor_index:
                    raise PackValidationError(f"duplicate tensor name: {name}")
                destination_relative = Path("planes") / relative
                actual_mode = _link_file(
                    source, output / destination_relative, link_mode
                )
                metadata = _npy_metadata(output / destination_relative)
                metadata["path"] = destination_relative.as_posix()
                metadata["storage"] = {
                    "kind": "npy",
                    "path": destination_relative.as_posix(),
                }
                tensor_index[name] = metadata
                linked.append(
                    {
                        "path": destination_relative.as_posix(),
                        "mode": actual_mode,
                        "role": "npy_plane",
                    }
                )
            config = json.loads(config_source.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise PackValidationError("source config.json must contain an object")
        else:
            for source in planes:
                descriptor = _banana_smasher_plane_descriptor(source, layer=layer)
                name = str(descriptor["name"])
                if name in tensor_index:
                    raise PackValidationError(f"duplicate tensor name: {name}")
                relative = (
                    Path("planes")
                    / "layers"
                    / f"layer_{layer:03d}"
                    / "truevq_d4"
                    / source.name
                )
                actual_mode = _link_file(source, output / relative, link_mode)
                metadata = _raw_metadata(
                    output / relative,
                    dtype=descriptor["dtype"],
                    shape=descriptor["shape"],
                )
                metadata.update(
                    {
                        "path": relative.as_posix(),
                        "encoding": descriptor["encoding"],
                        "subtier": descriptor["subtier"],
                        "projection": descriptor["projection"],
                        "storage": {"kind": "raw", "path": relative.as_posix()},
                    }
                )
                tensor_index[name] = metadata
                linked.append(
                    {
                        "path": relative.as_posix(),
                        "mode": actual_mode,
                        "role": "banana_smasher_raw_plane",
                    }
                )
            generated = {
                "tier_map": tier_map,
                "subtier_map": subtier_map,
            }
            for field, array in generated.items():
                relative = (
                    Path("planes")
                    / "layers"
                    / f"layer_{layer:03d}"
                    / "experts"
                    / f"{field}.npy"
                )
                (output / relative).parent.mkdir(parents=True, exist_ok=True)
                np.save(output / relative, array, allow_pickle=False)
                name = f"layers.{layer}.experts.{field}"
                metadata = _npy_metadata(output / relative)
                metadata["path"] = relative.as_posix()
                metadata["storage"] = {"kind": "npy", "path": relative.as_posix()}
                tensor_index[name] = metadata
                linked.append(
                    {
                        "path": relative.as_posix(),
                        "mode": "generated",
                        "role": "derived_index_plane",
                    }
                )
            provenance_relative = Path("provenance/LAYER_RECEIPT.json")
            (output / provenance_relative).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(banana_smasher_receipt, output / provenance_relative)
            config = {
                "_name_or_path": model_id,
                "model_type": "deepseek_v4",
                "bs_pack_scope": f"layer_{layer:03d}",
            }

        config["quantization_config"] = {
            "quant_method": QUANT_METHOD,
            "format": SCHEMA,
            "format_version": SCHEMA_VERSION,
            "pack_manifest": MANIFEST_NAME,
            "pack_root": ".",
            "kernel_cache_root": "kernel-cache",
            "architecture": "sm_120",
            "tensor_container": None,
            "kernel_cache_manifest": "BS_KERNEL_CACHE_MANIFEST.json",
        }
        (output / "config.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        layers = sorted(
            {
                int(name.split(".")[1])
                for name in tensor_index
                if name.endswith(".experts.tier_map")
            }
        )
        for layer in layers:
            relative = Path("planes/layers") / f"layer_{layer:03d}" / "meta.json"
            (output / relative).write_text(
                json.dumps(_layer_meta(layer, tensor_index), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            linked.append(
                {"path": relative.as_posix(), "mode": "generated", "role": "layer_meta"}
            )

        (output / COMPLETE_MARKER_NAME).write_text(
            json.dumps(
                _complete_marker(instance_id, layout_sha256()),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        linked.append(
            {
                "path": COMPLETE_MARKER_NAME,
                "mode": "generated",
                "role": "pack_complete",
            }
        )

        file_entries = [_file_entry(output, Path("config.json"), "model_config")]
        file_entries.extend(
            _file_entry(output, Path(row["path"]), row["role"]) for row in linked
        )
        if source_format == "banana_smasher-materialized-layer-v1":
            file_entries.append(
                _file_entry(
                    output,
                    Path("provenance/LAYER_RECEIPT.json"),
                    "source_layer_receipt",
                )
            )
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "source_format": source_format,
            "model_id": model_id,
            "instance_id": instance_id,
            "quant_method": QUANT_METHOD,
            "layers": layers,
            "experts_per_layer": 256,
            "expert_partitions": [64, 64, 64, 64],
            "tier_codes": TIER_CODES,
            "tensor_layout_sha256": layout_sha256(),
            "tensor_index": dict(sorted(tensor_index.items())),
            "files": sorted(file_entries, key=lambda row: row["path"]),
            "link_mode_requested": link_mode,
            "links": linked,
            "container": None,
            "provenance": {
                "source_root": str(source_root),
                "source_layer_receipt_sha256": source_receipt_sha256,
                "port_base": "glm52-ds4-bq3-ptq-opd/docker/scripts/export_pack.py",
            },
        }
        (output / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        verify_pack(output)
        return manifest
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise


def load_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    path = root / MANIFEST_NAME
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackValidationError(f"cannot read {MANIFEST_NAME}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise PackValidationError("pack manifest must contain an object")
    return manifest


def _verify_manifest_identity(manifest: dict[str, Any]) -> None:
    expected = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "quant_method": QUANT_METHOD,
        "experts_per_layer": 256,
        "expert_partitions": [64, 64, 64, 64],
        "tier_codes": TIER_CODES,
        "tensor_layout_sha256": layout_sha256(),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise PackValidationError(
                f"manifest {key} mismatch: expected {value!r}, got {manifest.get(key)!r}"
            )


def _verify_files(root: Path, manifest: dict[str, Any]) -> None:
    marker_path = root / COMPLETE_MARKER_NAME
    if not marker_path.is_file() or marker_path.is_symlink():
        raise PackValidationError("missing PACK_COMPLETE marker")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise PackValidationError("manifest files must be a list")
    expected_paths = {MANIFEST_NAME}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise PackValidationError("malformed manifest file row")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PackValidationError(f"unsafe manifest path: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise PackValidationError(f"missing/non-regular pack file: {relative}")
        actual_bytes = path.stat().st_size
        if actual_bytes != row.get("bytes"):
            raise PackValidationError(
                f"byte count mismatch for {relative}: expected {row.get('bytes')}, got {actual_bytes}"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != row.get("sha256"):
            raise PackValidationError(
                f"sha256 mismatch for {relative}: expected {row.get('sha256')}, got {actual_sha}"
            )
        expected_paths.add(relative.as_posix())
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    extras = sorted(actual_paths - expected_paths)
    missing = sorted(expected_paths - actual_paths)
    if extras or missing:
        raise PackValidationError(
            f"pack file-set mismatch: extras={extras}, missing={missing}"
        )


def _verify_complete_marker(root: Path, manifest: dict[str, Any]) -> None:
    rows = manifest.get("files", [])
    marker_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("path") == COMPLETE_MARKER_NAME
    ]
    if len(marker_rows) != 1 or marker_rows[0].get("role") != "pack_complete":
        raise PackValidationError(
            "PACK_COMPLETE must be manifest-bound exactly once with role pack_complete"
        )
    try:
        marker = json.loads((root / COMPLETE_MARKER_NAME).read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackValidationError(f"invalid PACK_COMPLETE marker: {exc}") from exc
    expected = _complete_marker(
        str(manifest.get("instance_id")), str(manifest.get("tensor_layout_sha256"))
    )
    if marker != expected:
        raise PackValidationError(
            f"PACK_COMPLETE marker mismatch: expected {expected!r}, got {marker!r}"
        )


def _verify_config(root: Path) -> None:
    try:
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        quant = config["quantization_config"]
    except Exception as exc:
        raise PackValidationError(
            f"invalid config.json quantization_config: {exc}"
        ) from exc
    expected = {
        "quant_method": QUANT_METHOD,
        "format": SCHEMA,
        "format_version": SCHEMA_VERSION,
        "pack_manifest": MANIFEST_NAME,
        "pack_root": ".",
        "kernel_cache_root": "kernel-cache",
        "architecture": "sm_120",
    }
    for key, value in expected.items():
        if quant.get(key) != value:
            raise PackValidationError(
                f"config quantization_config.{key} mismatch: expected {value!r}, got {quant.get(key)!r}"
            )


def _verify_layer_meta(root: Path, manifest: dict[str, Any]) -> None:
    index = manifest.get("tensor_index")
    layers = manifest.get("layers")
    if not isinstance(index, dict) or not isinstance(layers, list):
        raise PackValidationError("cannot verify layer meta without tensor_index/layers")
    file_roles = {
        row.get("path"): row.get("role")
        for row in manifest.get("files", [])
        if isinstance(row, dict)
    }
    for layer in layers:
        relative = Path("planes/layers") / f"layer_{layer:03d}" / "meta.json"
        if file_roles.get(relative.as_posix()) != "layer_meta":
            raise PackValidationError(
                f"layer meta is not manifest-bound with role layer_meta: {relative}"
            )
        try:
            actual = json.loads((root / relative).read_text(encoding="utf-8"))
        except Exception as exc:
            raise PackValidationError(f"cannot read layer meta {relative}: {exc}") from exc
        expected = _layer_meta(layer, index)
        if actual != expected:
            raise PackValidationError(
                f"layer meta mismatch for layer {layer}: expected {expected!r}, got {actual!r}"
            )


def _verify_tensors(root: Path, manifest: dict[str, Any]) -> tuple[int, list[int]]:
    index = manifest.get("tensor_index")
    if not isinstance(index, dict) or not index:
        raise PackValidationError("tensor_index must be a non-empty object")
    layer_fields: dict[int, dict[str, set[str]]] = {}
    tier_layers: list[int] = []
    for name, recorded in sorted(index.items()):
        match = TENSOR_RE.fullmatch(name)
        if match is None:
            raise PackValidationError(f"invalid tensor name: {name}")
        if not isinstance(recorded, dict):
            raise PackValidationError(f"invalid tensor metadata: {name}")
        layer = int(match.group(1))
        storage = recorded.get("storage")
        if storage is None:
            storage = {"kind": "npy", "path": recorded.get("path")}
        if not isinstance(storage, dict):
            raise PackValidationError(f"invalid tensor storage metadata: {name}")
        storage_kind = storage.get("kind")
        if storage_kind == "npy":
            path = root / str(storage.get("path"))
            metadata = _npy_metadata(path)
        elif storage_kind == "raw":
            path = root / str(storage.get("path"))
            if not path.is_file() or path.is_symlink():
                raise PackValidationError(
                    f"invalid raw tensor plane for {name}: {path}"
                )
            try:
                dtype = np.dtype(recorded.get("dtype"))
                shape = recorded.get("shape")
                if not isinstance(shape, list) or not all(
                    isinstance(value, int) and value >= 0 for value in shape
                ):
                    raise ValueError(f"invalid shape {shape!r}")
            except Exception as exc:
                raise PackValidationError(
                    f"invalid raw tensor metadata for {name}: {exc}"
                ) from exc
            metadata = _raw_metadata(path, dtype=dtype, shape=shape)
        elif storage_kind == "safetensors":
            from .repack import verify_tensor_storage

            metadata = verify_tensor_storage(root, name, recorded)
        else:
            raise PackValidationError(
                f"unsupported tensor storage kind for {name}: {storage_kind!r}"
            )
        for key in ("dtype", "shape", "data_bytes", "data_sha256"):
            if metadata[key] != recorded.get(key):
                raise PackValidationError(
                    f"tensor metadata mismatch for {name}.{key}: expected {recorded.get(key)!r}, got {metadata[key]!r}"
                )
        suffix = match.group(2)
        if suffix == "experts.tier_map":
            if storage_kind == "npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            else:
                from .repack import load_tensor_numpy

                array = load_tensor_numpy(root, name, recorded)
            if array.dtype != np.dtype("uint8") or tuple(array.shape) != (256,):
                raise PackValidationError(
                    f"{name} must be uint8[256], got {array.dtype}{tuple(array.shape)}"
                )
            codes = {int(code) for code in np.unique(array)}
            invalid = sorted(codes - set(TIER_CODES.values()))
            if invalid:
                raise PackValidationError(
                    f"{name} contains unknown tier codes: {invalid}"
                )
            layer_fields.setdefault(layer, {})["__used_codes__"] = {
                str(code) for code in codes
            }
            tier_layers.append(layer)
        elif suffix == "experts.subtier_map":
            if storage_kind == "npy":
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            else:
                from .repack import load_tensor_numpy

                array = load_tensor_numpy(root, name, recorded)
            if array.dtype != np.dtype("uint16") or tuple(array.shape) != (256,):
                raise PackValidationError(
                    f"{name} must be uint16[256], got {array.dtype}{tuple(array.shape)}"
                )
            invalid = sorted(
                {int(value) for value in np.unique(array)} - set(BANANA_SMASHER_SUBTIERS)
            )
            if invalid:
                raise PackValidationError(
                    f"{name} contains unknown trueVQ d4 subtiers: {invalid}"
                )
            layer_fields.setdefault(layer, {})["__subtier_map__"] = {"present"}
        else:
            family, field = suffix.split(".", 1)
            layer_fields.setdefault(layer, {}).setdefault(family, set()).add(field)

    declared_layers = manifest.get("layers")
    if sorted(tier_layers) != declared_layers:
        raise PackValidationError(
            f"tier-map layers mismatch: manifest={declared_layers}, tensors={sorted(tier_layers)}"
        )
    for layer in tier_layers:
        used_codes = layer_fields[layer].get("__used_codes__", set())
        for code_text in used_codes:
            family = TIER_FAMILIES[int(code_text)]
            fields = layer_fields[layer].get(family, set())
            if family == "truevq_d4" and any(
                field.startswith("d4_k") for field in fields
            ):
                expected = {
                    f"d4_k{subtier}.{projection}.{role}"
                    for subtier in BANANA_SMASHER_SUBTIERS
                    for projection in BANANA_SMASHER_PROJECTIONS
                    for role in BANANA_SMASHER_ROLES
                }
                missing = sorted(expected - fields)
                if not layer_fields[layer].get("__subtier_map__"):
                    missing.append("experts.subtier_map")
            else:
                missing = sorted(REQUIRED_FAMILY_FIELDS[family] - fields)
            if missing:
                raise PackValidationError(
                    f"layer {layer} family {family} missing required tensors: {missing}"
                )
    return len(index), sorted(tier_layers)


def verify_pack(root: str | Path) -> dict[str, Any]:
    """Verify every bs-pack v1 manifest, file, tensor, tier, and config invariant."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise PackValidationError(f"pack root is not a directory: {root}")
    manifest = load_manifest(root)
    _verify_manifest_identity(manifest)
    _verify_files(root, manifest)
    _verify_complete_marker(root, manifest)
    _verify_config(root)
    _verify_layer_meta(root, manifest)
    tensor_count, layers = _verify_tensors(root, manifest)
    return {
        "status": "PASS",
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "instance_id": manifest.get("instance_id"),
        "tensor_count": tensor_count,
        "layers": layers,
        "tensor_layout_sha256": manifest["tensor_layout_sha256"],
    }


def _required_families(root: Path, manifest: dict[str, Any]) -> list[str]:
    used: set[str] = set()
    for name, metadata in manifest["tensor_index"].items():
        if not name.endswith(".experts.tier_map"):
            continue
        storage = metadata.get("storage", {"kind": "npy", "path": metadata.get("path")})
        if storage.get("kind") == "npy":
            tier_map = np.load(
                root / storage["path"], mmap_mode="r", allow_pickle=False
            )
        else:
            from .repack import load_tensor_numpy

            tier_map = load_tensor_numpy(root, name, metadata)
        used.update(TIER_FAMILIES[int(code)] for code in np.unique(tier_map))
    return sorted(used)


def verify_serve_compatibility(
    pack_root: str | Path,
    kernel_cache_root: str | Path,
    *,
    architecture: str,
) -> dict[str, Any]:
    """Fail closed unless a verified kernel cache exactly matches this pack ABI."""
    pack_root = Path(pack_root).resolve()
    kernel_cache_root = Path(kernel_cache_root).resolve()
    pack_receipt = verify_pack(pack_root)
    pack_manifest = load_manifest(pack_root)
    required_families = _required_families(pack_root, pack_manifest)
    path = kernel_cache_root / KERNEL_MANIFEST_NAME
    try:
        kernel_manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackValidationError(f"cannot read {KERNEL_MANIFEST_NAME}: {exc}") from exc
    expected = {
        "schema": "bs-kernel-cache",
        "schema_version": 1,
        "quant_method": QUANT_METHOD,
        "pack_schema": SCHEMA,
        "pack_schema_version": SCHEMA_VERSION,
    }
    for key, value in expected.items():
        if kernel_manifest.get(key) != value:
            raise PackValidationError(
                f"kernel manifest {key} mismatch: expected {value!r}, got {kernel_manifest.get(key)!r}"
            )
    if kernel_manifest.get("tensor_layout_sha256") != pack_manifest.get(
        "tensor_layout_sha256"
    ):
        raise PackValidationError(
            "kernel cache tensor layout is incompatible with the pack: "
            f"cache={kernel_manifest.get('tensor_layout_sha256')}, "
            f"pack={pack_manifest.get('tensor_layout_sha256')}"
        )
    architectures = kernel_manifest.get("architectures")
    if not isinstance(architectures, list) or architecture not in architectures:
        raise PackValidationError(
            f"kernel cache does not support architecture {architecture!r}: {architectures!r}"
        )
    supported_families = kernel_manifest.get("families")
    if not isinstance(supported_families, list):
        raise PackValidationError("kernel manifest families must be a list")
    missing_families = sorted(set(required_families) - set(supported_families))
    if missing_families:
        raise PackValidationError(
            f"kernel cache is missing required families: {missing_families}"
        )
    runtime_adapter = kernel_manifest.get("runtime_adapter")
    if not isinstance(runtime_adapter, dict):
        raise PackValidationError("kernel manifest runtime_adapter must be an object")
    adapter_path = runtime_adapter.get("path")
    adapter_class = runtime_adapter.get("class")
    if (
        not isinstance(adapter_path, str)
        or Path(adapter_path).is_absolute()
        or ".." in Path(adapter_path).parts
        or not isinstance(adapter_class, str)
        or not adapter_class.isidentifier()
        or runtime_adapter.get("api_version") != 1
    ):
        raise PackValidationError(
            f"invalid runtime_adapter contract: {runtime_adapter!r}"
        )
    rows = kernel_manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackValidationError("kernel manifest files must be a non-empty list")
    expected_paths = {KERNEL_MANIFEST_NAME}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise PackValidationError("malformed kernel manifest file row")
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise PackValidationError(f"unsafe kernel cache path: {relative}")
        kernel_path = kernel_cache_root / relative
        if not kernel_path.is_file() or kernel_path.is_symlink():
            raise PackValidationError(
                f"missing/non-regular kernel cache file: {relative}"
            )
        if kernel_path.stat().st_size != row.get("bytes"):
            raise PackValidationError(f"kernel byte count mismatch: {relative}")
        if _sha256_file(kernel_path) != row.get("sha256"):
            raise PackValidationError(f"kernel sha256 mismatch: {relative}")
        expected_paths.add(relative.as_posix())
    actual_paths = {
        item.relative_to(kernel_cache_root).as_posix()
        for item in kernel_cache_root.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    if actual_paths != expected_paths:
        raise PackValidationError(
            "kernel cache file-set mismatch: "
            f"extras={sorted(actual_paths - expected_paths)}, "
            f"missing={sorted(expected_paths - actual_paths)}"
        )
    if adapter_path not in expected_paths:
        raise PackValidationError(
            f"runtime adapter is not covered by kernel file manifest: {adapter_path}"
        )
    return {
        "status": "PASS",
        "quant_method": QUANT_METHOD,
        "architecture": architecture,
        "required_families": required_families,
        "tensor_layout_sha256": pack_receipt["tensor_layout_sha256"],
        "kernel_file_count": len(rows),
        "runtime_adapter": runtime_adapter,
    }
