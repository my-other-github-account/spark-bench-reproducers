from __future__ import annotations

import ctypes
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .repair import (
    REPAIR_MANIFEST_PATH,
    REPAIR_STATE_PATH,
    RepairBundle,
    materialize_codebook_plane,
    verify_repair_payload,
    write_repair_payload,
)

SERVING_METADATA_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)
SERVING_METADATA_ROLES = {
    "tokenizer.json": "tokenizer",
    "tokenizer_config.json": "tokenizer_config",
    "generation_config.json": "generation_config",
}
BASE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
BASE_WEIGHTS_SHARD_ROLE = "base_weights_shard"
BASE_WEIGHTS_INDEX_ROLE = "base_weights_index"
BASE_WEIGHTS_ROLES = (BASE_WEIGHTS_SHARD_ROLE, BASE_WEIGHTS_INDEX_ROLE)

MANIFEST_NAME = "BANANA_PACK_MANIFEST.json"
COMPLETE_MARKER_NAME = "PACK_COMPLETE"
KERNEL_MANIFEST_NAME = "BS_KERNEL_CACHE_MANIFEST.json"
SCHEMA = "bs-pack"
SCHEMA_VERSION = 1
QUANT_METHOD = "banana_smasher"
PACKED_INDEX_ENCODING = "little-endian-packed-index-rows-v1"
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
P1016_META_RE = re.compile(r"^layer_(\d{3})\.meta\.json$")
P1016_PLANE_RE = re.compile(
    r"^layer_(\d{3})\.(.+)\.(13|2)\.([A-Za-z0-9_]+)\.npy$"
)
TENSOR_RE = re.compile(
    r"^layers\.(\d+)\.(experts\.(?:tier_map|subtier_map)|"
    r"(?:qtip2|qtip3|truevq_d4|truevq_d8|native_mxfp4)\."
    r"((?:[a-z0-9_]+\.)*[a-z0-9_]+))$"
)
GENESIS_LAYER_RE = re.compile(r"^layer_(\d{3})$")
GENESIS_PLANE_RE = re.compile(
    r"^(d4_k(256|1024|2048|4096))\.(down|fused13)\."
    r"(codebook\.fp16|codes\.le(8|10|11|12)|expert_ids\.i16|scales\.e8m0)\.bin$"
)
GENESIS_SUBTIERS = (256, 1024, 2048, 4096)
GENESIS_PROJECTIONS = ("down", "fused13")
GENESIS_ROLES = ("codebooks", "codes", "expert_ids", "scales")


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


def _index_bits(codebook_size: int) -> int:
    if codebook_size <= 1 or codebook_size & (codebook_size - 1):
        raise PackValidationError(
            f"D4 codebook size must be a power of two, got {codebook_size}"
        )
    bits = codebook_size.bit_length() - 1
    if not 1 <= bits <= 16:
        raise PackValidationError(f"unsupported D4 index width: {bits}")
    return bits


def _packed_row_bytes(values_per_row: int, bits: int) -> int:
    if values_per_row < 0 or not 1 <= bits <= 16:
        raise PackValidationError(
            f"invalid packed-index geometry: values_per_row={values_per_row} bits={bits}"
        )
    return (values_per_row * bits + 7) // 8


def unpack_index_rows(
    packed: np.ndarray, *, bits: int, values_per_row: int
) -> np.ndarray:
    """Decode independently packed little-endian index rows as uint16."""
    source = np.asarray(packed, dtype=np.uint8)
    if source.ndim < 1:
        raise PackValidationError("packed index tensor must have at least one dimension")
    expected = _packed_row_bytes(values_per_row, bits)
    if source.shape[-1] != expected:
        raise PackValidationError(
            f"packed row has {source.shape[-1]} bytes; expected {expected}"
        )
    rows = int(np.prod(source.shape[:-1], dtype=np.int64))
    source_rows = source.reshape(rows, expected)
    padded = np.pad(source_rows, ((0, 0), (0, 2)))
    decoded = np.empty((rows, values_per_row), dtype=np.uint16)
    mask = (1 << bits) - 1
    for column in range(values_per_row):
        bit = column * bits
        byte, shift = divmod(bit, 8)
        word = (
            padded[:, byte].astype(np.uint32)
            | (padded[:, byte + 1].astype(np.uint32) << 8)
            | (padded[:, byte + 2].astype(np.uint32) << 16)
        )
        decoded[:, column] = ((word >> shift) & mask).astype(np.uint16)
    return decoded.reshape(*source.shape[:-1], values_per_row)


def _pack_index_npy(
    source: Path,
    destination: Path,
    *,
    bits: int,
    working_bytes: int = 64 << 20,
) -> dict[str, Any]:
    """Stream one integer NPY into the row-addressable V4 wire encoding."""
    try:
        values = np.load(source, mmap_mode="r", allow_pickle=False)
    except Exception as exc:
        raise PackValidationError(f"cannot mmap D4 code indices {source}: {exc}") from exc
    if values.ndim < 1 or values.dtype.kind not in "iu":
        raise PackValidationError(
            f"D4 code indices must be an integer tensor, got {values.dtype}{values.shape}"
        )
    before = source.stat()
    decoded_shape = list(values.shape)
    decoded_dtype = values.dtype
    decoded_data_bytes = int(values.nbytes)
    decoded_data_sha256 = _sha256_npy_payload(source)
    values_per_row = int(values.shape[-1])
    row_bytes = _packed_row_bytes(values_per_row, bits)
    rows = int(np.prod(values.shape[:-1], dtype=np.int64))
    destination.parent.mkdir(parents=True, exist_ok=True)
    packed_shape = (*values.shape[:-1], row_bytes)
    packed = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.uint8, shape=packed_shape
    )
    source_rows = values.reshape(rows, values_per_row)
    packed_rows = packed.reshape(rows, row_bytes)
    expanded_bytes_per_row = max(
        1,
        values_per_row * (bits + decoded_dtype.itemsize) + row_bytes,
    )
    rows_per_chunk = max(1, working_bytes // expanded_bytes_per_row)
    shifts = np.arange(bits, dtype=np.uint16)
    limit = 1 << bits
    for start in range(0, rows, rows_per_chunk):
        stop = min(rows, start + rows_per_chunk)
        chunk = np.asarray(source_rows[start:stop])
        if chunk.size:
            minimum = int(chunk.min())
            maximum = int(chunk.max())
            if minimum < 0 or maximum >= limit:
                raise PackValidationError(
                    f"D4 index outside {bits}-bit range in {source}: "
                    f"chunk_rows={start}:{stop} min={minimum} max={maximum}"
                )
        bit_rows = (
            ((chunk[..., None].astype(np.uint16, copy=False) >> shifts) & 1)
            .astype(np.uint8)
            .reshape(stop - start, -1)
        )
        packed_rows[start:stop] = np.packbits(
            bit_rows, axis=-1, bitorder="little"
        )
    packed.flush()
    del packed_rows, packed, values
    after = source.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise PackValidationError(f"D4 source changed while packing: {source}")
    metadata = _npy_metadata(destination)
    metadata.update(
        {
            "encoding": PACKED_INDEX_ENCODING,
            "index_bits": bits,
            "values_per_row": values_per_row,
            "packed_row_bytes": row_bytes,
            "decoded_dtype": decoded_dtype.name,
            "decoded_shape": decoded_shape,
            "decoded_data_bytes": decoded_data_bytes,
            "decoded_data_sha256": decoded_data_sha256,
        }
    )
    return metadata


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


def _p1016_tensor_name(relative: Path, *, payload_family: str) -> tuple[int, str]:
    normalized = relative.as_posix()
    if len(relative.parts) != 1:
        raise PackValidationError(
            f"p1016 native plane must be a root-level file: {normalized}"
        )
    match = P1016_PLANE_RE.fullmatch(normalized)
    if match is None:
        raise PackValidationError(f"unsupported p1016 native plane name: {normalized}")
    layer = int(match.group(1))
    tier = match.group(2).lower()
    projection = {"13": "fused13", "2": "down"}[match.group(3)]
    field = match.group(4).lower()
    family = {
        "qtip2": "qtip2",
        "qtip3": "qtip3",
        "d4": "truevq_d4",
        "d8": "truevq_d8",
        "native": "native_mxfp4",
        "native_mxfp4": "native_mxfp4",
    }.get(payload_family)
    if family is None:
        raise PackValidationError(
            f"unsupported p1016 payload family {payload_family!r} in {normalized}"
        )
    name = f"layers.{layer}.{family}.{tier}.{projection}.{field}"
    if TENSOR_RE.fullmatch(name) is None:
        raise PackValidationError(f"unsupported p1016 tensor name: {name}")
    return layer, name


def _verify_p1016_source(
    source_root: Path,
) -> tuple[
    list[int],
    list[Path],
    list[Path],
    dict[int, dict[str, Any]],
    dict[str, list[tuple[dict[str, Any], str]]],
]:
    meta_paths = sorted(source_root.glob("layer_*.meta.json"))
    if not meta_paths:
        raise PackValidationError("p1016 source contains no layer metadata")
    layers: list[int] = []
    for path in meta_paths:
        match = P1016_META_RE.fullmatch(path.name)
        if match is None:
            raise PackValidationError(f"unsupported p1016 metadata name: {path.name}")
        layer = int(match.group(1))
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise PackValidationError(f"cannot read p1016 metadata {path}: {exc}") from exc
        if (
            not isinstance(document, dict)
            or document.get("format") != "p1016-true-c-native-planes-v1"
            or document.get("layer") != layer
            or document.get("E") != 256
        ):
            raise PackValidationError(f"invalid p1016 layer metadata: {path}")
        for key in ("family13", "family2", "tier13", "tier2", "slot13", "slot2"):
            if not isinstance(document.get(key), list) or len(document[key]) != 256:
                raise PackValidationError(f"invalid p1016 metadata vector {path.name}.{key}")
        layers.append(layer)
    if len(set(layers)) != len(layers):
        raise PackValidationError("duplicate p1016 layer metadata")
    documents: dict[int, dict[str, Any]] = {}
    selected_paths: dict[str, Path] = {}
    references: dict[str, list[tuple[dict[str, Any], str]]] = {}
    for path, layer in zip(meta_paths, layers, strict=True):
        document = json.loads(path.read_text(encoding="utf-8"))
        selected_document = copy.deepcopy(document)
        selected_payload_maps: dict[str, dict[str, Any]] = {}
        experts = int(document["E"])
        for projection, suffix in (("fused13", "13"), ("down", "2")):
            tiers = document[f"tier{suffix}"]
            slots = document[f"slot{suffix}"]
            families = document[f"family{suffix}"]
            if not all(
                isinstance(values, list) and len(values) == experts
                for values in (tiers, slots, families)
            ):
                raise PackValidationError(
                    f"invalid p1016 selected route {path.name}/{projection}"
                )
            payload_map = (document.get("payloads") or {}).get(projection)
            if not isinstance(payload_map, dict):
                raise PackValidationError(
                    f"missing p1016 payload map {path.name}/{projection}"
                )
            selected_tiers = {str(tier) for tier in tiers}
            missing = sorted(selected_tiers - set(payload_map))
            if missing:
                raise PackValidationError(
                    f"manifest-owned route {path.name}/{projection} is missing payloads: {missing}"
                )
            selected_payload_maps[projection] = {
                tier: copy.deepcopy(payload_map[tier]) for tier in sorted(selected_tiers)
            }
            family_codes = document.get("family_codes")
            if not isinstance(family_codes, dict):
                raise PackValidationError(
                    f"missing p1016 family code map {path.name}/{projection}"
                )
            used_slots: set[tuple[str, int]] = set()
            for expert, (tier_value, slot_value, family_value) in enumerate(
                zip(tiers, slots, families, strict=True)
            ):
                tier = str(tier_value)
                if not isinstance(slot_value, int) or slot_value < 0:
                    raise PackValidationError(
                        f"invalid selected slot {path.name}/{projection}/{tier}/{expert}: {slot_value!r}"
                    )
                binding = (tier, slot_value)
                if binding in used_slots:
                    raise PackValidationError(
                        f"selected cell binds more than once {path.name}/{projection}/{binding}"
                    )
                used_slots.add(binding)
                payload = selected_payload_maps[projection][tier]
                payload_family = (
                    payload.get("family") if isinstance(payload, dict) else None
                )
                if family_codes.get(payload_family) != family_value:
                    raise PackValidationError(
                        f"selected family binding drift {path.name}/{projection}/{tier}: "
                        f"expert={expert} family={family_value!r} payload_family={payload_family!r}"
                    )
            for tier, payload in selected_payload_maps[projection].items():
                tensors = payload.get("tensors") if isinstance(payload, dict) else None
                if not isinstance(tensors, dict) or not tensors:
                    raise PackValidationError(
                        f"selected payload {path.name}/{projection}/{tier} has no tensor map"
                    )
                for role, spec in tensors.items():
                    relative_text = spec.get("file") if isinstance(spec, dict) else None
                    if not isinstance(relative_text, str):
                        raise PackValidationError(
                            f"selected payload {path.name}/{projection}/{tier}/{role} has no file"
                        )
                    relative = Path(relative_text)
                    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
                        raise PackValidationError(f"unsafe selected p1016 path: {relative}")
                    source = source_root / relative
                    if not source.is_file() or source.is_symlink():
                        raise PackValidationError(f"missing selected p1016 tensor: {source}")
                    parsed_layer, _ = _p1016_tensor_name(
                        relative,
                        payload_family=str(payload.get("family")),
                    )
                    if parsed_layer != layer:
                        raise PackValidationError(
                            f"selected p1016 tensor layer mismatch: {relative} != {layer}"
                        )
                    selected_paths[relative.as_posix()] = source
                    references.setdefault(relative.as_posix(), []).append((payload, role))
                expert_spec = tensors.get("expert_ids")
                if not isinstance(expert_spec, dict):
                    raise PackValidationError(
                        f"selected payload {path.name}/{projection}/{tier} has no expert_ids"
                    )
                expert_ids = np.load(
                    source_root / str(expert_spec["file"]), mmap_mode="r", allow_pickle=False
                ).reshape(-1)
                expected = [
                    (expert, int(slots[expert]))
                    for expert, routed_tier in enumerate(tiers)
                    if str(routed_tier) == tier
                ]
                if len(expert_ids) != len(expected) or {slot for _, slot in expected} != set(
                    range(len(expert_ids))
                ):
                    raise PackValidationError(
                        f"selected payload rows are not exact for {path.name}/{projection}/{tier}"
                    )
                for expert, slot in expected:
                    if int(expert_ids[slot]) != expert:
                        raise PackValidationError(
                            f"selected payload binding drift {path.name}/{projection}/{tier}: "
                            f"expert={expert} slot={slot} stored={int(expert_ids[slot])}"
                        )
        selected_document["payloads"] = selected_payload_maps
        documents[layer] = selected_document
    planes = [selected_paths[name] for name in sorted(selected_paths)]
    if not planes:
        raise PackValidationError("p1016 manifest selection contains no tensor planes")
    plane_layers: set[int] = set()
    for plane in planes:
        match = P1016_PLANE_RE.fullmatch(plane.relative_to(source_root).as_posix())
        if match is None:  # already validated above; keep this gate fail-closed
            raise PackValidationError(f"unsupported selected p1016 tensor: {plane}")
        plane_layers.add(int(match.group(1)))
    if plane_layers != set(layers):
        raise PackValidationError(
            f"p1016 plane/meta layer mismatch: planes={sorted(plane_layers)} meta={layers}"
        )
    return layers, planes, meta_paths, documents, references


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


def _verify_packed_index_metadata(name: str, recorded: dict[str, Any]) -> None:
    if recorded.get("encoding") != PACKED_INDEX_ENCODING:
        return
    bits = recorded.get("index_bits")
    values_per_row = recorded.get("values_per_row")
    row_bytes = recorded.get("packed_row_bytes")
    decoded_shape = recorded.get("decoded_shape")
    decoded_dtype = recorded.get("decoded_dtype")
    decoded_bytes = recorded.get("decoded_data_bytes")
    decoded_sha = recorded.get("decoded_data_sha256")
    shape = recorded.get("shape")
    if not isinstance(bits, int) or not 1 <= bits <= 16:
        raise PackValidationError(f"invalid packed index width for {name}: {bits!r}")
    if not isinstance(values_per_row, int) or values_per_row < 0:
        raise PackValidationError(
            f"invalid packed values_per_row for {name}: {values_per_row!r}"
        )
    if row_bytes != _packed_row_bytes(values_per_row, bits):
        raise PackValidationError(f"invalid packed row byte count for {name}: {row_bytes!r}")
    if (
        not isinstance(shape, list)
        or not isinstance(decoded_shape, list)
        or decoded_shape[:-1] != shape[:-1]
        or decoded_shape[-1:] != [values_per_row]
        or shape[-1:] != [row_bytes]
    ):
        raise PackValidationError(
            f"packed/decoded shape mismatch for {name}: {shape!r} -> {decoded_shape!r}"
        )
    try:
        dtype = np.dtype(decoded_dtype)
    except Exception as exc:
        raise PackValidationError(
            f"invalid packed decoded dtype for {name}: {decoded_dtype!r}"
        ) from exc
    expected_decoded_bytes = int(np.prod(decoded_shape, dtype=np.int64)) * dtype.itemsize
    if decoded_bytes != expected_decoded_bytes:
        raise PackValidationError(
            f"invalid packed decoded byte count for {name}: "
            f"{decoded_bytes!r} != {expected_decoded_bytes}"
        )
    if not isinstance(decoded_sha, str) or len(decoded_sha) != 64:
        raise PackValidationError(f"invalid packed decoded SHA256 for {name}")
    if np.dtype(recorded.get("dtype")) != np.dtype("uint8"):
        raise PackValidationError(f"packed index payload must be uint8 for {name}")


def _genesis_plane_descriptor(path: Path, *, layer: int) -> dict[str, Any]:
    match = GENESIS_PLANE_RE.fullmatch(path.name)
    if match is None:
        raise PackValidationError(f"unsupported genesis plane name: {path.name}")
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


def _verify_genesis_source(source_root: Path) -> tuple[int, list[Path], str]:
    receipt_path = source_root / "LAYER_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise PackValidationError(
            f"cannot read genesis LAYER_RECEIPT.json: {exc}"
        ) from exc
    if not isinstance(receipt, dict) or receipt.get("status") != "PASS":
        raise PackValidationError("genesis LAYER_RECEIPT.json must be a PASS object")
    layer_match = GENESIS_LAYER_RE.fullmatch(source_root.name)
    receipt_layer = receipt.get("layer")
    if layer_match is None or receipt_layer != int(layer_match.group(1)):
        raise PackValidationError(
            f"genesis layer identity mismatch: directory={source_root.name!r}, receipt={receipt_layer!r}"
        )
    rows = receipt.get("files")
    if not isinstance(rows, list) or not rows:
        raise PackValidationError("genesis receipt files must be a non-empty list")
    expected: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise PackValidationError("malformed genesis receipt file row")
        relative = Path(row["path"])
        if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
            raise PackValidationError(f"unsafe genesis receipt path: {relative}")
        path = source_root / relative
        if not path.is_file() or path.is_symlink():
            raise PackValidationError(
                f"missing/non-regular genesis source file: {relative}"
            )
        actual_bytes = path.stat().st_size
        if actual_bytes != row.get("bytes"):
            raise PackValidationError(
                f"genesis source byte count mismatch for {relative}: "
                f"expected {row.get('bytes')}, got {actual_bytes}"
            )
        actual_sha = _sha256_file(path)
        if actual_sha != row.get("sha256"):
            raise PackValidationError(
                f"genesis source sha256 mismatch for {relative}: "
                f"expected {row.get('sha256')}, got {actual_sha}"
            )
        if GENESIS_PLANE_RE.fullmatch(relative.name) is None:
            raise PackValidationError(f"unsupported genesis receipt plane: {relative}")
        expected.add(relative.name)
    actual = {path.name for path in source_root.glob("*.bin") if path.is_file()}
    if actual != expected:
        raise PackValidationError(
            f"genesis source file-set mismatch: extras={sorted(actual - expected)}, "
            f"missing={sorted(expected - actual)}"
        )
    planes = [source_root / name for name in sorted(expected)]
    return int(receipt_layer), planes, _sha256_file(receipt_path)


def _genesis_tier_maps(planes: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    tier_map = np.full(256, TIER_CODES["truevq_d4"], dtype=np.uint8)
    subtier_map = np.zeros(256, dtype=np.uint16)
    seen_by_projection: dict[str, set[int]] = {
        projection: set() for projection in GENESIS_PROJECTIONS
    }
    ids_by_tier_projection: dict[tuple[int, str], np.ndarray] = {}
    for path in planes:
        descriptor = _genesis_plane_descriptor(path, layer=0)
        if descriptor["role"] != "expert_ids":
            continue
        ids = np.fromfile(path, dtype="<i2")
        if np.any(ids < 0) or np.any(ids >= 256) or len(np.unique(ids)) != len(ids):
            raise PackValidationError(f"invalid/duplicate genesis expert ids: {path}")
        projection = str(descriptor["projection"])
        overlap = seen_by_projection[projection].intersection(
            int(value) for value in ids
        )
        if overlap:
            raise PackValidationError(
                f"genesis tier expert overlap for {projection}: {sorted(overlap)}"
            )
        seen_by_projection[projection].update(int(value) for value in ids)
        key = (int(descriptor["subtier"]), projection)
        ids_by_tier_projection[key] = ids
        subtier_map[ids] = int(descriptor["subtier"])
    expected_ids = set(range(256))
    for projection, seen in seen_by_projection.items():
        if seen != expected_ids:
            raise PackValidationError(
                f"genesis {projection} expert partition is incomplete: "
                f"missing={sorted(expected_ids - seen)}, extras={sorted(seen - expected_ids)}"
            )
    for subtier in GENESIS_SUBTIERS:
        down = ids_by_tier_projection.get((subtier, "down"))
        fused = ids_by_tier_projection.get((subtier, "fused13"))
        if down is None or fused is None or not np.array_equal(down, fused):
            raise PackValidationError(
                f"genesis expert ids disagree across projections for d4_k{subtier}"
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
            "allowed_values": list(GENESIS_SUBTIERS),
            "semantics": "subtier_map[e] stores trueVQ d4 codebook cardinality K for expert e",
        },
        "genesis_raw_tensor_name": (
            "layers.{layer}.truevq_d4.d4_k{K}.{projection}.{role}"
        ),
        "genesis_raw_storage": "headerless little-endian source bytes, manifest-bound dtype/shape/encoding",
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


def _read_serving_metadata_files(
    root: Path,
) -> dict[str, bytes]:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise PackValidationError(f"cannot open serving model root safely: {root}: {exc}") from exc
    payloads: dict[str, bytes] = {}
    try:
        for name in ("config.json", *SERVING_METADATA_FILES):
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_fd,
                )
            except OSError as exc:
                raise PackValidationError(
                    f"serving metadata is missing/non-regular: {root / name}: {exc}"
                ) from exc
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise PackValidationError(
                        f"serving metadata is not a regular file: {root / name}"
                    )
                blocks: list[bytes] = []
                while True:
                    block = os.read(descriptor, 8 * 1024 * 1024)
                    if not block:
                        break
                    blocks.append(block)
                after = os.fstat(descriptor)
                identity_before = (
                    before.st_dev,
                    before.st_ino,
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                )
                identity_after = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if identity_after != identity_before:
                    raise PackValidationError(
                        f"serving metadata changed while reading: {root / name}"
                    )
                payloads[name] = b"".join(blocks)
            finally:
                os.close(descriptor)
    finally:
        os.close(root_fd)
    return payloads


def _load_serving_model_metadata(
    serving_model_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, bytes]]:
    root = Path(serving_model_root).resolve()
    payloads = _read_serving_metadata_files(root)
    try:
        config = json.loads(payloads["config.json"].decode("utf-8"))
    except Exception as exc:
        raise PackValidationError(f"cannot read serving config.json: {exc}") from exc
    if not isinstance(config, dict):
        raise PackValidationError("serving config.json must contain an object")
    architectures = config.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or not all(isinstance(value, str) and value for value in architectures)
    ):
        raise PackValidationError(
            "serving config.json must contain a non-empty architectures list"
        )
    return root, config, payloads


def _materialize_serving_metadata(
    metadata_payloads: dict[str, bytes],
    output: Path,
    config: dict[str, Any],
) -> list[dict[str, str]]:
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows: list[dict[str, str]] = []
    for name in SERVING_METADATA_FILES:
        (output / name).write_bytes(metadata_payloads[name])
        rows.append(
            {"path": name, "mode": "copy", "role": SERVING_METADATA_ROLES[name]}
        )
    return rows


def _load_base_weights_plan(
    serving_root: Path,
) -> tuple[list[tuple[str, Path]], bytes] | None:
    """Read the base-model safetensors index, validating every referenced shard.

    Returns (shard_name, resolved_source_path) pairs. Symlinked shards (e.g. a
    model root whose shards point into an NFS mirror) are resolved to their
    real targets; the resolved target must be a regular file.
    """
    index_path = serving_root / BASE_WEIGHTS_INDEX_NAME
    if index_path.is_symlink() or not index_path.is_file():
        return None
    payload = index_path.read_bytes()
    try:
        index = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        raise PackValidationError(
            f"cannot read base weights index: {index_path}: {exc}"
        ) from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise PackValidationError(
            f"base weights index must contain a non-empty weight_map: {index_path}"
        )
    shards: list[tuple[str, Path]] = []
    for shard in sorted({str(value) for value in weight_map.values()}):
        if "/" in shard or "\\" in shard or shard.startswith("."):
            raise PackValidationError(
                f"unsafe shard name in base weights index: {shard!r}"
            )
        resolved = (serving_root / shard).resolve()
        if not resolved.is_file() or resolved.is_symlink():
            raise PackValidationError(
                f"base weights shard missing/non-regular: {serving_root / shard}"
            )
        shards.append((shard, resolved))
    return shards, payload


def _materialize_base_weights(
    serving_root: Path,
    output: Path,
    *,
    link_mode: Literal["hardlink", "copy", "auto"] = "hardlink",
) -> list[dict[str, str]]:
    """Link base-model dense/full weight shards + index into the pack root.

    Hardlink mode never duplicates bytes on the same filesystem and never
    rewrites tensor payloads; a cross-device hardlink fails loudly (use
    ``--link-mode copy``/``auto`` deliberately instead). Returns manifest link
    rows (empty when the serving model root carries no safetensors index —
    metadata-only root).
    """
    plan = _load_base_weights_plan(serving_root)
    if plan is None:
        return []
    shards, index_payload = plan
    rows: list[dict[str, str]] = []
    for shard, resolved in shards:
        mode = _link_file(resolved, output / shard, link_mode)
        rows.append({"path": shard, "mode": mode, "role": BASE_WEIGHTS_SHARD_ROLE})
    (output / BASE_WEIGHTS_INDEX_NAME).write_bytes(index_payload)
    rows.append(
        {
            "path": BASE_WEIGHTS_INDEX_NAME,
            "mode": "copy",
            "role": BASE_WEIGHTS_INDEX_ROLE,
        }
    )
    return rows


def _clone_pack_with_hardlinks(
    source: Path,
    destination: Path,
    *,
    excluded: set[str],
) -> None:
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if relative.as_posix() in excluded:
            continue
        target = destination / relative
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file() and not path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, target)
        else:
            raise PackValidationError(f"cannot clone non-regular pack path: {relative}")


def _exchange_directories(first: Path, second: Path) -> None:
    """Atomically exchange two Linux directories using renameat2."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PackValidationError(
            "atomic metadata refresh requires Linux renameat2(RENAME_EXCHANGE)"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(first),
        -100,
        os.fsencode(second),
        2,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    parent_fd = os.open(first.parent, os.O_RDONLY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def refresh_serving_metadata(
    pack_root: str | Path,
    *,
    serving_model_root: str | Path,
    link_mode: Literal["hardlink", "copy", "auto"] = "hardlink",
) -> dict[str, Any]:
    """Atomically refresh serving metadata without rewriting tensor payloads."""
    pack_root = Path(pack_root).resolve()
    manifest_path = pack_root / MANIFEST_NAME
    manifest_before = manifest_path.read_bytes()
    config_before = (pack_root / "config.json").read_bytes()
    current_config = json.loads(config_before)
    current_manifest = json.loads(manifest_before)
    quantization_config = dict(current_config["quantization_config"])
    quantization_config["quant_method"] = QUANT_METHOD
    serving_root, serving_config, serving_payloads = _load_serving_model_metadata(
        serving_model_root
    )
    merged_config = dict(serving_config)
    merged_config["quantization_config"] = quantization_config

    base_plan = _load_base_weights_plan(serving_root)
    base_paths: set[str] = set()
    if base_plan is not None:
        base_paths = {name for name, _ in base_plan[0]} | {BASE_WEIGHTS_INDEX_NAME}
    stale_base_paths = {
        str(row.get("path"))
        for row in current_manifest.get("files", [])
        if isinstance(row, dict) and row.get("role") in BASE_WEIGHTS_ROLES
    }

    metadata_paths = {"config.json", MANIFEST_NAME, *SERVING_METADATA_FILES}
    replaced_paths = metadata_paths | base_paths | stale_base_paths
    staging = Path(
        tempfile.mkdtemp(prefix=f".{pack_root.name}.serving-metadata-", dir=pack_root.parent)
    )
    exchanged = False
    try:
        _clone_pack_with_hardlinks(pack_root, staging, excluded=replaced_paths)
        _materialize_serving_metadata(serving_payloads, staging, merged_config)
        base_rows = _materialize_base_weights(serving_root, staging, link_mode=link_mode)
        manifest = current_manifest
        manifest["quant_method"] = QUANT_METHOD
        manifest["files"] = [
            row for row in manifest["files"] if row.get("path") not in replaced_paths
        ]
        manifest["links"] = [
            row for row in manifest.get("links", []) if row.get("path") not in replaced_paths
        ]
        manifest["files"].append(_file_entry(staging, Path("config.json"), "model_config"))
        for name in SERVING_METADATA_FILES:
            manifest["files"].append(
                _file_entry(staging, Path(name), SERVING_METADATA_ROLES[name])
            )
            manifest["links"].append(
                {
                    "path": name,
                    "mode": "copy",
                    "role": SERVING_METADATA_ROLES[name],
                }
            )
        for row in base_rows:
            manifest["files"].append(
                _file_entry(staging, Path(row["path"]), row["role"])
            )
            manifest["links"].append(dict(row))
        manifest["files"].sort(key=lambda row: row["path"])
        selection = manifest.get("selected_payloads")
        if isinstance(selection, dict):
            selection["dense_base_bytes"] = sum(
                int(row["bytes"])
                for row in manifest["files"]
                if row.get("role") == BASE_WEIGHTS_SHARD_ROLE
            )
        manifest.setdefault("provenance", {})["serving_model_root"] = str(serving_root)
        staged_manifest = staging / MANIFEST_NAME
        staged_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt = verify_pack(staging)
        staged_config_sha256 = _sha256_file(staging / "config.json")
        staged_manifest_sha256 = _sha256_file(staged_manifest)
        _exchange_directories(pack_root, staging)
        exchanged = True
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if not exchanged:
        raise PackValidationError("serving metadata exchange did not commit")
    return {
        **receipt,
        "root": str(pack_root),
        "mode": "refresh-metadata",
        "serving_model_root": str(serving_root),
        "architectures": merged_config["architectures"],
        "copied_files": list(SERVING_METADATA_FILES),
        "base_weights_shards": len(base_plan[0]) if base_plan is not None else 0,
        "base_weights_index": base_plan is not None,
        "config_sha256_before": _sha256_bytes(config_before),
        "config_sha256_after": staged_config_sha256,
        "manifest_sha256_before": _sha256_bytes(manifest_before),
        "manifest_sha256_after": staged_manifest_sha256,
        "tensor_payloads_rewritten": False,
        "commit": "renameat2(RENAME_EXCHANGE)",
    }


def export_pack(
    *,
    source_root: str | Path,
    output: str | Path,
    model_id: str,
    instance_id: str,
    link_mode: Literal["hardlink", "copy", "auto"] = "hardlink",
    repair: RepairBundle | None = None,
    serving_model_root: str | Path | None = None,
    runtime_floor_bytes: int | None = None,
) -> dict[str, Any]:
    """Export canonical planes, optionally materializing a bound repair checkpoint."""
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    config_source = source_root / "config.json"
    serving_root: Path | None = None
    serving_config: dict[str, Any] | None = None
    serving_payloads: dict[str, bytes] | None = None
    if serving_model_root is not None:
        serving_root, serving_config, serving_payloads = _load_serving_model_metadata(
            serving_model_root
        )
    genesis_receipt = source_root / "LAYER_RECEIPT.json"
    source_receipt_sha256: str | None = None
    p1016_layers: list[int] = []
    p1016_meta_paths: list[Path] = []
    p1016_documents: dict[int, dict[str, Any]] = {}
    p1016_references: dict[str, list[tuple[dict[str, Any], str]]] = {}
    if runtime_floor_bytes is not None and (
        not isinstance(runtime_floor_bytes, int) or runtime_floor_bytes < 0
    ):
        raise PackValidationError("runtime_floor_bytes must be a non-negative integer")
    if genesis_receipt.is_file():
        layer, planes, source_receipt_sha256 = _verify_genesis_source(source_root)
        tier_map, subtier_map = _genesis_tier_maps(planes)
        source_format = "genesis-materialized-layer-v1"
    elif any(source_root.glob("layer_*.meta.json")):
        (
            p1016_layers,
            planes,
            p1016_meta_paths,
            p1016_documents,
            p1016_references,
        ) = _verify_p1016_source(source_root)
        source_format = "p1016-true-c-native-planes-v1"
        if runtime_floor_bytes is None:
            raise PackValidationError(
                "p1016 selected-payload export requires an explicit runtime_floor_bytes receipt"
            )
    else:
        if not config_source.is_file() and repair is None:
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
    repair_rows: list[dict[str, Any]] = []
    repair_summary: dict[str, Any] | None = None
    try:
        if source_format in {
            "canonical-npy-v1",
            "p1016-true-c-native-planes-v1",
        }:
            for source in planes:
                relative = source.relative_to(source_root)
                references = p1016_references.get(relative.as_posix(), [])
                if source_format == "p1016-true-c-native-planes-v1":
                    families = {
                        str(payload.get("family")) for payload, _role in references
                    }
                    if len(families) != 1:
                        raise PackValidationError(
                            f"selected p1016 tensor has ambiguous payload families: "
                            f"{relative} -> {sorted(families)}"
                        )
                    name = _p1016_tensor_name(
                        relative,
                        payload_family=next(iter(families)),
                    )[1]
                else:
                    name = _tensor_name(relative)
                if name in tensor_index:
                    raise PackValidationError(f"duplicate tensor name: {name}")
                destination_relative = Path("planes") / relative
                packed_bits = {
                    _index_bits(int(payload["k"]))
                    for payload, role in references
                    if payload.get("family") == "d4" and role == "codes"
                }
                if len(packed_bits) > 1:
                    raise PackValidationError(
                        f"selected D4 code tensor has conflicting index widths: "
                        f"{relative} -> {sorted(packed_bits)}"
                    )
                repair_row = None
                if repair is not None:
                    repair_row = materialize_codebook_plane(
                        source,
                        output / destination_relative,
                        repair.codebooks,
                    )
                if packed_bits:
                    if repair_row is not None:
                        raise PackValidationError(
                            f"repair attempted to rewrite D4 code indices: {relative}"
                        )
                    metadata = _pack_index_npy(
                        source,
                        output / destination_relative,
                        bits=next(iter(packed_bits)),
                    )
                    actual_mode = "generated-v4-row-pack"
                else:
                    if repair_row is None:
                        actual_mode = _link_file(
                            source, output / destination_relative, link_mode
                        )
                    else:
                        actual_mode = "materialized-repair"
                        repair_rows.extend(repair_row)
                    metadata = _npy_metadata(output / destination_relative)
                metadata["path"] = destination_relative.as_posix()
                metadata["storage"] = {
                    "kind": "npy",
                    "path": destination_relative.as_posix(),
                }
                for payload, role in references:
                    spec = payload["tensors"][role]
                    spec.update(
                        {
                            "file": relative.as_posix(),
                            "shape": metadata["shape"],
                            "dtype": np.dtype(metadata["dtype"]).name,
                            "data_bytes": metadata["data_bytes"],
                            "data_sha256": metadata["data_sha256"],
                        }
                    )
                    if metadata.get("encoding") == PACKED_INDEX_ENCODING:
                        spec.update(
                            {
                                key: metadata[key]
                                for key in (
                                    "encoding",
                                    "index_bits",
                                    "values_per_row",
                                    "packed_row_bytes",
                                    "decoded_dtype",
                                    "decoded_shape",
                                    "decoded_data_bytes",
                                    "decoded_data_sha256",
                                )
                            }
                        )
                tensor_index[name] = metadata
                linked.append(
                    {
                        "path": destination_relative.as_posix(),
                        "mode": actual_mode,
                        "role": "npy_plane",
                    }
                )
            if source_format == "p1016-true-c-native-planes-v1":
                for source_meta in p1016_meta_paths:
                    relative = Path("planes") / source_meta.name
                    match = P1016_META_RE.fullmatch(source_meta.name)
                    if match is None:
                        raise PackValidationError(
                            f"unsupported p1016 metadata name: {source_meta.name}"
                        )
                    selected_document = p1016_documents[int(match.group(1))]
                    (output / relative).write_text(
                        json.dumps(selected_document, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    linked.append(
                        {
                            "path": relative.as_posix(),
                            "mode": "selected-manifest",
                            "role": "source_layer_meta",
                        }
                    )
            if config_source.is_file():
                config = json.loads(config_source.read_text(encoding="utf-8"))
                if not isinstance(config, dict):
                    raise PackValidationError("source config.json must contain an object")
            else:
                config = {
                    "_name_or_path": model_id,
                    "model_type": "deepseek_v4",
                }
        else:
            for source in planes:
                descriptor = _genesis_plane_descriptor(source, layer=layer)
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
                        "role": "genesis_raw_plane",
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
            shutil.copy2(genesis_receipt, output / provenance_relative)
            config = {
                "_name_or_path": model_id,
                "model_type": "deepseek_v4",
                "bs_pack_scope": f"layer_{layer:03d}",
            }

        if serving_config is not None:
            config = dict(serving_config)
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
        if repair is not None:
            config["quantization_config"].update(
                {
                    "repair_manifest": REPAIR_MANIFEST_PATH.as_posix(),
                    "repair_state": REPAIR_STATE_PATH.as_posix(),
                    "repair_format": repair.checkpoint_format,
                    "repair_update": repair.update,
                }
            )
        if serving_root is None:
            (output / "config.json").write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            assert serving_payloads is not None
            linked.extend(_materialize_serving_metadata(serving_payloads, output, config))
            linked.extend(
                _materialize_base_weights(serving_root, output, link_mode=link_mode)
            )

        if source_format == "p1016-true-c-native-planes-v1":
            layers = p1016_layers
        else:
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

        if repair is not None:
            repair_summary = write_repair_payload(output, repair, repair_rows)
            linked.extend(
                [
                    {
                        "path": REPAIR_MANIFEST_PATH.as_posix(),
                        "mode": "generated",
                        "role": "repair_manifest",
                    },
                    {
                        "path": REPAIR_STATE_PATH.as_posix(),
                        "mode": "generated",
                        "role": "repair_state",
                    },
                ]
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
        if source_format == "genesis-materialized-layer-v1":
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
        if source_format == "p1016-true-c-native-planes-v1":
            dense_base_bytes = sum(
                int(row["bytes"])
                for row in file_entries
                if row.get("role") == BASE_WEIGHTS_SHARD_ROLE
            )
            selected_layers: dict[str, dict[str, Any]] = {}
            for selected_layer, document in sorted(p1016_documents.items()):
                selected_layers[str(selected_layer)] = {}
                for projection, suffix in (("fused13", "13"), ("down", "2")):
                    selected_layers[str(selected_layer)][projection] = {
                        "tiers": document[f"tier{suffix}"],
                        "slots": document[f"slot{suffix}"],
                        "families": document[f"family{suffix}"],
                        "payloads": document["payloads"][projection],
                    }
            manifest["selected_payloads"] = {
                "schema": "bs-pack-selected-payloads-v1",
                "producer_stage": "smash export:v4-row-packed-selected-wire-v1",
                "runtime_floor_bytes": runtime_floor_bytes,
                "dense_base_bytes": dense_base_bytes,
                "additional_resident_role_bytes": {
                    "repair_state": sum(
                        int(row["bytes"])
                        for row in file_entries
                        if row.get("role") == "repair_state"
                    )
                },
                "layers": selected_layers,
            }
        if serving_root is not None:
            manifest["provenance"]["serving_model_root"] = str(serving_root)
        if repair_summary is not None:
            manifest["repair"] = repair_summary
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
    if manifest.get("source_format") == "p1016-true-c-native-planes-v1":
        for layer in layers:
            relative = Path("planes") / f"layer_{layer:03d}.meta.json"
            if file_roles.get(relative.as_posix()) != "source_layer_meta":
                raise PackValidationError(
                    f"p1016 layer meta is not manifest-bound: {relative}"
                )
            try:
                actual = json.loads((root / relative).read_text(encoding="utf-8"))
            except Exception as exc:
                raise PackValidationError(
                    f"cannot read p1016 layer meta {relative}: {exc}"
                ) from exc
            if (
                not isinstance(actual, dict)
                or actual.get("format") != "p1016-true-c-native-planes-v1"
                or actual.get("layer") != layer
                or actual.get("E") != 256
            ):
                raise PackValidationError(f"p1016 layer meta identity drift: {relative}")
        return
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
        _verify_packed_index_metadata(name, recorded)
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
                {int(value) for value in np.unique(array)} - set(GENESIS_SUBTIERS)
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
    if manifest.get("source_format") == "p1016-true-c-native-planes-v1":
        tensor_layers = sorted(layer_fields)
        if tensor_layers != declared_layers:
            raise PackValidationError(
                f"p1016 tensor layers mismatch: manifest={declared_layers}, tensors={tensor_layers}"
            )
        return len(index), tensor_layers
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
                    for subtier in GENESIS_SUBTIERS
                    for projection in GENESIS_PROJECTIONS
                    for role in GENESIS_ROLES
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
    repair_receipt = None
    if "repair" in manifest:
        try:
            repair_receipt = verify_repair_payload(root, manifest["repair"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PackValidationError(f"repair payload verification failed: {exc}") from exc
    receipt = {
        "status": "PASS",
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_format": manifest.get("source_format"),
        "instance_id": manifest.get("instance_id"),
        "tensor_count": tensor_count,
        "layers": layers,
        "tensor_layout_sha256": manifest["tensor_layout_sha256"],
    }
    if repair_receipt is not None:
        receipt["repair"] = repair_receipt
    return receipt


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
