from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from safetensors import safe_open

from .contract import (
    MANIFEST_NAME,
    PackValidationError,
    _canonical_json_bytes,
    _file_entry,
    _sha256_file,
    _write_bytes_durable,
    load_manifest,
    verify_pack,
)

CONTAINER_NAME = "bs-pack.safetensors"
_DTYPE_TO_SAFE = {
    np.dtype("bool"): "BOOL",
    np.dtype("int8"): "I8",
    np.dtype("uint8"): "U8",
    np.dtype("int16"): "I16",
    np.dtype("uint16"): "U16",
    np.dtype("int32"): "I32",
    np.dtype("uint32"): "U32",
    np.dtype("int64"): "I64",
    np.dtype("uint64"): "U64",
    np.dtype("float16"): "F16",
    np.dtype("float32"): "F32",
    np.dtype("float64"): "F64",
}
_SAFE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_SAFE.items()}


def _sha256_range(
    path: Path, offset: int, length: int, chunk_bytes: int = 8 << 20
) -> str:
    digest = hashlib.sha256()
    remaining = length
    with path.open("rb") as stream:
        stream.seek(offset)
        while remaining:
            chunk = stream.read(min(remaining, chunk_bytes))
            if not chunk:
                raise PackValidationError(
                    f"unexpected EOF hashing {path} at {length - remaining}/{length}"
                )
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _copy_range(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    offset: int,
    length: int,
    chunk_bytes: int = 8 << 20,
) -> None:
    source.seek(offset)
    remaining = length
    while remaining:
        chunk = source.read(min(remaining, chunk_bytes))
        if not chunk:
            raise PackValidationError(
                f"unexpected EOF copying tensor payload at {length - remaining}/{length}"
            )
        destination.write(chunk)
        remaining -= len(chunk)


def _read_safetensors_header(path: Path) -> tuple[dict[str, Any], int]:
    try:
        with path.open("rb") as stream:
            raw_length = stream.read(8)
            if len(raw_length) != 8:
                raise PackValidationError(f"truncated safetensors prefix: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            if header_length <= 0 or header_length > 128 * 1024 * 1024:
                raise PackValidationError(
                    f"invalid safetensors header length {header_length}: {path}"
                )
            header_bytes = stream.read(header_length)
            if len(header_bytes) != header_length:
                raise PackValidationError(f"truncated safetensors header: {path}")
        header = json.loads(header_bytes.decode("utf-8"))
    except PackValidationError:
        raise
    except Exception as exc:
        raise PackValidationError(f"invalid safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise PackValidationError(f"safetensors header must be an object: {path}")
    return header, 8 + header_length


def _source_payload(
    root: Path, metadata: dict[str, Any]
) -> tuple[Path, int, int, np.dtype[Any], list[int]]:
    storage = metadata.get("storage", {"kind": "npy", "path": metadata.get("path")})
    if not isinstance(storage, dict) or storage.get("kind") not in {"npy", "raw"}:
        raise PackValidationError("repack input tensors must use npy or raw storage")
    path = root / str(storage.get("path"))
    if not path.is_file() or path.is_symlink():
        raise PackValidationError(f"invalid repack source plane: {path}")
    if storage["kind"] == "npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if not array.flags.c_contiguous:
            raise PackValidationError(
                f"only C-contiguous arrays can be repacked: {path}"
            )
        return (
            path,
            int(array.offset),
            int(array.nbytes),
            array.dtype,
            list(array.shape),
        )
    try:
        dtype = np.dtype(metadata.get("dtype"))
        shape = metadata.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(value, int) and value >= 0 for value in shape
        ):
            raise ValueError(f"invalid shape {shape!r}")
    except Exception as exc:
        raise PackValidationError(
            f"invalid raw source metadata for {path}: {exc}"
        ) from exc
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if path.stat().st_size != nbytes:
        raise PackValidationError(
            f"raw source byte count mismatch for {path}: {path.stat().st_size} != {nbytes}"
        )
    return path, 0, nbytes, dtype, shape


def _write_container(
    root: Path,
    destination: Path,
    tensors: Iterable[tuple[str, dict[str, Any]]],
    *,
    tensor_layout_sha256: str,
) -> tuple[dict[str, dict[str, Any]], int, int]:
    ordered = list(tensors)
    if not ordered:
        raise PackValidationError("no tensors selected for repack")
    entries: dict[str, dict[str, Any]] = {}
    sources: list[tuple[Path, int, int]] = []
    cursor = 0
    for name, metadata in ordered:
        source, source_offset, nbytes, dtype, shape = _source_payload(root, metadata)
        safe_dtype = _DTYPE_TO_SAFE.get(dtype)
        if safe_dtype is None:
            raise PackValidationError(
                f"unsupported safetensors dtype {dtype} for {name}"
            )
        entries[name] = {
            "dtype": safe_dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + nbytes],
        }
        sources.append((source, source_offset, nbytes))
        cursor += nbytes
    header: dict[str, Any] = dict(entries)
    header["__metadata__"] = {
        "bs_pack_schema": "1",
        "tensor_layout_sha256": tensor_layout_sha256,
        "byte_order": "little",
    }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    padding = (-len(encoded)) % 8
    padded = encoded + b" " * padding
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(struct.pack("<Q", len(padded)))
            output.write(padded)
            for source, offset, length in sources:
                with source.open("rb") as input_stream:
                    _copy_range(
                        input_stream,
                        output,
                        offset=offset,
                        length=length,
                    )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return entries, cursor, 8 + len(padded)


def verify_tensor_storage(
    root: Path, name: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    storage = metadata.get("storage")
    if not isinstance(storage, dict) or storage.get("kind") != "safetensors":
        raise PackValidationError(f"{name} is not safetensors-backed")
    container = root / str(storage.get("path"))
    header, data_start = _read_safetensors_header(container)
    entry = header.get(name)
    if not isinstance(entry, dict):
        raise PackValidationError(f"container is missing tensor {name}")
    dtype = _SAFE_TO_DTYPE.get(entry.get("dtype"))
    if dtype is None:
        raise PackValidationError(
            f"container tensor {name} has unsupported dtype {entry.get('dtype')!r}"
        )
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(shape, list)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        raise PackValidationError(f"invalid safetensors entry for {name}: {entry!r}")
    if storage.get("data_offsets") != offsets:
        raise PackValidationError(
            f"manifest/container data_offsets mismatch for {name}: "
            f"manifest={storage.get('data_offsets')}, container={offsets}"
        )
    data_bytes = offsets[1] - offsets[0]
    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if data_bytes != expected_bytes:
        raise PackValidationError(
            f"container byte count is inconsistent for {name}: {data_bytes} != {expected_bytes}"
        )
    return {
        "dtype": dtype.str,
        "shape": shape,
        "data_bytes": data_bytes,
        "data_sha256": _sha256_range(container, data_start + offsets[0], data_bytes),
    }


def load_tensor_numpy(root: Path, name: str, metadata: dict[str, Any]) -> np.ndarray:
    """Lazily open one named tensor; safe_open keeps the mmap lifetime scoped."""
    storage = metadata.get("storage")
    if not isinstance(storage, dict) or storage.get("kind") != "safetensors":
        raise PackValidationError(f"{name} is not safetensors-backed")
    container = root / str(storage.get("path"))
    with safe_open(container, framework="np") as handle:
        return handle.get_tensor(name)


def _replace_file_row(
    manifest: dict[str, Any], root: Path, relative: Path, role: str
) -> None:
    manifest["files"] = [
        row for row in manifest["files"] if row.get("path") != relative.as_posix()
    ]
    manifest["files"].append(_file_entry(root, relative, role))
    manifest["files"].sort(key=lambda row: row["path"])


def _prune_empty_directories(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories:
        try:
            directory.rmdir()
        except OSError:
            pass


def repack_to_safetensors(
    root: str | Path,
    *,
    output_name: str = CONTAINER_NAME,
    layers: set[int] | None = None,
    drop_planes: bool = False,
) -> dict[str, Any]:
    """Stream npy/raw payload ranges into one mmap-safe safetensors container."""
    root = Path(root).resolve()
    verify_pack(root)
    manifest = load_manifest(root)
    destination_relative = Path(output_name)
    if destination_relative.is_absolute() or ".." in destination_relative.parts:
        raise PackValidationError(f"unsafe container path: {destination_relative}")
    destination = root / destination_relative
    if destination.exists():
        raise FileExistsError(f"container already exists: {destination}")
    selected: list[tuple[str, dict[str, Any]]] = []
    for name, metadata in sorted(manifest["tensor_index"].items()):
        layer = int(name.split(".")[1])
        if layers is None or layer in layers:
            selected.append((name, metadata))
    entries, payload_bytes, data_start = _write_container(
        root,
        destination,
        selected,
        tensor_layout_sha256=manifest["tensor_layout_sha256"],
    )
    for name, metadata in selected:
        entry = entries[name]
        actual = verify_tensor_storage(
            root,
            name,
            {
                **metadata,
                "storage": {
                    "kind": "safetensors",
                    "path": destination_relative.as_posix(),
                    "data_offsets": entry["data_offsets"],
                },
            },
        )
        if actual["data_sha256"] != metadata["data_sha256"]:
            destination.unlink(missing_ok=True)
            raise PackValidationError(
                f"byte-exact repack verification failed for {name}: "
                f"source={metadata['data_sha256']}, container={actual['data_sha256']}"
            )

    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"]["tensor_container"] = destination_relative.as_posix()
    _write_bytes_durable(config_path, _canonical_json_bytes(config))
    _replace_file_row(manifest, root, Path("config.json"), "model_config")
    _replace_file_row(manifest, root, destination_relative, "safetensors_container")

    for name, metadata in selected:
        source_path = metadata.get("storage", {}).get("path", metadata.get("path"))
        entry = entries[name]
        metadata["storage"] = {
            "kind": "safetensors",
            "path": destination_relative.as_posix(),
            "data_offsets": entry["data_offsets"],
            "mmap": True,
        }
        metadata["source_plane_path"] = source_path
        metadata.pop("path", None)
        if drop_planes:
            relative = Path(str(source_path))
            (root / relative).unlink()
            manifest["files"] = [
                row
                for row in manifest["files"]
                if row.get("path") != relative.as_posix()
            ]
    manifest["container"] = {
        "format": "safetensors",
        "path": destination_relative.as_posix(),
        "tensor_count": len(selected),
        "payload_bytes": payload_bytes,
        "data_start": data_start,
        "mmap": True,
        "byte_exact": True,
        "selected_layers": sorted({int(name.split(".")[1]) for name, _ in selected}),
        "sha256": _sha256_file(destination),
    }
    manifest["files"].sort(key=lambda row: row["path"])
    _write_bytes_durable(root / MANIFEST_NAME, _canonical_json_bytes(manifest))
    if drop_planes:
        _prune_empty_directories(root / "planes")
    verify_pack(root)
    roundtrip = verify_repack_roundtrip(root)
    return {
        **roundtrip,
        "container": destination_relative.as_posix(),
        "container_sha256": manifest["container"]["sha256"],
        "payload_bytes": payload_bytes,
        "drop_planes": drop_planes,
    }


def verify_repack_roundtrip(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = load_manifest(root)
    byte_exact = 0
    containers: set[str] = set()
    for name, metadata in sorted(manifest["tensor_index"].items()):
        storage = metadata.get("storage")
        if not isinstance(storage, dict) or storage.get("kind") != "safetensors":
            continue
        containers.add(str(storage.get("path")))
        actual = verify_tensor_storage(root, name, metadata)
        if actual["data_sha256"] != metadata.get("data_sha256"):
            raise PackValidationError(f"round-trip payload mismatch for {name}")
        if actual["dtype"] != metadata.get("dtype") or actual["shape"] != metadata.get(
            "shape"
        ):
            raise PackValidationError(f"round-trip metadata mismatch for {name}")
        byte_exact += 1
    if byte_exact == 0:
        raise PackValidationError("pack has no safetensors-backed tensors")
    return {
        "status": "PASS",
        "byte_exact_tensors": byte_exact,
        "containers": sorted(containers),
        "mmap_container": True,
        "comparison": "tensor name + dtype + shape + raw C-order payload SHA-256",
    }
