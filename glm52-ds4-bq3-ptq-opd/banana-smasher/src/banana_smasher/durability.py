from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np


class DurabilityError(ValueError):
    """Raised when a durable artifact violates its declared identity."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_directory(path: str | Path) -> None:
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def atomic_json(path: str | Path, value: Any) -> Path:
    return atomic_bytes(path, canonical_json_bytes(value))


def atomic_npz(path: str | Path, arrays: Mapping[str, np.ndarray[Any, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **cast(dict[str, Any], dict(arrays)))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        temporary = None
        fsync_directory(destination.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DurabilityError(f"{label}_INVALID: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise DurabilityError(f"{label}_NOT_OBJECT: {source}")
    return value


def safe_relative_path(root: str | Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise DurabilityError(f"{label}_PATH_UNSAFE: {relative!r}")
    directory = Path(root).resolve()
    path = directory / candidate
    current = directory
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise DurabilityError(f"{label}_SYMLINK_FORBIDDEN: {relative}")
    try:
        path.resolve().relative_to(directory)
    except ValueError as exc:
        raise DurabilityError(f"{label}_PATH_ESCAPE: {relative!r}") from exc
    if not path.is_file():
        raise DurabilityError(f"{label}_MISSING: {relative}")
    return path


def file_identity(path: str | Path, *, root: str | Path | None = None) -> dict[str, Any]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise DurabilityError(f"FILE_INVALID: {source}")
    relative = source.name if root is None else source.relative_to(root).as_posix()
    return {
        "path": relative,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def tree_identity(
    root: str | Path,
    *,
    excluded_names: Sequence[str] = (),
    excluded_suffixes: Sequence[str] = (".tmp",),
) -> dict[str, Any]:
    directory = Path(root)
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise DurabilityError(f"TREE_SYMLINK_FORBIDDEN: {path}")
        if not path.is_file() or path.name in excluded_names:
            continue
        relative = path.relative_to(directory).as_posix()
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        if any(relative.endswith(suffix) for suffix in excluded_suffixes):
            continue
        rows.append(file_identity(path, root=directory))
    return {
        "sha256": canonical_sha256(rows),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }


@contextlib.contextmanager
def output_lock(root: str | Path) -> Iterator[None]:
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".banana-smasher.lock"
    with lock_path.open("a+b") as stream:
        try:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DurabilityError(f"OUTPUT_LOCKED: {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
