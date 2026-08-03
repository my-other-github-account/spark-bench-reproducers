#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

_CACHE_KEY = re.compile(r"[0-9a-f]{64}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_cache_path(
    cache_path: str | Path,
    *,
    expected_version: str,
    expected_arch: str,
) -> dict[str, Any]:
    """Validate one vLLM FlashInfer cache at <version>/<architecture>."""
    unresolved = Path(cache_path).expanduser()
    if unresolved.is_symlink() or not unresolved.is_dir():
        raise ValueError(f"cache path is not a local directory: {unresolved}")
    path = unresolved.resolve()
    if path.name != expected_arch or path.parent.name != expected_version:
        raise ValueError(
            "cache path version/architecture mismatch: "
            f"actual={path.parent.name}/{path.name} "
            f"expected={expected_version}/{expected_arch}"
        )

    members_on_disk = sorted(path.rglob("*"))
    symlinks = [candidate.relative_to(path).as_posix() for candidate in members_on_disk if candidate.is_symlink()]
    if symlinks:
        raise ValueError(f"cache contains symlink members: {symlinks}")
    top_level = sorted(path.iterdir())
    malformed_directories = [
        candidate.relative_to(path).as_posix()
        for candidate in top_level
        if not candidate.is_dir() or not _CACHE_KEY.fullmatch(candidate.name)
    ]
    if malformed_directories:
        raise ValueError(f"cache contains malformed cache-key directories: {malformed_directories}")
    files = sorted(path.rglob("autotune_configs.json"))
    if not files:
        raise ValueError(f"cache has no autotune_configs.json files: {path}")
    unexpected = sorted(
        candidate.relative_to(path).as_posix()
        for candidate in members_on_disk
        if candidate.is_file() and candidate.name != "autotune_configs.json"
    )
    if unexpected:
        raise ValueError(f"cache contains unexpected files: {unexpected}")

    members: list[dict[str, Any]] = []
    for unresolved_file in files:
        if unresolved_file.is_symlink() or not unresolved_file.is_file():
            raise ValueError(f"cache member is not a local regular file: {unresolved_file}")
        relative = unresolved_file.relative_to(path)
        if len(relative.parts) != 2 or not _CACHE_KEY.fullmatch(relative.parts[0]):
            raise ValueError(f"cache member path is not <cache-key>/autotune_configs.json: {relative}")
        try:
            payload = json.loads(unresolved_file.read_text())
        except Exception as exc:
            raise ValueError(f"invalid cache JSON {relative}: {exc}") from exc
        metadata = payload.get("_metadata") if isinstance(payload, dict) else None
        actual_version = metadata.get("flashinfer_version") if isinstance(metadata, dict) else None
        if actual_version != expected_version:
            raise ValueError(
                f"flashinfer_version mismatch in {relative}: "
                f"actual={actual_version!r} expected={expected_version!r}"
            )
        members.append(
            {
                "path": relative.as_posix(),
                "bytes": unresolved_file.stat().st_size,
                "sha256": _sha256(unresolved_file),
            }
        )

    return {
        "schema": "banana-smasher-flashinfer-cache-capture-v1",
        "status": "VALID",
        "flashinfer_version": expected_version,
        "architecture": expected_arch,
        "file_count": len(members),
        "bytes": sum(member["bytes"] for member in members),
        "members": members,
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a version-aware vLLM FlashInfer autotune cache capture."
    )
    parser.add_argument("cache_path", type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--write-manifest", type=Path)
    args = parser.parse_args()
    result = validate_cache_path(
        args.cache_path,
        expected_version=args.version,
        expected_arch=args.arch,
    )
    if args.write_manifest is not None:
        _atomic_write(args.write_manifest, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
