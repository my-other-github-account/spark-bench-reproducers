#!/usr/bin/env python3
"""Capacity-gated parallel transfer for independent immutable objects.

Inventory format (tab separated, comments and blank lines allowed):
    <sha256>\t<bytes>\t<source-relative-path>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BUFFER_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class Item:
    sha256: str
    size: int
    relative: str


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(BUFFER_BYTES):
            h.update(chunk)
    return h.hexdigest()


def parse_inventory(path: Path) -> list[Item]:
    items: list[Item] = []
    seen: set[str] = set()
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split("\t", 2)
        if len(parts) != 3:
            raise SystemExit(f"inventory line {number}: expected three tab-separated fields")
        expected_sha, size_text, relative = parts
        if len(expected_sha) != 64 or any(ch not in "0123456789abcdef" for ch in expected_sha):
            raise SystemExit(f"inventory line {number}: invalid lowercase SHA-256")
        try:
            size = int(size_text)
        except ValueError as exc:
            raise SystemExit(f"inventory line {number}: invalid byte count") from exc
        pure = PurePosixPath(relative)
        if size < 0 or pure.is_absolute() or ".." in pure.parts or relative in ("", "."):
            raise SystemExit(f"inventory line {number}: unsafe relative path")
        if relative in seen:
            raise SystemExit(f"inventory line {number}: duplicate path {relative}")
        seen.add(relative)
        items.append(Item(expected_sha, size, relative))
    if not items:
        raise SystemExit("inventory is empty")
    return items


def fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def transfer_one(item: Item, source_root: Path, dest_root: Path) -> dict:
    source = source_root / item.relative
    destination = dest_root / item.relative
    started = time.monotonic()
    if not source.is_file():
        raise RuntimeError(f"missing source {item.relative}")
    source_size = source.stat().st_size
    if source_size != item.size:
        raise RuntimeError(
            f"source size mismatch {item.relative}: {source_size} != {item.size}"
        )
    source_sha = digest(source)
    if source_sha != item.sha256:
        raise RuntimeError(f"source hash mismatch {item.relative}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == item.size:
        if digest(destination) == item.sha256:
            elapsed = time.monotonic() - started
            return {
                "path": item.relative,
                "bytes": item.size,
                "sha256": item.sha256,
                "elapsed_seconds": elapsed,
                "bytes_per_second": 0.0,
                "action": "REUSED_VERIFIED",
            }

    temporary = destination.with_name(f".{destination.name}.part.{os.getpid()}")
    try:
        with source.open("rb") as src, temporary.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=BUFFER_BYTES)
            dst.flush()
            os.fsync(dst.fileno())
        if temporary.stat().st_size != item.size or digest(temporary) != item.sha256:
            raise RuntimeError(f"staged readback mismatch {item.relative}")
        os.replace(temporary, destination)
        fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)

    elapsed = time.monotonic() - started
    return {
        "path": item.relative,
        "bytes": item.size,
        "sha256": item.sha256,
        "elapsed_seconds": elapsed,
        "bytes_per_second": item.size / elapsed if elapsed else None,
        "action": "COPIED_VERIFIED_ATOMIC",
    }


def atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dest-root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--min-free-bytes", type=int, default=16 * 1024**3)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    if not 1 <= args.jobs <= 64:
        raise SystemExit("--jobs must be in [1, 64]")
    if args.min_free_bytes < 0:
        raise SystemExit("--min-free-bytes must be nonnegative")
    source_root = args.source_root.resolve()
    dest_root = args.dest_root.resolve()
    if not source_root.is_dir():
        raise SystemExit("source root is not a directory")
    dest_root.mkdir(parents=True, exist_ok=True)

    items = parse_inventory(args.inventory)
    total_bytes = sum(item.size for item in items)
    free_before = shutil.disk_usage(dest_root).free
    required = total_bytes + args.min_free_bytes
    if free_before < required:
        raise SystemExit(
            f"capacity gate failed: free={free_before} required={required} "
            f"(payload={total_bytes} reserve={args.min_free_bytes})"
        )

    started_wall = time.time()
    started = time.monotonic()
    failures: list[str] = []
    rows: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        future_to_item = {
            pool.submit(transfer_one, item, source_root, dest_root): item for item in items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                rows.append(future.result())
            except Exception as exc:  # every failure is reported in the receipt
                failures.append(f"{item.relative}: {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - started
    rows.sort(key=lambda row: row["path"])
    copied_bytes = sum(row["bytes"] for row in rows if row["action"].startswith("COPIED"))
    document = {
        "schema": "immutable-parallel-transfer-v1",
        "status": "PASS" if not failures and len(rows) == len(items) else "FAIL",
        "started_unix": started_wall,
        "elapsed_seconds": elapsed,
        "jobs": args.jobs,
        "inventory_objects": len(items),
        "inventory_bytes": total_bytes,
        "copied_bytes": copied_bytes,
        "aggregate_copied_bytes_per_second": copied_bytes / elapsed if elapsed else None,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(dest_root).free,
        "min_free_bytes": args.min_free_bytes,
        "objects": rows,
        "failures": failures,
    }
    atomic_json(args.receipt, document)
    if failures or len(rows) != len(items):
        raise SystemExit("TRANSFER_FAIL receipt written")
    print(
        f"TRANSFER_PASS objects={len(rows)} bytes={total_bytes} "
        f"elapsed={elapsed:.6f}s copied_Bps={document['aggregate_copied_bytes_per_second']:.3f}"
    )


if __name__ == "__main__":
    main()
