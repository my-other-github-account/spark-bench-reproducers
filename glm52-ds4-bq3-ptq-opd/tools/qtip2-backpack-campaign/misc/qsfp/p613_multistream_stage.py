#!/usr/bin/env python3
"""Four-stream direct-QSFP durable stage of immutable GENESIS wire package."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

ROOT = Path("$HOME/run-bundles/P613_ACTCACHE_ACCEL_PUBLIC_TASK_s5w")
SOURCE = Path("$HOME/run-bundles/GENESIS_FANIN_PUBLIC_TASK_s8/package/wire43")
DEST = ROOT / "inputs/compute-node-wire.example.invalid"
OUT = ROOT / "TRANSFER_STAGE.json"
STREAMS = 4
BUF = 16 * 1024 * 1024


def copy_worker(paths):
    started = time.perf_counter()
    copied = skipped = total = 0
    digest = hashlib.sha256()
    for raw, rel, expected in paths:
        source = Path(raw)
        destination = DEST / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size == expected:
            skipped += 1; total += expected
            continue
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with source.open("rb", buffering=0) as src, temporary.open("wb", buffering=0) as dst:
            while True:
                data = src.read(BUF)
                if not data:
                    break
                dst.write(data)
                total += len(data)
                digest.update(data[:64]); digest.update(data[-64:])
            dst.flush(); os.fsync(dst.fileno())
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(src.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(dst.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        if temporary.stat().st_size != expected:
            raise RuntimeError(f"short copy {source}: {temporary.stat().st_size} != {expected}")
        os.replace(temporary, destination)
        copied += 1
    return {
        "bytes": total,
        "copied_files": copied,
        "skipped_files": skipped,
        "seconds": time.perf_counter() - started,
        "sample_sha256": digest.hexdigest(),
    }


def atomic_json(path, obj):
    data = (json.dumps(obj, indent=2, sort_keys=True) + "\n").encode()
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    files = sorted((p for p in SOURCE.rglob("*") if p.is_file()),
                   key=lambda p: p.stat().st_size, reverse=True)
    bins = [[] for _ in range(STREAMS)]
    sizes = [0] * STREAMS
    for path in files:
        slot = min(range(STREAMS), key=sizes.__getitem__)
        size = path.stat().st_size
        rel = str(path.relative_to(SOURCE))
        bins[slot].append((str(path), rel, size)); sizes[slot] += size
    DEST.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=STREAMS) as pool:
        results = list(pool.map(copy_worker, bins))
    wall = time.perf_counter() - started
    staged = [p for p in DEST.rglob("*") if p.is_file()]
    staged_bytes = sum(p.stat().st_size for p in staged)
    expected_bytes = sum(sizes)
    if len(staged) != len(files) or staged_bytes != expected_bytes:
        raise RuntimeError(
            f"stage verification failed files={len(staged)}/{len(files)} "
            f"bytes={staged_bytes}/{expected_bytes}"
        )
    complete = DEST / "P613_STAGE.COMPLETE"
    complete.write_text(json.dumps({
        "schema": "p613-wire43-stage-complete-v1",
        "source": str(SOURCE), "files": len(files), "bytes": expected_bytes,
        "completed_unix": time.time(),
    }, sort_keys=True) + "\n")
    value = {
        "schema": "p613-qsfp-four-stream-durable-stage-v1",
        "host": os.uname().nodename,
        "source": str(SOURCE), "destination": str(DEST),
        "streams": STREAMS, "files": len(files), "bytes": expected_bytes,
        "wall_seconds": wall,
        "bytes_per_second": expected_bytes / wall,
        "decimal_GB_per_second": expected_bytes / wall / 1e9,
        "minimum_required_decimal_GB_per_second": 5.0,
        "environment_defect": expected_bytes / wall < 5e9,
        "stream_expected_bytes": sizes, "stream_results": results,
        "size_verification_pass": True,
        "complete_marker": str(complete), "completed_unix": time.time(),
    }
    atomic_json(OUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
