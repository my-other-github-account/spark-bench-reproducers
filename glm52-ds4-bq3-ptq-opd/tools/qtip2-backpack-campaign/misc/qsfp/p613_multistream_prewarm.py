#!/usr/bin/env python3
"""Four-stream direct-QSFP read/prefetch of immutable BANANA_SMASHER wire package."""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path("$HOME/run-bundles/P613_ACTCACHE_ACCEL_PUBLIC_TASK_s5w")
SOURCE = Path("$HOME/run-bundles/BANANA_SMASHER_FANIN_PUBLIC_TASK_s8/package/wire43")
OUT = ROOT / "TRANSFER_PROFILE.json"
STREAMS = 4
CHUNK = 16 * 1024 * 1024


def worker(paths):
    started = time.perf_counter()
    total = 0
    digest = hashlib.sha256()
    for raw in paths:
        path = Path(raw)
        with path.open("rb", buffering=0) as handle:
            while True:
                data = handle.read(CHUNK)
                if not data:
                    break
                total += len(data)
                # Sparse sample protects against a read-to-nowhere shortcut without
                # spending CPU hashing every immutable wire byte.
                digest.update(data[:64])
                digest.update(data[-64:])
    seconds = time.perf_counter() - started
    return {"bytes": total, "seconds": seconds, "sample_sha256": digest.hexdigest()}


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
        bins[slot].append(str(path)); sizes[slot] += size
    total = sum(sizes)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=STREAMS) as pool:
        results = list(pool.map(worker, bins))
    wall = time.perf_counter() - started
    throughput = total / wall
    value = {
        "schema": "p613-qsfp-four-stream-prefetch-v1",
        "host": os.uname().nodename,
        "source": str(SOURCE),
        "streams": STREAMS,
        "files": len(files),
        "bytes": total,
        "wall_seconds": wall,
        "bytes_per_second": throughput,
        "decimal_GB_per_second": throughput / 1e9,
        "minimum_required_decimal_GB_per_second": 5.0,
        "environment_defect": throughput < 5e9,
        "stream_expected_bytes": sizes,
        "stream_results": results,
        "purpose": "direct QSFP pull into node-work kernel page cache before empirical benchmark",
        "completed_unix": time.time(),
    }
    atomic_json(OUT, value)
    print(json.dumps(value, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
