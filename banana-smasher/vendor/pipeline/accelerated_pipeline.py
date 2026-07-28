#!/usr/bin/env python3
"""Package-local pipeline accelerations with deterministic receipt surfaces."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def materialize(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and sha256(destination) == sha256(source):
            return {"status": "SKIP_EXISTING", "method": "sealed-existing", "sha256": sha256(destination)}
        raise FileExistsError("destination collision: " + str(destination))
    temporary = destination.with_name("." + destination.name + ".partial")
    try:
        try:
            os.link(source, temporary)
            method = "hardlink"
        except OSError:
            shutil.copy2(source, temporary)
            method = "copy"
        if sha256(temporary) != sha256(source):
            raise RuntimeError("materialization readback mismatch")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {"status": "PASS", "method": method, "bytes": destination.stat().st_size, "sha256": sha256(destination)}


def bulk_move(rows: Sequence[tuple[Path, Path]], streams: int = 8) -> list[dict[str, Any]]:
    if streams < 8:
        raise ValueError("bulk mover requires at least 8 streams")
    with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
        futures = [pool.submit(materialize, source, destination) for source, destination in rows]
        return [future.result() for future in futures]


def shard_ranges(count: int, shards: int = 4) -> list[tuple[int, int]]:
    if count < 0 or shards != 4:
        raise ValueError("GPU fleet contract requires nonnegative count and exactly 4 shards")
    return [((count * index) // shards, (count * (index + 1)) // shards) for index in range(shards)]


def stream_codebooks(
    codebooks: Iterable[str],
    produce: Callable[[str], Any],
    rebuild: Callable[[str], Any],
    stream: Callable[[str], Any],
) -> list[dict[str, Any]]:
    receipts = []
    seen = set()
    for codebook in codebooks:
        if codebook in seen:
            raise ValueError("duplicate codebook: " + codebook)
        seen.add(codebook)
        receipts.append({
            "codebook": codebook,
            "produce": produce(codebook),
            "rebuild": rebuild(codebook),
            "stream": stream(codebook),
            "order": ["produce", "rebuild", "stream"],
        })
    return receipts


def speculative_publish(
    expected_seal: str,
    current_seal: Callable[[], str],
    warm: Callable[[], Any],
    publish: Callable[[Any], Any],
) -> dict[str, Any]:
    warmed = warm()
    observed = current_seal()
    if observed != expected_seal:
        return {"status": "REVOKED", "expected_seal": expected_seal, "observed_seal": observed, "published": False}
    result = publish(warmed)
    return {"status": "PASS", "expected_seal": expected_seal, "observed_seal": observed, "published": True, "result": result}


def replay_gate(reference: Sequence[float], accelerated: Sequence[float], tolerance: float = 1e-12) -> dict[str, Any]:
    if len(reference) != len(accelerated) or not reference:
        raise ValueError("replay vectors must be non-empty and equal length")
    maximum = max(abs(left - right) for left, right in zip(reference, accelerated))
    return {"status": "PASS" if maximum <= tolerance else "FAIL", "max_abs": maximum, "tolerance": tolerance}
