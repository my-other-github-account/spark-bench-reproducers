#!/usr/bin/env python3
"""Exact ARM4 plane conversion, loading, caching, and prefetch primitives."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

_META_KEY = "arm4_meta_json"
_PLANE_KEYS = ("codes13", "sc13", "codes2", "sc2", "cb13", "cb2")


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_plane(payload: Mapping[str, Any]) -> None:
    missing = sorted(set(_PLANE_KEYS) - set(payload))
    if missing:
        raise ValueError(f"missing ARM4 plane tensors: {missing}")
    for key in _PLANE_KEYS:
        if not isinstance(payload[key], torch.Tensor):
            raise TypeError(f"{key} is not a tensor")
    if not isinstance(payload.get("meta"), dict):
        raise TypeError("ARM4 plane meta must be a dict")


def convert_pt_plane(source: Path | str, target: Path | str) -> dict[str, Any]:
    """Convert one legacy torch ARM4 plane to hash-bound safetensors exactly."""
    source = Path(source).resolve()
    target = Path(target).resolve()
    started = time.time()
    payload = torch.load(source, map_location="cpu", mmap=True, weights_only=True)
    _validate_plane(payload)
    tensors = {key: payload[key] for key in _PLANE_KEYS}
    metadata = {_META_KEY: json.dumps(payload["meta"], sort_keys=True, separators=(",", ":"))}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    save_file(tensors, temporary, metadata=metadata)
    os.replace(temporary, target)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    return {
        "schema": "arm4-plane-conversion-v1",
        "status": "PASS",
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "target": str(target),
        "target_bytes": target.stat().st_size,
        "target_sha256": sha256_file(target),
        "tensor_bytes": tensor_bytes,
        "meta": payload["meta"],
        "elapsed_seconds": time.time() - started,
    }


class PlaneStore:
    """Bounded exact plane handle cache with optional asynchronous prefetch."""

    def __init__(
        self,
        mode: str = "torch-mmap",
        device: str = "cpu",
        cache_size: int = 2,
        prefetch_workers: int = 1,
    ) -> None:
        if mode not in {
            "torch-eager", "torch-mmap", "safetensors", "fastsafetensors"
        }:
            raise ValueError(f"unsupported plane loading mode: {mode}")
        if cache_size < 1 or prefetch_workers < 1:
            raise ValueError("cache_size and prefetch_workers must be positive")
        self.mode = mode
        self.device = "cuda:0" if device == "cuda" else device
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[int, str], dict[str, Any]] = OrderedDict()
        self._resources: dict[tuple[int, str], Any] = {}
        self._futures: dict[tuple[int, str], Future] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=prefetch_workers, thread_name_prefix="arm4-prefetch"
        )
        self._lock = threading.RLock()
        self._closed = False

    @staticmethod
    def _key(layer: int, path: Path | str) -> tuple[int, str]:
        return int(layer), str(Path(path).resolve())

    def _open(self, path: str) -> tuple[dict[str, Any], Any]:
        if self.mode == "torch-eager":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            _validate_plane(payload)
            return payload, None
        if self.mode == "torch-mmap":
            payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
            _validate_plane(payload)
            return payload, None
        if self.mode == "safetensors":
            handle = safe_open(path, framework="pt", device=self.device)
            metadata = handle.metadata() or {}
            payload = {key: handle.get_tensor(key) for key in handle.keys()}
            payload["meta"] = json.loads(metadata[_META_KEY])
            _validate_plane(payload)
            return payload, handle

        from fastsafetensors import fastsafe_open

        handle = fastsafe_open(path, framework="pt", device=self.device, nogds=True)
        metadata_by_path = handle.metadata()
        if len(metadata_by_path) != 1:
            handle.__exit__(None, None, None)
            raise ValueError(f"expected one fastsafetensors file, got {len(metadata_by_path)}")
        metadata = next(iter(metadata_by_path.values())) or {}
        payload = {key: handle.get_tensor(key) for key in handle.keys()}
        payload["meta"] = json.loads(metadata[_META_KEY])
        _validate_plane(payload)
        return payload, handle

    @staticmethod
    def _close_resource(resource: Any) -> None:
        if resource is None:
            return
        exit_method = getattr(resource, "__exit__", None)
        if exit_method is not None:
            exit_method(None, None, None)
        else:
            close_method = getattr(resource, "close", None)
            if close_method is not None:
                close_method()

    def _load_and_cache(self, key: tuple[int, str]) -> dict[str, Any]:
        payload, resource = self._open(key[1])
        evicted_resource = None
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self._close_resource(resource)
                return existing
            self._cache[key] = payload
            self._resources[key] = resource
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                evicted_key, _ = self._cache.popitem(last=False)
                evicted_resource = self._resources.pop(evicted_key, None)
            self._futures.pop(key, None)
        self._close_resource(evicted_resource)
        return payload

    def load(self, layer: int, path: Path | str) -> dict[str, Any]:
        key = self._key(layer, path)
        with self._lock:
            if self._closed:
                raise RuntimeError("PlaneStore is closed")
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached
            future = self._futures.get(key)
        if future is not None:
            return future.result()
        return self._load_and_cache(key)

    def prefetch(self, layer: int, path: Path | str) -> Future:
        key = self._key(layer, path)
        with self._lock:
            if self._closed:
                raise RuntimeError("PlaneStore is closed")
            if key in self._cache:
                ready: Future = Future()
                ready.set_result(self._cache[key])
                return ready
            existing = self._futures.get(key)
            if existing is not None:
                return existing
            future = self._executor.submit(self._load_and_cache, key)
            self._futures[key] = future
            return future

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._lock:
            resources = list(self._resources.values())
            self._resources.clear()
            self._cache.clear()
            self._futures.clear()
        for resource in resources:
            self._close_resource(resource)

    def __enter__(self) -> "PlaneStore":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def build_wire_index(
    directory: Path | str,
    output: Path | str,
    expected_layers: list[int] | range = range(43),
) -> dict[str, Any]:
    directory = Path(directory).resolve()
    output = Path(output).resolve()
    by_layer: dict[int, Path] = {}
    pattern = re.compile(r"vq3u_layer_(\d{3})\.safetensors$")
    for path in directory.glob("vq3u_layer_*.safetensors"):
        match = pattern.match(path.name)
        if match:
            by_layer[int(match.group(1))] = path.resolve()
    expected = list(expected_layers)
    missing = [layer for layer in expected if layer not in by_layer]
    if missing:
        raise ValueError(f"wire index missing layers: {missing}")
    rows = [
        {
            "layer": layer,
            "path": str(by_layer[layer]),
            "bytes": by_layer[layer].stat().st_size,
            "sha256": sha256_file(by_layer[layer]),
        }
        for layer in expected
    ]
    result = {
        "schema": "arm4-wire-safetensors-index-v1",
        "status": "PASS",
        "directory": str(directory),
        "layers": rows,
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("target", type=Path)
    index_parser = subparsers.add_parser("index")
    index_parser.add_argument("directory", type=Path)
    index_parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "convert":
        result = convert_pt_plane(args.source, args.target)
    else:
        result = build_wire_index(args.directory, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
