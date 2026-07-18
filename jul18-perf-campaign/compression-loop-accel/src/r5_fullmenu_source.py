#!/usr/bin/env python3
"""Layer a sparse assignment-delta pack with sealed R5 codebooks at runtime.

ACCEL VARIANT (<source-task>): identical numerics to the sealed R5 source, plus
  1) fill_layer(): bulk tier-batched expert materialization (one codebook
     upload + fp16-wire cast per (tier, projection) per layer; batched row
     gather; vectorized dequant). Every op is elementwise or gather — no
     reductions — so each output element sees the same scalar inputs and op
     order as the sealed per-expert path: bitwise identical by construction.
  2) local next-layer readahead: while layer L forwards, a background thread
     sequentially reads layer L+1 component files (page-cache warm) and
     opens them with torch.load(mmap=True). Pure prefetch; no numerical
     effect. Only active for local packs (remote streaming keeps the sealed
     rsync prefetch behavior).

Required environment:
  FULLMENU_BASE_SELECTED=/path/to/base/selected
  FULLMENU_BASE_MANIFEST=/path/to/base_manifest.json
  FULLMENU_DELTA_DIR=/path/to/delta_mission
  R5_RAIL_CHECKPOINT=/path/to/sealed/best.pt
  R5_RAIL_CHECKPOINT_SHA256=<sealed sha256>
  FULLMENU_TEACHER_ROOT=~/missions/DS4_TEACHER  (optional)
"""

from __future__ import annotations

import gc
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any

import torch

TEACHER = Path(os.environ.get("FULLMENU_TEACHER_ROOT", "~/missions/DS4_TEACHER")).expanduser()
if str(TEACHER) not in sys.path:
    sys.path.insert(0, str(TEACHER))
import t8192_ds4_build_v3 as v3  # noqa: E402

DIMS = (256, 4096, 4096, 4096, 2048)
STATIC_TIERS = {"vqa", "fp4"}
VQ_TIERS = {
    "d4_k1024", "d4_k2048", "d4_k4096",
    "d8_k256", "d8_k512", "d8_k1024", "d8_k2048", "d8_k4096",
}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tier(entry: object, projection: str) -> str:
    if isinstance(entry, dict):
        return str(entry["fused13" if projection == "13" else "down"])
    return str(entry)


def _ids_list(value: object) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return [int(item) for item in value]


def resolve_row(
    *,
    target_tier: str,
    base_tier: str,
    expert_id: int,
    projection: str,
    base_ids: dict[str, dict[str, object]],
    delta_ids: dict[str, dict[str, object]],
) -> tuple[str, int]:
    """Return (base|delta, row index), failing closed on a missing row."""
    changed = target_tier != base_tier
    source_name = "delta" if changed else "base"
    source = delta_ids if changed else base_ids
    ids = _ids_list(source.get(target_tier, {}).get(projection, []))
    message = f"missing {source_name} row: tier={target_tier} expert={expert_id} proj={projection}"
    assert ids.count(expert_id) == 1, message
    return source_name, ids.index(expert_id)


def component(tier_name: str) -> str:
    return "static" if tier_name in STATIC_TIERS else tier_name


def component_path(root: Path, tier_name: str, layer: int) -> Path:
    return root / component(tier_name) / f"layer_{layer:03d}.pt"


class FullMenuDeltaSource:
    def __init__(self, target_manifest_path: str | os.PathLike[str]):
        self.target_path = Path(target_manifest_path).expanduser().resolve()
        self.base_manifest_path = Path(os.environ["FULLMENU_BASE_MANIFEST"]).expanduser().resolve()
        self.delta_dir = Path(os.environ["FULLMENU_DELTA_DIR"]).expanduser().resolve()
        self.checkpoint_path = Path(
            os.environ["R5_RAIL_CHECKPOINT"]
        ).expanduser().resolve()
        expected_checkpoint_sha256 = os.environ["R5_RAIL_CHECKPOINT_SHA256"]
        assert self.checkpoint_path.is_file(), self.checkpoint_path
        checkpoint_sha256 = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()
        assert checkpoint_sha256 == expected_checkpoint_sha256, (
            checkpoint_sha256,
            expected_checkpoint_sha256,
        )
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )
        assert checkpoint.get("format") == "r5-fullmenu-combo-v1"
        assert checkpoint.get("mechanism") == (
            "codebooks-plus-all-rmsnorms-plus-attention-output-gains"
        )
        assignment_md5 = hashlib.md5(
            json.dumps(
                json.loads(self.target_path.read_text())["assignment"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        assert checkpoint.get("manifest_md5") == assignment_md5
        self.r5_codebooks = checkpoint["state"]["codebooks"]
        assert set(self.r5_codebooks) == {f"L{layer}" for layer in range(43)}
        assert sum(
            value.numel()
            for layer_state in self.r5_codebooks.values()
            for value in layer_state.values()
        ) == 6_694_912
        self.base_host = os.environ.get("FULLMENU_BASE_HOST")
        if self.base_host:
            self.base_remote_selected = Path(os.environ["FULLMENU_BASE_REMOTE_SELECTED"])
            self.base_selected = Path(
                os.environ.get("FULLMENU_BASE_CACHE", self.delta_dir / "base-layer-cache")
            ).expanduser().resolve()
        else:
            self.base_remote_selected = None
            self.base_selected = Path(os.environ["FULLMENU_BASE_SELECTED"]).expanduser().resolve()
        self.delta_host = os.environ.get("FULLMENU_DELTA_HOST")
        if self.delta_host:
            self.delta_remote_selected = Path(os.environ["FULLMENU_DELTA_REMOTE_SELECTED"])
            self.delta_selected = Path(
                os.environ.get("FULLMENU_DELTA_CACHE", self.delta_dir / "selected")
            ).expanduser().resolve()
        else:
            self.delta_remote_selected = None
            self.delta_selected = self.delta_dir / "selected"
        workers = int(
            os.environ.get(
                "FULLMENU_PREFETCH_WORKERS",
                os.environ.get("FULLMENU_BASE_PREFETCH_WORKERS", "8"),
            )
        )
        if workers < 1:
            raise ValueError("FULLMENU_PREFETCH_WORKERS must be positive")
        self._executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(max_workers=workers)
            if self.base_host or self.delta_host
            else None
        )
        self.target = json.loads(self.target_path.read_text())
        self.base = json.loads(self.base_manifest_path.read_text())
        self._verify_pack()
        self.cache: dict[str, dict[str, Any]] = {"base": {}, "delta": {}}
        self._prefetch: dict[tuple[str, int], dict[str, Future[Path]]] = {}
        self._cached_remote_layer: int | None = None
        ledger = os.environ.get("FULLMENU_STREAM_LEDGER")
        self._stream_ledger = Path(ledger).expanduser().resolve() if ledger else None
        self._stream_lock = threading.Lock()
        self._stream_seen: set[tuple[str, str, int]] = set()
        # ACCEL: local next-layer readahead (inert when remote streaming is on).
        self._warm_executor = ThreadPoolExecutor(max_workers=1)
        self._warm_futures: dict[int, Future[dict[str, dict[str, Any]]]] = {}

    def _r5_codebook(self, layer: int, tier_name: str, projection: str) -> torch.Tensor:
        key = f"{tier_name}__{projection}"
        layer_state = self.r5_codebooks[f"L{layer}"]
        assert key in layer_state, (layer, key, sorted(layer_state))
        value = layer_state[key]
        assert value.ndim == 2 and value.shape[1] in (4, 8), (
            layer,
            key,
            value.shape,
        )
        # Trainer evaluation wires each fp32 master through fp16 first.
        return value.to(v3.DEV).to(torch.float16).float()

    def _verify_pack(self) -> None:
        pack_path = self.delta_dir / "DELTA_PACK_MANIFEST.json"
        complete_path = self.delta_dir / "DELTA_PACK.COMPLETE"
        assert pack_path.is_file() and complete_path.is_file(), self.delta_dir
        pack_digest = md5(pack_path)
        assert complete_path.read_text().strip() == pack_digest, "delta COMPLETE digest mismatch"
        pack = json.loads(pack_path.read_text())
        assert pack["status"] == "ASSIGNMENT_DELTA_PACK_COMPLETE", pack.get("status")
        import os as _os
        if not _os.environ.get("FULLMENU_SKIP_PACK_BASE_MD5"):
            assert pack["base_manifest_md5"] == md5(self.base_manifest_path)
        assert pack["target_manifest_md5"] == md5(self.target_path)
        self.pack = pack
        self._delta_files = {
            (str(item["component"]), Path(str(item["path"])).name): item
            for item in pack.get("files", [])
        }

    def _required_components(self, layer: int) -> tuple[set[str], set[str]]:
        base_needed: set[str] = set()
        delta_needed: set[str] = set()
        target_map = self.target["assignment"][str(layer)]
        base_map = self.base["assignment"][str(layer)]
        assert set(target_map) == set(base_map), f"expert domain mismatch at layer {layer}"
        for expert_id, entry in target_map.items():
            for projection in ("13", "2"):
                target_tier = tier(entry, projection)
                base_tier = tier(base_map[expert_id], projection)
                (delta_needed if target_tier != base_tier else base_needed).add(target_tier)
        return base_needed, delta_needed

    @staticmethod
    def _validate_component(data: dict[str, Any], expected_manifest_md5: str, path: Path) -> None:
        meta = data.get("meta", {})
        import os as _os
        _extra = set(filter(None, _os.environ.get("FULLMENU_ACCEPT_MD5S", "").split(",")))
        _got = meta.get("manifest_md5")
        assert _got == expected_manifest_md5 or _got in _extra, path

    @staticmethod
    def _rsync_with_retry(remote: str, local: Path, attempts: int = 6) -> None:
        """Resume an atomic component transfer across transient QSFP SSH resets."""
        command = [
            "rsync", "-a", "--partial", "--append-verify", "--timeout=180",
            "-e", (
                "ssh -o BatchMode=yes -o ConnectTimeout=10 "
                "-o ControlMaster=auto -o ControlPersist=120 "
                "-o ControlPath=/tmp/fullmenu-%C"
            ),
            remote,
            str(local),
        ]
        last_error: subprocess.CalledProcessError | None = None
        for attempt in range(1, attempts + 1):
            try:
                subprocess.run(command, check=True)
                return
            except subprocess.CalledProcessError as exc:
                last_error = exc
                if attempt == attempts:
                    break
                time.sleep(min(2 ** attempt, 20))
        assert last_error is not None
        raise last_error

    def _copy_remote_component(self, source_name: str, name: str, layer: int) -> Path:
        assert source_name in {"base", "delta"}
        if source_name == "base":
            host = self.base_host
            remote_selected = self.base_remote_selected
            selected = self.base_selected
        else:
            host = self.delta_host
            remote_selected = self.delta_remote_selected
            selected = self.delta_selected
        assert host and remote_selected is not None
        destination = selected / name
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"layer_{layer:03d}.pt"
        target = destination / filename
        receipt_path = target.with_suffix(".DONE.json")
        transfer_path = target.with_suffix(target.suffix + ".partial")
        transfer_receipt = target.with_suffix(".TRANSFER.json")
        expected_manifest_md5 = (
            md5(self.base_manifest_path) if source_name == "base" else md5(self.target_path)
        )

        def valid() -> bool:
            try:
                receipt = json.loads(transfer_receipt.read_text())
                return (
                    target.is_file()
                    and target.stat().st_size == int(receipt["bytes"])
                    and md5(target) == receipt["md5"]
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                return False

        if not valid():
            remote = remote_selected / name
            self._rsync_with_retry(f"{host}:{remote / filename}", transfer_path)
            os.replace(transfer_path, target)
            observed_md5 = md5(target)
            tmp_receipt = transfer_receipt.with_suffix(transfer_receipt.suffix + ".tmp")
            tmp_receipt.write_text(json.dumps({
                "bytes": target.stat().st_size,
                "md5": observed_md5,
                "remote": f"{host}:{remote / filename}",
                "verified_transport": "rsync-success-plus-local-md5",
            }, sort_keys=True) + "\n")
            os.replace(tmp_receipt, transfer_receipt)
            if source_name == "base":
                receipt_partial = receipt_path.with_suffix(receipt_path.suffix + ".partial")
                try:
                    self._rsync_with_retry(
                        f"{host}:{remote / receipt_path.name}", receipt_partial, attempts=3
                    )
                    os.replace(receipt_partial, receipt_path)
                except subprocess.CalledProcessError:
                    receipt_partial.unlink(missing_ok=True)
        assert valid(), f"invalid streamed component: {target}"
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        item = self._delta_files.get((name, filename)) if source_name == "delta" else None
        observed_md5 = json.loads(transfer_receipt.read_text())["md5"]
        key = (source_name, name, layer)
        if self._stream_ledger is not None:
            with self._stream_lock:
                if key not in self._stream_seen:
                    self._stream_ledger.parent.mkdir(parents=True, exist_ok=True)
                    with self._stream_ledger.open("a") as handle:
                        handle.write(json.dumps({
                            "source": source_name,
                            "component": name,
                            "layer": layer,
                            "bytes": target.stat().st_size,
                            "md5": observed_md5,
                            "remote": f"{host}:{remote_selected / name / filename}",
                            "upstream_manifest_md5": expected_manifest_md5,
                            "upstream_receipt_md5": receipt.get("md5") if receipt else None,
                            "upstream_receipt_bytes": receipt.get("bytes") if receipt else None,
                            "upstream_receipt_matches_payload": bool(
                                receipt
                                and receipt.get("md5") == observed_md5
                                and int(receipt.get("bytes", -1)) == target.stat().st_size
                            ),
                            "delta_pack_manifest_md5": item.get("md5") if item else None,
                            "delta_pack_manifest_bytes": item.get("bytes") if item else None,
                            "delta_pack_manifest_matches_payload": bool(
                                item
                                and item.get("md5") == observed_md5
                                and int(item.get("bytes", -1)) == target.stat().st_size
                            ),
                            "verified": True,
                            "ts": time.time(),
                        }, sort_keys=True) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    self._stream_seen.add(key)
        return target

    def _submit_remote_layer(self, source_name: str, layer: int, tiers: set[str]) -> None:
        host = self.base_host if source_name == "base" else self.delta_host
        if self._executor is None or not host:
            return
        key = (source_name, layer)
        futures = self._prefetch.setdefault(key, {})
        for name in sorted({component(tier_name) for tier_name in tiers}):
            if name not in futures:
                futures[name] = self._executor.submit(
                    self._copy_remote_component, source_name, name, layer
                )

    def _wait_remote_layer(self, source_name: str, layer: int, tiers: set[str]) -> None:
        host = self.base_host if source_name == "base" else self.delta_host
        if self._executor is None or not host:
            return
        self._submit_remote_layer(source_name, layer, tiers)
        futures = self._prefetch.pop((source_name, layer))
        for name in sorted({component(tier_name) for tier_name in tiers}):
            futures[name].result()

    def _drop_remote_layer(self, layer: int | None) -> None:
        if self._executor is None or layer is None:
            return
        for host, selected in (
            (self.base_host, self.base_selected),
            (self.delta_host, self.delta_selected),
        ):
            if not host:
                continue
            for name in VQ_TIERS | {"static"}:
                path = selected / name / f"layer_{layer:03d}.pt"
                path.unlink(missing_ok=True)
                path.with_suffix(".DONE.json").unlink(missing_ok=True)
                path.with_suffix(".TRANSFER.json").unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".partial").unlink(missing_ok=True)

    def _read_layer_cache(self, layer: int) -> dict[str, dict[str, Any]]:
        """Warm page cache sequentially, then open components (mmap) + validate.

        ACCEL: pure I/O staging — identical tensors to the sealed inline loop.
        """
        cache: dict[str, dict[str, Any]] = {"base": {}, "delta": {}}
        base_needed, delta_needed = self._required_components(layer)
        for source_name, root, tiers, expected_md5 in (
            ("base", self.base_selected, base_needed, md5(self.base_manifest_path)),
            ("delta", self.delta_selected, delta_needed, md5(self.target_path)),
        ):
            for tier_name in sorted(tiers):
                name = component(tier_name)
                if name in cache[source_name]:
                    continue
                path = component_path(root, tier_name, layer)
                assert path.is_file(), path
                with path.open("rb") as handle:
                    while handle.read(64 << 20):
                        pass
                data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                self._validate_component(data, expected_md5, path)
                cache[source_name][name] = data
        return cache

    def _load_layer(self, layer: int) -> None:
        # Evict mmap-backed prior-layer pages before opening the next layer.
        # Unified-memory Sparks otherwise risk page-cache pressure starving CUDA.
        try:
            import lp4_train as trainer

            for by_component in self.cache.values():
                for payload in by_component.values():
                    for value in payload.values():
                        if (
                            isinstance(value, torch.Tensor)
                            and value.device.type == "cpu"
                            and value.untyped_storage().nbytes() > (16 << 20)
                        ):
                            trainer.evict_tensor(value)
        except (ImportError, AttributeError):
            pass
        self.cache = {"base": {}, "delta": {}}
        gc.collect()
        if self._executor is not None:
            # Remote streaming path: sealed behavior (rsync prefetch overlaps).
            base_needed, delta_needed = self._required_components(layer)
            self._drop_remote_layer(self._cached_remote_layer)
            self._wait_remote_layer("base", layer, base_needed)
            self._wait_remote_layer("delta", layer, delta_needed)
            if layer < 42:
                next_base_needed, next_delta_needed = self._required_components(layer + 1)
                self._submit_remote_layer("base", layer + 1, next_base_needed)
                self._submit_remote_layer("delta", layer + 1, next_delta_needed)
            for source_name, root, tiers, expected_md5 in (
                ("base", self.base_selected, base_needed, md5(self.base_manifest_path)),
                ("delta", self.delta_selected, delta_needed, md5(self.target_path)),
            ):
                for tier_name in sorted(tiers):
                    name = component(tier_name)
                    if name in self.cache[source_name]:
                        continue
                    path = component_path(root, tier_name, layer)
                    assert path.is_file(), path
                    data = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
                    self._validate_component(data, expected_md5, path)
                    self.cache[source_name][name] = data
        else:
            # ACCEL local path: consume the background-prefetched layer cache
            # (or read inline on first layer / cache miss), then stage L+1.
            future = self._warm_futures.pop(layer, None)
            self.cache = future.result() if future is not None else self._read_layer_cache(layer)
            if layer < 42 and layer + 1 not in self._warm_futures:
                self._warm_futures[layer + 1] = self._warm_executor.submit(
                    self._read_layer_cache, layer + 1
                )
        self._cached_remote_layer = layer

    def _ids(self, source_name: str, tier_name: str, projection: str) -> object:
        data = self.cache[source_name][component(tier_name)]
        prefix = tier_name if tier_name in STATIC_TIERS else "expert"
        return data[f"{prefix}_ids{projection}"]

    def fill_layer(self, layer: int, gate_up: Any, down: Any) -> None:
        """Bulk tier-batched materialization; bitwise-identical to layer()/expert().

        Grouping: (source, tier, projection) -> [(expert_id, row), ...].
        Per group: ONE codebook upload + fp16-wire cast, batched row gather,
        vectorized dequant (gather/exp2/repeat_interleave/mul only), disjoint
        destination row writes. No reductions anywhere => batch-shape invariant.
        """
        self._load_layer(layer)
        target_map = self.target["assignment"][str(layer)]
        base_map = self.base["assignment"][str(layer)]
        row_index_cache: dict[tuple[str, str, str], dict[int, int]] = {}

        def row_index(source_name: str, tier_name: str, projection: str) -> dict[int, int]:
            key = (source_name, tier_name, projection)
            if key not in row_index_cache:
                index: dict[int, int] = {}
                for row, expert_id in enumerate(_ids_list(self._ids(source_name, tier_name, projection))):
                    assert expert_id not in index, f"duplicate expert id {expert_id}"
                    index[expert_id] = row
                row_index_cache[key] = index
            return row_index_cache[key]

        groups: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        for expert_key, entry in target_map.items():
            expert_id = int(expert_key)
            for projection in ("13", "2"):
                target_tier = tier(entry, projection)
                base_tier = tier(base_map[expert_key], projection)
                source_name = "delta" if target_tier != base_tier else "base"
                index = row_index(source_name, target_tier, projection)
                message = (
                    f"missing {source_name} row: tier={target_tier} "
                    f"expert={expert_id} proj={projection}"
                )
                assert expert_id in index, message
                groups.setdefault((source_name, target_tier, projection), []).append(
                    (expert_id, index[expert_id])
                )

        batch_size = int(os.environ.get("FULLMENU_ASSEMBLY_BATCH", "16"))
        if batch_size < 1:
            raise ValueError("FULLMENU_ASSEMBLY_BATCH must be positive")
        filled = 0
        for (source_name, tier_name, projection), entries in sorted(groups.items()):
            data = self.cache[source_name][component(tier_name)]
            destination = gate_up if projection == "13" else down
            codebook = None
            prefix = ""
            if tier_name.startswith(("d4_k", "d8_k")) or tier_name == "vqa":
                prefix = "" if tier_name.startswith(("d4_k", "d8_k")) else "vqa_"
                codebook = self._r5_codebook(layer, tier_name, projection)
            for batch_start in range(0, len(entries), batch_size):
                batch = entries[batch_start:batch_start + batch_size]
                expert_ids = [item[0] for item in batch]
                rows = [item[1] for item in batch]
                if codebook is not None:
                    codes = data[f"{prefix}codes{projection}"][rows].to(v3.DEV)
                    scales = data[f"{prefix}sc{projection}"][rows].to(v3.DEV)
                    scale_columns = torch.exp2(
                        scales.float() - 127.0
                    ).repeat_interleave(32, dim=-1)
                    weights = codebook[codes.long()].reshape(
                        codes.shape[0], codes.shape[1], -1
                    ) * scale_columns
                elif tier_name == "fp4":
                    weights = v3.deq_fp4_block32(
                        data[f"fp4_wb{projection}"][rows].to(v3.DEV),
                        data[f"fp4_sb{projection}"][rows].to(v3.DEV),
                        "e2m1",
                    )
                else:
                    raise KeyError(tier_name)
                expected = (len(batch), 4096, 4096 if projection == "13" else 2048)
                assert tuple(weights.shape) == expected, (
                    layer, projection, tier_name, weights.shape, expected
                )
                destination[expert_ids] = weights.to(torch.bfloat16)
                filled += len(batch)
                del weights
        assert filled == 2 * len(target_map), (layer, filled, len(target_map))
        print(
            f"[FullMenuDeltaSource] L{layer:03d} bulk-filled "
            f"base={self.base_manifest_path.name} target={self.target_path.name}",
            flush=True,
        )

    def layer(self, layer: int):
        self._load_layer(layer)
        target_map = self.target["assignment"][str(layer)]
        base_map = self.base["assignment"][str(layer)]
        base_ids: dict[str, dict[str, object]] = {}
        delta_ids: dict[str, dict[str, object]] = {}
        for source_name, destination in (("base", base_ids), ("delta", delta_ids)):
            for tier_name in {
                tier(entry, projection)
                for entry in target_map.values()
                for projection in ("13", "2")
            }:
                name = component(tier_name)
                if name not in self.cache[source_name]:
                    continue
                destination.setdefault(tier_name, {})
                for projection in ("13", "2"):
                    destination[tier_name][projection] = self._ids(
                        source_name, tier_name, projection
                    )

        def expert(expert_id: int, which: str):
            projection = "13" if which == "13" else "2"
            target_tier = tier(target_map[str(expert_id)], projection)
            base_tier = tier(base_map[str(expert_id)], projection)
            source_name, row = resolve_row(
                target_tier=target_tier,
                base_tier=base_tier,
                expert_id=expert_id,
                projection=projection,
                base_ids=base_ids,
                delta_ids=delta_ids,
            )
            data = self.cache[source_name][component(target_tier)]
            if target_tier.startswith(("d4_k", "d8_k")):
                codes = data[f"codes{projection}"][row].to(v3.DEV)
                scales = data[f"sc{projection}"][row].to(v3.DEV)
                codebook = self._r5_codebook(layer, target_tier, projection)
                scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
                weights = codebook[codes.long()].reshape(codes.shape[0], -1) * scale_columns
            elif target_tier == "vqa":
                codes = data[f"vqa_codes{projection}"][row].to(v3.DEV)
                scales = data[f"vqa_sc{projection}"][row].to(v3.DEV)
                codebook = self._r5_codebook(layer, target_tier, projection)
                scale_columns = torch.exp2(scales.float() - 127.0).repeat_interleave(32, dim=1)
                weights = codebook[codes.long()].reshape(codes.shape[0], -1) * scale_columns
            elif target_tier == "fp4":
                weights = v3.deq_fp4_block32(
                    data[f"fp4_wb{projection}"][row].to(v3.DEV),
                    data[f"fp4_sb{projection}"][row].to(v3.DEV),
                    "e2m1",
                )
            else:
                raise KeyError(target_tier)
            expected = (4096, 4096) if projection == "13" else (4096, 2048)
            assert tuple(weights.shape) == expected, (
                layer,
                expert_id,
                projection,
                target_tier,
                weights.shape,
            )
            return weights.to(torch.bfloat16)

        print(
            f"[FullMenuDeltaSource] L{layer:03d} base={self.base_manifest_path.name} "
            f"target={self.target_path.name}",
            flush=True,
        )
        return expert, DIMS
