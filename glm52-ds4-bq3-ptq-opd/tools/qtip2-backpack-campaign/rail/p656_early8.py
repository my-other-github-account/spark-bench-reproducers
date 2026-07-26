#!/usr/bin/env python3
"""P656 compute-node-6 EARLY_8 adapter: P623 baseline + pinned P651 sparse mechanics.

The numerical builder/reader/reducer remain byte-pinned.  This adapter only binds
P651's sparse changed-cell mechanics to compute-node-6's sealed P653 manifest and streams
the immutable current-GENESIS base from compute-node-8 one layer at a time.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P656_EARLY8_PUBLIC_TASK_s6")
CLAIM = Path("$HOME/HOST_CLAIM.json")
P640 = Path("$HOME/run-bundles/P640_GENESIS_QTIP2_WIRE_PUBLIC_TASK_s6")
P623 = Path("$HOME/run-bundles/P623_GENESIS_BASELINE_PUBLIC_TASK_s6")
P492 = Path("$HOME/run-bundles/P492_SWAP_LADDER_PUBLIC_TASK_s6")
P653 = P640 / "WIRE_STREAM_IN/P647_RESPENT/P653_ASSEMBLY/P653_EXACT_ASSEMBLED_WIRE_MANIFEST.json"
ASSIGNMENT = P640 / "inputs/ASSIGNMENT_RESPENT.json"
BASE_ASSIGNMENT = ROOT / "inputs/NOMINATED_ASSIGNMENT.json"
P623_VIEW = ROOT / "inputs/P623_BASELINE_FULL512_VIEW.json"
P623_PARITY = ROOT / "receipts/P623_BASELINE_PARITY_EARLY8.json"
P632_SCORE = ROOT / "code/p632_score.py"
P651_MECHANICS = ROOT / "code/p651_overlay_rail.py"
P651_READER = ROOT / "code/p651_baseline_parity.py"
QTIP_SOURCE = Path("$HOME/run-bundles/P605R_QTIP2_ANCHORS_C_PUBLIC_TASK_s6/code/run_qtip_anchor.py")
QTIP_KERNEL = Path("$HOME/run-bundles/P532_QTIP2_SHARD_B_PUBLIC_TASK_s6/inputs/qtip-canonical/lib/utils/kernel_decompress.py")
QTIP_TLUT = Path("$HOME/run-bundles/P532_QTIP2_SHARD_B_PUBLIC_TASK_s6/inputs/tlut/PINNED_TLUT.pt")
BASE_SOURCE_HOST = "203.0.113.9"
BASE_REMOTE_PACKAGE = "$HOME/run-bundles/GENESIS_FANIN_PUBLIC_TASK_s8/package/wire43"
MODEL = Path("$HOME/models/hf/DeepSeek-V4-Flash")
TEACHER = Path("$HOME/run-bundles/DS4_TEACHER")

PINS = {
    "p653": "e03bc8919d51bbf1a9cf1f54f342e9f43dea625839ad8aad23578f7b8f9d98fa",
    "assignment": "c030883fddb1217529d67444d08257c4a1df18e2adbc93be092aba3d3611bc65",
    "assignment_map": "36d0841986d5781186f766b3815e4b3c6332eece2090d3e6d73e7e3ffa33dc07",
    "base_assignment": "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d",
    "base_wire": "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755",
    "compact": "d9421f1f6d0e696608bb0ce9b09131e63790c18e9cd536e440b1884b727db00d",
    "labels": "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88",
    "window_contract": "91a33069d7d2f5648d63ef10b4a11eb122dbce740eec2ac9acd0bc202325fbad",
    "p623_source": "c3ba83fddf8f39d4b300c2baf8ad242bfdef21d3a90ac758b005fd01b078d3d5",
    "p623_scorer": "844d7e06c5c221e4138a4be931f61e9616ec6ac46123aba1bdd02902b58dffb9",
    "p651_reader": "c6f13c3e82a9d1a21ac7f50a3ab0f59239158b7736cd077f28c751087fca725c",
    "p651_mechanics": "2170cabd7e240e89bce8ecd1778d9cbc1a498f2aaf783beaa5e119bc19591430",
    "p632_score": "5c16e62c32e6936223c54e2b3cf9394a1d0f87833cc409360e82e0341954c12f",
    "reader": "bc0920b8865376463e58d11686e888524122b9bc995668fca23fa1ec24312b42",
    "builder": "d56677ed63711aac24181463d7ef8ac45499c4b507919b3ad4d5dcb63da205bb",
    "delta": "2aeed7527631050ad440a52fe796502ff01dcd98096f86dd20e8ca9e9187625f",
    "lp4": "7a8e48547824a87a48db4c7142ec53f73303a91ce6a0c95cf1a88b1b87d22350",
    "planes": "aeb3e473a00b48426f56b9f80aefc6bc086b7791ec2372606c724e90db126334",
    "contracts": "0842784bfba78032f122c8e859f2a1df1d67885823e1aa323cc020d3ae6fccbf",
    "safety": "b45f6eef933ac51d2c5f1693f21f1859945a9ad9d18741dd0732fa3956275e0c",
    "loader": "155310d1e6701d6cb2d1c04558514366a2304cb2a8d6d26402ed7c800b8b6c89",
    "model_index": "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
    "teacher_done": "6338af84f907a26dfdf0f784edc322aa672738542ed884b70e4d9b6e96aa33b0",
    "corpus": "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd",
    "qtip_source": "b2a9b6c60e95aa387129246fd0f30354f356d9b89409d71089d5fddffb7eea4a",
    "qtip_kernel": "01b6520c8f39982ac5f35de58364f31f79f00555350c8c1776e4fd9b1ca0a63f",
    "qtip_tlut_file": "be7e69b5b18419afc333dc3ef7841bda2ed8207114eeae0ac5bcd7bcab79b93c",
}
EXACT_WIRE_BYTES = 101_346_521_679
BASELINE_FULL512 = 0.08394998423027422
RAW_WITHOUT = 0.06708283585873699
RAW_WITH = 0.05541288213586761
PREDICTED_FULL512 = 0.07228003050740484


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def claim_snapshot() -> tuple[bytes, dict[str, Any]]:
    raw = CLAIM.read_bytes()
    claim = json.loads(raw)
    expected = {"host": "compute-node-6", "owner": TASK, "task": TASK, "task_id": TASK, "mission": str(ROOT)}
    drift = {k: (claim.get(k), v) for k, v in expected.items() if claim.get(k) != v}
    if drift or claim.get("status") not in ("CLAIMED", "ACTIVE"):
        raise RuntimeError(f"P656 claim drift: drift={drift} status={claim.get('status')}")
    return raw, claim


def gpu_snapshot(*, own_pid: int | None = None, require_zero_util: bool = False) -> dict[str, Any]:
    apps_raw = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    apps = []
    for line in apps_raw.splitlines():
        if not line.strip():
            continue
        pid = line.split(",", 1)[0].strip()
        if own_pid is not None and pid == str(own_pid):
            continue
        apps.append(line.strip())
    if apps:
        raise RuntimeError(f"foreign GPU applications present: {apps}")
    util = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return {"foreign_compute_apps": apps, "nvidia_reported_utilization": util, "zero_util_required": require_zero_util, "exclusivity_gate": "no foreign compute apps"}


def flatten_assignment(path: Path) -> dict[tuple[int, int, str], str]:
    payload = json.loads(path.read_text())["assignment"]
    out = {}
    for layer in range(43):
        row = payload[str(layer)]
        for expert in range(256):
            cell = row[str(expert)]
            for projection in ("fused13", "down"):
                out[(layer, expert, projection)] = str(cell[projection])
    if len(out) != 43 * 256 * 2:
        raise RuntimeError("assignment surface drift")
    return out


def preflight_p653() -> dict[str, Any]:
    if sha256(P653) != PINS["p653"] or sha256(ASSIGNMENT) != PINS["assignment"] or sha256(BASE_ASSIGNMENT) != PINS["base_assignment"]:
        raise RuntimeError("P653/assignment byte identity drift")
    doc = json.loads(P653.read_text())
    if (
        doc.get("status") != "PASS_EXACT_ASSEMBLED_LOGICAL_WIRE"
        or doc.get("assignment", {}).get("sha256") != PINS["assignment"]
        or doc.get("assignment_map_sha256") != PINS["assignment_map"]
        or doc.get("base_wire", {}).get("manifest_sha256") != PINS["base_wire"]
        or int(doc.get("exact_wire_bytes", -1)) != EXACT_WIRE_BYTES
    ):
        raise RuntimeError("P653 authority drift")
    raw_rows = doc.get("overlay_rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 1411:
        raise RuntimeError("P653 overlay row count drift")
    base = flatten_assignment(BASE_ASSIGNMENT)
    final = flatten_assignment(ASSIGNMENT)
    expected_changed = {key for key in final if final[key] != base[key]}
    rows: list[dict[str, Any]] = []
    identities = []
    source_specs: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        layer, expert, projection = int(raw["layer"]), int(raw["expert"]), str(raw["projection"])
        key = (layer, expert, projection)
        if raw.get("key") != f"{layer}:{expert}:{projection}" or raw.get("old") != base[key] or raw.get("new") != final[key]:
            raise RuntimeError(f"P653 transition drift {key}")
        artifact = raw.get("artifact")
        if not isinstance(artifact, dict):
            raise RuntimeError(f"artifact missing {key}")
        artifact_path = Path(str(artifact.get("consumer_source_path_spark6")))
        if not artifact_path.is_relative_to(P640):
            raise RuntimeError(f"artifact escaped P640 root: {artifact_path}")
        kind = "qtip2_exact" if raw["new"] == "qtip2_2.0117" else "genesis_vq"
        row = {
            "layer": layer, "expert": expert, "projection": projection,
            "old_tier": str(raw["old"]), "new_tier": str(raw["new"]), "kind": kind,
            "artifact": artifact_path.name,
            "artifact_source_abs": str(artifact_path),
            "artifact_source_rel": f"cells/{artifact_path.name}",
            "artifact_sha256": str(artifact["sha256"]),
            "artifact_physical_bytes": int(artifact["bytes"]),
            "codebook": None, "codebook_source_abs": None, "codebook_source_rel": None,
            "codebook_sha256": None, "codebook_physical_bytes": 0,
        }
        if kind == "genesis_vq":
            cb = raw.get("codebook")
            if not isinstance(cb, dict):
                raise RuntimeError(f"VQ codebook missing {key}")
            cb_path = Path(str(cb.get("consumer_source_path_spark6")))
            if not cb_path.is_relative_to(P640):
                raise RuntimeError(f"codebook escaped P640 root: {cb_path}")
            row.update({
                "codebook": cb_path.name,
                "codebook_source_abs": str(cb_path),
                "codebook_source_rel": f"codebooks/{cb_path.name}",
                "codebook_sha256": str(cb["sha256"]),
                "codebook_physical_bytes": int(cb["bytes"]),
            })
        for role, source, rel, digest, size in [
            ("cell", row["artifact_source_abs"], row["artifact_source_rel"], row["artifact_sha256"], row["artifact_physical_bytes"]),
            ("codebook", row["codebook_source_abs"], row["codebook_source_rel"], row["codebook_sha256"], row["codebook_physical_bytes"]),
        ]:
            if source is None:
                continue
            spec = {"role": role, "source": source, "rel": rel, "sha256": digest, "bytes": size}
            prior = source_specs.setdefault(source, spec)
            if prior != spec:
                raise RuntimeError(f"conflicting source identity: {source}")
        rows.append(row)
        identities.append(key)
    if set(identities) != expected_changed or len(set(identities)) != 1411:
        raise RuntimeError("P653 assignment/overlay identity mismatch")
    by_layer = {layer: [row for row in rows if row["layer"] == layer] for layer in range(43)}
    qtip = sum(row["kind"] == "qtip2_exact" for row in rows)
    vq = len(rows) - qtip
    if (qtip, vq, len(final) - len(rows)) != (406, 1005, 20605):
        raise RuntimeError("P653 changed/copy-through count drift")
    return {
        "rows": rows,
        "by_layer": by_layer,
        "manifest_rows": [{"name": "P653_EXACT_ASSEMBLED_WIRE_MANIFEST", "path": str(P653), "sha256": PINS["p653"], "rows": len(rows), "bytes": P653.stat().st_size}],
        "changed_cells": len(rows), "unchanged_cells": len(final) - len(rows),
        "qtip2_cells": qtip, "vq_cells": vq,
        "identity_set_sha256": canonical_sha(sorted(identities)),
        "unique_sparse_files": len(source_specs),
        "unique_sparse_bytes": sum(int(spec["bytes"]) for spec in source_specs.values()),
    }


def stage_layer_local(layer: int, rows: list[dict[str, Any]], cache_root: Path):
    if not rows:
        return None, {"layer": layer, "streams": 0, "files": 0, "bytes": 0, "elapsed_seconds": 0.0, "bytes_per_second": None, "transport": "local sparse copy"}
    specs: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidates = [
            {"source": row["artifact_source_abs"], "rel": row["artifact_source_rel"], "sha256": row["artifact_sha256"], "bytes": row["artifact_physical_bytes"], "role": "cell"},
        ]
        if row["codebook_source_abs"]:
            candidates.append({"source": row["codebook_source_abs"], "rel": row["codebook_source_rel"], "sha256": row["codebook_sha256"], "bytes": row["codebook_physical_bytes"], "role": "codebook"})
        for spec in candidates:
            prior = specs.setdefault(spec["source"], spec)
            if prior != spec:
                raise RuntimeError(f"conflicting layer source identity: {spec['source']}")
    ordered = sorted(specs.values(), key=lambda x: int(x["bytes"]), reverse=True)
    stage = cache_root / f"overlay_layer_{layer:03d}"
    partial = cache_root / f".overlay_layer_{layer:03d}.{uuid.uuid4().hex}.partial"
    if stage.exists() or partial.exists():
        raise RuntimeError(f"once-only overlay stage exists L{layer}")
    partial.mkdir(parents=True)
    started = time.time()
    streams = min(4, len(ordered))
    buckets = [[] for _ in range(streams)]
    loads = [0] * streams
    for spec in ordered:
        idx = min(range(streams), key=loads.__getitem__)
        buckets[idx].append(spec)
        loads[idx] += int(spec["bytes"])

    def copy_bucket(idx: int) -> dict[str, Any]:
        for spec in buckets[idx]:
            src, dst = Path(spec["source"]), partial / spec["rel"]
            if not src.is_file() or src.stat().st_size != int(spec["bytes"]):
                raise RuntimeError(f"source size drift L{layer}: {src}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        return {"worker": idx, "files": len(buckets[idx]), "bytes": loads[idx]}

    try:
        with ThreadPoolExecutor(max_workers=streams) as pool:
            workers = list(pool.map(copy_bucket, range(streams)))
        for spec in ordered:
            dst = partial / spec["rel"]
            if dst.stat().st_size != int(spec["bytes"]) or sha256(dst) != spec["sha256"]:
                raise RuntimeError(f"staged sparse identity drift L{layer}: {spec['source']}")
        os.replace(partial, stage)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    elapsed = time.time() - started
    total = sum(int(spec["bytes"]) for spec in ordered)
    return stage, {
        "layer": layer, "streams": streams, "workers": workers, "files": len(ordered),
        "cells": len(rows), "bytes": total, "elapsed_seconds": elapsed,
        "bytes_per_second": total / elapsed if elapsed else None,
        "source_host": "compute-node-6-local", "source_root": str(P640), "transport": "four-way bounded local sparse copy+SHA256",
    }


class CorrectQtip2Decoder:
    """Exact P605R QTIP2 decoder bound to the pinned compute-node-6 TLUT/kernel."""

    def __init__(self, p651: Any) -> None:
        if sha256(QTIP_SOURCE) != PINS["qtip_source"] or sha256(QTIP_KERNEL) != PINS["qtip_kernel"] or sha256(QTIP_TLUT) != PINS["qtip_tlut_file"]:
            raise RuntimeError("pinned P605R QTIP2 source/kernel/TLUT drift")
        qsource = load_module("p656_p605r_qtip_source", QTIP_SOURCE)
        self.kernel = load_module("p656_p605r_qtip_kernel", QTIP_KERNEL)
        self.fwht = qsource.QtipResolver.fwht
        payload = p651.torch.load(QTIP_TLUT, map_location="cpu", mmap=True, weights_only=True)
        tlut = payload.get("tlut")
        if not isinstance(tlut, p651.torch.Tensor) or tuple(tlut.shape) != (512, 2):
            raise RuntimeError("P605R QTIP2 TLUT tensor surface drift")
        tlut = tlut.float().contiguous()
        tensor_hash = hashlib.sha256(tlut.numpy().tobytes()).hexdigest()
        if tensor_hash != "000c7985f6ac0cbece4a9850d3913102f9a6cf6ccb20cacf582d4fa95b569c19":
            raise RuntimeError("P605R QTIP2 TLUT tensor hash drift")
        index = p651.torch.arange(1 << 16, device="cuda")
        quadratic = (index + 1) * index
        sign_flip = 1 - ((quadratic >> 15) & 1) * 2
        lookup = (quadratic >> (16 - 9 - 1)) & 511
        expanded = tlut.to("cuda")[lookup]
        expanded[:, 0] *= sign_flip
        self.expanded = expanded.contiguous()
        self.tlut_tensor_sha256 = tensor_hash

    def decode(self, path: Path, row: Mapping[str, Any], destination: Any) -> dict[str, Any]:
        p = __import__("torch")
        payload = p.load(path, map_location="cpu", mmap=True, weights_only=True)
        identity = payload.get("identity")
        geometry = payload.get("geometry")
        if (
            payload.get("schema") not in ("qtip-rate-rung-unit-v1", "qtip-hyb-wire-unit-v1")
            or not isinstance(identity, dict)
            or int(identity.get("layer", -1)) != row["layer"]
            or int(identity.get("expert", -1)) != row["expert"]
            or identity.get("projection") != row["projection"]
            or geometry != {"L": 16, "K": 2, "V": 2, "tlut_bits": 9, "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16}
            or payload.get("tlut_sha256") != self.tlut_tensor_sha256
        ):
            raise RuntimeError(f"P605R QTIP2 payload identity/geometry drift: {path}")
        shape = tuple(map(int, payload["shape"]))
        if shape != tuple(destination.shape):
            raise RuntimeError(f"P605R QTIP2 destination shape drift: {path}")
        raw = self.kernel.decode_compressed(
            16, 9, 2, 1, shape[0], shape[1],
            payload["trellis"].to("cuda", non_blocking=False).reshape(-1), self.expanded,
        ) * payload["Wscale"].to("cuda")
        reconstructed = self.fwht(raw.T).T * payload["SV"].float().to("cuda")[:, None]
        reconstructed = self.fwht(reconstructed) * payload["SU"].float().to("cuda")
        if tuple(reconstructed.shape) != shape or not bool(p.isfinite(reconstructed).all()):
            raise RuntimeError(f"P605R QTIP2 decoded surface invalid: {path}")
        destination.copy_(reconstructed.to(p.bfloat16))
        del payload, raw, reconstructed
        return {"shape": list(shape), "finite": True, "decoder": "exact-p605r-qtip2", "tlut_tensor_sha256": self.tlut_tensor_sha256}


def install_remote_stream_source(p651: Any, base: Any, manifest: Mapping[str, Any], cache: Path, mode: str):
    by_layer = manifest["by_layer"]
    stage_rows: list[dict[str, Any]] = []
    applied: dict[tuple[int, int, str], dict[str, Any]] = {}
    decoder_holder: dict[str, Any] = {}

    class P640StreamSource(base.GenesisTierSource):
        def _stage_remote(self, layer: int, row: dict) -> Path:
            return base.GenesisTierSource._stage_remote(self, layer, row)

        def fill_layer(self, layer: int, gate_up: Any, down: Any) -> None:
            super().fill_layer(layer, gate_up, down)
            rows = by_layer[layer]
            if not rows:
                stage_rows.append({"layer": layer, "changed_cells": 0, "streams": 0, "files": 0, "bytes": 0, "elapsed_seconds": 0.0, "stage_retired": True})
                return
            stage, transfer = stage_layer_local(layer, rows, cache)
            assert stage is not None
            qtip_rows = [row for row in rows if row["kind"] == "qtip2_exact"]
            try:
                if qtip_rows and "decoder" not in decoder_holder:
                    decoder_holder["decoder"] = CorrectQtip2Decoder(p651)
                for row in rows:
                    destination = gate_up[row["expert"]] if row["projection"] == "fused13" else down[row["expert"]]
                    artifact = stage / row["artifact_source_rel"]
                    if row["kind"] == "qtip2_exact":
                        decode_info = decoder_holder["decoder"].decode(artifact, row, destination)
                    else:
                        payload = p651.torch.load(artifact, map_location="cpu", mmap=True, weights_only=True)
                        meta = payload.get("meta")
                        if (
                            not isinstance(meta, dict)
                            or meta.get("schema") != "p640-genesis-vq-overlay-cell-v1"
                            or meta.get("assignment_sha256") != PINS["assignment"]
                            or meta.get("canonical_builder_sha256") != "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
                            or int(meta.get("layer", -1)) != row["layer"]
                            or int(meta.get("expert", -1)) != row["expert"]
                            or meta.get("projection") != row["projection"]
                            or meta.get("tier") != row["new_tier"]
                            or meta.get("codebook_sha256") != row["codebook_sha256"]
                            or meta.get("fp16_codebook_replay_exact") is not True
                        ):
                            raise RuntimeError(f"VQ overlay payload metadata drift: {artifact}")
                        d, k = int(meta["d"]), int(meta["k"])
                        codebook_path = stage / row["codebook_source_rel"]
                        codebook = p651.torch.from_file(str(codebook_path), dtype=p651.torch.float16, size=k * d).reshape(k, d).clone()
                        codes = payload["codes"].unsqueeze(0)
                        scales = payload["scales"].unsqueeze(0)
                        if not bool(p651.torch.isfinite(codebook).all()) or int(codes.min()) < 0 or int(codes.max()) >= k:
                            raise RuntimeError(f"VQ overlay numerical surface drift: {artifact}")
                        base.GenesisTierSource._launch_vq(codes, scales, codebook, [row["expert"]], gate_up if row["projection"] == "fused13" else down, d)
                        decode_info = {"d": d, "k": k, "finite_codebook": True, "fp16_codebook_replay_exact": True}
                        del payload, codebook, codes, scales
                    key = (row["layer"], row["expert"], row["projection"])
                    record = {
                        "layer": row["layer"], "expert": row["expert"], "projection": row["projection"],
                        "old_tier": row["old_tier"], "new_tier": row["new_tier"], "kind": row["kind"],
                        "artifact_sha256": row["artifact_sha256"], "artifact_physical_bytes": row["artifact_physical_bytes"],
                        "codebook_sha256": row["codebook_sha256"], "decode": decode_info,
                    }
                    prior = applied.setdefault(key, record)
                    if prior != record:
                        raise RuntimeError(f"non-deterministic changed-cell application: {key}")
                p651.torch.cuda.synchronize()
            finally:
                gc.collect()
                shutil.rmtree(stage, ignore_errors=True)
            transfer.update({
                "schema": "p656-layer-overlay-consumption-v1", "status": "PASS",
                "task_id": TASK, "mode": mode, "changed_cells": len(rows),
                "qtip2_cells": len(qtip_rows), "vq_cells": len(rows) - len(qtip_rows),
                "stage_retired": not stage.exists(),
            })
            stage_rows.append(transfer)
            p651.atomic_json(ROOT / f"run/{mode}/LAYER_{layer:03d}_OVERLAY.json", transfer)

    return P640StreamSource, {"stage_rows": stage_rows, "applied": applied, "decoder_holder": decoder_holder}


def configure_p632(p: Any) -> None:
    package = ROOT / "code/eval_package"
    inputs = ROOT / "inputs"
    p.TASK = TASK
    p.ROOT = ROOT
    p.CLAIM = CLAIM
    p.SOURCE_HOST = BASE_SOURCE_HOST
    p.REMOTE_PACKAGE = BASE_REMOTE_PACKAGE
    p.PACKAGE = package
    p.MODEL = MODEL
    p.TEACHER = TEACHER
    p.CORPUS = TEACHER / "static/windows_ds4_eval.json"
    p.INPUTS = inputs
    p.COMPACT_MANIFEST = inputs / "GENESIS_COMPACT_FANIN.json"
    p.ASSIGNMENT = inputs / "NOMINATED_ASSIGNMENT.json"
    p.LABELS = inputs / "BQ3_STEP0_PER_CLASS.json"
    p.WINDOW_CONTRACT = inputs / "WINDOW_CONTRACT.json"
    p.WIRE_MANIFEST = inputs / "WIRE_43_MANIFEST.json"
    p.BASELINE_FULL512 = P623_VIEW
    p.CANONICAL_READER = ROOT / "code/genesis_remote_full512.py"
    p.CANONICAL_BUILDER = package / "t8192_ds4_build_v3.py"
    p.CANONICAL_DELTA = package / "delta_pack_sources.py"
    p.CANONICAL_LP4_PACK = package / "lp4_pack.py"
    p.CANONICAL_PLANES_UNPACK = package / "planes_unpack.py"
    p.CANONICAL_EVAL_CONTRACTS = package / "readapt_eval_contracts.py"
    p.CANONICAL_SAFETY = ROOT / "code/full512_safety.py"
    p.LOADER_SOURCE = ROOT / "code/rail_loading.py"
    p.PHYSICAL_MARKER = inputs / "PHYSICAL_CODE76.json"
    p.PHYSICAL_PASS_MARKER = inputs / "PHYSICAL_CODE76.PASS.json"
    p.ASSIGNMENT_SHA = PINS["base_assignment"]
    p.COMPACT_SHA = PINS["compact"]
    p.WIRE_SHA = PINS["base_wire"]
    p.MODEL_INDEX_SHA = PINS["model_index"]
    p.TEACHER_DONE_SHA = PINS["teacher_done"]
    p.CORPUS_SHA = PINS["corpus"]
    p.CANONICAL_SHA256 = {
        p.CANONICAL_READER: PINS["reader"], p.CANONICAL_BUILDER: PINS["builder"],
        p.CANONICAL_DELTA: PINS["delta"], p.CANONICAL_LP4_PACK: PINS["lp4"],
        p.CANONICAL_PLANES_UNPACK: PINS["planes"], p.CANONICAL_EVAL_CONTRACTS: PINS["contracts"],
        p.CANONICAL_SAFETY: PINS["safety"], p.LOADER_SOURCE: PINS["loader"],
    }
    p.EXPECTED_INPUT_SHA256 = {p.BASELINE_FULL512: sha256(P623_VIEW)}
    p.current_claim = claim_snapshot
    p.gpu_snapshot = gpu_snapshot
    original_configure_parent = p.configure_parent_module

    def configure_parent_module(base: Any, *, cache: Path, progress: Path, sentinel: Path) -> None:
        # Let the exact P632 binder install every canonical identity.  Keep the
        # canonical reader's transport pointed at compute-node-8, but keep its native
        # PHYSICAL_PACKAGE read root on the task-local one-layer cache.  P651's
        # original main assigns PHYSICAL_PACKAGE from p.REMOTE_PACKAGE after
        # this call, so expose the cache there only after binding base.REMOTE_PACKAGE.
        original_configure_parent(base, cache=cache, progress=progress, sentinel=sentinel)
        base.REMOTE_PACKAGE = BASE_REMOTE_PACKAGE
        base.PHYSICAL_PACKAGE = cache
        p.REMOTE_PACKAGE = str(cache)

    p.configure_parent_module = configure_parent_module

    def preflight_contract(mode: str) -> dict[str, Any]:
        if mode != "early8":
            raise RuntimeError("P656 scope is exactly EARLY_8")
        required_hashes = {
            p.CANONICAL_READER: PINS["reader"], p.CANONICAL_BUILDER: PINS["builder"],
            p.CANONICAL_DELTA: PINS["delta"], p.CANONICAL_LP4_PACK: PINS["lp4"],
            p.CANONICAL_PLANES_UNPACK: PINS["planes"], p.CANONICAL_EVAL_CONTRACTS: PINS["contracts"],
            p.CANONICAL_SAFETY: PINS["safety"], p.LOADER_SOURCE: PINS["loader"],
            p.COMPACT_MANIFEST: PINS["compact"], p.ASSIGNMENT: PINS["base_assignment"],
            p.LABELS: PINS["labels"], p.WINDOW_CONTRACT: PINS["window_contract"],
            p.WIRE_MANIFEST: PINS["base_wire"], MODEL / "model.safetensors.index.json": PINS["model_index"],
            TEACHER / "t8192_eval/DONE.jsonl": PINS["teacher_done"], p.CORPUS: PINS["corpus"],
        }
        drift = {str(path): {"expected": expected, "observed": sha256(path) if path.is_file() else None} for path, expected in required_hashes.items() if not path.is_file() or sha256(path) != expected}
        if drift:
            raise RuntimeError(f"canonical physical input drift: {drift}")
        parity = json.loads(P623_PARITY.read_text())
        if parity.get("status") != "PASS_EXACT_SAME_HOST_INSTRUMENT" or float(parity.get("maximum_absolute_window_mean_delta", 1.0)) != 0.0:
            raise RuntimeError("P623 parity prerequisite drift")
        labels = json.loads(p.LABELS.read_text())["per_window"]
        counts = Counter(str(row["source_class"]) for row in labels)
        if [int(row["win"]) for row in labels] != list(range(512)):
            raise RuntimeError("label order drift")
        return {"window_contract": {"full512_class_counts": dict(counts)}, "artifacts": {}, "p656_p623_authority": True}

    p.preflight_contract = preflight_contract


def main() -> int:
    sys.path.insert(0, str(ROOT / "code"))
    if sha256(P651_MECHANICS) != PINS["p651_mechanics"] or sha256(P651_READER) != PINS["p651_reader"] or sha256(P632_SCORE) != PINS["p632_score"]:
        raise RuntimeError("pinned P651/P632 source drift")
    p651 = load_module("p656_pinned_p651_mechanics", P651_MECHANICS)
    p651.TASK = TASK
    p651.ROOT = ROOT
    p651.CLAIM = CLAIM
    p651.P632_ROOT = ROOT
    p651.P632_SCORE = P632_SCORE
    p651.P640_ROOT = P640
    p651.P640_HOST = "compute-node-6-local"
    p651.ASSIGNMENT = ASSIGNMENT
    p651.BASE_ASSIGNMENT = BASE_ASSIGNMENT
    p651.BASELINE_PARITY = P623_PARITY
    p651.QTIP_SOURCE = QTIP_SOURCE
    p651.QTIP_KERNEL = QTIP_KERNEL
    p651.QTIP_TLUT = QTIP_TLUT
    p651.PRE_REPAIR_GLOBAL = BASELINE_FULL512
    p651.PREDICTED_MEASURED_GLOBAL = PREDICTED_FULL512
    p651.RAW_WITHOUT = RAW_WITHOUT
    p651.RAW_WITH = RAW_WITH
    p651.current_claim = claim_snapshot
    p651.preflight_manifests = preflight_p653
    p651.install_stream_source = lambda base, manifest, cache, mode: install_remote_stream_source(p651, base, manifest, cache, mode)
    real_load = p651.load_module

    def patched_load(name: str, path: Path):
        module = real_load(name, path)
        if Path(path).resolve() == P632_SCORE.resolve():
            configure_p632(module)
        return module

    p651.load_module = patched_load
    claim_raw_before, _ = claim_snapshot()
    if not P623_PARITY.is_file():
        raise RuntimeError("P623 EARLY_8 parity receipt missing")
    # P651's adapter passes nested progress/sentinel paths to the canonical
    # reader but does not create their parent before the first mmap sentinel.
    (ROOT / "run/early8").mkdir(parents=True, exist_ok=True)
    rc = p651.main()
    if rc:
        return int(rc)
    if "--preflight-only" in sys.argv:
        print(json.dumps({"status": "PASS_P656_PREFLIGHT_ONLY", "p653_manifest_sha256": PINS["p653"], "p623_parity_receipt_sha256": sha256(P623_PARITY)}, sort_keys=True), flush=True)
        return 0
    if claim_snapshot()[0] != claim_raw_before:
        raise RuntimeError("claim changed during P656 physical row")
    raw_path = ROOT / "receipts/RAIL_EARLY8.json"
    raw = json.loads(raw_path.read_text())
    raw_sha = sha256(raw_path)
    final = dict(raw)
    final.update({
        "schema": "p656-p640-p653-physical-early8-v1",
        "status": "PASS_VALIDATED_PRELIMINARY_RECEIPT",
        "task_id": TASK,
        "host": "compute-node-6",
        "source_host": "immutable current-GENESIS base from compute-node-8 over direct QSFP; sealed P653 sparse overlays local on compute-node-6",
        "measurement_label": "PRE_REPAIR_UNDOSED_WIRE / EARLY_8 / PRELIM_NOT_DECISION_GRADE",
        "decision_grade": False,
        "canonical_p651_mechanical_receipt": str(raw_path),
        "canonical_p651_mechanical_receipt_sha256": raw_sha,
        "pre_repair_baseline_receipt": str(P623 / "inputs/SOURCE_FULL512_RECEIPT.json"),
        "pre_repair_baseline_receipt_sha256": PINS["p623_source"],
    })
    final["prediction_vs_measurement"].update({
        "prediction_basis": "P623 sealed current-GENESIS full512 anchor plus P637/P653 raw solver delta; EARLY_8 comparison is preliminary only",
        "measured_pre_repair_baseline_global": BASELINE_FULL512,
        "predicted_measured_global": PREDICTED_FULL512,
        "raw_solver_delta": RAW_WITH - RAW_WITHOUT,
        "decision_grade": False,
    })
    final["instrument"].update({
        "p623_exact_scorer_sha256": PINS["p623_scorer"],
        "p623_source_receipt_sha256": PINS["p623_source"],
        "p651_physical_reader_sha256": PINS["p651_reader"],
        "p651_sparse_mechanics_sha256": PINS["p651_mechanics"],
        "p653_exact_assembled_manifest_sha256": PINS["p653"],
        "final_assignment_sha256": PINS["assignment"],
        "final_assignment_map_sha256": PINS["assignment_map"],
        "exact_wire_bytes": EXACT_WIRE_BYTES,
        "base_transport": "canonical reader QSFP receipt-first layer stream; retired after every layer",
        "overlay_transport": "bounded local four-way sparse copy+SHA256; retired after every layer",
    })
    final_path = ROOT / "receipts/P656_EARLY8.json"
    p651.atomic_json(final_path, final, exclusive=True)
    seal = {
        "schema": "p656-early8-seal-v1", "status": "PASS_SEALED",
        "task_id": TASK, "host": "compute-node-6",
        "label": final["measurement_label"],
        "receipt": str(final_path), "receipt_sha256": sha256(final_path),
        "p623_parity_receipt": str(P623_PARITY), "p623_parity_receipt_sha256": sha256(P623_PARITY),
        "p653_manifest_sha256": PINS["p653"], "created_unix": time.time(),
    }
    seal_path = ROOT / "receipts/P656_EARLY8_SEAL.json"
    p651.atomic_json(seal_path, seal, exclusive=True)
    print(json.dumps({
        "status": final["status"], "label": final["measurement_label"],
        "global": final["global"]["mean"],
        "paired_delta": final["matched_delta_vs_measured_pre_repair"]["global"]["candidate_minus_pre_repair_mean"],
        "predicted_full512": PREDICTED_FULL512,
        "receipt": str(final_path), "receipt_sha256": sha256(final_path),
        "seal": str(seal_path), "seal_sha256": sha256(seal_path),
    }, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        failure = ROOT / f"receipts/P656_EARLY8_FAILURE_{int(time.time())}.json"
        try:
            tmp = failure.with_name(f".{failure.name}.{os.getpid()}.tmp")
            tmp.write_text(json.dumps({"schema": "p656-early8-failure-v1", "status": "FAIL_CLOSED", "task_id": TASK, "error_type": type(exc).__name__, "error": str(exc), "created_unix": time.time()}, indent=2, sort_keys=True) + "\n")
            os.replace(tmp, failure)
        finally:
            raise
