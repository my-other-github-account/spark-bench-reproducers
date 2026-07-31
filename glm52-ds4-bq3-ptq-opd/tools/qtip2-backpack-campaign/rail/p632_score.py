#!/usr/bin/env python3
"""Exact P600 whole-wire scorer with P623 ordered-prefix measurements.

The immutable physical base is read through the exact released parent reader and
canonical builder.  The only mutable candidate payload copied across hosts is a
sidecar-authenticated BASIC checkpoint.  Its codebook masters are converted to
exact fp16 wire bytes in retired scratch, its RMSNorm masters replace only the
235 safetensors reads, and its 43 bounded output gains are installed with the
trainer's exact BF16 hook semantics.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch
import triton

from checkpoint_state import (
    CONTAMINATED_IDS,
    EXPECTED_CLEAN72_IDS,
    EXPECTED_CLEAN72_SHA256,
    EXPECTED_CODE76_IDS,
    GAIN_CLAMP,
    NATURAL_UPDATES,
    SCORABLE_UPDATES,
    OverlaySafeOpen,
    expected_norm_keys,
    expected_output_keys,
    load_checkpoint,
    sha256_file,
)

TASK = "PUBLIC_TASK"
ROOT = Path("$HOME/run-bundles/P632_DIRECTIONAL_PUBLIC_TASK_s7")
CLAIM = Path("$HOME/HOST_CLAIM.json")
SOURCE_HOST = "compute-node-8-local"
SOURCE_TASK = "PUBLIC_TASK"
SOURCE_ROOT = Path("$HOME/run-bundles/P600_CLASS_REWEIGHTED_PUBLIC_TASK_s8_attempt2")
# P602/P625 copied this exact 43-layer receipt/artifact surface to compute-node-7.
REMOTE_PACKAGE = "$HOME/run-bundles/P602_CONTAINER_PUBLIC_TASK_s7/pack/planes"
PARENT = ROOT
PACKAGE = ROOT / "code/eval_package"
MODEL = Path("$HOME/models/hf/DeepSeek-V4-Flash")
TEACHER = Path("$HOME/run-bundles/DS4_TEACHER")
CORPUS = TEACHER / "static/windows_ds4_eval.json"
TRAIN_CORPUS = ROOT / "inputs/windows_ds4_TRAIN.json"
TRAIN_REFS = ROOT / "inputs/TRAIN_REFS"
INPUTS = ROOT / "inputs"
SOURCE_MANIFEST = INPUTS / "SOURCE_MANIFEST.json"
TEACHER_MANIFEST = INPUTS / "TEACHER_MANIFEST.json"
MODEL_MANIFEST = INPUTS / "MODEL_MANIFEST.json"
WINDOW_CONTRACT = INPUTS / "WINDOW_CONTRACT.json"
COMPACT_MANIFEST = INPUTS / "BANANA_SMASHER_COMPACT_FANIN.json"
ASSIGNMENT = INPUTS / "NOMINATED_ASSIGNMENT.json"
LABELS = INPUTS / "BQ3_STEP0_PER_CLASS.json"
PHYSICAL_MARKER = INPUTS / "PHYSICAL_CODE76.json"
PHYSICAL_PASS_MARKER = INPUTS / "PHYSICAL_CODE76.PASS.json"
WIRE_MANIFEST = INPUTS / "WIRE_43_MANIFEST.json"
BASELINE_FULL512 = INPUTS / "PRE_REPAIR_FULL512.json"
PREPARE_RECEIPT = ROOT / "receipts/PREPARE_MISSION.json"
SOURCE_RELEASE_INPUT = ROOT / "inputs/P600_HOST_CLAIM.json"
SOURCE_RELEASE_SHA256 = "a104adfe0f3e9171b3bf14e12ff6c6e6d0bf2b87eeaf5064a91c50b6c4da5ac7"
CANONICAL_READER = ROOT / "code/banana_smasher_remote_full512.py"
CANONICAL_BUILDER = PACKAGE / "t8192_ds4_build_v3.py"
CANONICAL_DELTA = PACKAGE / "delta_pack_sources.py"
CANONICAL_LP4_PACK = PACKAGE / "lp4_pack.py"
CANONICAL_PLANES_UNPACK = PACKAGE / "planes_unpack.py"
CANONICAL_EVAL_CONTRACTS = PACKAGE / "readapt_eval_contracts.py"
CANONICAL_SAFETY = ROOT / "code/full512_safety.py"
LOADER_SOURCE = ROOT / "code/rail_loading.py"
CANONICAL_SHA256 = {
    CANONICAL_READER: "bc0920b8865376463e58d11686e888524122b9bc995668fca23fa1ec24312b42",
    CANONICAL_BUILDER: "d56677ed63711aac24181463d7ef8ac45499c4b507919b3ad4d5dcb63da205bb",
    CANONICAL_DELTA: "2aeed7527631050ad440a52fe796502ff01dcd98096f86dd20e8ca9e9187625f",
    CANONICAL_LP4_PACK: "7a8e48547824a87a48db4c7142ec53f73303a91ce6a0c95cf1a88b1b87d22350",
    CANONICAL_PLANES_UNPACK: "aeb3e473a00b48426f56b9f80aefc6bc086b7791ec2372606c724e90db126334",
    CANONICAL_EVAL_CONTRACTS: "0842784bfba78032f122c8e859f2a1df1d67885823e1aa323cc020d3ae6fccbf",
    CANONICAL_SAFETY: "52a9d550b1a02fed2a524f00a66e85dee5653464ed2eb1b0ab597cb4873d9519",
    LOADER_SOURCE: "155310d1e6701d6cb2d1c04558514366a2304cb2a8d6d26402ed7c800b8b6c89",
}
EXPECTED_INPUT_SHA256 = {
    COMPACT_MANIFEST: "d9421f1f6d0e696608bb0ce9b09131e63790c18e9cd536e440b1884b727db00d",
    ASSIGNMENT: "c9fb72e2bf7416ef48f33df229f9a3b5b5dd4f9e9b35a610d83fb1c49f4a050d",
    LABELS: "5a49b0d92cf7f1c403b2d6bb49487c6d97f273211d6b1c68efb27782a8a20a88",
    PHYSICAL_MARKER: "8735f6047d22582ee64f774a9081d25c5a7a50ee5ef25e3911ea401df885ba95",
    PHYSICAL_PASS_MARKER: "98e3a621f0de12db7c78c72c0ceeaa17daff10e1d13ed68130c551521987b6ca",
    WIRE_MANIFEST: "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755",
    BASELINE_FULL512: "4823f90fb1df1f7d0351c585954799f1fd69a98f3709cfcb721f1c8e0183ef0c",
    WINDOW_CONTRACT: "91a33069d7d2f5648d63ef10b4a11eb122dbce740eec2ac9acd0bc202325fbad",
}
ASSIGNMENT_SHA = EXPECTED_INPUT_SHA256[ASSIGNMENT]
COMPACT_SHA = EXPECTED_INPUT_SHA256[COMPACT_MANIFEST]
WIRE_SHA = EXPECTED_INPUT_SHA256[WIRE_MANIFEST]
MODEL_INDEX_SHA = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
TEACHER_DONE_SHA = "6338af84f907a26dfdf0f784edc322aa672738542ed884b70e4d9b6e96aa33b0"
CORPUS_SHA = "5aadaacbb486ae4f528c5e51ae70beff863337bd908fc727e6e49fc3ac520ebd"
TRAIN_CORPUS_SHA = "16575db7fd180ca193aa13c4e642400b9ed416dbd0c36c3c5302422b31f5cbae"
TRAIN_DONE_SHA = "e7dc46a2069386a372f7b5f1f1ed78c3a5335ea635a4f58264e44065cdadfde0"
DISK_FLOOR = 6 * (1 << 30)
HELDOUT_PREFIX_COUNTS = {"early8": 8, "interim64": 64, "full512": 512}
FROZEN_CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")
CURRENT_BANANA_SMASHER_GLOBAL = 0.08394998423027422
CURRENT_BANANA_SMASHER_CODE = 0.0417040064907229


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def atomic_bytes(path: Path, payload: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_json(path: Path, value: object, *, exclusive: bool = False) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), exclusive=exclusive)


@contextmanager
def force_weights_only_torch_loads():
    """Force every canonical-reader/builder torch.load through the safe unpickler."""
    original = torch.load
    stats = {"calls": 0, "explicit_false_overridden": 0, "all_weights_only": True}

    def guarded(*args, **kwargs):
        stats["calls"] += 1
        if kwargs.get("weights_only") is False:
            stats["explicit_false_overridden"] += 1
        kwargs["weights_only"] = True
        return original(*args, **kwargs)

    torch.load = guarded
    try:
        yield stats
    finally:
        torch.load = original


def validate_artifact_manifests(mode: str) -> dict[str, Any]:
    source = json.loads(SOURCE_MANIFEST.read_text())
    if source.get("schema") != "banana_smasher-repair-second-host-source-manifest-v2":
        raise RuntimeError("source manifest schema drift")
    teacher = json.loads(TEACHER_MANIFEST.read_text())
    model = json.loads(MODEL_MANIFEST.read_text())
    if (
        source.get("teacher", {}).get("manifest_sha256") != sha256_file(TEACHER_MANIFEST)
        or source.get("model", {}).get("manifest_sha256") != sha256_file(MODEL_MANIFEST)
    ):
        raise RuntimeError("artifact manifest binding drift")
    teacher_rows = teacher.get("rows")
    if (
        teacher.get("schema") != "banana_smasher-repair-teacher-manifest-v1"
        or teacher.get("status") != "PASS_FULL_CONTENT_REHASH"
        or not isinstance(teacher_rows, list)
        or [row.get("win") for row in teacher_rows] != list(range(512))
    ):
        raise RuntimeError("teacher manifest coverage drift")
    selected = list(EXPECTED_CODE76_IDS) if mode == "fast" else list(range(HELDOUT_PREFIX_COUNTS.get(mode, 512)))
    if mode == "train":
        selected = []
    selected_rows = []
    for win in selected:
        row = teacher_rows[win]
        path = TEACHER / "t8192_eval" / str(row.get("file"))
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
            raise RuntimeError(f"teacher file size drift win={win}")
        actual_sha = sha256_file(path)
        if actual_sha != row.get("sha256"):
            raise RuntimeError(f"teacher file SHA-256 drift win={win}")
        selected_rows.append({"win": win, "bytes": path.stat().st_size, "sha256": actual_sha})
    train_rows = []
    if mode == "train":
        if sha256_file(TRAIN_CORPUS) != TRAIN_CORPUS_SHA or sha256_file(TRAIN_REFS / "DONE.jsonl") != TRAIN_DONE_SHA:
            raise RuntimeError("sealed TRAIN corpus/reference ledger drift")
        ledger = [json.loads(line) for line in (TRAIN_REFS / "DONE.jsonl").read_text().splitlines() if line.strip()]
        if [row.get("win") for row in ledger] != list(range(8)):
            raise RuntimeError("TRAIN reference ordered coverage drift")
        for win, row in enumerate(ledger):
            path = TRAIN_REFS / f"t8192_win{win}.pt"
            actual_sha = sha256_file(path)
            if (
                row.get("file") != path.name
                or int(row.get("size", -1)) != path.stat().st_size
                or row.get("sha256") != actual_sha
            ):
                raise RuntimeError(f"TRAIN reference identity drift win={win}")
            train_rows.append({"win": win, "bytes": path.stat().st_size, "sha256": actual_sha})
    model_rows = model.get("rows")
    if (
        model.get("schema") != "banana_smasher-repair-model-manifest-v1"
        or model.get("status") != "PASS_INDEX_AND_FILE_SURFACE"
        or not isinstance(model_rows, list)
        or len(model_rows) != 46
    ):
        raise RuntimeError("model manifest coverage drift")
    for row in model_rows:
        path = MODEL / str(row.get("name"))
        if not path.is_file() or path.stat().st_size != int(row.get("bytes", -1)):
            raise RuntimeError(f"model shard file surface drift: {path}")
    return {
        "teacher_manifest_sha256": sha256_file(TEACHER_MANIFEST),
        "teacher_selected_file_count": len(selected_rows),
        "teacher_selected_total_bytes": sum(row["bytes"] for row in selected_rows),
        "teacher_selected_sha256_set": canonical_json_sha256(selected_rows),
        "train_selected_file_count": len(train_rows),
        "train_selected_total_bytes": sum(row["bytes"] for row in train_rows),
        "train_selected_sha256_set": canonical_json_sha256(train_rows),
        "model_manifest_sha256": sha256_file(MODEL_MANIFEST),
        "model_file_count": len(model_rows),
        "model_total_bytes": sum(int(row["bytes"]) for row in model_rows),
    }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_claim() -> tuple[bytes, dict[str, Any]]:
    """Preserve the force-ordered, parked QTIP claim byte-for-byte.

    P632 is explicitly scoped to compute-node-7 after P602/P625 used the same idle
    host, but it must not steal or rewrite the resumable QTIP claim.  The GPU
    emptiness check is separate and mandatory immediately after this check.
    """
    raw = CLAIM.read_bytes()
    claim = json.loads(raw)
    exact = {
        "schema": "qtip-anchor-wire-host-claim-v1",
        "host": "compute-node-7",
        "owner": "PUBLIC_TASK",
        "task": "PUBLIC_TASK",
        "task_id": "PUBLIC_TASK",
        "mission": "$HOME/run-bundles/QTIP_ANCHOR_WIRE_PUBLIC_TASK_s7",
    }
    drift = {key: (claim.get(key), expected) for key, expected in exact.items() if claim.get(key) != expected}
    if drift:
        raise RuntimeError(f"preserved local QTIP claim drift: {drift}")
    if time.time() > float(claim.get("expected_release_epoch", 0)):
        raise RuntimeError("preserved local QTIP claim expired")
    return raw, claim


def source_claim(expected_sha: str) -> tuple[bytes, dict[str, Any]]:
    raw = SOURCE_RELEASE_INPUT.read_bytes()
    if sha256_file(SOURCE_RELEASE_INPUT) != SOURCE_RELEASE_SHA256:
        raise RuntimeError("sealed P487 release input drift")
    current = json.loads(raw)
    if hashlib.sha256(raw).hexdigest() == expected_sha:
        claim = current
    elif (
        current.get("schema") == "banana_smasher-seams-basic-repair-host-claim-v1"
        and current.get("host") == "compute-node-8"
        and current.get("owner") == SOURCE_TASK
        and current.get("task") == SOURCE_TASK
        and current.get("task_id") == SOURCE_TASK
        and current.get("mission") == str(SOURCE_ROOT)
    ):
        # UPDATE_000 and UPDATE_006 retain predecessor producer-claim hashes.
        # Their immutable transfer/lineage receipts authenticate the payload;
        # keep the live P487 claim byte-stable during this read.
        claim = current
    else:
        if (
            current.get("owner") != "UNCLAIMED"
            or current.get("released_from") != SOURCE_TASK
            or current.get("previous_claim_sha256") != expected_sha
            or not isinstance(current.get("previous_claim"), dict)
        ):
            raise RuntimeError("source claim does not match checkpoint identity or exact source-task release")
        claim = current["previous_claim"]
    exact = {
        "schema": "banana_smasher-seams-basic-repair-host-claim-v1",
        "host": "compute-node-8",
        "owner": claim.get("owner"),
        "task": claim.get("task"),
        "task_id": claim.get("task_id"),
        "mission": claim.get("mission"),
    }
    drift = {key: (claim.get(key), expected) for key, expected in exact.items() if claim.get(key) != expected}
    if drift:
        raise RuntimeError(f"source claim drift: {drift}")
    return raw, current


def gpu_snapshot(*, own_pid: int | None = None, require_zero_util: bool = False) -> dict[str, Any]:
    apps_raw = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    util_raw = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    apps = []
    for line in apps_raw.splitlines():
        if not line.strip():
            continue
        pid_text = line.split(",", 1)[0].strip()
        if own_pid is not None and pid_text == str(own_pid):
            continue
        apps.append(line.strip())
    util = [[int(value.strip()) for value in row.split(",")] for row in util_raw.splitlines() if row.strip()]
    if apps:
        raise RuntimeError(f"foreign GPU applications present: {apps}")
    if require_zero_util and (util != [[0, 0]]):
        raise RuntimeError(f"GPU utilization is not zero: {util}")
    return {"foreign_compute_apps": apps, "utilization_gpu_memory_percent": util}


def preflight_contract(mode: str) -> dict[str, Any]:
    for path, expected_sha in {**CANONICAL_SHA256, **EXPECTED_INPUT_SHA256}.items():
        if not path.is_file() or sha256_file(path) != expected_sha:
            raise RuntimeError(f"sealed source/input drift: {path}")
    if sha256_file(MODEL / "model.safetensors.index.json") != MODEL_INDEX_SHA:
        raise RuntimeError("model index drift")
    if sha256_file(TEACHER / "t8192_eval/DONE.jsonl") != TEACHER_DONE_SHA:
        raise RuntimeError("teacher DONE drift")
    if sha256_file(CORPUS) != CORPUS_SHA:
        raise RuntimeError("corpus drift")
    if not SOURCE_MANIFEST.is_file() or not PREPARE_RECEIPT.is_file():
        raise RuntimeError("sealed source manifest/prepare receipt missing")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text())
    prepare = json.loads(PREPARE_RECEIPT.read_text())
    if (
        source_manifest.get("schema") != "banana_smasher-repair-second-host-source-manifest-v2"
        or source_manifest.get("status") != "PASS_SEALED_LOCAL_INPUTS"
        or prepare.get("schema") != "banana_smasher-repair-second-host-prepare-receipt-v2"
        or prepare.get("status") != "PASS"
    ):
        raise RuntimeError("source preparation status drift")
    if prepare.get("source_manifest_sha256") != sha256_file(SOURCE_MANIFEST):
        raise RuntimeError("source manifest binding drift")
    contract = json.loads(WINDOW_CONTRACT.read_text())
    if (
        contract.get("schema") != "repair-rail-window-contract-v1"
        or contract.get("status") != "PASS"
        or tuple(contract.get("code76", ())) != EXPECTED_CODE76_IDS
        or contract.get("full512") != list(range(512))
        or contract.get("selection_source_sha256") != EXPECTED_INPUT_SHA256[LABELS]
        or contract.get("source_corpus_sha256") != CORPUS_SHA
    ):
        raise RuntimeError("window contract drift")
    if mode not in {"train", "fast", *HELDOUT_PREFIX_COUNTS}:
        raise RuntimeError(f"unsupported scoring mode {mode}")
    implementation = source_manifest.get("scorer_implementation", {})
    implementation_rows = implementation.get("rows")
    if not isinstance(implementation_rows, list) or len(implementation_rows) != 6:
        raise RuntimeError("persisted scorer implementation manifest drift")
    if implementation.get("rows_sha256") != canonical_json_sha256(implementation_rows):
        raise RuntimeError("persisted scorer implementation row-set drift")
    for row in implementation_rows:
        path = Path(str(row.get("path", "")))
        if (
            not path.is_file()
            or path.stat().st_size != int(row.get("bytes", -1))
            or sha256_file(path) != row.get("sha256")
        ):
            raise RuntimeError(f"persisted scorer implementation changed after seal: {path}")
    artifacts = validate_artifact_manifests(mode)
    return {"source_manifest": source_manifest, "window_contract": contract, "artifacts": artifacts}


def configure_parent_module(base: Any, *, cache: Path, progress: Path, sentinel: Path) -> None:
    base.TASK = TASK
    base.ROOT = ROOT
    base.CLAIM = CLAIM
    base.SOURCE_HOST = SOURCE_HOST
    base.REMOTE_PACKAGE = REMOTE_PACKAGE
    base.MODEL = MODEL
    base.PACKAGE = PACKAGE
    base.TEACHER = TEACHER
    base.CORPUS = CORPUS
    base.LABELS = LABELS
    base.COMPACT_MANIFEST = COMPACT_MANIFEST
    base.ASSIGNMENT = ASSIGNMENT
    base.PHYSICAL_MARKER = PHYSICAL_MARKER
    base.PHYSICAL_PASS_MARKER = PHYSICAL_PASS_MARKER
    base.WIRE_MANIFEST = WIRE_MANIFEST
    base.CACHE = cache
    base.PHYSICAL_PACKAGE = cache
    base.PROGRESS = progress
    base.LOADER_SOURCE = LOADER_SOURCE
    base.LOADER_SHA = CANONICAL_SHA256[LOADER_SOURCE]
    base.LOADER_SENTINEL = sentinel
    base.RESUME_AUTHORIZATION = ROOT / "run/NO_RESUME_AUTHORIZATION_FOR_CHECKPOINT_SCORER"
    base.DISALLOW_RESUME = True
    base.np = np
    base.torch = torch
    base.triton = triton
    base.current_claim = current_claim


def install_environment() -> dict[str, str]:
    contract = {
        "TWOBIN_KLD_STREAM_OUT": "1",
        "TWOBIN_ATTN_IMPL": "eager",
        "TWOBIN_LAYER_OVERLAP": "0",
        "ARM4_LOADER_MODE": "torch-mmap",
        "ARM4_LOADER_FALLBACK": "torch-eager",
        "FULL512_LOADER_SHA256": CANONICAL_SHA256[LOADER_SOURCE],
        "FULL512_LOADER_INPUT_SHA256": WIRE_SHA,
        "FULL512_TASK_ID": TASK,
        "FULLMENU_ASSEMBLY_BATCH": "8",
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDA_MODULE_LOADING": "EAGER",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    # The released resume builder's in-band chunk proof is resume-specific
    # (chunks 1..7). This scorer independently validates fresh 0-based coverage,
    # so those four optional variables must remain unset.
    for name in (
        "FULL512_LOADER_PROGRESS_PATH", "FULL512_LOADER_SENTINEL_PATH",
        "TWOBIN_TEACHER_CACHE", "TWOBIN_STATELESS_CACHE", "TWOBIN_REUSE_MASK",
        "TWOBIN_PREFIX_LOGITS", "TWOBIN_STREAMS",
    ):
        os.environ.pop(name, None)
    os.environ.update(contract)
    return contract


def install_overlay(
    base: Any,
    builder: Any,
    state: Mapping[str, Any],
    *,
    update: int,
) -> tuple[type, dict[str, Any]]:
    codebook_rows: list[dict[str, Any]] = []
    codebook_unique: dict[tuple[int, str], dict[str, Any]] = {}
    norm_seen: set[str] = set()
    output_seen: set[str] = set()

    class CheckpointTierSource(base.BananaSmasherTierSource):
        def _stage_remote(self, layer: int, row: dict) -> Path:
            # Source-local relocation of the canonical receipt-first allowlist stage.
            # The staged bytes and downstream torch-mmap numerical path are unchanged.
            self._cleanup_stage()
            stage = self.CACHE / f"layer_{layer:03d}" if hasattr(self, "CACHE") else base.CACHE / f"layer_{layer:03d}"
            cache_root = stage.parent
            temp = cache_root / f".layer_{layer:03d}.partial"
            if stage.exists():
                shutil.rmtree(stage)
            if temp.exists():
                shutil.rmtree(temp)
            temp.mkdir(parents=True)
            wire_row = self.wire_rows[layer]
            required_bytes = int(wire_row["physical_wire_bytes"])
            free = shutil.disk_usage(cache_root).free
            if free - required_bytes < DISK_FLOOR:
                raise RuntimeError(f"disk floor before source-local stage L{layer}: free={free} required={required_bytes}")
            source_layer = Path(REMOTE_PACKAGE) / f"layer_{layer:03d}"
            receipt_path = temp / "LAYER_RECEIPT.json"
            shutil.copy2(source_layer / "LAYER_RECEIPT.json", receipt_path)
            if sha256_file(receipt_path) != wire_row["receipt_sha256"]:
                raise RuntimeError(f"L{layer} receipt SHA drift after source-local stage")
            receipt = json.loads(receipt_path.read_text())
            if (
                receipt.get("schema") != "banana_smasher-materialized-layer-v1"
                or receipt.get("status") != "PASS"
                or int(receipt.get("layer", -1)) != layer
                or receipt.get("assignment_sha256") != ASSIGNMENT_SHA
                or receipt.get("builder_sha256") != base.BUILD_BUILDER_SHA
                or int(receipt.get("physical_wire_bytes", -1)) != required_bytes
            ):
                raise RuntimeError(f"L{layer} receipt identity drift")
            receipt_files = receipt.get("files")
            if not isinstance(receipt_files, list) or not receipt_files:
                raise RuntimeError(f"L{layer} receipt file list missing")
            allowed_paths = []
            for item in receipt_files:
                rel = item.get("path")
                rel_path = Path(rel) if isinstance(rel, str) else Path("/")
                if not isinstance(rel, str) or rel_path.is_absolute() or ".." in rel_path.parts or rel in allowed_paths:
                    raise RuntimeError(f"L{layer} receipt path unsafe or duplicate: {rel}")
                allowed_paths.append(rel)
                target = temp / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_layer / rel, target)
            stage_accounting = base.validate_staged_layer(
                temp, receipt, required_bytes=required_bytes,
                free_bytes_after=shutil.disk_usage(cache_root).free, floor=DISK_FLOOR,
            )
            os.replace(temp, stage)
            self.active_stage = stage
            atomic_json(ROOT / f"run/LAYER_{layer:03d}_STAGE.json", {
                "schema": "banana_smasher-full512-layer-stage-v2", "task_id": TASK, "layer": layer,
                "receipt_sha256": wire_row["receipt_sha256"], **stage_accounting,
                "allowed_files": allowed_paths, "stage": str(stage),
                "transport": "source-local receipt-first allowlist copy; canonical torch-mmap path unchanged",
                "created_unix": time.time(),
            })
            saved = state["codebooks"][f"L{layer}"]
            visit_rows = []
            for name, master in sorted(saved.items()):
                tier, projection = name.rsplit("__", 1)
                suffix = "fused13" if projection == "13" else "down"
                path = stage / f"{tier}.{suffix}.codebook.fp16.bin"
                if not path.is_file():
                    raise RuntimeError(f"physical codebook target missing L{layer}: {path.name}")
                before_bytes = path.read_bytes()
                before_sha = hashlib.sha256(before_bytes).hexdigest()
                wire = master.to(torch.float16).contiguous().numpy().tobytes(order="C")
                if len(wire) != len(before_bytes):
                    raise RuntimeError(f"codebook wire byte drift L{layer}/{name}")
                atomic_bytes(path, wire)
                after_sha = sha256_file(path)
                if after_sha != hashlib.sha256(wire).hexdigest():
                    raise RuntimeError(f"codebook overlay readback drift L{layer}/{name}")
                item = {
                    "layer": layer,
                    "name": name,
                    "path": path.name,
                    "bytes": len(wire),
                    "base_sha256": before_sha,
                    "checkpoint_wire_sha256": after_sha,
                    "changed": before_sha != after_sha,
                }
                visit_rows.append(item)
                prior = codebook_unique.setdefault((layer, name), item)
                if prior != item:
                    raise RuntimeError(f"non-deterministic codebook overlay L{layer}/{name}")
            codebook_rows.append({"layer": layer, "overlays": visit_rows})
            return stage

    real_safe_open = None
    import safetensors
    real_safe_open = safetensors.safe_open

    def safe_open_overlay(*args, **kwargs):
        return OverlaySafeOpen(real_safe_open(*args, **kwargs), state["norms"], norm_seen)

    safetensors.safe_open = safe_open_overlay
    original_materialize = builder.materialize_layer

    def materialize_with_gain(model, layer, layer_state, config):
        module = original_materialize(model, layer, layer_state, config)
        key = f"model.layers.{layer}.self_attn.o_b_proj.output_log_gain"
        gain_master = state["outputs"][key]
        projection = module.self_attn.o_b_proj
        if not hasattr(projection, "_banana_smasher_repair_gain_hook"):
            captured = gain_master.detach().clone()

            def output_gain_hook(_module, _inputs, output, *, log_gain=captured):
                gain = torch.exp(log_gain.to(output.device).clamp(-GAIN_CLAMP, GAIN_CLAMP)).to(output.dtype)
                return output * gain

            projection._banana_smasher_repair_gain_hook = projection.register_forward_hook(output_gain_hook)
            projection._banana_smasher_repair_gain_value = float(gain_master)
        elif projection._banana_smasher_repair_gain_value != float(gain_master):
            raise RuntimeError(f"output gain changed during run L{layer}")
        output_seen.add(key)
        return module

    builder.materialize_layer = materialize_with_gain
    overlay = {
        "codebook_rows": codebook_rows,
        "codebook_unique": codebook_unique,
        "norm_seen": norm_seen,
        "output_seen": output_seen,
        "restore": lambda: (
            setattr(safetensors, "safe_open", real_safe_open),
            setattr(builder, "materialize_layer", original_materialize),
        ),
    }
    return CheckpointTierSource, overlay


def reduce_outputs(out: Path, wins: list[int], classes: Mapping[int, str]) -> dict[str, Any]:
    done_rows = [json.loads(line) for line in (out / "DONE.jsonl").read_text().splitlines() if line.strip()]
    if [row.get("win") for row in done_rows] != wins:
        raise RuntimeError("output DONE ledger order/coverage drift")
    rows = []
    values_by_win: dict[int, torch.Tensor] = {}
    for win in wins:
        path = out / f"kld_win{win}.pt"
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        tensor = payload.get("kld")
        if payload.get("win") != win or payload.get("support") != 8192 or payload.get("cutoff") != 1024:
            raise RuntimeError(f"KLD output identity drift win={win}")
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (1024,) or not bool(torch.isfinite(tensor).all()) or bool((tensor < -1e-6).any()):
            raise RuntimeError(f"KLD output tensor drift win={win}")
        tensor64 = tensor.double()
        values_by_win[win] = tensor64
        rows.append({
            "win": win,
            "source_class": classes[win],
            "mean": float(tensor64.mean()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    joined = torch.cat([values_by_win[win] for win in wins])
    if not bool(torch.isfinite(joined).all()):
        raise RuntimeError("non-finite reduced KLD surface")

    def summary(selected: list[int], label: str) -> dict[str, Any]:
        vals = torch.cat([values_by_win[win] for win in selected])
        window_means = [float(values_by_win[win].mean()) for win in selected]
        se = statistics.stdev(window_means) / math.sqrt(len(window_means)) if len(window_means) > 1 else 0.0
        mean = float(vals.mean())
        return {
            "source_class": label,
            "mean": mean,
            "n_windows": len(selected),
            "n_positions": int(vals.numel()),
            "window_mean_se": se,
            "window_mean_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        }

    by_class = {}
    for label in sorted(set(classes.values())):
        selected = [win for win in wins if classes[win] == label]
        if selected:
            by_class[label] = summary(selected, label)
    output_set = [{"win": row["win"], "sha256": row["sha256"]} for row in rows]
    return {
        "global": summary(wins, "global"),
        "by_class": by_class,
        "per_window": rows,
        "values_by_win": values_by_win,
        "window_output_set_sha256": canonical_json_sha256(output_set),
    }


def paired_delta(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]], selected: list[int], label: str) -> dict[str, Any]:
    candidate = {int(row["win"]): float(row["mean"]) for row in candidate_rows}
    baseline = {int(row["win"]): float(row["mean"]) for row in baseline_rows}
    if not set(selected).issubset(candidate) or not set(selected).issubset(baseline):
        raise RuntimeError(f"paired delta surface missing for {label}")
    deltas = [candidate[win] - baseline[win] for win in selected]
    mean = statistics.fmean(deltas)
    se = statistics.stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    return {
        "label": label,
        "candidate_minus_pre_repair_mean": mean,
        "window_mean_se": se,
        "window_mean_ci95": [mean - 1.96 * se, mean + 1.96 * se],
        "n_windows": len(selected),
        "window_ids": selected,
    }


def validate_update0_same_host_baseline(
    candidate_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    candidate = {int(row["win"]): float(row["mean"]) for row in candidate_rows}
    baseline = {int(row["win"]): float(row["mean"]) for row in baseline_rows}
    expected = list(EXPECTED_CODE76_IDS)
    if sorted(candidate) != sorted(expected) or not set(expected).issubset(baseline):
        raise RuntimeError("update-0 same-host baseline window surface drift")
    deltas = [candidate[win] - baseline[win] for win in expected]
    if not all(math.isfinite(value) for value in deltas):
        raise RuntimeError("update-0 same-host baseline non-finite delta")
    maximum = max(abs(value) for value in deltas)
    if maximum > 1e-12:
        raise RuntimeError(f"update-0 same-host instrument identity mismatch max_abs={maximum}")
    return {
        "status": "PASS_EXACT_SAME_HOST_INSTRUMENT",
        "baseline_receipt": str(BASELINE_FULL512),
        "baseline_receipt_sha256": EXPECTED_INPUT_SHA256[BASELINE_FULL512],
        "windows": len(expected),
        "maximum_absolute_window_mean_delta": maximum,
        "mean_window_delta": statistics.fmean(deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", required=True, type=int)
    parser.add_argument(
        "--mode", choices=("train", "fast", *HELDOUT_PREFIX_COUNTS), default="early8"
    )
    args = parser.parse_args()
    update = args.update
    mode = args.mode
    if update not in SCORABLE_UPDATES:
        raise RuntimeError(f"not the frozen P600 dose-2 update: {update}")
    if mode == "full512" and update == 0:
        raise RuntimeError("terminal full512 is only valid for a mechanically selected trained checkpoint")
    started = time.time()
    sealed = preflight_contract(mode)
    claim_raw_before, _claim = current_claim()
    claim_sha = hashlib.sha256(claim_raw_before).hexdigest()
    gpu_before = gpu_snapshot(require_zero_util=True)
    if shutil.disk_usage(ROOT).free < DISK_FLOOR:
        raise RuntimeError("disk free below scorer floor")

    matches = sorted(ROOT.glob(f"checkpoints/UPDATE_{update:03d}_*.pt"))
    matches = [path for path in matches if path.name.count("_") == 2]
    if len(matches) != 1:
        raise RuntimeError(f"expected one collision-safe checkpoint for update {update}, found {matches}")
    checkpoint = matches[0]
    checkpoint_sha = checkpoint.stem.rsplit("_", 1)[1]
    sidecar_path = ROOT / f"checkpoints/UPDATE_{update:03d}_{checkpoint_sha}.source.json"
    transfer_path = ROOT / f"receipts/TRANSFER_UPDATE_{update:03d}_{checkpoint_sha}.json"
    if not sidecar_path.is_file() or not transfer_path.is_file():
        raise RuntimeError("authenticated checkpoint transfer artifacts missing")
    transfer = json.loads(transfer_path.read_text())
    if transfer.get("status") != "PASS_AUTHENTICATED_CHECKPOINT_BYTES" or transfer.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError("checkpoint transfer receipt drift")
    state, sidecar = load_checkpoint(checkpoint, sidecar_path, update)
    source_claim_sha = sidecar["identity"]["claim_sha256"]
    source_claim_raw_before, _source_claim = source_claim(source_claim_sha)

    run_id = f"U{update:03d}_{checkpoint_sha}_{mode}"
    out = ROOT / f"out/{run_id}"
    receipt_path = ROOT / f"receipts/SCORE_{run_id}.json"
    cache = ROOT / f"scratch/{run_id}"
    progress = ROOT / f"run/PROGRESS_{run_id}.json"
    sentinel = ROOT / f"run/LOADER_SENTINEL_{run_id}.json"
    if receipt_path.exists() or out.exists() or cache.exists() or progress.exists() or sentinel.exists():
        raise RuntimeError(f"once-only score target already exists: {run_id}")
    cache.mkdir(parents=True)
    progress.parent.mkdir(parents=True, exist_ok=True)

    base = load_module(f"canonical_parent_full512_{run_id}", CANONICAL_READER)
    configure_parent_module(base, cache=cache, progress=progress, sentinel=sentinel)
    env_contract = install_environment()
    sys.path.insert(0, str(PACKAGE))
    import t8192_ds4_build_v3 as builder
    CheckpointTierSource, overlay = install_overlay(base, builder, state, update=update)
    builder.PlaneSource = CheckpointTierSource
    labels_payload = json.loads(LABELS.read_text())
    classes = {int(row["win"]): str(row["source_class"]) for row in labels_payload["per_window"]}
    counts = Counter(classes.values())
    if set(classes) != set(range(512)) or counts != Counter(sealed["window_contract"]["full512_class_counts"]):
        raise RuntimeError("class label surface drift")
    if mode == "train":
        wins = list(range(8))
        classes = {win: "train" for win in wins}
        counts = Counter(classes.values())
        chunk = 8
        reference_dir = TRAIN_REFS
        corpus_path = TRAIN_CORPUS
    else:
        wins = list(EXPECTED_CODE76_IDS) if mode == "fast" else list(range(HELDOUT_PREFIX_COUNTS[mode]))
        chunk = len(wins)
        reference_dir = TEACHER / "t8192_eval"
        corpus_path = CORPUS
    out.mkdir(parents=True)
    original_argv = sys.argv
    original_cwd = Path.cwd()
    rc = -1
    try:
        sys.argv = [
            "t8192_ds4_build_v3.py", "--mode", "planes",
            "--planes-dir", str(COMPACT_MANIFEST),
            "--ref-dir", str(reference_dir),
            "--corpus", str(corpus_path),
            "--meta-dir", str(MODEL), "--local-dir", str(MODEL),
            "--out", str(out), "--cand-pos-limit", "1024",
            "--count", str(len(wins)), "--chunk", str(chunk), "--mb", "2",
            "--windows", ",".join(map(str, wins)),
            "--tag", f"BANANA_SMASHER_REPAIR_CHECKPOINT_{run_id}",
        ]
        os.chdir(TEACHER)
        with force_weights_only_torch_loads() as weights_only_stats:
            rc = int(builder.main() or 0)
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)
        overlay["restore"]()
        base.retire_scratch(cache)
    if rc:
        raise RuntimeError(f"canonical builder rc={rc}")
    if any(cache.iterdir()):
        raise RuntimeError("layer scratch not retired")
    if set(overlay["norm_seen"]) != expected_norm_keys():
        raise RuntimeError("RMSNorm overlay was not fully consumed")
    if set(overlay["output_seen"]) != expected_output_keys():
        raise RuntimeError("output-gain overlay was not fully consumed")
    expected_codebook_pairs = {
        (layer, name)
        for layer in range(43)
        for name in state["codebooks"][f"L{layer}"]
    }
    if set(overlay["codebook_unique"]) != expected_codebook_pairs:
        raise RuntimeError("codebook overlay coverage drift")
    if update == 0 and any(item["changed"] for item in overlay["codebook_unique"].values()):
        raise RuntimeError("update-0 codebook overlay differs from sealed physical wire")

    expected_chunks = 1
    progress_payload = json.loads(progress.read_text())
    expected_visits = list(range(43)) * expected_chunks
    if (
        progress_payload.get("completed_layers") != expected_visits
        or progress_payload.get("mmap_completed_layers") != expected_visits
        or progress_payload.get("completed_chunks") != expected_chunks
        or progress_payload.get("local_stage_retired") is not True
        or progress_payload.get("mmap_loader_mode") != "torch-mmap"
        or progress_payload.get("mmap_loader_sha256") != CANONICAL_SHA256[LOADER_SOURCE]
        or progress_payload.get("mmap_input_identity_sha256") != WIRE_SHA
    ):
        raise RuntimeError("fresh scorer loader coverage/proof drift")
    if not sentinel.is_file():
        raise RuntimeError("loader sentinel missing")
    reduced = reduce_outputs(out, wins, classes)
    per_window = reduced["per_window"]
    parent_physical = json.loads(PHYSICAL_MARKER.read_text())
    baseline = json.loads(BASELINE_FULL512.read_text())

    clean72 = None
    code76 = None
    contaminated4 = None
    paired = None
    same_host_update0 = None
    terminal_comparison = None
    if mode == "fast":
        code76 = reduce_outputs(out, list(EXPECTED_CODE76_IDS), classes)["global"]
        code76["source_class"] = "code76-diagnostic"
        clean_rows = [row for row in per_window if row["win"] in EXPECTED_CLEAN72_IDS]
        contaminated_rows = [row for row in per_window if row["win"] in CONTAMINATED_IDS]
        clean72_values = torch.cat([reduced["values_by_win"][win] for win in EXPECTED_CLEAN72_IDS])
        contaminated_values = torch.cat([reduced["values_by_win"][win] for win in CONTAMINATED_IDS])
        clean_means = [float(reduced["values_by_win"][win].mean()) for win in EXPECTED_CLEAN72_IDS]
        clean_se = statistics.stdev(clean_means) / math.sqrt(len(clean_means))
        clean_mean = float(clean72_values.mean())
        clean72 = {
            "mean": clean_mean,
            "n_windows": 72,
            "n_positions": int(clean72_values.numel()),
            "window_ids": list(EXPECTED_CLEAN72_IDS),
            "window_ids_sha256": EXPECTED_CLEAN72_SHA256,
            "window_mean_se": clean_se,
            "window_mean_ci95": [clean_mean - 1.96 * clean_se, clean_mean + 1.96 * clean_se],
            "selection_authority": True,
            "known_contaminated_excluded": list(CONTAMINATED_IDS),
        }
        contaminated4 = {
            "mean": float(contaminated_values.mean()),
            "n_windows": 4,
            "n_positions": int(contaminated_values.numel()),
            "window_ids": list(CONTAMINATED_IDS),
            "diagnostic_only": True,
            "selection_authority": False,
        }
        paired = paired_delta(
            per_window,
            parent_physical["per_window"],
            list(EXPECTED_CODE76_IDS),
            "spark1-canonical-minus-spark8-physical-code76-diagnostic",
        )
        if update == 0:
            same_host_update0 = validate_update0_same_host_baseline(
                per_window, baseline["per_window"]
            )
    elif mode in HELDOUT_PREFIX_COUNTS:
        selected_by_class = {
            label: [win for win in wins if classes[win] == label]
            for label in FROZEN_CLASSES
        }
        terminal_comparison = {
            "label": "candidate-minus-current-BANANA_SMASHER paired ordered-window mean",
            "global": paired_delta(per_window, baseline["per_window"], wins, "global"),
            "code76": (
                paired_delta(per_window, baseline["per_window"], list(EXPECTED_CODE76_IDS), "code76")
                if mode == "full512" else None
            ),
            "six_classes": {
                label: (
                    paired_delta(per_window, baseline["per_window"], selected_by_class[label], label)
                    if selected_by_class[label]
                    else {
                        "label": label,
                        "candidate_minus_pre_repair_mean": None,
                        "window_mean_se": None,
                        "window_mean_ci95": None,
                        "n_windows": 0,
                        "window_ids": [],
                    }
                )
                for label in FROZEN_CLASSES
            },
            "baseline_receipt": str(BASELINE_FULL512),
            "baseline_receipt_sha256": EXPECTED_INPUT_SHA256[BASELINE_FULL512],
            "baseline_anchor_global_full512": CURRENT_BANANA_SMASHER_GLOBAL,
            "baseline_anchor_code_full512": CURRENT_BANANA_SMASHER_CODE,
        }

    claim_raw_after, _ = current_claim()
    if claim_raw_after != claim_raw_before:
        raise RuntimeError("local claim changed during scoring")
    source_claim_raw_after, _ = source_claim(source_claim_sha)
    if source_claim_raw_after != source_claim_raw_before:
        raise RuntimeError("source claim changed during scoring")
    completed = time.time()
    overlay_unique = list(overlay["codebook_unique"].values())
    instrument = {
        "canonical_reader_sha256": CANONICAL_SHA256[CANONICAL_READER],
        "canonical_builder_sha256": CANONICAL_SHA256[CANONICAL_BUILDER],
        "canonical_delta_source_sha256": CANONICAL_SHA256[CANONICAL_DELTA],
        "canonical_safety_sha256": CANONICAL_SHA256[CANONICAL_SAFETY],
        "loader_sha256": CANONICAL_SHA256[LOADER_SOURCE],
        "adapter_sha256": sha256_file(Path(__file__)),
        "checkpoint_state_loader_sha256": sha256_file(Path(__file__).with_name("checkpoint_state.py")),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "window_contract_sha256": EXPECTED_INPUT_SHA256[WINDOW_CONTRACT],
        "model_index_sha256": MODEL_INDEX_SHA,
        "teacher_done_sha256": TEACHER_DONE_SHA,
        "artifact_validation": sealed["artifacts"],
        "corpus_sha256": TRAIN_CORPUS_SHA if mode == "train" else CORPUS_SHA,
        "assignment_sha256": ASSIGNMENT_SHA,
        "compact_manifest_sha256": COMPACT_SHA,
        "wire_manifest_sha256": WIRE_SHA,
        "claim_sha256": claim_sha,
        "environment_contract": env_contract,
        "attention": "eager",
        "microbatch": 2,
        "chunk_size": chunk,
        "torch_load_safety": weights_only_stats,
        "checkpoint_overlay_semantics": {
            "codebooks": "fp32 master -> fp16 wire scratch replacement before canonical mmap dequant",
            "norms": "235 fail-closed safetensors reads replaced by fp32 master -> BF16 canonical load",
            "outputs": "43 trainer-equivalent exp(clamp(log_gain,-0.25,0.25)) BF16 forward hooks",
        },
    }
    receipt = {
        "schema": (
            "banana_smasher-repair-train-directional-v1" if mode == "train"
            else "banana_smasher-repair-checkpoint-score-v1" if mode == "fast"
            else "p632-p600-p623-whole-wire-prefix-v1"
        ),
        "status": "PASS_VALIDATED_RECEIPT",
        "measurement_label": "TRAIN_MID_DOSE_DIRECTIONAL_NOT_VERDICT" if mode == "train" else "MEASURED_HELDOUT",
        "task_id": TASK,
        "host": "compute-node-7",
        "source_task_id": transfer.get("source_task_id", SOURCE_TASK),
        "source_host": "compute-node-8 read-only sealed checkpoint over QSFP",
        "mode": mode,
        "exactness_label": "EXACT_P600_CANONICAL_WHOLE_WIRE_PLUS_P623_ORDERED_PREFIX_REDUCTION",
        "update": update,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sidecar": str(sidecar_path),
        "checkpoint_sidecar_sha256": sha256_file(sidecar_path),
        "transfer_receipt": str(transfer_path),
        "transfer_receipt_sha256": sha256_file(transfer_path),
        "instrument": instrument,
        "instrument_id_sha256": canonical_json_sha256(instrument),
        "local_claim_sha256": claim_sha,
        "source_claim_sha256": source_claim_sha,
        "source_manifest": str(SOURCE_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "direction": "KL(teacher||candidate)",
        "support": 8192,
        "cutoff": 1024,
        "windows": len(wins),
        "window_ids": wins,
        "window_ids_sha256": hashlib.sha256(",".join(map(str, wins)).encode()).hexdigest(),
        "global": reduced["global"],
        "by_class": reduced["by_class"],
        "clean72": clean72,
        "code76_diagnostic": code76,
        "contaminated4_diagnostic": contaminated4,
        "update0_physical_identity_comparison": paired if mode == "fast" else None,
        "update0_same_host_baseline_identity": same_host_update0,
        "terminal_paired_comparison": terminal_comparison,
        "selection": {
            "eligible": False,
            "authoritative_metric": None,
            "tie_break": None,
            "contaminated_windows_can_select": False,
        },
        "coverage": {
            "expected_chunks": expected_chunks,
            "expected_layer_visits": 43 * expected_chunks,
            "completed_layers": progress_payload["completed_layers"],
            "mmap_completed_layers": progress_payload["mmap_completed_layers"],
            "local_stage_retired": True,
            "remote_physical_package_mutated": False,
            "persistent_model_mutated": False,
        },
        "overlay": {
            "codebook_unique_count": len(overlay_unique),
            "codebook_changed_count": sum(bool(row["changed"]) for row in overlay_unique),
            "codebook_unchanged_count": sum(not bool(row["changed"]) for row in overlay_unique),
            "codebook_wire_set_sha256": canonical_json_sha256([
                {key: row[key] for key in ("layer", "name", "bytes", "base_sha256", "checkpoint_wire_sha256", "changed")}
                for row in sorted(overlay_unique, key=lambda row: (row["layer"], row["name"]))
            ]),
            "norms_consumed": len(overlay["norm_seen"]),
            "output_gains_consumed": len(overlay["output_seen"]),
            "gain_clamp": GAIN_CLAMP,
            "weights_only_checkpoint_load": True,
        },
        "outputs": {
            "directory": str(out),
            "window_output_set_sha256": reduced["window_output_set_sha256"],
            "per_window": per_window,
        },
        "loader_proof": {
            "mode": "torch-mmap",
            "loader_sha256": CANONICAL_SHA256[LOADER_SOURCE],
            "input_identity_sha256": WIRE_SHA,
            "progress": str(progress),
            "progress_sha256": sha256_file(progress),
            "sentinel": str(sentinel),
            "sentinel_sha256": sha256_file(sentinel),
            "fresh_zero_based_coverage_validated_out_of_band": True,
            "released_resume_builder_in_band_receipt_disabled": True,
        },
        "gpu_snapshot_before": gpu_before,
        "gpu_snapshot_before_child_exit": gpu_snapshot(own_pid=os.getpid(), require_zero_util=False),
        "disk_free_bytes_after": shutil.disk_usage(ROOT).free,
        "started_unix": started,
        "completed_unix": completed,
        "elapsed_seconds": completed - started,
        "windows_per_second": len(wins) / (completed - started),
        "windows_per_minute": len(wins) * 60 / (completed - started),
        "source_checkpoint_mtime_ns": int(transfer["source_checkpoint_mtime_ns"]),
        "dump_to_validated_receipt_seconds": completed - int(transfer["source_checkpoint_mtime_ns"]) / 1e9,
        "source_local_payload_policy": "authenticated compute-node-8 checkpoint bytes staged directly over QSFP; P602/P625 receipt-first local base-wire staging; canonical reader/builder/mmap bytes unchanged",
    }
    if shutil.disk_usage(ROOT).free < DISK_FLOOR:
        raise RuntimeError("disk floor not restored before receipt")
    atomic_json(receipt_path, receipt, exclusive=True)
    print(json.dumps({
        "status": receipt["status"],
        "mode": mode,
        "update": update,
        "checkpoint_sha256": checkpoint_sha,
        "clean72": clean72["mean"] if clean72 else None,
        "code76": code76["mean"] if code76 else reduced["by_class"].get("code", {}).get("mean"),
        "global": reduced["global"]["mean"],
        "receipt": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "dump_to_validated_receipt_seconds": receipt["dump_to_validated_receipt_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
