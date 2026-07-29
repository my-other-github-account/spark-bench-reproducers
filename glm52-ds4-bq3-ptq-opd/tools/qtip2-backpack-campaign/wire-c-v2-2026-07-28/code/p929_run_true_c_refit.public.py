#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Re-exec into the exact P760/P647 torch runtime before importing torch.
RUNTIME_PYTHON = "$SOURCE_ROOT/humming_env/bin/python3"
if os.environ.get("P925_TORCH_REEXEC") != "1":
    env = os.environ.copy()
    env["P925_TORCH_REEXEC"] = "1"
    os.execve(RUNTIME_PYTHON, [RUNTIME_PYTHON, *sys.argv], env)
if os.path.realpath(sys.executable) != os.path.realpath(RUNTIME_PYTHON):
    raise RuntimeError(f"runtime re-exec bypass detected: {sys.executable}")

import torch

SIGNAL_STOP = False


def install_signal_handlers() -> None:
    def request_stop(_signum, _frame) -> None:
        global SIGNAL_STOP
        SIGNAL_STOP = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)


def stop_requested(run_dir: Path) -> bool:
    return SIGNAL_STOP or (run_dir / "STOP_REQUESTED").exists()

TASK_ID = "PUBLIC_TASK"
WIRE_LABEL = "WIRE_C_BASELINE_R"
TRUE_C_LABEL = "WIRE_C_TRUE_C_f521_T"
QTIP_TIERS = {"qtip3_3.0117", "qtip2_2.0117"}
NATIVE = "native_mxfp4"
BUILDER_SHA = "60b594ac38e4973eaaecb76c708b555418406eb697414d2563aeb1e978268a7e"
CKPT_INDEX_SHA = "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"
EXPECTED_PINS = {
    "build_plan": "8c92ce62167db7980fde20b8e32cecc6934a816bb2a4b65dd78e99ecbf8f29c4",
    "active_overlay": "64c2ddc4248d71d738a23f461868b1072e49cd3e6ef6fa6eca43ac0e21e86113",
    "physical_manifest": "398441d16f1a251079b518a55095c568353b9f3e542f2ec55d4139e0ac6e7ffd",
    "assignment": "f521cf07e0dce3c39739c7493b6eda82cd78d6b1566fadb2101691321566ca39",
    "builder": BUILDER_SHA,
}
EXPECTED_TARGET_ROWS = 2860
EXPECTED_TARGET_CODEBOOKS = 80
EXPECTED_TARGET_LAYERS = 25
EXPECTED_TARGET_BY_TIER = {"d4_k1024": 173, "d4_k2048": 1374, "d4_k4096": 1313}
EXPECTED_TARGET_BY_PROJECTION = {"down": 1603, "fused13": 1257}
EXPECTED_ASSIGNMENT_MAP_SHA = "786b01a3f8c0197407e0025c80ca92c29b347a9c18de4b1ca48b7cf52ae08df6"
EXPECTED_SOURCE_ROWS_SHA = "8ed54f3bb922f9f79e0df590258142bbd75e8b06aba785764a7c07674ff32126"
EXPECTED_BASE_WIRE_MANIFEST_SHA = "c24a1c0568a00fcb8460d7edfb7630187ef10c98e9d0c25c87aa0bccb1d89755"
SOURCE_ROOT = Path("$SOURCE_ROOT/missions/P885_WIRE_C_PREREPAIR_PUBLIC_TASK_s6/f521_PUBLIC_TASK")
BASE_CODEBOOK_ROOT = Path("$SOURCE_ROOT/missions/P640_GENESIS_QTIP2_WIRE_PUBLIC_TASK_s6/inputs/base_codebooks")
PILOT_ROOT = Path("$SOURCE_ROOT/missions/P640_GENESIS_QTIP2_WIRE_PUBLIC_TASK_s6/code/pilot_code")
PILOT_LEDGER = Path("$SOURCE_ROOT/PUBLIC_TASK/run/VQ3_LEDGER.jsonl")
CKPT_ROOT = Path("$SOURCE_ROOT/models/hf/DeepSeek-V4-Flash")


def sha256(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(rows: list[dict]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def gpu_apps() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return [line.strip() for line in cp.stdout.splitlines() if line.strip()]


def tier_params(tier: str) -> tuple[int, int]:
    ds, ks = tier.split("_k")
    return int(ds.removeprefix("d")), int(ks)


def status_writer(run_dir: Path, started: float):
    def write(state: str, **extra: object) -> None:
        atomic_json(
            run_dir / "STATUS.json",
            {
                **extra,
                "schema": "p925-true-c-refit-status-v1",
                "state": state,
                "task_id": TASK_ID,
                "host": os.uname().nodename,
                "pid": os.getpid(),
                "pgid": os.getpgid(0),
                "sid": os.getsid(0),
                "elapsed_seconds": time.time() - started,
                "updated_unix": time.time(),
            },
        )
    return write


def load_inputs(mission: Path) -> tuple[dict, dict, dict, dict, dict[str, str]]:
    inp = mission / "inputs"
    code = mission / "code"
    paths = {
        "build_plan": inp / "BUILD_PLAN_WIRE_C_V2_PREVIEW.json",
        "active_overlay": inp / "WIRE_C_BASELINE_R_ACTIVE_OVERLAY.json",
        "physical_manifest": inp / "RESTORED_F521_PHYSICAL_MANIFEST.json",
        "assignment": inp / "ASSIGNMENT_WIRE_C_V2.json",
        "builder": code / "canonical_shared_builder.py",
    }
    pins = {name: sha256(path) for name, path in paths.items()}
    if pins != EXPECTED_PINS:
        raise RuntimeError(f"input pin mismatch: {pins}")
    plan = json.loads(paths["build_plan"].read_text())
    active = json.loads(paths["active_overlay"].read_text())
    physical = json.loads(paths["physical_manifest"].read_text())
    assignment = json.loads(paths["assignment"].read_text())
    return plan, active, physical, assignment, pins


def derive_target(plan: dict, active: dict) -> tuple[list[dict], dict[tuple[int, int, str], dict], dict]:
    if plan.get("status") != "PASS_BUILD_PLAN" or int(plan.get("changed_cells", -1)) != 21472:
        raise RuntimeError("source build plan contract mismatch")
    if active.get("status") != "PASS_EXACT_ACTIVE_LAYERS" or active.get("wire_label") != WIRE_LABEL:
        raise RuntimeError("source active overlay contract mismatch")
    if active.get("active_assignment_sha256") != EXPECTED_PINS["assignment"]:
        raise RuntimeError("active assignment mismatch")
    if active.get("final_assignment_map_sha256") != EXPECTED_ASSIGNMENT_MAP_SHA:
        raise RuntimeError("assignment map mismatch")
    if active.get("active_rows_sha256") != EXPECTED_SOURCE_ROWS_SHA:
        raise RuntimeError("source active rows pin mismatch")
    if active.get("base_wire_manifest_sha256") != EXPECTED_BASE_WIRE_MANIFEST_SHA:
        raise RuntimeError("base wire manifest mismatch")

    active_by_id = {
        (int(row["layer"]), int(row["expert"]), row["projection"]): row
        for row in active["rows"]
    }
    if len(active_by_id) != len(active["rows"]) or len(active_by_id) != 21472:
        raise RuntimeError("active overlay duplicate/coverage mismatch")

    target: list[dict] = []
    base_hashes: dict[str, str] = {}
    vq_rows = 0
    for row in plan["rows"]:
        tier = row["new"]
        if tier in QTIP_TIERS or tier == NATIVE:
            continue
        vq_rows += 1
        ident = (int(row["layer"]), int(row["expert"]), row["projection"])
        active_row = active_by_id.get(ident)
        if not active_row or active_row.get("new") != tier:
            raise RuntimeError(f"active/plan VQ identity mismatch: {ident}")
        base_path = BASE_CODEBOOK_ROOT / f"layer_{ident[0]:03d}" / f"{tier}.{ident[2]}.codebook.fp16.bin"
        key = str(base_path)
        if key not in base_hashes:
            if not base_path.is_file():
                raise RuntimeError(f"base codebook missing: {base_path}")
            base_hashes[key] = sha256(base_path)
        if active_row.get("codebook_sha256") == base_hashes[key]:
            target.append(
                {
                    **row,
                    "identity": [ident[0], ident[1], ident[2]],
                    "source_active_artifact": active_row["artifact"],
                    "source_active_artifact_bytes": int(active_row["artifact_bytes"]),
                    "source_active_artifact_sha256": active_row["artifact_sha256"],
                    "base_codebook_path": key,
                    "base_codebook_bytes": int(base_path.stat().st_size),
                    "base_codebook_sha256": base_hashes[key],
                }
            )

    target.sort(key=lambda r: (int(r["layer"]), int(r["expert"]), r["projection"]))
    unique_cb = {(int(r["layer"]), r["new"], r["projection"]) for r in target}
    layers = {int(r["layer"]) for r in target}
    by_tier = dict(sorted(Counter(r["new"] for r in target).items()))
    by_projection = dict(sorted(Counter(r["projection"] for r in target).items()))
    if len(target) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(f"target row count drift: {len(target)} != {EXPECTED_TARGET_ROWS}")
    if len(unique_cb) != EXPECTED_TARGET_CODEBOOKS:
        raise RuntimeError(f"target codebook count drift: {len(unique_cb)} != {EXPECTED_TARGET_CODEBOOKS}")
    if len(layers) != EXPECTED_TARGET_LAYERS:
        raise RuntimeError(f"target layer count drift: {len(layers)} != {EXPECTED_TARGET_LAYERS}")
    if by_tier != EXPECTED_TARGET_BY_TIER or by_projection != EXPECTED_TARGET_BY_PROJECTION:
        raise RuntimeError(f"target distribution drift: tiers={by_tier}, projections={by_projection}")
    if vq_rows != 4173:
        raise RuntimeError(f"source VQ count drift: {vq_rows}")

    output_bytes_estimate = sum(int(r["source_active_artifact_bytes"]) for r in target)
    contract = {
        "schema": "p925-true-c-refit-target-v1",
        "status": "PASS_EXACT_CURRENT_BASE_BINDING",
        "task_id": TASK_ID,
        "wire_label": WIRE_LABEL,
        "true_c_label": TRUE_C_LABEL,
        "selection_rule": "every current VQ overlay row whose active codebook SHA exactly equals the frozen P640 base codebook for the same layer/tier/projection",
        "source_vq_rows": vq_rows,
        "target_rows": len(target),
        "preserved_nonbase_vq_rows": vq_rows - len(target),
        "unique_target_codebooks": len(unique_cb),
        "target_layers": sorted(layers),
        "target_layer_count": len(layers),
        "target_by_tier": by_tier,
        "target_by_projection": by_projection,
        "estimated_payload_bytes": output_bytes_estimate,
        "builder_method": {
            "builder_sha256": BUILDER_SHA,
            "scope": "layer-shared per projection and tier",
            "fit_experts": [17, 77, 177],
            "scale_grid": "W3v2 e43 LUT with SSE offsets -4..+2",
            "objective": "scale-squared weighted kmeans++ plus 15-iteration Lloyd",
            "assignment": "nearest against serialized fp16 codebook",
            "seed": 0,
            "corpus_windows_used": False,
        },
        "rows_sha256": canonical_sha(target),
        "rows": target,
    }
    return target, active_by_id, contract


def create_build_identity(mission: Path, target: list[dict], target_contract: dict, pins: dict[str, str]) -> tuple[dict, str]:
    source_index_path = CKPT_ROOT / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text())
    weight_map = source_index.get("weight_map") or {}
    target_layers = {int(row["layer"]) for row in target}
    shard_names = sorted(
        {
            shard
            for key, shard in weight_map.items()
            if any(key.startswith(f"layers.{layer}.") for layer in target_layers)
        }
    )
    if not shard_names:
        raise RuntimeError("checkpoint shard derivation produced an empty set")
    shards = []
    for name in shard_names:
        path = CKPT_ROOT / name
        if not path.is_file():
            raise RuntimeError(f"checkpoint shard missing: {path}")
        shards.append({"name": name, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    shard_manifest = {
        "schema": "p925-source-shard-manifest-v1",
        "checkpoint_index": str(source_index_path),
        "checkpoint_index_sha256": CKPT_INDEX_SHA,
        "target_layers": sorted(target_layers),
        "shard_count": len(shards),
        "shards": shards,
        "shards_sha256": canonical_sha(shards),
    }
    shard_manifest_path = mission / "inputs/SOURCE_SHARD_MANIFEST.json"
    atomic_json(shard_manifest_path, shard_manifest)

    runtime_path = Path(os.path.realpath(sys.executable))
    torch_path = Path(torch.__file__).resolve()
    pilot_paths = {
        "gptqv2_pilot": PILOT_ROOT / "gptqv2_pilot.py",
        "vqw2_pilot": PILOT_ROOT / "vqw2_pilot.py",
        "pilot_ledger": PILOT_LEDGER,
    }
    pilot_pins = {}
    for name, path in pilot_paths.items():
        if not path.is_file():
            raise RuntimeError(f"pilot input missing: {path}")
        pilot_pins[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
    unique_base_codebooks = sorted(
        {
            (int(row["layer"]), row["new"], row["projection"], row["base_codebook_path"], row["base_codebook_sha256"])
            for row in target
        }
    )
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if smi.returncode != 0 or not smi.stdout.strip():
        raise RuntimeError(f"CUDA runtime inventory failed: {smi.stderr.strip()}")
    build_identity = {
        "schema": "p925-true-c-build-identity-v1",
        "status": "PINNED",
        "task_id": TASK_ID,
        "host": os.uname().nodename,
        "input_pins": pins,
        "builder_sha256": BUILDER_SHA,
        "checkpoint_index_sha256": CKPT_INDEX_SHA,
        "source_shard_manifest": str(shard_manifest_path),
        "source_shard_manifest_sha256": sha256(shard_manifest_path),
        "source_shards_sha256": shard_manifest["shards_sha256"],
        "source_shard_count": len(shards),
        "pilot_pins": pilot_pins,
        "runtime": {
            "python": str(runtime_path),
            "python_sha256": sha256(runtime_path),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cudnn_version": torch.backends.cudnn.version(),
            "cuda_compiled_version": torch._C._cuda_getCompiledVersion(),
            "nvidia_smi_gpu_driver": [line.strip() for line in smi.stdout.splitlines() if line.strip()],
            "torch_file": str(torch_path),
            "torch_file_sha256": sha256(torch_path),
        },
        "target_rows": len(target),
        "target_rows_sha256": target_contract["rows_sha256"],
        "target_layers": target_contract["target_layers"],
        "unique_target_codebooks": target_contract["unique_target_codebooks"],
        "unique_base_codebooks_sha256": canonical_sha(unique_base_codebooks),
        "method": target_contract["builder_method"],
    }
    path = mission / "inputs/P925_BUILD_IDENTITY.json"
    atomic_json(path, build_identity)
    return build_identity, sha256(path)


def load_builder(path: Path, tier: str):
    d, k = tier_params(tier)
    name = f"p925_builder_{tier}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # The imported canonical builder installs handlers that only set its own
    # unused STOP Event. Restore the task handler immediately after import so
    # SIGTERM/SIGINT cannot be swallowed.
    install_signal_handlers()
    module.D = d
    module.CB_K = k
    module.CKPT = CKPT_ROOT
    module.PILOT = PILOT_ROOT
    module.PILOT_LEDGER = PILOT_LEDGER
    return module


def identity(row: dict) -> tuple[int, int, str]:
    return int(row["layer"]), int(row["expert"]), str(row["projection"])


def expected_identity_rows(rows: list[dict]) -> list[list[object]]:
    return [list(identity(row)) for row in sorted(rows, key=identity)]


def quarantine_receipt(path: Path, reason: str) -> None:
    quarantine = path.parent / "quarantine"
    quarantine.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time())}.{os.getpid()}"
    moved = quarantine / f"{path.name}.invalid.{stamp}"
    os.replace(path, moved)
    atomic_json(
        quarantine / f"{path.name}.invalid.{stamp}.reason.json",
        {
            "schema": "p925-quarantined-resume-receipt-v1",
            "status": "QUARANTINED_FAIL_CLOSED",
            "task_id": TASK_ID,
            "source": str(path),
            "quarantined": str(moved),
            "source_sha256": sha256(moved),
            "reason": reason,
            "quarantined_unix": time.time(),
        },
    )
    raise RuntimeError(f"resume receipt quarantined fail-closed: {path}: {reason}")


def validate_payload_row(row: dict, expected: dict, expected_target_sha: str, expected_build_sha: str) -> None:
    ident = identity(expected)
    if identity(row) != ident or row.get("identity") != list(ident):
        raise RuntimeError(f"row identity mismatch: got={row.get('identity')} expected={ident}")
    if row.get("tier") != expected.get("new") or row.get("fp16_codebook_replay_exact") is not True:
        raise RuntimeError(f"row tier/replay mismatch: {ident}")
    artifact = Path(row["artifact"])
    cb = Path(row["codebook"])
    if not artifact.is_file() or artifact.stat().st_size != int(row["artifact_bytes"]) or sha256(artifact) != row["artifact_sha256"]:
        raise RuntimeError(f"artifact readback mismatch: {ident}")
    if not cb.is_file() or cb.stat().st_size != int(row["codebook_bytes"]) or sha256(cb) != row["codebook_sha256"]:
        raise RuntimeError(f"codebook readback mismatch: {ident}")
    if row["codebook_sha256"] == expected["base_codebook_sha256"]:
        raise RuntimeError(f"refit codebook equals base: {ident}")
    payload = torch.load(artifact, map_location="cpu", weights_only=True)
    meta = payload.get("meta") or {}
    exact_meta = {
        "schema": "p925-true-c-refit-vq-cell-v2",
        "task_id": TASK_ID,
        "layer": ident[0],
        "expert": ident[1],
        "projection": ident[2],
        "tier": expected["new"],
        "target_contract_sha256": expected_target_sha,
        "build_identity_sha256": expected_build_sha,
        "codebook_sha256": row["codebook_sha256"],
        "fp16_codebook_replay_exact": True,
    }
    drift = {key: {"expected": value, "observed": meta.get(key)} for key, value in exact_meta.items() if meta.get(key) != value}
    if drift:
        raise RuntimeError(f"payload metadata mismatch {ident}: {drift}")


def valid_layer_receipt(
    path: Path,
    expected_rows: int,
    expected_target_sha: str,
    expected_rows_contract: list[dict] | None = None,
    expected_build_sha: str | None = None,
) -> tuple[bool, list[dict]]:
    """Strict layer receipt validation; malformed existing receipts are never reusable."""
    if not path.is_file():
        return False, []
    if expected_rows_contract is None or expected_build_sha is None:
        return False, []
    try:
        rec = json.loads(path.read_text())
        expected_sorted = sorted(expected_rows_contract, key=identity)
        expected_ids = expected_identity_rows(expected_sorted)
        rows = rec.get("rows")
        exact = {
            "schema": "p925-true-c-refit-layer-v2",
            "status": "PASS",
            "task_id": TASK_ID,
            "target_rows": expected_rows,
            "target_contract_sha256": expected_target_sha,
            "build_identity_sha256": expected_build_sha,
            "expected_identities_sha256": canonical_sha(expected_ids),
        }
        drift = {key: {"expected": value, "observed": rec.get(key)} for key, value in exact.items() if rec.get(key) != value}
        if drift or not isinstance(rows, list) or len(rows) != expected_rows:
            raise RuntimeError(f"layer receipt contract mismatch: drift={drift} rows={len(rows) if isinstance(rows, list) else None}")
        if rec.get("rows_sha256") != canonical_sha(rows):
            raise RuntimeError("layer rows SHA mismatch")
        if [list(identity(row)) for row in rows] != expected_ids:
            raise RuntimeError("layer identity/order mismatch")
        expected_groups = sorted({(int(row["layer"]), row["new"], row["projection"]) for row in expected_sorted})
        got_groups = sorted({(int(row["layer"]), row["tier"], row["projection"]) for row in rows})
        if got_groups != expected_groups or rec.get("unique_codebooks") != len(expected_groups):
            raise RuntimeError("layer codebook-group mismatch")
        for row, expected in zip(rows, expected_sorted):
            validate_payload_row(row, expected, expected_target_sha, expected_build_sha)
        return True, rows
    except Exception as exc:
        quarantine_receipt(path, repr(exc))
    raise AssertionError("unreachable")


def load_codebook_receipt(
    path: Path,
    group: tuple[int, str, str],
    expected_rows_contract: list[dict],
    expected_target_sha: str,
    expected_build_sha: str,
) -> tuple[str, list[dict], dict] | None:
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text())
        expected_sorted = sorted(expected_rows_contract, key=identity)
        expected_ids = expected_identity_rows(expected_sorted)
        rows = rec.get("rows")
        exact = {
            "schema": "p925-true-c-refit-codebook-v2",
            "task_id": TASK_ID,
            "codebook_group": [group[0], group[1], group[2]],
            "expected_rows": len(expected_sorted),
            "expected_identities_sha256": canonical_sha(expected_ids),
            "target_contract_sha256": expected_target_sha,
            "build_identity_sha256": expected_build_sha,
        }
        drift = {key: {"expected": value, "observed": rec.get(key)} for key, value in exact.items() if rec.get(key) != value}
        if drift or rec.get("status") not in {"PARTIAL", "PASS"} or not isinstance(rows, list):
            raise RuntimeError(f"codebook receipt contract mismatch: {drift}")
        if rec.get("completed_rows") != len(rows) or rec.get("rows_sha256") != canonical_sha(rows):
            raise RuntimeError("codebook completed count/rows SHA mismatch")
        if rec["status"] == "PASS" and len(rows) != len(expected_sorted):
            raise RuntimeError("PASS codebook receipt is incomplete")
        if rec["status"] == "PARTIAL" and len(rows) >= len(expected_sorted):
            raise RuntimeError("PARTIAL codebook receipt is complete/overflowed")
        completed_ids = [list(identity(row)) for row in rows]
        if completed_ids != expected_ids[: len(rows)]:
            raise RuntimeError("codebook completed identities are not the exact expected prefix")
        codebook = Path(rec["codebook"])
        d, k = tier_params(group[1])
        if (
            not codebook.is_file()
            or codebook.stat().st_size != d * k * 2
            or codebook.stat().st_size != int(rec["codebook_bytes"])
            or sha256(codebook) != rec["codebook_sha256"]
        ):
            raise RuntimeError("codebook payload readback mismatch")
        if rec["codebook_sha256"] == expected_sorted[0]["base_codebook_sha256"]:
            raise RuntimeError("refit codebook equals frozen base")
        if any(row.get("codebook_sha256") != rec["codebook_sha256"] for row in rows):
            raise RuntimeError("row/codebook hash inconsistency")
        for row, expected in zip(rows, expected_sorted):
            validate_payload_row(row, expected, expected_target_sha, expected_build_sha)
        return rec["status"], rows, rec
    except Exception as exc:
        quarantine_receipt(path, repr(exc))
    raise AssertionError("unreachable")


def write_codebook_receipt(
    path: Path,
    group: tuple[int, str, str],
    expected_rows_contract: list[dict],
    rows: list[dict],
    codebook: Path,
    codebook_sha: str,
    target_sha: str,
    build_sha: str,
) -> dict:
    expected_sorted = sorted(expected_rows_contract, key=identity)
    complete = len(rows) == len(expected_sorted)
    receipt = {
        "schema": "p925-true-c-refit-codebook-v2",
        "status": "PASS" if complete else "PARTIAL",
        "task_id": TASK_ID,
        "host": os.uname().nodename,
        "codebook_group": [group[0], group[1], group[2]],
        "expected_rows": len(expected_sorted),
        "expected_identities_sha256": canonical_sha(expected_identity_rows(expected_sorted)),
        "completed_rows": len(rows),
        "target_contract_sha256": target_sha,
        "build_identity_sha256": build_sha,
        "codebook": str(codebook),
        "codebook_bytes": codebook.stat().st_size,
        "codebook_sha256": codebook_sha,
        "rows_sha256": canonical_sha(rows),
        "rows": rows,
        "updated_unix": time.time(),
    }
    atomic_json(path, receipt)
    return receipt


def immutable_done_fast_path(mission: Path, expected_target_rows: list[dict]) -> dict | None:
    done_path = mission / "out/DONE.json"
    if not done_path.is_file():
        return None
    try:
        target_path = mission / "inputs/P925_EXACT_REFIT_TARGET.json"
        if not target_path.is_file():
            raise RuntimeError("DONE exists without target contract")
        target = json.loads(target_path.read_text())
        target_sha = sha256(target_path)
        if (
            target.get("task_id") != TASK_ID
            or target.get("target_rows") != EXPECTED_TARGET_ROWS
            or target.get("unique_target_codebooks") != EXPECTED_TARGET_CODEBOOKS
            or target.get("rows_sha256") != canonical_sha(target["rows"])
            or canonical_sha(target["rows"]) != canonical_sha(expected_target_rows)
        ):
            raise RuntimeError("existing DONE target contract drift")
        build_path = Path(target["build_identity"])
        build_sha = target["build_identity_sha256"]
        if not build_path.is_file() or sha256(build_path) != build_sha:
            raise RuntimeError("existing DONE build identity drift")
        build = json.loads(build_path.read_text())
        if build.get("task_id") != TASK_ID or build.get("target_rows_sha256") != target["rows_sha256"]:
            raise RuntimeError("existing DONE build/target cross-pin drift")
        done = json.loads(done_path.read_text())
        if (
            done.get("schema") != "p925-true-c-refit-done-v2"
            or done.get("status") != "PASS"
            or done.get("task_id") != TASK_ID
            or done.get("target_contract_sha256") != target_sha
            or done.get("build_identity_sha256") != build_sha
            or done.get("target_rows") != EXPECTED_TARGET_ROWS
            or done.get("unique_refit_codebooks") != EXPECTED_TARGET_CODEBOOKS
        ):
            raise RuntimeError("existing DONE contract drift")
        delta_path = Path(done["delta_manifest"])
        overlay_path = Path(done["active_overlay"])
        if not delta_path.is_file() or sha256(delta_path) != done["delta_manifest_sha256"]:
            raise RuntimeError("existing DONE delta pin drift")
        if not overlay_path.is_file() or sha256(overlay_path) != done["active_overlay_sha256"]:
            raise RuntimeError("existing DONE overlay pin drift")
        delta = json.loads(delta_path.read_text())
        rows = delta.get("rows")
        expected_sorted = sorted(expected_target_rows, key=identity)
        if (
            delta.get("task_id") != TASK_ID
            or delta.get("target_contract_sha256") != target_sha
            or delta.get("build_identity_sha256") != build_sha
            or delta.get("delta_rows") != EXPECTED_TARGET_ROWS
            or delta.get("unique_refit_codebooks") != EXPECTED_TARGET_CODEBOOKS
            or not isinstance(rows, list)
            or delta.get("delta_rows_sha256") != canonical_sha(rows)
            or [list(identity(row)) for row in rows] != expected_identity_rows(expected_sorted)
        ):
            raise RuntimeError("existing DONE delta contract drift")
        for row, expected in zip(rows, expected_sorted):
            validate_payload_row(row, expected, target_sha, build_sha)
        overlay = json.loads(overlay_path.read_text())
        if (
            overlay.get("task_id") != TASK_ID
            or overlay.get("p925_target_contract_sha256") != target_sha
            or overlay.get("active_rows_sha256") != canonical_sha(overlay["rows"])
            or overlay.get("active_rows_sha256") != done.get("merged_active_rows_sha256")
        ):
            raise RuntimeError("existing DONE overlay contract drift")
        return done
    except Exception as exc:
        quarantine_receipt(done_path, repr(exc))
    raise AssertionError("unreachable")


def main() -> int:
    install_signal_handlers()
    ap = argparse.ArgumentParser()
    ap.add_argument("mission", type=Path)
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0, help="test-only cap; a capped run never emits final manifests")
    args = ap.parse_args()

    mission = args.mission.resolve()
    inp = mission / "inputs"
    code = mission / "code"
    out = mission / "out"
    cells = out / "cells"
    codebooks = out / "codebooks"
    receipts = mission / "receipts"
    run = mission / "run"
    logs = mission / "logs"
    for path in (cells, codebooks, receipts, run, logs):
        path.mkdir(parents=True, exist_ok=True)
    started = time.time()
    write_status = status_writer(run, started)
    write_status("PREFLIGHT")

    claim_path = mission / "CLAIM.json"
    if not claim_path.is_file():
        raise RuntimeError("task-owned CLAIM.json missing")
    claim = json.loads(claim_path.read_text())
    allowed_claim_status = {"CLAIMED"} if not args.preflight_only else {"STAGED_NO_GPU", "CLAIMED"}
    if (
        claim.get("task_id") != TASK_ID
        or claim.get("host") != os.uname().nodename
        or claim.get("status") not in allowed_claim_status
    ):
        raise RuntimeError(f"claim contract mismatch: {claim}")

    plan, active, physical, assignment, pins = load_inputs(mission)
    if physical.get("status") != "PASS" or physical.get("rows_sha256") != "c2f1c56c7baad772d7e7927f65f4873590161e2c4600410b5ea1b40e848f75b1":
        raise RuntimeError("physical manifest contract mismatch")
    if assignment.get("schema") is None:
        raise RuntimeError("assignment schema missing")
    source_index = CKPT_ROOT / "model.safetensors.index.json"
    if not source_index.is_file() or sha256(source_index) != CKPT_INDEX_SHA:
        raise RuntimeError("checkpoint index mismatch")

    target, active_by_id, target_contract = derive_target(plan, active)
    existing_done = immutable_done_fast_path(mission, target)
    if existing_done is not None:
        write_status("IMMUTABLE_DONE", done_sha256=sha256(out / "DONE.json"), target_rows=existing_done["target_rows"])
        print(json.dumps(existing_done, sort_keys=True))
        return 0
    adapter_path = code / "p925_true_c_overlay_adapter.py"
    verifier_path = code / "verify_true_c.py"
    supervisor_path = code / "supervise_true_c.py"
    for required_code in (adapter_path, verifier_path, supervisor_path):
        if not required_code.is_file():
            raise RuntimeError(f"required output/control code missing: {required_code}")

    # GPU-resume identity contains only immutable algorithm/input/runtime facts.
    # Adapter/verifier/supervisor/runner provenance is separate so a consumer-only
    # correction never invalidates already-sealed GPU codebooks.
    target_contract["input_pins"] = pins
    build_identity, build_identity_sha = create_build_identity(mission, target, target_contract, pins)
    target_contract["build_identity"] = str(inp / "P925_BUILD_IDENTITY.json")
    target_contract["build_identity_sha256"] = build_identity_sha
    atomic_json(inp / "P925_EXACT_REFIT_TARGET.json", target_contract)
    target_sha = sha256(inp / "P925_EXACT_REFIT_TARGET.json")
    output_provenance = {
        "schema": "p925-output-control-provenance-v1",
        "status": "PINNED_SEPARATE_FROM_GPU_BUILD_IDENTITY",
        "task_id": TASK_ID,
        "target_contract_sha256": target_sha,
        "build_identity_sha256": build_identity_sha,
        "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        "consumer_adapter": {"path": str(adapter_path), "sha256": sha256(adapter_path)},
        "verifier": {"path": str(verifier_path), "sha256": sha256(verifier_path)},
        "supervisor": {"path": str(supervisor_path), "sha256": sha256(supervisor_path)},
        "written_unix": time.time(),
    }
    atomic_json(inp / "P925_OUTPUT_PROVENANCE.json", output_provenance)
    free = shutil.disk_usage(mission).free
    required = int(target_contract["estimated_payload_bytes"]) + 12 * (1 << 30)
    preflight = {
        "schema": "p925-true-c-refit-preflight-v1",
        "status": "PASS" if free >= required else "FAIL_DISK_FLOOR",
        "task_id": TASK_ID,
        "host": os.uname().nodename,
        "input_pins": pins,
        "target_contract": str(inp / "P925_EXACT_REFIT_TARGET.json"),
        "target_contract_sha256": target_sha,
        "build_identity": str(inp / "P925_BUILD_IDENTITY.json"),
        "build_identity_sha256": build_identity_sha,
        "source_shards_sha256": build_identity["source_shards_sha256"],
        "pilot_pins": build_identity["pilot_pins"],
        "runtime": build_identity["runtime"],
        "target_rows": len(target),
        "unique_target_codebooks": target_contract["unique_target_codebooks"],
        "target_layers": target_contract["target_layers"],
        "estimated_payload_bytes": target_contract["estimated_payload_bytes"],
        "disk_free_bytes": free,
        "required_free_bytes": required,
        "gpu_apps_observed": gpu_apps(),
        "completed_unix": time.time(),
    }
    atomic_json(receipts / "PREFLIGHT.json", preflight)
    if preflight["status"] != "PASS":
        raise RuntimeError(f"disk floor failed: {preflight}")
    if args.preflight_only:
        write_status("PREFLIGHT_PASS", target_rows=len(target), target_contract_sha256=target_sha)
        print(json.dumps(preflight, sort_keys=True))
        return 0

    apps = gpu_apps()
    if apps:
        raise RuntimeError(f"pre-existing GPU apps at launch: {apps}")
    write_status("BUILD_START", target_rows=len(target), target_contract_sha256=target_sha)

    os.environ["VQ3U_PILOT"] = str(PILOT_ROOT)
    os.environ["VQ3U_PILOT_LEDGER"] = str(PILOT_LEDGER)
    os.environ["VQ3U_CKPT"] = str(CKPT_ROOT)
    builder_path = code / "canonical_shared_builder.py"
    modules: dict[str, object] = {}
    layers = list(target_contract["target_layers"])
    if args.max_layers:
        layers = layers[: args.max_layers]
    all_rows: list[dict] = []

    sealed_codebooks = 0
    for layer_index, layer in enumerate(layers, 1):
        if stop_requested(run):
            write_status("PARTIAL_PREEMPTED", completed_rows=len(all_rows), sealed_codebooks=sealed_codebooks, next_layer=layer)
            return 75
        layer_target = sorted([row for row in target if int(row["layer"]) == layer], key=identity)
        layer_receipt_path = receipts / f"LAYER_{layer:03d}.json"
        valid, prior_rows = valid_layer_receipt(
            layer_receipt_path,
            len(layer_target),
            target_sha,
            expected_rows_contract=layer_target,
            expected_build_sha=build_identity_sha,
        )
        if valid:
            all_rows.extend(prior_rows)
            sealed_codebooks += len({(row["tier"], row["projection"]) for row in prior_rows})
            write_status(
                "RESUME_SKIP_LAYER",
                active_layer=layer,
                layer_index=layer_index,
                completed_rows=len(all_rows),
                sealed_codebooks=sealed_codebooks,
            )
            continue

        groups = sorted({(layer, row["new"], row["projection"]) for row in layer_target})
        group_targets = {group: [row for row in layer_target if row["new"] == group[1] and row["projection"] == group[2]] for group in groups}
        group_paths = {group: receipts / f"CODEBOOK_L{layer:03d}.{group[1]}.{group[2]}.json" for group in groups}
        group_state: dict[tuple[int, str, str], tuple[str, list[dict], dict] | None] = {
            group: load_codebook_receipt(group_paths[group], group, group_targets[group], target_sha, build_identity_sha)
            for group in groups
        }
        resumed_pass_groups = sum(1 for state in group_state.values() if state is not None and state[0] == "PASS")
        sealed_codebooks += resumed_pass_groups
        if resumed_pass_groups == len(groups):
            layer_rows = sorted([row for state in group_state.values() for row in state[1]], key=identity)
            layer_receipt = {
                "schema": "p925-true-c-refit-layer-v2",
                "status": "PASS",
                "task_id": TASK_ID,
                "host": os.uname().nodename,
                "layer": layer,
                "target_rows": len(layer_target),
                "expected_identities_sha256": canonical_sha(expected_identity_rows(layer_target)),
                "unique_codebooks": len(groups),
                "target_contract_sha256": target_sha,
                "build_identity_sha256": build_identity_sha,
                "codebook_receipts": [str(group_paths[group]) for group in groups],
                "codebook_receipts_sha256": canonical_sha([sha256(group_paths[group]) for group in groups]),
                "rows_sha256": canonical_sha(layer_rows),
                "rows": layer_rows,
                "completed_unix": time.time(),
            }
            atomic_json(layer_receipt_path, layer_receipt)
            all_rows.extend(layer_rows)
            write_status("RESUME_ASSEMBLE_LAYER", active_layer=layer, completed_rows=len(all_rows), sealed_codebooks=sealed_codebooks)
            continue

        write_status(
            "EXTRACT_LAYER",
            active_layer=layer,
            layer_index=layer_index,
            layer_rows=len(layer_target),
            completed_rows=len(all_rows),
            sealed_codebooks=sealed_codebooks,
        )
        tiers = sorted({group[1] for group in groups if group_state[group] is None or group_state[group][0] != "PASS"})
        for tier in tiers:
            if tier not in modules:
                modules[tier] = load_builder(builder_path, tier)
        extraction_module = modules[tiers[0]]
        packed = extraction_module.extract_layer(layer)
        bundle = extraction_module.mem_bundle(packed)

        for tier in tiers:
            if stop_requested(run):
                write_status("PARTIAL_PREEMPTED", active_layer=layer, active_tier=tier, completed_rows=len(all_rows), sealed_codebooks=sealed_codebooks)
                return 75
            module = modules[tier]
            tier_groups = [group for group in groups if group[1] == tier and (group_state[group] is None or group_state[group][0] != "PASS")]
            missing_groups = [group for group in tier_groups if group_state[group] is None]
            fitted = None
            if missing_groups:
                write_status("FIT_CODEBOOKS", active_layer=layer, active_tier=tier, completed_rows=len(all_rows), sealed_codebooks=sealed_codebooks)
                fitted = module.fit_layer_cbs(bundle, layer)
                d, k = tier_params(tier)
                # Persist every newly fitted codebook before building cells. A
                # signal after fitting therefore resumes from bytes, not k-means.
                for group in missing_groups:
                    projection = group[2]
                    source_projection = "fused13" if projection == "fused13" else "down"
                    cb_cpu = fitted[source_projection].to(torch.float16).cpu().contiguous()
                    dst = codebooks / f"L{layer:03d}.{tier}.{source_projection}.codebook.fp16.bin"
                    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                    tmp.write_bytes(cb_cpu.numpy().tobytes())
                    if tmp.stat().st_size != d * k * 2:
                        raise RuntimeError(f"refit codebook bytes mismatch: {tmp}")
                    os.replace(tmp, dst)
                    cb_hash = sha256(dst)
                    if cb_hash == group_targets[group][0]["base_codebook_sha256"]:
                        raise RuntimeError(f"refit unexpectedly equals base codebook: {group}")
                    receipt = write_codebook_receipt(
                        group_paths[group], group, group_targets[group], [], dst, cb_hash, target_sha, build_identity_sha
                    )
                    group_state[group] = ("PARTIAL", [], receipt)
                del fitted
                torch.cuda.empty_cache()

            for group in tier_groups:
                state = group_state[group]
                if state is None:
                    raise AssertionError(f"missing codebook state after fit: {group}")
                if state[0] == "PASS":
                    continue
                projection = group[2]
                group_target = sorted(group_targets[group], key=identity)
                group_rows = list(state[1])
                cb_path = Path(state[2]["codebook"])
                cb_hash = state[2]["codebook_sha256"]
                d, k = tier_params(tier)
                cb_cpu = torch.from_file(str(cb_path), dtype=torch.float16, size=d * k).reshape(k, d).clone().contiguous()
                cb16 = cb_cpu.to("cuda")
                cb32 = cb16.float()
                for group_cell_index in range(len(group_rows), len(group_target)):
                    if stop_requested(run):
                        write_status(
                            "PARTIAL_PREEMPTED",
                            active_layer=layer,
                            active_tier=tier,
                            active_projection=projection,
                            completed_rows=len(all_rows) + sum(len(s[1]) for s in group_state.values() if s is not None),
                            sealed_codebooks=sealed_codebooks,
                            next_identity=list(identity(group_target[group_cell_index])),
                        )
                        return 75
                    source_row = group_target[group_cell_index]
                    expert = int(source_row["expert"])
                    W, sb = bundle.fused13(expert) if projection == "fused13" else bundle.down(expert)
                    if not bool(torch.isfinite(W).all().item()):
                        raise RuntimeError(f"nonfinite source: {(layer, expert, projection)}")
                    codes, scales, metrics = module.build_unit(W, sb, cb32, cb16)
                    s_col = module.gp.sbytes_to_scol(scales)
                    replay = module.vp.assign_chunk((W / s_col).view(-1, d), cb16.float()).view_as(codes)
                    if not bool(replay.eq(codes.long()).all().item()):
                        raise RuntimeError(f"code replay mismatch: {(layer, expert, projection)}")
                    if not all(math.isfinite(float(value)) for value in metrics.values()):
                        raise RuntimeError(f"nonfinite metrics: {(layer, expert, projection)}")
                    dtype = torch.uint8 if k <= 256 else torch.int16
                    payload = {
                        "codes": codes.to(dtype).cpu(),
                        "scales": scales.cpu(),
                        "meta": {
                            "schema": "p925-true-c-refit-vq-cell-v2",
                            "task_id": TASK_ID,
                            "wire_label": TRUE_C_LABEL,
                            "layer": layer,
                            "expert": expert,
                            "projection": projection,
                            "tier": tier,
                            "d": d,
                            "k": k,
                            "assignment_sha256": EXPECTED_PINS["assignment"],
                            "assignment_map_sha256": EXPECTED_ASSIGNMENT_MAP_SHA,
                            "source_active_overlay_sha256": EXPECTED_PINS["active_overlay"],
                            "target_contract_sha256": target_sha,
                            "build_identity_sha256": build_identity_sha,
                            "builder_sha256": BUILDER_SHA,
                            "old_base_codebook_sha256": source_row["base_codebook_sha256"],
                            "codebook_sha256": cb_hash,
                            "refit_codebook_sha256": cb_hash,
                            "fp16_codebook_replay_exact": True,
                            "metrics": metrics,
                        },
                    }
                    cell_dir = cells / f"L{layer:03d}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    artifact = cell_dir / f"L{layer:03d}_E{expert:03d}_{projection}__{tier}.pt"
                    tmp = artifact.with_name(f".{artifact.name}.{os.getpid()}.tmp")
                    torch.save(payload, tmp)
                    os.replace(tmp, artifact)
                    row = {
                        "schema": "p925-true-c-refit-delta-row-v2",
                        "status": "PASS",
                        "task_id": TASK_ID,
                        "identity": [layer, expert, projection],
                        "layer": layer,
                        "expert": expert,
                        "projection": projection,
                        "tier": tier,
                        "old_artifact": source_row["source_active_artifact"],
                        "old_artifact_bytes": source_row["source_active_artifact_bytes"],
                        "old_artifact_sha256": source_row["source_active_artifact_sha256"],
                        "old_base_codebook": source_row["base_codebook_path"],
                        "old_base_codebook_bytes": source_row["base_codebook_bytes"],
                        "old_base_codebook_sha256": source_row["base_codebook_sha256"],
                        "artifact": str(artifact),
                        "artifact_bytes": artifact.stat().st_size,
                        "artifact_sha256": sha256(artifact),
                        "codebook": str(cb_path),
                        "codebook_bytes": cb_path.stat().st_size,
                        "codebook_sha256": cb_hash,
                        "d": d,
                        "k": k,
                        "codes_dtype": str(dtype),
                        "fp16_codebook_replay_exact": True,
                        "metrics": metrics,
                    }
                    group_rows.append(row)
                    receipt = write_codebook_receipt(
                        group_paths[group], group, group_target, group_rows, cb_path, cb_hash, target_sha, build_identity_sha
                    )
                    group_state[group] = (receipt["status"], list(group_rows), receipt)
                    del W, sb, codes, scales, s_col, replay, payload
                    if (group_cell_index + 1) % 8 == 0:
                        torch.cuda.empty_cache()
                    if (group_cell_index + 1) % 4 == 0 or group_cell_index + 1 == len(group_target):
                        write_status(
                            "BUILD_CELL",
                            active_layer=layer,
                            active_tier=tier,
                            active_projection=projection,
                            codebook_cell_index=group_cell_index + 1,
                            codebook_cell_count=len(group_target),
                            completed_rows=len(all_rows) + sum(len(s[1]) for s in group_state.values() if s is not None),
                            sealed_codebooks=sealed_codebooks,
                            disk_free_bytes=shutil.disk_usage(mission).free,
                        )
                    if shutil.disk_usage(mission).free < 10 * (1 << 30):
                        raise RuntimeError("root disk floor below 10 GiB; current codebook receipt is resumable")
                del cb16, cb32, cb_cpu
                sealed = load_codebook_receipt(group_paths[group], group, group_target, target_sha, build_identity_sha)
                if sealed is None or sealed[0] != "PASS":
                    raise RuntimeError(f"codebook did not seal: {group}")
                group_state[group] = sealed
                sealed_codebooks += 1
                write_status(
                    "CODEBOOK_SEALED",
                    active_layer=layer,
                    active_tier=tier,
                    active_projection=projection,
                    sealed_codebooks=sealed_codebooks,
                    codebook_receipt=str(group_paths[group]),
                    codebook_receipt_sha256=sha256(group_paths[group]),
                    completed_rows=len(all_rows) + sum(len(s[1]) for s in group_state.values() if s is not None),
                )
                if stop_requested(run):
                    write_status("PARTIAL_PREEMPTED", active_layer=layer, completed_rows=len(all_rows), sealed_codebooks=sealed_codebooks)
                    return 75

        layer_rows = sorted([row for state in group_state.values() if state is not None for row in state[1]], key=identity)
        if len(layer_rows) != len(layer_target) or [list(identity(row)) for row in layer_rows] != expected_identity_rows(layer_target):
            raise RuntimeError(f"layer codebook fan-in mismatch: L{layer:03d}")
        layer_receipt = {
            "schema": "p925-true-c-refit-layer-v2",
            "status": "PASS",
            "task_id": TASK_ID,
            "host": os.uname().nodename,
            "layer": layer,
            "target_rows": len(layer_target),
            "expected_identities_sha256": canonical_sha(expected_identity_rows(layer_target)),
            "unique_codebooks": len(groups),
            "target_contract_sha256": target_sha,
            "build_identity_sha256": build_identity_sha,
            "codebook_receipts": [str(group_paths[group]) for group in groups],
            "codebook_receipts_sha256": canonical_sha([sha256(group_paths[group]) for group in groups]),
            "rows_sha256": canonical_sha(layer_rows),
            "rows": layer_rows,
            "completed_unix": time.time(),
        }
        atomic_json(layer_receipt_path, layer_receipt)
        all_rows.extend(layer_rows)
        del bundle, packed
        torch.cuda.empty_cache()

    if args.max_layers:
        write_status("TEST_CAP_COMPLETE", completed_layers=len(layers), completed_rows=len(all_rows))
        return 0

    if len(all_rows) != EXPECTED_TARGET_ROWS:
        raise RuntimeError(f"final target coverage mismatch: {len(all_rows)}")
    all_rows.sort(key=lambda r: (r["layer"], r["expert"], r["projection"]))
    expected_ids = {tuple(r["identity"]) for r in target}
    got_ids = {tuple(r["identity"]) for r in all_rows}
    if got_ids != expected_ids or len(got_ids) != len(all_rows):
        raise RuntimeError("final target identity mismatch")

    for index, row in enumerate(all_rows, 1):
        artifact = Path(row["artifact"])
        cb = Path(row["codebook"])
        if artifact.stat().st_size != int(row["artifact_bytes"]) or sha256(artifact) != row["artifact_sha256"]:
            raise RuntimeError(f"final artifact readback mismatch: {row['identity']}")
        if cb.stat().st_size != int(row["codebook_bytes"]) or sha256(cb) != row["codebook_sha256"]:
            raise RuntimeError(f"final codebook readback mismatch: {row['identity']}")
        if index % 64 == 0:
            write_status("FINAL_READBACK", verified_rows=index, total_rows=len(all_rows))

    delta_by_id = {tuple(row["identity"]): row for row in all_rows}
    merged_rows: list[dict] = []
    for ident, source_active_row in active_by_id.items():
        if ident not in delta_by_id:
            merged_rows.append(source_active_row)
            continue
        delta = delta_by_id[ident]
        stale_artifact_fields = {
            "schema",
            "status",
            "task_id",
            "source_task_id",
            "origin_task_id",
            "origin",
            "kind",
            "metrics",
            "provenance",
            "source_provenance",
            "producer_receipt",
            "build_receipt",
            "receipt",
            "builder_sha256",
            "canonical_builder_sha256",
            "completed_unix",
            "created_unix",
        }
        replacement = {
            **{key: value for key, value in source_active_row.items() if key not in stale_artifact_fields},
            "schema": "p925-true-c-active-row-v2",
            "status": "PASS_REPLACED_REFIT",
            "task_id": TASK_ID,
            "artifact": delta["artifact"],
            "artifact_bytes": delta["artifact_bytes"],
            "artifact_sha256": delta["artifact_sha256"],
            "codebook": delta["codebook"],
            "codebook_bytes": delta["codebook_bytes"],
            "codebook_sha256": delta["codebook_sha256"],
            "effective_method": "fresh_canonical_vq_refit_p925",
            "source_host": os.uname().nodename,
            "pack_fraction": 1.0,
            "metrics": delta["metrics"],
            "builder_sha256": BUILDER_SHA,
            "p925_target_contract_sha256": target_sha,
            "p925_build_identity_sha256": build_identity_sha,
            "p925_old_base_codebook_sha256": delta["old_base_codebook_sha256"],
            "provenance": {
                "schema": "p925-true-c-replacement-provenance-v1",
                "producer_task_id": TASK_ID,
                "source_active_overlay_sha256": EXPECTED_PINS["active_overlay"],
                "source_artifact_sha256": delta["old_artifact_sha256"],
                "source_base_codebook_sha256": delta["old_base_codebook_sha256"],
                "target_contract_sha256": target_sha,
                "build_identity_sha256": build_identity_sha,
                "builder_sha256": BUILDER_SHA,
            },
        }
        merged_rows.append(replacement)
    merged_rows.sort(key=lambda r: (int(r["layer"]), int(r["expert"]), r["projection"]))
    if len(merged_rows) != len(active["rows"]):
        raise RuntimeError("merged overlay row count mismatch")
    merged_rows_sha = canonical_sha(merged_rows)

    delta_manifest = {
        "schema": "p925-true-c-refit-delta-manifest-v2",
        "status": "PASS",
        "task_id": TASK_ID,
        "host": os.uname().nodename,
        "base_wire_label": WIRE_LABEL,
        "true_c_label": TRUE_C_LABEL,
        "assembly_mode": "immutable WIRE_C_BASELINE_R active overlay plus exact replacement of every currently base-codebook-bound VQ row",
        "source_active_overlay_sha256": EXPECTED_PINS["active_overlay"],
        "source_active_rows_sha256": EXPECTED_SOURCE_ROWS_SHA,
        "source_physical_manifest_sha256": EXPECTED_PINS["physical_manifest"],
        "assignment_sha256": EXPECTED_PINS["assignment"],
        "assignment_map_sha256": EXPECTED_ASSIGNMENT_MAP_SHA,
        "base_wire_manifest_sha256": EXPECTED_BASE_WIRE_MANIFEST_SHA,
        "builder_sha256": BUILDER_SHA,
        "consumer_adapter": str(code / "p925_true_c_overlay_adapter.py"),
        "consumer_adapter_sha256": sha256(code / "p925_true_c_overlay_adapter.py"),
        "target_contract": str(inp / "P925_EXACT_REFIT_TARGET.json"),
        "target_contract_sha256": target_sha,
        "build_identity": str(inp / "P925_BUILD_IDENTITY.json"),
        "build_identity_sha256": build_identity_sha,
        "output_provenance": str(inp / "P925_OUTPUT_PROVENANCE.json"),
        "output_provenance_sha256": sha256(inp / "P925_OUTPUT_PROVENANCE.json"),
        "delta_rows": len(all_rows),
        "unique_refit_codebooks": len({(r["layer"], r["tier"], r["projection"]) for r in all_rows}),
        "delta_rows_sha256": canonical_sha(all_rows),
        "merged_active_rows_sha256": merged_rows_sha,
        "all_vq_fp16_codebook_replay_exact": True,
        "changed_cell_hash_readback_pass": True,
        "rows": all_rows,
        "completed_unix": time.time(),
    }
    delta_path = out / "P925_REFIT_DELTA_MANIFEST.json"
    atomic_json(delta_path, delta_manifest)

    true_c_overlay = {
        **{k: v for k, v in active.items() if k != "rows"},
        "schema": "p885-wire-c-active-overlay-v1",
        "status": "PASS_EXACT_ACTIVE_LAYERS",
        "task_id": TASK_ID,
        "source_task_id": active.get("task_id"),
        "wire_label": TRUE_C_LABEL,
        "base_wire_label": WIRE_LABEL,
        "active_rows_sha256": merged_rows_sha,
        "rows": merged_rows,
        "codebook_deviation_disclosure": {
            "policy": "TRUE-C P925 exact refit overlay",
            "restored_vq_cells_use_frozen_base_wire_codebooks": False,
            "p925_refit_delta_rows": len(all_rows),
            "p925_refit_unique_codebooks": delta_manifest["unique_refit_codebooks"],
            "p925_refit_delta_manifest": str(delta_path),
            "p925_refit_delta_manifest_sha256": sha256(delta_path),
            "builder_sha256": BUILDER_SHA,
        },
        "p925_refit_delta_manifest": str(delta_path),
        "p925_refit_delta_manifest_sha256": sha256(delta_path),
        "p925_target_contract_sha256": target_sha,
        "p925_build_identity_sha256": build_identity_sha,
        "p925_consumer_adapter": str(code / "p925_true_c_overlay_adapter.py"),
        "p925_consumer_adapter_sha256": sha256(code / "p925_true_c_overlay_adapter.py"),
        "locally_altered": True,
        "stale": False,
        "definitive": True,
    }
    overlay_path = out / "WIRE_C_TRUE_C_ACTIVE_OVERLAY.json"
    atomic_json(overlay_path, true_c_overlay)

    done = {
        "schema": "p925-true-c-refit-done-v2",
        "status": "PASS",
        "task_id": TASK_ID,
        "host": os.uname().nodename,
        "true_c_label": TRUE_C_LABEL,
        "assignment_sha256": EXPECTED_PINS["assignment"],
        "assignment_map_sha256": EXPECTED_ASSIGNMENT_MAP_SHA,
        "target_contract_sha256": target_sha,
        "build_identity_sha256": build_identity_sha,
        "target_rows": len(all_rows),
        "unique_refit_codebooks": delta_manifest["unique_refit_codebooks"],
        "delta_manifest": str(delta_path),
        "delta_manifest_sha256": sha256(delta_path),
        "active_overlay": str(overlay_path),
        "active_overlay_sha256": sha256(overlay_path),
        "merged_active_rows_sha256": merged_rows_sha,
        "completed_unix": time.time(),
    }
    atomic_json(out / "DONE.json", done)
    write_status(
        "PASS",
        done_schema=done["schema"],
        done_status=done["status"],
        done_sha256=sha256(out / "DONE.json"),
        target_contract_sha256=target_sha,
        build_identity_sha256=build_identity_sha,
        target_rows=done["target_rows"],
        unique_refit_codebooks=done["unique_refit_codebooks"],
    )
    print(json.dumps(done, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        try:
            mission_arg = Path(sys.argv[1]).resolve()
            atomic_json(
                mission_arg / "run" / "STATUS.json",
                {
                    "schema": "p925-true-c-refit-status-v1",
                    "state": "FAILED",
                    "task_id": TASK_ID,
                    "host": os.uname().nodename,
                    "pid": os.getpid(),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                    "updated_unix": time.time(),
                },
            )
        except Exception:
            pass
        raise
    raise SystemExit(exit_code)
