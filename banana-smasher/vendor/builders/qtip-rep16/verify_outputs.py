#!/usr/bin/env python3
"""Independent structural/hash verifier and sealed config emitter for PUBLIC_TASK."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from typing import Any

import torch


TASK = "PUBLIC_TASK"
MISSION = Path.home() / "run-bundles/QTIP_RATE_VERIFY_PUBLIC_TASK_s7"
PRIMARY_STATUS = (
    Path.home()
    / "run-bundles/QTIP_ANCHOR_WIRE_PUBLIC_TASK_s7/status/WIRE_DRIVER.json"
)
CLAIM = Path.home() / "HOST_CLAIM.json"
EXPECTED_PROJECTIONS = {"fused13", "down"}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def wait_primary_claim_catchup(expected_claim_sha: str) -> dict[str, Any]:
    deadline = time.time() + 30
    last: dict[str, Any] | None = None
    while time.time() < deadline:
        last = read_json(PRIMARY_STATUS)
        if last.get("claim_sha256") == expected_claim_sha:
            return last
        time.sleep(2)
    raise RuntimeError(
        f"primary status did not catch restored claim within 30s: "
        f"{last.get('claim_sha256') if last else None} != {expected_claim_sha}"
    )


def main() -> int:
    if socket.gethostname() != "compute-node-7":
        raise RuntimeError(f"hard host scope violation: {socket.gethostname()}")
    config_path = MISSION / "receipts/QTIP_2BPW_CONFIG_PASS.json"
    driver_path = MISSION / "status/RATE_SMOKE.DONE.json"
    release_path = MISSION / "receipts/TEMP_GPU_SUBLEASE_RELEASE.json"
    rate_path = MISSION / "receipts/RATE_REACHABILITY.json"
    stage_path = MISSION / "receipts/FIT_STAGE_DONE.json"
    manifest_path = MISSION / "receipts/SOURCE_MANIFEST.json"
    config = read_json(config_path)
    driver = read_json(driver_path)
    release = read_json(release_path)
    rate = read_json(rate_path)
    stage = read_json(stage_path)
    manifest = read_json(manifest_path)

    if config.get("status") != "PASS" or config.get("task") != TASK:
        raise RuntimeError("2.0 config receipt is not current-task PASS")
    if driver.get("status") != "PASS" or driver.get("task") != TASK:
        raise RuntimeError("rate-smoke driver receipt is not current-task PASS")
    if sha256(config_path) != driver["config_receipt_sha256"]:
        raise RuntimeError("driver/config receipt SHA mismatch")
    if release.get("status") != "PASS" or not release.get("primary_owner_restored"):
        raise RuntimeError("temporary GPU sublease release did not restore primary")
    if rate.get("status") != "PASS" or rate.get("task") != TASK:
        raise RuntimeError("rate reachability receipt is not current-task PASS")
    if stage.get("status") != "PASS" or stage.get("capture_count") != 128:
        raise RuntimeError("fit stage receipt is not exact 128-window PASS")
    if manifest.get("status") != "PASS" or manifest.get("task") != TASK:
        raise RuntimeError("source manifest is not current-task PASS")

    targets = {float(row["target_bpw"]): row for row in rate["targets"]}
    if targets[1.5]["reachable_current_uniform_codebook"]:
        raise RuntimeError("1.5 unexpectedly marked reachable")
    if [row["K"] for row in targets[2.0]["passing_configs"]] != [2]:
        raise RuntimeError("2.0 passing config is not exact K=2")

    units = []
    seen = set()
    for unit in config.get("units", []):
        identity = unit["identity"]
        projection = identity["projection"]
        seen.add(projection)
        done_path = Path(unit["done_path"])
        artifact_path = Path(unit["artifact"]["path"])
        done = read_json(done_path)
        artifact_sha = sha256(artifact_path)
        payload = torch.load(
            artifact_path, map_location="cpu", mmap=True, weights_only=True
        )
        shape = tuple(int(value) for value in payload["shape"])
        logical_bytes = sum(
            payload[key].numel() * payload[key].element_size()
            for key in ("trellis", "SU", "SV", "Wscale")
        )
        logical_bpw = logical_bytes * 8.0 / math.prod(shape)
        checks = {
            "config_status": unit.get("status") == "PASS",
            "done_status": done.get("status") == "PASS",
            "task": unit.get("task") == TASK and done.get("task") == TASK,
            "layer": int(identity["layer"]) == 22,
            "expert": int(identity["expert"]) == 2,
            "projection": projection in EXPECTED_PROJECTIONS,
            "target": float(identity["target_bpw"]) == 2.0,
            "geometry": payload["geometry"] == {"L": 16, "K": 2, "V": 2, "tlut_bits": 9, "decode_mode": "quantlut_sym", "td_x": 16, "td_y": 16},
            "artifact_bytes": artifact_path.stat().st_size == int(unit["artifact"]["bytes"]),
            "artifact_sha256": artifact_sha == unit["artifact"]["sha256"],
            "done_sha256": sha256(done_path) == unit["done_sha256"],
            "logical_bytes": logical_bytes == int(unit["logical_wire_bytes_excluding_shared_tlut"]),
            "logical_bpw": abs(logical_bpw - float(unit["logical_bpw_excluding_shared_tlut"])) < 1e-12,
            "target_tolerance": abs(logical_bpw - 2.0) <= 0.15,
            "build_seconds": math.isfinite(float(unit["build"]["build_wall_seconds"])) and float(unit["build"]["build_wall_seconds"]) > 0,
            "all_gates": all(bool(value) for value in unit["gates"].values()),
            "decode_loader_open": bool(unit["packed_readback"]["loader_open"]),
            "decode_finite": bool(unit["packed_readback"]["packed_decode_finite"]),
            "decode_fp16_bit_exact": bool(unit["packed_readback"]["packed_decode_fp16_bit_exact"]),
            "decode_sha_match": unit["packed_readback"]["packed_decode_fp16_sha256"] == unit["packed_readback"]["expected_reconstruction_fp16_sha256"],
            "trellis_kernel_16bit": payload["trellis"].dtype in (torch.int16, torch.uint16),
            "su_fp16": payload["SU"].dtype == torch.float16,
            "sv_fp16": payload["SV"].dtype == torch.float16,
        }
        if not all(checks.values()):
            raise RuntimeError(f"unit verification failed {projection}: {checks}")
        units.append(
            {
                "identity": identity,
                "shape": list(shape),
                "logical_wire_bytes_excluding_shared_tlut": logical_bytes,
                "logical_bpw_excluding_shared_tlut": logical_bpw,
                "build_wall_seconds": float(unit["build"]["build_wall_seconds"]),
                "quant_seconds": float(unit["build"]["quant_seconds"]),
                "trellis_dtype": str(payload["trellis"].dtype),
                "artifact": {**unit["artifact"], "sha256_readback": artifact_sha},
                "done_path": str(done_path),
                "done_sha256": unit["done_sha256"],
                "decode": unit["packed_readback"],
                "checks": checks,
            }
        )
        del payload
    if seen != EXPECTED_PROJECTIONS or len(units) != 2:
        raise RuntimeError(f"unit identity closure failed: {seen}, count={len(units)}")

    claim_sha = sha256(CLAIM)
    claim = read_json(CLAIM)
    if claim.get("owner") != "PUBLIC_TASK" or claim.get("task_id") != "PUBLIC_TASK":
        raise RuntimeError("canonical primary claim was not restored")
    if claim.get("temporary_gpu_sublease"):
        raise RuntimeError("temporary GPU sublease remains present")
    if claim_sha != release["claim_restored_sha256"]:
        raise RuntimeError("restored claim SHA does not match release receipt")
    primary = wait_primary_claim_catchup(claim_sha)
    primary_checks = {
        "task": primary.get("task") == "PUBLIC_TASK",
        "status": primary.get("status") == "RUNNING",
        "stage": primary.get("stage") == "wait_s3_shard",
        "layer": int(primary.get("layer", -1)) in (34, 36),
        "pid_live": Path(f"/proc/{int(primary.get('pid', -1))}/cmdline").is_file(),
        "claim_sha256": primary.get("claim_sha256") == claim_sha,
    }
    if not all(primary_checks.values()):
        raise RuntimeError(f"primary post-smoke state failed: {primary_checks}")
    gpu_apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory,process_name",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    if gpu_apps:
        raise RuntimeError(f"GPU not empty after smoke: {gpu_apps}")

    active = []
    exact_targets = {
        str(MISSION / "code/run_rate_smoke.sh"),
        str(MISSION / "code/qtip_rate_smoke.py"),
    }
    for proc_path in Path("/proc").glob("[0-9]*"):
        try:
            tokens = [
                item.decode(errors="replace")
                for item in (proc_path / "cmdline").read_bytes().split(b"\0")
                if item
            ]
            if any(token in exact_targets for token in tokens):
                active.append({"pid": int(proc_path.name), "tokens": tokens})
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if active:
        raise RuntimeError(f"task smoke process remains active: {active}")

    config_sha = sha256(config_path)
    unreachable = {
        "schema": "qtip-rate-config-seal-v1",
        "status": "UNREACHABLE_CURRENT_OPTIONS",
        "task": TASK,
        "target_bpw": 1.5,
        "tolerance_bpw": 0.15,
        "uniform_unit_config": None,
        "reason": rate["hybrid_note"],
        "math": rate["math"],
        "candidate_integer_rates": targets[1.5]["candidates"],
        "rate_reachability_receipt": str(rate_path),
        "rate_reachability_receipt_sha256": sha256(rate_path),
        "created_unix": time.time(),
    }
    reachable = {
        "schema": "qtip-rate-config-seal-v1",
        "status": "PASS",
        "task": TASK,
        "target_bpw": 2.0,
        "tolerance_bpw": 0.15,
        "uniform_unit_config": config["config"],
        "selected_representative_layer": config["selected_representative_layer"],
        "expert": 2,
        "unit_proofs": [
            {
                "projection": row["identity"]["projection"],
                "logical_bpw": row["logical_bpw_excluding_shared_tlut"],
                "build_wall_seconds": row["build_wall_seconds"],
                "artifact_sha256": row["artifact"]["sha256"],
                "done_sha256": row["done_sha256"],
            }
            for row in units
        ],
        "config_receipt": str(config_path),
        "config_receipt_sha256": config_sha,
        "source_files": config["source_files"],
        "tlut_sha256": config["units"][0]["tlut_sha256"],
        "created_unix": time.time(),
    }
    unreachable_path = MISSION / "receipts/SEALED_CONFIG_1P5_UNREACHABLE.json"
    reachable_path = MISSION / "receipts/SEALED_CONFIG_2P0.json"
    atomic_json(unreachable_path, unreachable)
    atomic_json(reachable_path, reachable)

    result = {
        "schema": "qtip-rate-verify-final-v1",
        "status": "PASS",
        "task": TASK,
        "host": socket.gethostname(),
        "rate_points": {
            "1.5": {
                "verdict": "UNREACHABLE_CURRENT_OPTIONS",
                "seal": str(unreachable_path),
                "seal_sha256": sha256(unreachable_path),
            },
            "2.0": {
                "verdict": "PASS",
                "config": config["config"],
                "seal": str(reachable_path),
                "seal_sha256": sha256(reachable_path),
                "units": units,
            },
        },
        "fit_stage": stage,
        "weight_transport": {
            "receipt": config["transport_receipt"],
            "receipt_sha256": config["transport_receipt_sha256"],
        },
        "claim_restore": {
            "claim_sha256": claim_sha,
            "owner": claim["owner"],
            "primary": primary,
            "primary_checks": primary_checks,
            "gpu_apps": gpu_apps,
            "task_processes": active,
        },
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "created_unix": time.time(),
    }
    output = MISSION / "receipts/FINAL_VERIFY.json"
    atomic_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "output_sha256": sha256(output),
                "rate_points": result["rate_points"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
