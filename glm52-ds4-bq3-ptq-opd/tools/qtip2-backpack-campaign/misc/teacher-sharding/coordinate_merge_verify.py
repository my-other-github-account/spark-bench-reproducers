#!/usr/bin/env python3
"""Atomically merge and verify the sole P490 two-host TRAIN gate on compute-node-work.

The peer payload moves directly over the QSFP fabric (203.0.113.1 -> compute-node-work).
No tensor or checkpoint transits the Mac.  The published merged_outputs directory
appears only after both shard receipts, exact coverage, tensor correctness, memory,
payload, and projection gates pass.
"""
import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_STARTED_EPOCH_NS = time.time_ns()
SCRIPT_STARTED_MONO = time.perf_counter()

import torch

TASK = "PUBLIC_TASK"
FLOOR = 8 << 30
EXPECTED_PARENT_RESULT_SHA = "c06a7f15f5dfda06ffc7224d9bc8000cf5bfda2d01e0162cb6d6ba756fba1c22"
EXPECTED_PARENT_TIMING_SHA = "ad458dd39a76e465eaeaf5c52f3151fe456b1e5a6a7b6faf7595972ab1344d1c"
EXPECTED_CONTENT_IDENTITY = "64afcd59f4b61c2f59e80ffeeebd0ea7ebb1fc7636bf672876440d8585a42ee7"
EXPECTED_CORPUS_SHA = "16575db7fd180ca193aa13c4e642400b9ed416dbd0c36c3c5302422b31f5cbae"
CALIBRATION = 0.7738554738722851
BASELINE_CALIBRATED_SECONDS = 7405.118363542616
TARGET_SECONDS = 5400.0
REQUIRED_SPEEDUP = 1.371318


def memavailable() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def require_memory(phase: str) -> int:
    available = memavailable()
    if available < FLOOR:
        raise MemoryError(f"{phase}: MemAvailable {available} below {FLOOR}")
    return available


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def sha256(path: Path) -> str:
    require_memory(f"before hash read {path}")
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            require_memory(f"during hash read {path}")
            block = f.read(16 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    require_memory(f"before JSON read {path}")
    return json.loads(path.read_text())


def gpu_apps():
    p = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
         "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return [line.strip() for line in p.stdout.splitlines() if line.strip()]


def phase_seconds(started: float) -> float:
    return time.perf_counter() - started


def validate_stage_intervals(timing, role):
    expected = list(range(43))
    layers = [row["layer"] for row in timing["layer_rows"]]
    if layers != expected:
        raise RuntimeError(f"{role}: layer rows are not exact 0..42")
    prior_end = None
    for row in timing["layer_rows"]:
        iv = row["intervals_epoch_ns"]
        names = ["weight_build", "materialize", "residency_gate", "forward", "dematerialize"]
        for name in names:
            pair = iv[name]
            if len(pair) != 2 or pair[0] > pair[1]:
                raise RuntimeError(f"{role}: invalid {name} interval layer {row['layer']}")
        for left, right in zip(names, names[1:]):
            if iv[left][1] > iv[right][0]:
                raise RuntimeError(f"{role}: overlapping serial intervals {left}/{right} layer {row['layer']}")
        if prior_end is not None and prior_end > iv["weight_build"][0]:
            raise RuntimeError(f"{role}: layer stage rows overlap at layer {row['layer']}")
        prior_end = iv["dematerialize"][1]
    process = timing["process_interval_epoch_ns"]
    if not (process[0] <= timing["layer_rows"][0]["intervals_epoch_ns"]["weight_build"][0]
            <= prior_end <= process[1]):
        raise RuntimeError(f"{role}: stage intervals escape process interval")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True)
    ap.add_argument("--peer", default="fleet-user@203.0.113.1")
    ap.add_argument("--peer-mission", required=True)
    ap.add_argument("--parent-mission", required=True)
    a = ap.parse_args()

    mission = Path(a.mission).resolve()
    peer_mission = Path(a.peer_mission)
    parent = Path(a.parent_mission).resolve()
    run = mission / "run"
    receipts = mission / "receipts"
    own_outputs = mission / "outputs"
    final_dir = mission / "merged_outputs"
    stage = mission / f"merged_outputs.pending.{os.getpid()}"
    status_path = run / "COORDINATOR_STATUS.json"
    done_path = run / "COORDINATOR_DONE.json"
    final_result = receipts / "FINAL_RESULT.json"
    memory_path = run / "COORDINATOR_MEMORY.tsv"
    memory_stop = run / "COORDINATOR_MEMORY_STOP.json"
    phases = {}
    monitor = None
    error = None

    if any(p.exists() for p in (status_path, done_path, final_result, final_dir)):
        raise RuntimeError("coordinator merge is once-only; published state already exists")
    if stage.exists():
        raise RuntimeError(f"unexpected pending merge directory {stage}")

    pid = os.getpid()
    launch = {
        "schema": "two-host-teacher-bank-coordinator-launch-v1",
        "task_id": TASK,
        "host": socket.gethostname(),
        "pid": pid,
        "pgid": os.getpgid(pid),
        "sid": os.getsid(pid),
        "started_epoch_ns": SCRIPT_STARTED_EPOCH_NS,
        "peer": a.peer,
        "peer_mission": str(peer_mission),
        "publish_target": str(final_dir),
        "status": "RUNNING",
    }
    atomic_json(run / "COORDINATOR_LAUNCH.json", launch)
    atomic_json(status_path, launch)
    monitor = subprocess.Popen([
        sys.executable, str(mission / "code" / "memory_monitor.py"),
        "--watched-pid", str(pid), "--pgid", str(launch["pgid"]),
        "--floor-bytes", str(FLOOR), "--tsv", str(memory_path),
        "--stop-json", str(memory_stop),
    ])

    try:
        initial_apps = gpu_apps()
        if initial_apps:
            raise RuntimeError(f"GPU compute apps present before coordinator: {initial_apps}")
        require_memory("coordinator start")
        peer_hostname = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", a.peer, "hostname"],
            capture_output=True, text=True, check=True).stdout.strip()
        if peer_hostname != "compute-node-1":
            raise RuntimeError(f"peer identity mismatch: {peer_hostname!r}")

        p = time.perf_counter()
        inbound = stage / "peer"
        inbound.mkdir(parents=True)
        peer_files = [
            "receipts/TIMING.json",
            "receipts/SHARD_DONE.json",
            "receipts/MODEL_CONTENT_IDENTITY.json",
            "receipts/HOST_CLAIM.json",
            "run/GATE_DONE.json",
        ] + [f"outputs/t8192_win{k}.pt" for k in range(4, 8)]
        needed_peer_bytes = sum(
            int(row["candidate"]["bytes"])
            for row in read_json(mission / "inputs" / "FINAL_RESULT.json")["window_rows"]
            if row["window"] >= 4)
        disk_free_before = shutil.disk_usage(mission).free
        if disk_free_before - needed_peer_bytes < FLOOR:
            raise RuntimeError(
                f"peer merge would breach 8GiB disk floor: free={disk_free_before} "
                f"payload={needed_peer_bytes}")
        rsync = subprocess.run(
            ["rsync", "-a", "--files-from=-", f"{a.peer}:{peer_mission}/", str(inbound) + "/"],
            input="\n".join(peer_files) + "\n", text=True, capture_output=True)
        if rsync.returncode:
            raise RuntimeError(f"direct QSFP rsync failed rc={rsync.returncode}: {rsync.stderr[-1000:]}")
        phases["direct_qsfp_peer_sync_seconds"] = phase_seconds(p)

        p = time.perf_counter()
        own_timing = read_json(receipts / "TIMING.json")
        own_done = read_json(receipts / "SHARD_DONE.json")
        own_identity = read_json(receipts / "MODEL_CONTENT_IDENTITY.json")
        own_claim = read_json(receipts / "HOST_CLAIM.json")
        peer_timing = read_json(inbound / "receipts" / "TIMING.json")
        peer_done = read_json(inbound / "receipts" / "SHARD_DONE.json")
        peer_identity = read_json(inbound / "receipts" / "MODEL_CONTENT_IDENTITY.json")
        peer_claim = read_json(inbound / "receipts" / "HOST_CLAIM.json")
        if own_done.get("status") != "PASS" or peer_done.get("status") != "PASS":
            raise RuntimeError("one or both shard seals are not PASS")
        if own_timing.get("status") != "PASS" or peer_timing.get("status") != "PASS":
            raise RuntimeError("one or both timing receipts are not PASS")
        if own_timing["windows"] != [0, 1, 2, 3] or peer_timing["windows"] != [4, 5, 6, 7]:
            raise RuntimeError("shard timing coverage is not exact/disjoint")
        if sorted(own_done["windows"] + peer_done["windows"]) != list(range(8)):
            raise RuntimeError("sealed shard coverage is not exact 0..7")
        if set(own_done["windows"]) & set(peer_done["windows"]):
            raise RuntimeError("sealed shard coverage overlaps")
        if own_timing["microbatch"] != 4 or peer_timing["microbatch"] != 4:
            raise RuntimeError("global microbatch4 semantics changed")
        if own_timing["readout_mode"] != "topk" or peer_timing["readout_mode"] != "topk":
            raise RuntimeError("top8192 readout semantics changed")
        for role, claim in (("coordinator", own_claim), ("peer", peer_claim)):
            if claim.get("owner") != TASK or claim.get("state") != "CLAIMED":
                raise RuntimeError(f"{role} exact claim drifted")
        identities = {own_identity["full_content_manifest_sha256"],
                      peer_identity["full_content_manifest_sha256"]}
        if identities != {EXPECTED_CONTENT_IDENTITY}:
            raise RuntimeError(f"checkpoint full-content identity mismatch: {identities}")
        if (own_identity["sealed_train_corpus"]["sha256"] != EXPECTED_CORPUS_SHA or
                peer_identity["sealed_train_corpus"]["sha256"] != EXPECTED_CORPUS_SHA):
            raise RuntimeError("sealed TRAIN identity mismatch")
        validate_stage_intervals(own_timing, "coordinator-shard-A")
        validate_stage_intervals(peer_timing, "peer-shard-B")
        phases["receipt_and_interval_validation_seconds"] = phase_seconds(p)

        p = time.perf_counter()
        for k in range(4):
            src = own_outputs / f"t8192_win{k}.pt"
            if not src.is_file():
                raise RuntimeError(f"missing coordinator shard output {src}")
            os.link(src, stage / src.name)
        for k in range(4, 8):
            src = inbound / "outputs" / f"t8192_win{k}.pt"
            if not src.is_file():
                raise RuntimeError(f"missing peer shard output {src}")
            os.link(src, stage / src.name)
        phases["atomic_stage_assembly_seconds"] = phase_seconds(p)

        p = time.perf_counter()
        parent_result_path = mission / "inputs" / "FINAL_RESULT.json"
        parent_timing_path = mission / "inputs" / "OVERLAP_TIMING.json"
        if sha256(parent_result_path) != EXPECTED_PARENT_RESULT_SHA:
            raise RuntimeError("P489 parent result SHA mismatch")
        if sha256(parent_timing_path) != EXPECTED_PARENT_TIMING_SHA:
            raise RuntimeError("P489 parent timing SHA mismatch")
        parent_result = read_json(parent_result_path)
        parent_timing = read_json(parent_timing_path)
        reference_nll = {row["window"]: row["nll1024"] for row in parent_timing["window_rows"]}
        candidate_nll = {
            row["window"]: row["nll1024"]
            for timing in (own_timing, peer_timing)
            for row in timing["window_rows"]
        }
        reference_paths = {
            row["window"]: Path(row["candidate"]["path"])
            for row in parent_result["window_rows"]
        }
        output_rows = []
        equal_count = total_index_count = top1_equal = top1_count = 0
        nll_abs = []
        all_finite = True
        all_shapes = True
        exact_file_sha_matches = True
        minimum_coordinator_memavailable = require_memory("before tensor verification")
        for k in range(8):
            require_memory(f"before candidate tensor read win{k}")
            cand_path = stage / f"t8192_win{k}.pt"
            ref_path = reference_paths[k]
            cand = torch.load(cand_path, map_location="cpu", weights_only=True, mmap=True)
            require_memory(f"before reference tensor read win{k}")
            ref = torch.load(ref_path, map_location="cpu", weights_only=True, mmap=True)
            if set(cand) != {"idx", "logprob"} or set(ref) != {"idx", "logprob"}:
                raise RuntimeError(f"window {k}: top8192 schema mismatch")
            shape_ok = (cand["idx"].shape == cand["logprob"].shape ==
                        ref["idx"].shape == ref["logprob"].shape and
                        cand["idx"].shape[1] == 8192)
            all_shapes &= bool(shape_ok)
            finite = bool(torch.isfinite(cand["logprob"]).all().item())
            all_finite &= finite
            eq = cand["idx"].eq(ref["idx"])
            equal_count += int(eq.sum().item())
            total_index_count += eq.numel()
            top = cand["idx"][:, 0].eq(ref["idx"][:, 0])
            top1_equal += int(top.sum().item())
            top1_count += top.numel()
            delta_nll = abs(candidate_nll[k] - reference_nll[k])
            nll_abs.append(delta_nll)
            cand_sha = sha256(cand_path)
            ref_sha = sha256(ref_path)
            exact_file_sha_matches &= cand_sha == ref_sha
            minimum_coordinator_memavailable = min(
                minimum_coordinator_memavailable, require_memory(f"after tensor verification win{k}"))
            output_rows.append({
                "window": k, "path": str(cand_path), "bytes": cand_path.stat().st_size,
                "sha256": cand_sha, "p489_sha256": ref_sha,
                "sha256_equal_to_p489": cand_sha == ref_sha,
                "shape": list(cand["idx"].shape), "shape_pass": bool(shape_ok),
                "finite": finite, "top8192_indices_equal": bool(eq.all().item()),
                "top1_agreement": float(top.float().mean().item()),
                "nll1024": candidate_nll[k], "p489_nll1024": reference_nll[k],
                "nll_abs": delta_nll,
            })
            del cand, ref, eq, top
        index_fraction = equal_count / total_index_count
        top1_agreement = top1_equal / top1_count
        nll_mean_abs = sum(nll_abs) / len(nll_abs)
        phases["tensor_correctness_and_hash_seconds"] = phase_seconds(p)

        p463_timing = read_json(mission / "inputs" / "P463_TIMING.json")
        p463_result = read_json(mission / "inputs" / "P463_FINAL_RESULT.json")
        p463_projected = p463_result["projection"]["candidate_stage_projection_seconds"]
        stage_critical = {
            name: max(own_timing["stage_seconds"].get(name, 0.0),
                      peer_timing["stage_seconds"].get(name, 0.0))
            for name in set(own_timing["stage_seconds"]) | set(peer_timing["stage_seconds"])
        }
        chunk_stages = {
            "input_prepare", "layer_weight_load", "layer_materialize",
            "layer_dematerialize", "empty_cache",
        }
        per_work_stages = {
            "forward", "hc_norm", "lm_head", "normalization_nll",
            "topk_or_sort", "device_to_cpu", "torch_save_rename", "hash_ledger",
        }
        projection_rows = {}
        compute_raw = 0.0
        for name in sorted(chunk_stages | per_work_stages | {"process_setup"}):
            parent_gate = p463_timing["stage_seconds"][name]
            parent_full = p463_projected[name]
            parent_factor = parent_full / parent_gate
            # One 4-window shard gate -> one 256-window half-bank.  Per-window and
            # per-microbatch work has the same 64x shape as P463's 8->512 mapping.
            # Per-chunk/layer-fixed work executes four rather than eight 64-window
            # chunks, so preserve P463's calibrated factor at exactly one half.
            applied_factor = parent_factor / 2 if name in chunk_stages else parent_factor
            raw_full = stage_critical[name] * applied_factor
            compute_raw += raw_full
            projection_rows[name] = {
                "critical_gate_seconds": stage_critical[name],
                "p463_gate_to_full_factor": parent_factor,
                "two_host_applied_factor": applied_factor,
                "raw_halfbank_critical_seconds": raw_full,
            }
        parent_other_factor = (
            p463_projected["other_unaccounted"] / p463_timing["other_unaccounted_seconds"])
        critical_other = max(own_timing["other_unaccounted_seconds"],
                             peer_timing["other_unaccounted_seconds"])
        other_full = critical_other * parent_other_factor / 2
        projection_rows["other_unaccounted"] = {
            "critical_gate_seconds": critical_other,
            "p463_gate_to_full_factor": parent_other_factor,
            "two_host_applied_factor": parent_other_factor / 2,
            "raw_halfbank_critical_seconds": other_full,
        }
        compute_raw += other_full
        # Conservative: charge every gate-exposed I/O wait four times even though
        # 64-window forward should hide all but the first layer, unlike the 4-window gate.
        exposed_io_full = stage_critical["weight_io_exposed_wait"] * 4
        residency_full = stage_critical["residency_gate"] * 4
        compute_raw += exposed_io_full + residency_full

        model_start = min(own_timing["process_interval_epoch_ns"][0],
                          peer_timing["process_interval_epoch_ns"][0])
        model_end = max(own_timing["process_interval_epoch_ns"][1],
                        peer_timing["process_interval_epoch_ns"][1])
        model_common_clock_wall = (model_end - model_start) / 1e9
        model_overlap = max(
            0, min(own_timing["process_interval_epoch_ns"][1], peer_timing["process_interval_epoch_ns"][1])
            - max(own_timing["process_interval_epoch_ns"][0], peer_timing["process_interval_epoch_ns"][0])) / 1e9

        coordinator_overhead = time.perf_counter() - SCRIPT_STARTED_MONO
        # Merge/verification touches 4 peer + 8 total files; full-bank has exactly
        # 64x each count. Scale the entire measured coordinator path conservatively.
        coordinator_full = coordinator_overhead * 64
        compute_calibrated = compute_raw * CALIBRATION
        calibrated_full = compute_calibrated + coordinator_full
        speedup = BASELINE_CALIBRATED_SECONDS / calibrated_full
        paired_critical_wall = model_common_clock_wall + coordinator_overhead

        external_min = min(own_done["minimum_external_memavailable_bytes"],
                           peer_done["minimum_external_memavailable_bytes"],
                           minimum_coordinator_memavailable)
        internal_min = min(own_done["minimum_internal_memavailable_bytes"],
                           peer_done["minimum_internal_memavailable_bytes"])
        payload_pass = all(
            done["max_logical_layer_payloads"] <= 2 and
            done["max_actual_resident_source_bytes"] <=
            done["max_layer_shard_bytes"] + done["resident_source_allowance_bytes"]
            for done in (own_done, peer_done))
        hard_gates = {
            "calibrated_full512_le_5400": calibrated_full <= TARGET_SECONDS,
            "speedup_ge_1_371318": speedup >= REQUIRED_SPEEDUP,
            "finite_outputs": all_finite,
            "shape_and_top8192_schema": all_shapes,
            "identical_top8192_indices": index_fraction == 1.0,
            "top1_agreement_eq_1": top1_agreement == 1.0,
            "nll_mean_abs_le_0_05": nll_mean_abs <= 0.05,
            "exact_disjoint_coverage_0_7": True,
            "memory_floor_ge_8gib": external_min >= FLOOR and internal_min >= FLOOR and not memory_stop.exists(),
            "payload_bound": payload_pass,
            "checkpoint_and_train_identity": True,
            "no_gpu_payload_after_shards": not gpu_apps(),
        }
        if not all(hard_gates.values()):
            raise RuntimeError(f"hard gate failure: {hard_gates}")

        # Remove transport-only peer tree before publishing.  Tensor hardlinks in
        # stage remain valid; only the eight root outputs and manifest are published.
        shutil.rmtree(inbound)
        manifest = {
            "schema": "two-host-teacher-bank-atomic-merge-manifest-v1",
            "task_id": TASK, "windows": list(range(8)),
            "coverage": {"compute-node-work": [0, 1, 2, 3], "compute-node-1": [4, 5, 6, 7]},
            "output_rows": output_rows,
            "created_epoch_ns": time.time_ns(),
        }
        atomic_json(stage / "MERGE_MANIFEST.json", manifest)
        for path in stage.iterdir():
            if path.is_file():
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        dfd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        os.replace(stage, final_dir)
        dfd = os.open(mission, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        phases["atomic_publish_seconds"] = phase_seconds(p)

        final = {
            "schema": "two-host-teacher-bank-sharded-gate-result-v1",
            "status": "PASS", "verdict": "PASS_TWO_HOST_SHARDING_TARGET",
            "task_id": TASK, "host": socket.gethostname(),
            "source_identity": {
                "checkpoint_full_content_manifest_sha256": EXPECTED_CONTENT_IDENTITY,
                "sealed_train_corpus_sha256": EXPECTED_CORPUS_SHA,
                "p489_result_sha256": EXPECTED_PARENT_RESULT_SHA,
                "p489_timing_sha256": EXPECTED_PARENT_TIMING_SHA,
                "teacher_bank_sharded_sha256": sha256(mission / "code" / "teacher_bank_sharded.py"),
                "coordinator_sha256": sha256(mission / "code" / "coordinate_merge_verify.py"),
                "seal_shard_sha256": sha256(mission / "code" / "seal_shard.py"),
                "patch_sha256": sha256(mission / "code" / "teacher_bank_two_host_sharding.patch"),
                "same_eager_semantics": True, "same_microbatch": 4,
                "same_topk": 8192,
            },
            "shards": {
                "compute-node-work": {"windows": [0, 1, 2, 3], "timing": own_timing,
                               "done": own_done, "identity": own_identity},
                "compute-node-1": {"windows": [4, 5, 6, 7], "timing": peer_timing,
                            "done": peer_done, "identity": peer_identity},
            },
            "paired_gate": {
                "process_intervals_epoch_ns": {
                    "compute-node-work": own_timing["process_interval_epoch_ns"],
                    "compute-node-1": peer_timing["process_interval_epoch_ns"],
                },
                "common_clock_model_wall_seconds": model_common_clock_wall,
                "concurrent_model_overlap_seconds": model_overlap,
                "coordinator_overhead_seconds": coordinator_overhead,
                "critical_path_wall_with_coordinator_seconds": paired_critical_wall,
                "coordinator_phase_seconds": phases,
                "atomic_merge_path": str(final_dir),
                "atomic_merge_manifest_sha256": sha256(final_dir / "MERGE_MANIFEST.json"),
            },
            "reported_stage_seconds": {
                role: {
                    "setup": timing["stage_seconds"]["process_setup"],
                    "weight_io_raw": timing["stage_seconds"]["weight_io"],
                    "weight_io_exposed_wait": timing["stage_seconds"]["weight_io_exposed_wait"],
                    "weight_io_forward_overlap": timing["stage_seconds"]["weight_io_forward_overlap"],
                    "weight_build_after_prefetch": timing["stage_seconds"]["layer_weight_load"],
                    "materialize": timing["stage_seconds"]["layer_materialize"],
                    "residency_gate": timing["stage_seconds"]["residency_gate"],
                    "forward": timing["stage_seconds"]["forward"],
                    "dematerialize": timing["stage_seconds"]["layer_dematerialize"],
                    "readout_total": sum(timing["stage_seconds"][name] for name in (
                        "hc_norm", "lm_head", "normalization_nll", "topk_or_sort",
                        "device_to_cpu", "torch_save_rename", "hash_ledger", "empty_cache")),
                    "wall": timing["wall_seconds"],
                }
                for role, timing in (("compute-node-work", own_timing), ("compute-node-1", peer_timing))
            },
            "correctness": {
                "all_finite": all_finite, "shape_pass": all_shapes,
                "top8192_index_equal_fraction": index_fraction,
                "top1_agreement": top1_agreement,
                "nll_mean_abs": nll_mean_abs,
                "all_output_file_sha256_equal_p489": exact_file_sha_matches,
                "window_rows": output_rows,
            },
            "memory_and_payload": {
                "floor_bytes": FLOOR,
                "minimum_external_or_coordinator_memavailable_bytes": external_min,
                "minimum_internal_memavailable_bytes": internal_min,
                "payload_pass": payload_pass,
                "compute-node-work_max_actual_resident_source_bytes": own_done["max_actual_resident_source_bytes"],
                "compute-node-1_max_actual_resident_source_bytes": peer_done["max_actual_resident_source_bytes"],
                "max_layer_shard_bytes": max(own_done["max_layer_shard_bytes"], peer_done["max_layer_shard_bytes"]),
                "resident_source_allowance_bytes": max(own_done["resident_source_allowance_bytes"], peer_done["resident_source_allowance_bytes"]),
                "max_logical_payloads": max(own_done["max_logical_layer_payloads"], peer_done["max_logical_layer_payloads"]),
                "coordinator_memory_watchdog_fired": memory_stop.exists(),
            },
            "projection": {
                "method": "Preserve P463 empirical gate-to-full stage factors and P489 raw-to-historical calibration. Apply unchanged factor to 4-window->256-window per-window/microbatch stages; half factor to per-chunk/layer-fixed stages because each host owns four rather than eight 64-window chunks; charge max host per stage, four times all gate-exposed I/O waits and mincore residency gates, plus 64x the entire measured coordinator path without calibration.",
                "stage_rows": projection_rows,
                "conservative_exposed_io_full_seconds": exposed_io_full,
                "residency_gate_full_seconds": residency_full,
                "raw_compute_critical_seconds": compute_raw,
                "raw_to_historical_calibration_factor": CALIBRATION,
                "calibrated_compute_seconds": compute_calibrated,
                "conservative_coordinator_full_seconds": coordinator_full,
                "calibrated_full512_seconds": calibrated_full,
                "baseline_calibrated_seconds": BASELINE_CALIBRATED_SECONDS,
                "speedup_vs_7405": speedup,
                "target_seconds": TARGET_SECONDS,
                "required_speedup": REQUIRED_SPEEDUP,
            },
            "hard_gates": hard_gates,
            "output_rows": output_rows,
            "completed_epoch_ns": time.time_ns(),
        }
        atomic_json(final_result, final)
        final["final_result_sha256"] = sha256(final_result)
        final["status"] = "PASS"
        atomic_json(status_path, final)
        atomic_json(done_path, final)
        print(json.dumps({
            "status": "PASS", "verdict": final["verdict"],
            "critical_path_wall_seconds": paired_critical_wall,
            "coordinator_overhead_seconds": coordinator_overhead,
            "calibrated_full512_seconds": calibrated_full,
            "speedup_vs_7405": speedup,
            "top8192_index_equal_fraction": index_fraction,
            "top1_agreement": top1_agreement, "nll_mean_abs": nll_mean_abs,
            "final_result_sha256": final["final_result_sha256"],
        }, sort_keys=True), flush=True)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        atomic_json(status_path, {
            **launch, "status": "FAIL", "error": error,
            "failed_epoch_ns": time.time_ns(), "phases": phases,
            "pending_stage": str(stage) if stage.exists() else None,
        })
        raise
    finally:
        if monitor is not None and monitor.poll() is None:
            monitor.terminate()
            try:
                monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()


if __name__ == "__main__":
    main()
