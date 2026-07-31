#!/usr/bin/env python3
"""Fail-closed validation shared by the BANANA_SMASHER full512 waiter and evaluator."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
from collections import Counter, defaultdict
from typing import Mapping

HEX64 = re.compile(r"[0-9a-f]{64}")
CLAIM_SCHEMA = "banana_smasher-pre-repair-full512-remediation-host-claim-v1"
EXPECTED_HOST = "compute-node-1"
EXPECTED_SOURCE = "203.0.113.9"
EXPECTED_TRANSPORT = "QSFP only, source read-only"
EXPECTED_CLASSES = {"agentic", "chat", "code", "multilingual", "prose", "reasoning"}
ORIGINAL_COMMAND_SHA = "a5ceec2c136d6b740d5222950609e01bac9141fb81ae963db609a72b04382f02"
ORIGINAL_READER_SHA = "57b4b1537bf2931f33f67b043b3d70b92c5559647e67cbb81b0976af413fdb69"
ORIGINAL_LATCH_SHA = "499b7ca2ba6965240d12df75b8da0c17199d1b9a04079bb79b27c46882d8fea0"
FLOOR = 20 * (1 << 30)


def require_unoptimized() -> None:
    if sys.flags.optimize:
        raise RuntimeError("optimized Python is forbidden for full512 safety gates")


def _finite(value: object, name: str, *, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} is not numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return number


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name} is not a lowercase sha256")
    return value


def sha256_path(path: Path, chunk: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_preserved_outputs(out: Path, done_path: Path, *, expected_windows: list[int]) -> str:
    """Authenticate a builder checkpoint before admitting a resume."""
    if expected_windows != list(range(len(expected_windows))):
        raise ValueError("preserved windows must be a contiguous prefix")
    rows: list[dict[str, object]] = []
    try:
        for raw in done_path.read_text().splitlines():
            if raw.strip():
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise ValueError("DONE row is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("preserved DONE ledger unreadable") from exc
    if [row.get("win") for row in rows] != expected_windows:
        raise ValueError("preserved DONE windows are not the exact ordered prefix")
    actual = sorted(
        int(match.group(1))
        for path in out.glob("kld_win*.pt")
        if (match := re.fullmatch(r"kld_win(\d+)\.pt", path.name))
    )
    if actual != expected_windows:
        raise ValueError("preserved output files are not the exact ordered prefix")
    output_set: list[dict[str, object]] = []
    for win, row in zip(expected_windows, rows, strict=True):
        path = out / f"kld_win{win}.pt"
        if row.get("file") != path.name or row.get("mode") != "planes":
            raise ValueError(f"preserved DONE identity mismatch win{win}")
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        if row.get("md5") != md5:
            raise ValueError(f"preserved DONE md5 mismatch win{win}")
        output_set.append({"win": win, "sha256": sha256_path(path)})
    return hashlib.sha256(
        json.dumps(output_set, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def validate_live_output_set(
    out: Path, done_path: Path, result: Mapping[str, object],
    authorization_path: Path, *, preserved_prefix_windows: int,
) -> None:
    """Rehash the exact live output/ledger transaction before host release."""
    per_window = result.get("per_window")
    if not isinstance(per_window, list) or [row.get("win") for row in per_window] != list(range(512)):
        raise ValueError("live output result surface invalid")
    try:
        ledger = [json.loads(line) for line in done_path.read_text().splitlines() if line.strip()]
        authorization = json.loads(authorization_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("live output ledger/authorization unreadable") from exc
    if [row.get("win") for row in ledger] != list(range(512)):
        raise ValueError("live output ledger is not exact ordered full512")
    actual = sorted(
        int(match.group(1))
        for path in out.glob("kld_win*.pt")
        if (match := re.fullmatch(r"kld_win(\d+)\.pt", path.name))
    )
    if actual != list(range(512)):
        raise ValueError("live output file set is not exact full512")
    output_set: list[dict[str, object]] = []
    for win, (receipt_row, ledger_row) in enumerate(zip(per_window, ledger, strict=True)):
        path = out / f"kld_win{win}.pt"
        sha_digest = hashlib.sha256()
        md5_digest = hashlib.md5()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1 << 20), b""):
                    sha_digest.update(block)
                    md5_digest.update(block)
        except OSError as exc:
            raise ValueError(f"live output unreadable win{win}") from exc
        observed_sha = sha_digest.hexdigest()
        if (
            receipt_row.get("sha256") != observed_sha
            or int(receipt_row.get("bytes", -1)) != path.stat().st_size
            or ledger_row.get("file") != path.name
            or ledger_row.get("mode") != "planes"
            or ledger_row.get("md5") != md5_digest.hexdigest()
        ):
            raise ValueError(f"live output identity mismatch win{win}")
        output_set.append({"win": win, "sha256": observed_sha})
    output_set_sha = hashlib.sha256(
        json.dumps(output_set, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if result.get("window_output_set_sha256") != output_set_sha:
        raise ValueError("live output full set hash mismatch")
    if preserved_prefix_windows <= 0 or preserved_prefix_windows >= 512:
        raise ValueError("live output preserved prefix boundary invalid")
    prefix_sha = hashlib.sha256(
        json.dumps(output_set[:preserved_prefix_windows], separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if (
        result.get("preserved_output_set_sha256") != prefix_sha
        or authorization.get("preserved_output_set_sha256") != prefix_sha
    ):
        raise ValueError("live output preserved prefix hash mismatch")


def find_scoped_processes(
    ps_rows: list[str], *, mission: str, waiter_script: str, child_script: str,
    protected_pgid: int | None = None, protected_sid: int | None = None,
) -> list[str]:
    """Return processes belonging to the exact mission, including relative launch argv."""
    needles = (mission, waiter_script, child_script)
    found: list[str] = []
    for raw in ps_rows:
        row = raw.strip()
        fields = row.split(maxsplit=4)
        in_protected_group = False
        if len(fields) >= 4:
            try:
                pgid, sid = int(fields[2]), int(fields[3])
                in_protected_group = (
                    (protected_pgid is not None and pgid == protected_pgid)
                    or (protected_sid is not None and sid == protected_sid)
                )
            except ValueError:
                pass
        if row and (any(needle in row for needle in needles) or in_protected_group):
            found.append(row)
    return found


def validate_resumed_window_loader_proof(
    per_window: object, proof: Mapping[str, object], *,
    expected_task: str, expected_loader_sha256: str,
    expected_input_identity_sha256: str,
    expected_loader_source_path: Path, expected_sentinel_path: Path,
) -> None:
    """Bind every resumed-window receipt to the active mmap loader and input set."""
    if not isinstance(per_window, list) or not isinstance(proof, Mapping):
        raise ValueError("loader proof per-window surface invalid")
    resumed_from = int(proof.get("resumed_from_window", -1))
    if resumed_from != 64 or len(per_window) != 512:
        raise ValueError("loader proof resume boundary invalid")
    pinned_loader = _hex(expected_loader_sha256, "expected loader source hash")
    pinned_input = _hex(expected_input_identity_sha256, "expected loader input identity")
    if not expected_loader_source_path.is_file() or sha256_path(expected_loader_source_path) != pinned_loader:
        raise ValueError("live loader source hash mismatch")
    if proof.get("mode") != "torch-mmap" or proof.get("loader_sha256") != pinned_loader:
        raise ValueError("loader proof does not match pinned loader")
    if proof.get("input_identity_sha256") != pinned_input:
        raise ValueError("loader proof input identity mismatch")
    if proof.get("sentinel") != str(expected_sentinel_path) or not expected_sentinel_path.is_file():
        raise ValueError("loader proof live sentinel path mismatch")
    live_sentinel_sha = sha256_path(expected_sentinel_path)
    if proof.get("sentinel_sha256") != live_sentinel_sha:
        raise ValueError("loader proof live sentinel hash mismatch")
    try:
        sentinel = json.loads(expected_sentinel_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("loader live sentinel unreadable") from exc
    exact_sentinel = {
        "schema": "banana_smasher-arm4-mmap-sentinel-v1",
        "status": "ACTIVE_ON_PATH",
        "task_id": expected_task,
        "mode": "torch-mmap",
        "fallback": "torch-eager",
        "loader_source": str(expected_loader_source_path),
        "loader_sha256": pinned_loader,
        "input_identity_sha256": pinned_input,
        "first_layer": 0,
    }
    for field, value in exact_sentinel.items():
        if sentinel.get(field) != value:
            raise ValueError(f"loader live sentinel {field} mismatch")
    expected = {
        "loader_mode": "torch-mmap",
        "loader_sha256": pinned_loader,
        "loader_sentinel_sha256": live_sentinel_sha,
        "input_identity_sha256": pinned_input,
    }
    chunk_rows = proof.get("chunk_receipts")
    if not isinstance(chunk_rows, list) or len(chunk_rows) != 7:
        raise ValueError("loader proof chunk receipt surface invalid")
    for chunk_index, proof_row in enumerate(chunk_rows, start=1):
        start = chunk_index * 64
        stop = start + 63
        expected_path = expected_sentinel_path.parent / f"ARM4_MMAP_CHUNK_{start:03d}_{stop:03d}.json"
        if not isinstance(proof_row, Mapping) or proof_row.get("path") != str(expected_path) or not expected_path.is_file():
            raise ValueError(f"loader chunk receipt path mismatch chunk{chunk_index}")
        live_chunk_sha = sha256_path(expected_path)
        progress_sha = _hex(proof_row.get("loader_progress_sha256"), "loader progress hash")
        if proof_row.get("sha256") != live_chunk_sha or proof_row.get("windows") != [start, stop]:
            raise ValueError(f"loader chunk receipt proof mismatch chunk{chunk_index}")
        try:
            chunk = json.loads(expected_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"loader chunk receipt unreadable chunk{chunk_index}") from exc
        exact_chunk = {
            "schema": "banana_smasher-arm4-mmap-chunk-v1",
            "status": "PASS_ON_PATH",
            "task_id": expected_task,
            "mode": "torch-mmap",
            "loader_sha256": pinned_loader,
            "input_identity_sha256": pinned_input,
            "sentinel_sha256": live_sentinel_sha,
            "window_ids": list(range(start, stop + 1)),
            "stream_completed_layers": list(range(43)) * (chunk_index + 1),
            "mmap_completed_layers": list(range(43)) * chunk_index,
            "loader_progress_sha256": progress_sha,
        }
        for field, value in exact_chunk.items():
            if chunk.get(field) != value:
                raise ValueError(f"loader chunk receipt {field} mismatch chunk{chunk_index}")
        for expected_win, row in enumerate(per_window[start:stop + 1], start=start):
            if not isinstance(row, Mapping) or row.get("win") != expected_win:
                raise ValueError("loader proof window identity invalid")
            row_expected = {
                **expected,
                "loader_progress_sha256": progress_sha,
                "loader_chunk_receipt_sha256": live_chunk_sha,
            }
            for field, value in row_expected.items():
                if row.get(field) != value:
                    raise ValueError(f"loader proof {field} mismatch win{expected_win}")


def validate_claim(claim: Mapping[str, object], *, task_id: str, mission: str, now: float | None = None) -> None:
    required = {
        "schema", "host", "owner", "task", "task_id", "purpose", "mission",
        "claimed_at_epoch", "lease_until_unix", "nonce", "exact_cas_from_sha256",
        "previous_claim_sha256", "previous_release_receipt",
        "previous_release_receipt_sha256", "gpu_apps_empty_at_claim",
        "gpu_utilization_zero_at_claim", "gpu_snapshot_at_claim", "free_bytes_at_claim",
        "minimum_free_bytes", "source_host", "transport", "scope", "launch_policy",
        "no_services",
    }
    missing = sorted(required - set(claim))
    if missing:
        raise ValueError(f"claim fields missing: {missing}")
    exact = {
        "schema": CLAIM_SCHEMA,
        "host": EXPECTED_HOST,
        "owner": task_id,
        "task": task_id,
        "task_id": task_id,
        "mission": mission,
        "source_host": EXPECTED_SOURCE,
        "transport": EXPECTED_TRANSPORT,
    }
    for key, expected in exact.items():
        if claim.get(key) != expected:
            raise ValueError(f"claim {key} mismatch")
    if claim.get("no_services") is not True:
        raise ValueError("claim no_services mismatch")
    if claim.get("gpu_apps_empty_at_claim") is not True or claim.get("gpu_utilization_zero_at_claim") is not True:
        raise ValueError("claim GPU-empty assertions missing")
    if not isinstance(claim.get("nonce"), str) or len(str(claim["nonce"])) < 16:
        raise ValueError("claim nonce invalid")
    for key in ("exact_cas_from_sha256", "previous_claim_sha256", "previous_release_receipt_sha256"):
        _hex(claim.get(key), f"claim {key}")
    claimed = _finite(claim.get("claimed_at_epoch"), "claimed_at_epoch", nonnegative=True)
    lease = _finite(claim.get("lease_until_unix"), "lease_until_unix", nonnegative=True)
    if lease <= claimed:
        raise ValueError("claim lease is not after claim time")
    if now is not None and now > lease:
        raise ValueError("claim lease expired")
    minimum = int(claim.get("minimum_free_bytes", -1))
    free = int(claim.get("free_bytes_at_claim", -1))
    if minimum < FLOOR or free < minimum:
        raise ValueError("claim free-space floor invalid")
    snapshot = claim.get("gpu_snapshot_at_claim")
    if not isinstance(snapshot, Mapping) or snapshot.get("compute_apps") != []:
        raise ValueError("claim GPU snapshot invalid")
    util = snapshot.get("utilization_gpu_memory_percent")
    if util != [[0, 0]]:
        raise ValueError("claim utilization snapshot invalid")
    scope = claim.get("scope")
    if not isinstance(scope, Mapping) or set(scope) != {"remediation", "tier_s"}:
        raise ValueError("claim scope invalid")
    policy = str(claim.get("launch_policy", ""))
    for required_term in ("setsid", "nohup", "PID", "STATUS", "RESUME", "DONE", "second-SSH", "no services"):
        if required_term not in policy:
            raise ValueError(f"claim launch policy missing {required_term}")


def validate_gate_bundle(
    passed: Mapping[str, object],
    receipt: Mapping[str, object],
    wire: Mapping[str, object],
    *,
    hashes: Mapping[str, str],
    expected_code: float,
    expected_reader_sha256: str,
    expected_assignment_sha256: str,
    expected_compact_sha256: str,
    expected_builder_sha256: str,
    expected_package: str,
    expected_artifact_hashes: Mapping[str, str],
) -> dict[str, object]:
    for key in ("pass_marker_sha256", "receipt_sha256", "wire_sha256"):
        _hex(hashes.get(key), key)
        if hashes.get(key) != _hex(expected_artifact_hashes.get(key), f"expected {key}"):
            raise ValueError(f"authoritative {key} mismatch")
    if passed.get("schema") != "banana_smasher-physical-code76-pass-v1" or passed.get("status") != "PASS":
        raise ValueError("physical PASS marker status/schema invalid")
    if passed.get("task_id") != "PUBLIC_TASK" or passed.get("package_mutated") is not False:
        raise ValueError("physical PASS marker owner/package state invalid")
    if passed.get("receipt_sha256") != hashes["receipt_sha256"]:
        raise ValueError("PASS marker receipt hash mismatch")
    if passed.get("reader_sha256") != expected_reader_sha256:
        raise ValueError("PASS marker reader hash mismatch")
    if int(passed.get("layers", -1)) != 43 or int(passed.get("positions", -1)) != 77824:
        raise ValueError("physical PASS coverage mismatch")
    pass_mean = _finite(passed.get("measured_code_kld"), "PASS measured code", nonnegative=True)
    streamed = _finite(passed.get("streamed_code_kld"), "PASS streamed code", nonnegative=True)
    paired_delta = _finite(passed.get("paired_delta"), "PASS paired delta")
    if pass_mean != expected_code or streamed != expected_code or paired_delta != 0.0:
        raise ValueError("physical PASS numerical identity mismatch")

    if receipt.get("schema") != "banana_smasher-physical-code76-v1" or receipt.get("status") != "PHYSICAL_MATCHES_STREAMED":
        raise ValueError("physical receipt status/schema invalid")
    if receipt.get("task_id") != "PUBLIC_TASK" or receipt.get("host") != "compute-node-8" or receipt.get("measurement_label") != "MEASURED":
        raise ValueError("physical receipt identity invalid")
    if receipt.get("finite_gate") is not True or int(receipt.get("windows", -1)) != 76 or int(receipt.get("positions", -1)) != 77824:
        raise ValueError("physical receipt finite/count gate invalid")
    coverage = receipt.get("coverage_gate")
    if not isinstance(coverage, Mapping) or int(coverage.get("layers", -1)) != 43 or coverage.get("stream_completed_layers") != list(range(43)):
        raise ValueError("physical receipt layer coverage invalid")
    comparison = receipt.get("physical_vs_streamed")
    if not isinstance(comparison, Mapping) or comparison.get("within_tolerance") is not True:
        raise ValueError("physical paired comparison invalid")
    measured = _finite(receipt.get("measured_code_kld"), "receipt measured code", nonnegative=True)
    tolerance = _finite(comparison.get("paired_tolerance_max_2se_or_5e6"), "paired tolerance", nonnegative=True)
    physical = _finite(comparison.get("physical_mean"), "physical mean", nonnegative=True)
    streamed_mean = _finite(comparison.get("streamed_mean"), "streamed mean", nonnegative=True)
    delta = _finite(comparison.get("paired_delta_physical_minus_streamed"), "paired delta")
    se = _finite(comparison.get("paired_se"), "paired se", nonnegative=True)
    if any(abs(value - expected_code) > tolerance for value in (measured, physical, streamed_mean)) or abs(delta) > tolerance or se > tolerance:
        raise ValueError("physical code76 values exceed tolerance")

    if wire.get("schema") != "banana_smasher-materialized-wire43-v1" or wire.get("status") != "PASS_MATERIALIZED":
        raise ValueError("wire status/schema invalid")
    if wire.get("task_id") != "PUBLIC_TASK" or wire.get("host") != "compute-node-8" or wire.get("source_graph_all_declared_size_sha_verified") is not True:
        raise ValueError("wire identity invalid")
    identities = {
        "assignment_sha256": ("assignment_sha256", expected_assignment_sha256),
        "compact_manifest_sha256": ("compact_manifest_sha256", expected_compact_sha256),
        "builder_sha256": ("build_builder_sha256", expected_builder_sha256),
    }
    for wire_key, (receipt_key, expected) in identities.items():
        if receipt.get(receipt_key) != expected or wire.get(wire_key) != expected:
            raise ValueError(f"marker/wire {wire_key} mismatch")
    if receipt.get("physical_reader_sha256") != expected_reader_sha256 or receipt.get("physical_package") != expected_package:
        raise ValueError("receipt reader/package mismatch")
    layers = wire.get("layers")
    if not isinstance(layers, list) or int(wire.get("layer_count", -1)) != 43 or [int(row.get("layer", -1)) for row in layers] != list(range(43)):
        raise ValueError("wire layer coverage invalid")
    for index, row in enumerate(layers):
        if row.get("path") != f"{expected_package}/layer_{index:03d}":
            raise ValueError("wire package path mismatch")
        _hex(row.get("receipt_sha256"), f"wire layer {index} receipt")
        if int(row.get("physical_wire_bytes", -1)) <= 0:
            raise ValueError("wire layer byte count invalid")
    arithmetic = wire.get("wire_arithmetic")
    if not isinstance(arithmetic, Mapping):
        raise ValueError("wire arithmetic missing")
    physical_bytes = int(arithmetic.get("physical_layer_wire_bytes", -1))
    actual_bytes = int(arithmetic.get("actual_serialized_wire_bytes", -1))
    if physical_bytes <= 0 or actual_bytes < physical_bytes:
        raise ValueError("wire byte arithmetic invalid")
    if int(receipt.get("actual_serialized_wire_bytes", -1)) != physical_bytes or int(passed.get("actual_serialized_wire_bytes", -1)) != physical_bytes:
        raise ValueError("marker/wire physical byte identity mismatch")
    return {
        "measured_code_kld": measured,
        "tolerance": tolerance,
        "physical_layer_wire_bytes": physical_bytes,
        "actual_serialized_wire_bytes": actual_bytes,
        "layers": list(range(43)),
    }


def validate_staged_layer(root: Path, receipt: Mapping[str, object], *, required_bytes: int, free_bytes_after: int, floor: int) -> dict[str, int]:
    if free_bytes_after < floor:
        raise ValueError(f"post-rsync disk floor breached: {free_bytes_after} < {floor}")
    files = receipt.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("layer receipt files missing")
    listed: set[str] = set()
    payload_bytes = 0
    for item in files:
        rel = item.get("path")
        if not isinstance(rel, str):
            raise ValueError("layer receipt path invalid")
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts or rel in listed:
            raise ValueError("layer receipt path unsafe or duplicate")
        listed.add(rel)
        path = root / rel_path
        expected_bytes = int(item.get("bytes", -1))
        if not path.is_file() or path.stat().st_size != expected_bytes:
            raise ValueError(f"staged size mismatch: {rel}")
        if sha256_path(path) != item.get("sha256"):
            raise ValueError(f"staged sha mismatch: {rel}")
        payload_bytes += expected_bytes
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "LAYER_RECEIPT.json"
    }
    extras = sorted(actual - listed)
    missing = sorted(listed - actual)
    if extras or missing:
        raise ValueError(f"unlisted or missing staged bytes extras={extras} missing={missing}")
    source_metadata_bytes = int(receipt.get("source_metadata_bytes", -1))
    if source_metadata_bytes < 0 or source_metadata_bytes > payload_bytes:
        raise ValueError("staged source metadata byte count invalid")
    physical_payload_bytes = payload_bytes - source_metadata_bytes
    if physical_payload_bytes != int(required_bytes):
        raise ValueError(f"staged physical bytes mismatch {physical_payload_bytes}/{required_bytes}")
    receipt_path = root / "LAYER_RECEIPT.json"
    if not receipt_path.is_file():
        raise ValueError("staged layer receipt missing from scratch tree")
    tree_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    expected_tree_bytes = payload_bytes + receipt_path.stat().st_size
    if tree_bytes != expected_tree_bytes:
        raise ValueError(f"staged scratch tree byte mismatch {tree_bytes}/{expected_tree_bytes}")
    return {"payload_bytes": payload_bytes, "physical_payload_bytes": physical_payload_bytes, "actual_tree_bytes": tree_bytes, "free_bytes_after": free_bytes_after}


def validate_layer_coverage(visits: object, *, expected_chunks: int = 1) -> None:
    if not isinstance(expected_chunks, int) or expected_chunks <= 0:
        raise ValueError("full512 physical stream coverage chunk count invalid")
    expected = list(range(43))
    if not isinstance(visits, list) or len(visits) != 43 * expected_chunks:
        raise ValueError("full512 physical stream coverage must contain exact 43-layer chunks")
    for chunk in range(expected_chunks):
        observed = visits[chunk * 43:(chunk + 1) * 43]
        if observed != expected:
            raise ValueError(
                f"full512 physical stream coverage chunk {chunk} must be exact ordered L000-L042 once"
            )


def make_child_env(base: Mapping[str, str], contract: Mapping[str, str]) -> dict[str, str]:
    child = dict(base)
    child.update({str(key): str(value) for key, value in contract.items()})
    validate_runtime_environment(contract, child)
    return child


def validate_runtime_environment(contract: Mapping[str, str], env: Mapping[str, str] | None = None) -> None:
    source = os.environ if env is None else env
    mismatches = {key: (source.get(key), value) for key, value in contract.items() if source.get(key) != value}
    if mismatches:
        raise ValueError(f"runtime environment contract mismatch: {mismatches}")


def retire_scratch(root: Path) -> None:
    if not root.exists():
        return
    for child in list(root.iterdir()):
        if child.name.startswith(("layer_", ".layer_", "overlay_layer_", ".overlay_layer_")):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    assert_scratch_empty(root)


def assert_scratch_empty(root: Path) -> None:
    survivors = [] if not root.exists() else sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.exists())
    if survivors:
        raise ValueError(f"scratch retirement incomplete: {survivors[:8]}")


def parse_gpu_snapshot(apps_text: str, utilization_text: str, *, own_pid: int | None, require_zero_util: bool) -> dict[str, object]:
    apps: list[dict[str, object]] = []
    for raw in apps_text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) < 3:
            raise ValueError(f"malformed GPU app row: {raw}")
        try:
            pid = int(parts[0])
        except ValueError as exc:
            raise ValueError(f"malformed GPU app pid: {raw}") from exc
        if own_pid is not None and pid == own_pid:
            continue
        apps.append({"pid": pid, "process_name": parts[1], "used_memory": ",".join(parts[2:]).strip()})
    utilization: list[list[int]] = []
    for raw in utilization_text.splitlines():
        if not raw.strip():
            continue
        parts = [part.strip() for part in raw.split(")")] if ")" in raw else [part.strip() for part in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"malformed GPU utilization row: {raw}")
        try:
            values = [int(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"malformed GPU utilization row: {raw}") from exc
        if any(value < 0 or value > 100 for value in values):
            raise ValueError(f"GPU utilization out of range: {raw}")
        utilization.append(values)
    if not utilization:
        raise ValueError("GPU utilization output empty")
    if apps:
        raise ValueError(f"foreign GPU applications present: {apps}")
    if require_zero_util and any(gpu != 0 or memory != 0 for gpu, memory in utilization):
        raise ValueError(f"GPU utilization nonzero: {utilization}")
    return {"compute_apps": [], "utilization_gpu_memory_percent": utilization}


def validate_ready_receipt(
    ready: Mapping[str, object],
    pid_receipt: Mapping[str, object],
    *,
    expected_task: str,
    expected_claim_sha256: str,
    expected_script: str,
) -> None:
    if ready.get("schema") != "banana_smasher-full512-waiter-ready-v1" or ready.get("state") != "READY_WAITING":
        raise ValueError("waiter readiness state/schema invalid")
    _hex(expected_claim_sha256, "expected claim hash")
    for key in ("pid", "start_ticks", "cmdline", "task_id", "claim_sha256"):
        if ready.get(key) != pid_receipt.get(key):
            raise ValueError(f"waiter readiness {key} not bound to PID receipt")
    if ready.get("task_id") != expected_task or ready.get("claim_sha256") != expected_claim_sha256:
        raise ValueError("waiter readiness task/claim mismatch")
    pid = int(ready.get("pid", -1))
    if pid <= 1 or int(ready.get("start_ticks", -1)) <= 0 or int(ready.get("process_group_id", -1)) != pid:
        raise ValueError("waiter PID/start/process-group identity invalid")
    cmdline = ready.get("cmdline")
    if not isinstance(cmdline, list) or expected_script not in cmdline:
        raise ValueError("waiter cmdline identity invalid")


def _validate_summary(row: object, name: str, *, windows: int | None = None) -> None:
    if not isinstance(row, Mapping) or row.get("source_class") != name:
        raise ValueError(f"{name} summary identity invalid")
    for key in ("mean", "window_mean_se"):
        _finite(row.get(key), f"{name} {key}", nonnegative=True)
    ci = row.get("window_mean_ci95")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError(f"{name} CI invalid")
    low = _finite(ci[0], f"{name} CI low")
    high = _finite(ci[1], f"{name} CI high")
    if low > high:
        raise ValueError(f"{name} CI order invalid")
    mean = float(row["mean"])
    se = float(row["window_mean_se"])
    expected_low = mean - 1.96 * se
    expected_high = mean + 1.96 * se
    if not math.isclose(low, expected_low, rel_tol=1e-12, abs_tol=1e-12) or not math.isclose(high, expected_high, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{name} CI does not match mean/SE")
    if int(row.get("n_windows", -1)) <= 0 or int(row.get("n_positions", -1)) != int(row.get("n_windows", -1)) * 1024:
        raise ValueError(f"{name} counts invalid")
    if windows is not None and int(row.get("n_windows", -1)) != windows:
        raise ValueError(f"{name} window count invalid")


def validate_result_receipt(
    result: Mapping[str, object], *, expected_task: str,
    expected_hashes: Mapping[str, str], expected_claim_sha256: str,
    expected_command_sha256: str, expected_reader_sha256: str,
    expected_wire_bytes: int, expected_assignment_sha256: str,
    expected_compact_sha256: str,
    resume_validation: Mapping[str, object] | None = None,
    loader_validation: Mapping[str, object] | None = None,
) -> None:
    if result.get("schema") != "banana_smasher-pre-repair-physical-full512-v2" or result.get("status") != "PASS_FULL512_MEASURED":
        raise ValueError("result schema/status invalid")
    if result.get("measurement_label") != "MEASURED" or result.get("task_id") != expected_task or result.get("host") != "compute-node-1":
        raise ValueError("result identity invalid")
    if int(result.get("windows", -1)) != 512 or result.get("window_ids") != list(range(512)) or int(result.get("positions", -1)) != 512 * 1024:
        raise ValueError("result ordered full512 coverage invalid")
    expected_window_ids_sha = hashlib.sha256(",".join(map(str, range(512))).encode()).hexdigest()
    if result.get("window_ids_sha256") != expected_window_ids_sha:
        raise ValueError("result window-id set hash invalid")
    if int(result.get("microbatch", -1)) != 2 or result.get("attention") != "eager":
        raise ValueError("result MB2/eager contract invalid")
    if result.get("finite_gate") is not True:
        raise ValueError("result finite gate invalid")
    _validate_summary(result.get("global"), "global", windows=512)
    classes = result.get("by_class")
    if not isinstance(classes, Mapping) or set(classes) != EXPECTED_CLASSES:
        raise ValueError("result six-class surface invalid")
    for name in EXPECTED_CLASSES:
        _validate_summary(classes[name], name)
    _validate_summary(result.get("code"), "code", windows=76)
    if result.get("code") != classes.get("code"):
        raise ValueError("result code row alias mismatch")
    per_window = result.get("per_window")
    if not isinstance(per_window, list) or [row.get("win") for row in per_window] != list(range(512)):
        raise ValueError("result per-window order invalid")
    counts: Counter[str] = Counter()
    means: defaultdict[str, list[float]] = defaultdict(list)
    for row in per_window:
        if row.get("source_class") not in EXPECTED_CLASSES:
            raise ValueError("result per-window class invalid")
        source_class = str(row["source_class"])
        value = _finite(row.get("mean"), "per-window mean", nonnegative=True)
        counts[source_class] += 1
        means[source_class].append(value)
        if int(row.get("bytes", -1)) <= 0:
            raise ValueError("result per-window bytes invalid")
        _hex(row.get("sha256"), "result per-window hash")
    if set(counts) != EXPECTED_CLASSES or counts["code"] != 76 or sum(counts.values()) != 512:
        raise ValueError(f"result per-window class counts invalid: {counts}")
    for name in EXPECTED_CLASSES:
        row = classes[name]
        if int(row.get("n_windows", -1)) != counts[name] or int(row.get("n_positions", -1)) != counts[name] * 1024:
            raise ValueError(f"result {name} summary/per-window count mismatch")
        derived = sum(means[name]) / counts[name]
        if not math.isclose(float(row["mean"]), derived, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"result {name} summary/per-window mean mismatch")
    derived_global = sum(value for values in means.values() for value in values) / 512
    if not math.isclose(float(result["global"]["mean"]), derived_global, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("result global/per-window mean mismatch")
    window_output_set = [{"win": int(row["win"]), "sha256": row["sha256"]} for row in per_window]
    expected_output_set_sha = hashlib.sha256(json.dumps(window_output_set, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    if result.get("window_output_set_sha256") != expected_output_set_sha:
        raise ValueError("result window output set hash invalid")
    layer_receipts = result.get("layer_receipt_set")
    if not isinstance(layer_receipts, list) or [row.get("layer") for row in layer_receipts] != list(range(43)):
        raise ValueError("result layer receipt set coverage invalid")
    for row in layer_receipts:
        _hex(row.get("receipt_sha256"), "result layer receipt hash")
    expected_layer_set_sha = hashlib.sha256(json.dumps(layer_receipts, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    if result.get("layer_receipt_set_sha256") != expected_layer_set_sha:
        raise ValueError("result layer receipt set hash invalid")
    coverage = result.get("coverage_gate")
    expected_chunks = [list(range(43)) for _ in range(8)]
    expected_visits = [layer for chunk in expected_chunks for layer in chunk]
    if not isinstance(coverage, Mapping):
        raise ValueError("result layer coverage invalid")
    validate_layer_coverage(coverage.get("stream_completed_layers"), expected_chunks=8)
    if (
        coverage.get("stream_completed_layers") != expected_visits
        or coverage.get("stream_completed_layers_by_chunk") != expected_chunks
        or coverage.get("exact_once_per_chunk_in_order") is not True
        or int(coverage.get("layers", -1)) != 43
        or int(coverage.get("chunks", -1)) != 8
        or int(coverage.get("layer_visits", -1)) != 344
    ):
        raise ValueError("result layer coverage invalid")
    restoration = result.get("baseline_restoration_gate")
    if not isinstance(restoration, Mapping) or restoration.get("local_layer_scratch_retired") is not True or restoration.get("verdict") != "PASS_NO_PERSISTENT_MUTATION":
        raise ValueError("result scratch/restoration gate invalid")
    if result.get("gpu_empty_after_child") is not True:
        raise ValueError("result post-child GPU-empty gate invalid")
    gpu = result.get("gpu_snapshot_after_child")
    if not isinstance(gpu, Mapping) or gpu.get("compute_apps") != [] or gpu.get("utilization_gpu_memory_percent") != [[0, 0]]:
        raise ValueError("result post-child GPU snapshot invalid")
    fields = {
        "pass": "physical_code76_pass_marker_sha256",
        "marker": "physical_code76_marker_sha256",
        "wire": "wire_manifest_sha256",
    }
    for short, field in fields.items():
        expected = _hex(expected_hashes.get(short), f"expected {short} hash")
        if result.get(field) != expected:
            raise ValueError(f"result {field} mismatch")
    exact_identity = {
        "claim_sha256": _hex(expected_claim_sha256, "expected claim hash"),
        "command_receipt_sha256": _hex(expected_command_sha256, "expected command hash"),
        "physical_reader_sha256": _hex(expected_reader_sha256, "expected reader hash"),
        "assignment_sha256": _hex(expected_assignment_sha256, "expected assignment hash"),
        "compact_manifest_sha256": _hex(expected_compact_sha256, "expected compact hash"),
        "actual_serialized_wire_bytes": int(expected_wire_bytes),
    }
    for field, expected in exact_identity.items():
        if result.get(field) != expected:
            raise ValueError(f"result {field} identity mismatch")
    resume_path = result.get("resume_authorization")
    if resume_validation is not None:
        try:
            authorization_path = Path(resume_validation["authorization_path"])
            original_command_path = Path(resume_validation["original_command_path"])
            launch_once_path = Path(resume_validation["launch_once_path"])
            resume_launch_once_path = Path(resume_validation["resume_launch_once_path"])
            prefix_windows = int(resume_validation["preserved_prefix_windows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("explicit resume validation context incomplete") from exc
        for path, name in (
            (authorization_path, "resume authorization"),
            (original_command_path, "original command"),
            (launch_once_path, "launch once"),
            (resume_launch_once_path, "resume launch once"),
        ):
            if not path.is_file():
                raise ValueError(f"live {name} evidence missing")
        live_authorization_sha = sha256_path(authorization_path)
        live_original_command_sha = sha256_path(original_command_path)
        live_launch_once_sha = sha256_path(launch_once_path)
        live_resume_launch_once_sha = sha256_path(resume_launch_once_path)
        if resume_path != str(authorization_path):
            raise ValueError("result resume authorization path mismatch")
        if result.get("resume_authorization_sha256") != live_authorization_sha:
            raise ValueError("result live resume authorization hash mismatch")
        if result.get("original_command_receipt") != str(original_command_path):
            raise ValueError("result original command path mismatch")
        if result.get("original_command_receipt_sha256") != live_original_command_sha:
            raise ValueError("result live original command hash mismatch")
        if result.get("launch_once") != str(launch_once_path):
            raise ValueError("result launch once path mismatch")
        if result.get("launch_once_sha256") != live_launch_once_sha:
            raise ValueError("result live launch once hash mismatch")
        if result.get("resume_launch_once") != str(resume_launch_once_path):
            raise ValueError("result resume launch once path mismatch")
        if result.get("resume_launch_once_sha256") != live_resume_launch_once_sha:
            raise ValueError("result live resume launch once hash mismatch")
        if prefix_windows <= 0 or prefix_windows >= len(per_window):
            raise ValueError("result preserved prefix boundary invalid")
        prefix = [
            {"win": int(row["win"]), "sha256": row["sha256"]}
            for row in per_window[:prefix_windows]
        ]
        prefix_sha = hashlib.sha256(
            json.dumps(prefix, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        if result.get("preserved_output_set_sha256") != prefix_sha:
            raise ValueError("result preserved output set hash is not derived from per-window prefix")
        expected_history = [
            {"windows": [0, prefix_windows - 1], "reader_sha256": ORIGINAL_READER_SHA, "command_sha256": live_original_command_sha},
            {"windows": [prefix_windows, 511], "reader_sha256": expected_reader_sha256, "command_sha256": expected_command_sha256},
        ]
        if result.get("physical_reader_history") != expected_history:
            raise ValueError("result reader/command history mismatch")
    elif resume_path is not None:
        raise ValueError("result resume evidence supplied without explicit validator authority")
    if loader_validation is not None:
        try:
            expected_loader_sha256 = str(loader_validation["expected_loader_sha256"])
            expected_input_identity_sha256 = str(loader_validation["expected_input_identity_sha256"])
            expected_loader_source_path = Path(loader_validation["expected_loader_source_path"])
            expected_sentinel_path = Path(loader_validation["expected_sentinel_path"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("explicit loader validation context incomplete") from exc
        proof = result.get("loader_proof")
        if not isinstance(proof, Mapping):
            raise ValueError("result loader proof missing")
        validate_resumed_window_loader_proof(
            per_window, proof,
            expected_task=expected_task,
            expected_loader_sha256=expected_loader_sha256,
            expected_input_identity_sha256=expected_input_identity_sha256,
            expected_loader_source_path=expected_loader_source_path,
            expected_sentinel_path=expected_sentinel_path,
        )
    for field in (
        "assignment_sha256", "compact_manifest_sha256", "claim_sha256",
        "command_receipt_sha256", "physical_reader_sha256",
    ):
        _hex(result.get(field), f"result {field}")
    if int(result.get("actual_serialized_wire_bytes", -1)) <= 0:
        raise ValueError("result actual wire bytes invalid")


def validate_done_receipt(
    done: Mapping[str, object],
    *,
    expected_task: str,
    result_sha256: str,
    command_sha256: str,
    claim_sha256: str,
    expected_hashes: Mapping[str, str],
    resume_launch_once_sha256: str | None = None,
    loader_sentinel_sha256: str | None = None,
) -> None:
    exact = {
        "schema": "banana_smasher-full512-done-v2",
        "task_id": expected_task,
        "status": "PASS",
        "result_sha256": _hex(result_sha256, "expected result hash"),
        "command_sha256": _hex(command_sha256, "expected command hash"),
        "claim_sha256": _hex(claim_sha256, "expected claim hash"),
        "physical_code76_pass_marker_sha256": _hex(expected_hashes.get("pass"), "expected pass hash"),
        "physical_code76_marker_sha256": _hex(expected_hashes.get("marker"), "expected marker hash"),
        "wire_manifest_sha256": _hex(expected_hashes.get("wire"), "expected wire hash"),
        "window_ids": list(range(512)),
        "stream_completed_layers": list(range(43)) * 8,
        "stream_completed_layers_by_chunk": [list(range(43)) for _ in range(8)],
        "chunks": 8,
        "layer_visits": 344,
        "six_classes": sorted(EXPECTED_CLASSES),
        "gpu_empty_after_child": True,
        "scratch_retired_after_child": True,
    }
    for key, expected in exact.items():
        if done.get(key) != expected:
            raise ValueError(f"DONE {key} mismatch")
    if resume_launch_once_sha256 is not None:
        if done.get("resume_launch_once_sha256") != _hex(resume_launch_once_sha256, "expected resume launch hash"):
            raise ValueError("DONE resume launch once hash mismatch")
    if loader_sentinel_sha256 is not None:
        if done.get("loader_sentinel_sha256") != _hex(loader_sentinel_sha256, "expected loader sentinel hash"):
            raise ValueError("DONE loader sentinel hash mismatch")
    _validate_summary(done.get("global"), "global", windows=512)
    _validate_summary(done.get("code"), "code", windows=76)


require_unoptimized()
