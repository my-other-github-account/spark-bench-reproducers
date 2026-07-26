#!/usr/bin/env python3
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
from typing import Any

TASK = "PUBLIC_TASK"
MISSION = Path("$HOME/run-bundles/P541_QTIP2_SHARDA_PUBLIC_TASK_s5w")
CLAIM = Path("$HOME/HOST_CLAIM.json")
PYTHON = "$HOME/humming_env/bin/python"
S7 = "compute-node-7"
S7_FIT_MISSION = "$HOME/run-bundles/P534_TRAIN_FIT_PUBLIC_TASK_s7"
S7_CAPTURES = "/dev/shm/P534_TRAIN_FIT_PUBLIC_TASK_s7/captures"
S3_MODEL = "$HOME/run-bundles/QTIP_PROOF1_SHARD_PUBLIC_TASK_s3/source_model"
LAYERS = [0, 2, 4, 6]
FIT_SHA = {
    0: "ef6bcbafe7d93027bbd2229218b3717dc75c4de7214820c4c042b3dde1790822",
    2: "42b0c3ec16dbd780b9afe025a48fde9b2976577d2af3b63c591322824d9c530b",
    4: "afecd20c33edc853347a8e2c1735b415bd225a8108eff8b2520d3b4f8c82e663",
    6: "fcd38cb59c4fd6b4959fe0856501a4bad9ddc14ef2e9bfaa576a41aaf95ee4bd",
}
PLANE = {
    0: ("$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/planes/vq3u_layer_000.pt", 3422621289, "bfeedd7bff25e1d814851c2e6d056e67f04b2275b0932a134636debafc5ddc4b"),
    2: ("$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/planes/vq3u_layer_002.pt", 3422621289, "44a498ef84a39b1e9f368c4e89cded74311aed359b9dc24a5eb8a1701f79fe38"),
    4: ("$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/planes/vq3u_layer_004.pt", 3422621289, "bf8f6a690ece0b902b849947f7f5e871cabdd22956f030709ccb58b4a429803c"),
    6: ("$HOME/run-bundles/BINREPAIR_PUBLIC_TASK/planes/vq3u_layer_006.pt", 3422621289, "a7d3a3aeffaf1d6617bfbaf8fbf231deb6c6181aa6a1ba9af8f4e276474f2843"),
}


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def md5_file(path: Path, chunk: int = 16 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def claim() -> tuple[bytes, str, dict[str, Any]]:
    raw = CLAIM.read_bytes()
    obj = json.loads(raw)
    checks = {
        "host": socket.gethostname() == "compute-node-5-work",
        "owner": obj.get("owner") == TASK,
        "task": obj.get("task_id") == TASK,
        "mission": obj.get("mission") == str(MISSION),
        "layers": obj.get("layers") == LAYERS,
    }
    if not all(checks.values()):
        raise RuntimeError(f"exclusive claim invalid: {checks}; claim={obj}")
    return raw, sha256_bytes(raw), obj


def completed_counts() -> tuple[int, dict[int, int]]:
    per = {}
    for layer in LAYERS:
        per[layer] = len(list((MISSION / "artifacts").glob(f"L{layer:03d}_E???_*_L16_K2_V2.DONE.json")))
    return sum(per.values()), per


def progress(stage: str, status: str, **extra: Any) -> None:
    total, per = completed_counts()
    old = {}
    try:
        old = json.loads((MISSION / "PROGRESS.json").read_text())
    except Exception:
        pass
    value = {
        **old,
        "schema": "p541-progress-v1",
        "task": TASK,
        "host": socket.gethostname(),
        "mission": str(MISSION),
        "status": status,
        "stage": stage,
        "layers": LAYERS,
        "forbidden_layers": [16, 11, 14, 19],
        "experts": [0, 255],
        "projections": ["fused13", "down"],
        "geometry": {"L": 16, "K": 2, "V": 2, "target_bpw": 2.0},
        "expected_units": 2048,
        "completed_units": total,
        "completed_units_by_layer": {str(k): v for k, v in per.items()},
        "remaining_units": 2048 - total,
        "heldout_used": False,
        "pid": os.getpid(),
        "updated_unix": time.time(),
        **extra,
    }
    atomic_json(MISSION / "PROGRESS.json", value)


def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    print("RUN", json.dumps(argv), flush=True)
    cp = subprocess.run(argv, **kwargs)
    if cp.returncode:
        raise RuntimeError(f"command failed rc={cp.returncode}: {argv}")
    return cp


def rsync(argv: list[str]) -> None:
    run(["rsync", "-aL", "--partial", "--protect-args", *argv])


def valid_staged_fit(layer: int, path: Path) -> bool:
    try:
        obj = json.loads(path.read_text())
        if obj.get("status") != "PASS" or int(obj.get("layer", -1)) != layer or len(obj.get("files", [])) != 128:
            return False
        for row in obj["files"]:
            p = Path(row["path"])
            d = Path(row["done_path"])
            if not p.is_file() or not d.is_file():
                return False
            if p.stat().st_size != int(row["bytes"]):
                return False
            if sha256_file(p) != row["sha256"] or sha256_file(d) != row["done_sha256"]:
                return False
        return True
    except Exception as exc:
        print(f"staged fit L{layer:03d} invalid/resume-miss: {exc!r}", flush=True)
        return False


def stage_fit(layer: int) -> Path:
    out = MISSION / f"inputs/FIT_L{layer:03d}_STAGED.json"
    if valid_staged_fit(layer, out):
        print(f"SKIP verified staged fit L{layer:03d}", flush=True)
        return out
    claim()
    source_receipt = MISSION / f"inputs/source_receipts/FIT_L{layer:03d}.json"
    source_receipt.parent.mkdir(parents=True, exist_ok=True)
    rsync([f"{S7}:{S7_FIT_MISSION}/receipts/FIT_L{layer:03d}.json", str(source_receipt)])
    raw = source_receipt.read_bytes()
    if sha256_bytes(raw) != FIT_SHA[layer]:
        raise RuntimeError(f"authoritative fit receipt SHA mismatch L{layer:03d}")
    source = json.loads(raw)
    rows = source.get("files", [])
    if source.get("status") != "PASS" or int(source.get("layer", -1)) != layer or len(rows) != 128:
        raise RuntimeError(f"authoritative fit receipt identity invalid L{layer:03d}")
    destination = MISSION / f"inputs/captures/L{layer:03d}"
    destination.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for row in rows:
        names.extend([Path(row["path"]).name, Path(row["done_path"]).name])
    if len(set(names)) != 256:
        raise RuntimeError(f"duplicate capture/sidecar basename L{layer:03d}")
    files_from = MISSION / f"status/L{layer:03d}_RSYNC_FILES.txt"
    atomic_bytes(files_from, ("\n".join(names) + "\n").encode())
    rsync([f"--files-from={files_from}", f"{S7}:{S7_CAPTURES}/", str(destination) + "/"])
    staged_rows = []
    for row in rows:
        p = destination / Path(row["path"]).name
        d = destination / Path(row["done_path"]).name
        if p.stat().st_size != int(row["bytes"]):
            raise RuntimeError(f"capture size mismatch: {p}")
        if sha256_file(p) != row["sha256"] or md5_file(p) != row["md5"]:
            raise RuntimeError(f"capture hash mismatch: {p}")
        if sha256_file(d) != row["done_sha256"]:
            raise RuntimeError(f"capture sidecar hash mismatch: {d}")
        staged_rows.append({
            **row,
            "authoritative_source_path": row["path"],
            "authoritative_source_done_path": row["done_path"],
            "path": str(p),
            "resolved_path": str(p.resolve()),
            "done_path": str(d),
            "done_resolved_path": str(d.resolve()),
        })
    staged = {
        **{k: v for k, v in source.items() if k != "files"},
        "files": staged_rows,
        "staged_by_task": TASK,
        "stage_host": "compute-node-5-work",
        "source_host": "compute-node-7",
        "source_qsfp": "203.0.113.8",
        "source_receipt": f"{S7}:{S7_FIT_MISSION}/receipts/FIT_L{layer:03d}.json",
        "source_receipt_sha256": FIT_SHA[layer],
        "rsync_copy_links": True,
        "staged_unix": time.time(),
        "heldout_used": False,
    }
    atomic_json(out, staged)
    if not valid_staged_fit(layer, out):
        raise RuntimeError(f"staged fit final verification failed L{layer:03d}")
    receipt = {
        "schema": "p541-fit-stage-v1",
        "status": "PASS",
        "task": TASK,
        "layer": layer,
        "source_receipt_sha256": FIT_SHA[layer],
        "staged_fit_receipt": str(out),
        "staged_fit_receipt_sha256": sha256_file(out),
        "capture_count": 128,
        "capture_payload_bytes": sum(int(row["bytes"]) for row in staged_rows),
        "rsync_copy_links": True,
        "source_read_only": True,
    }
    atomic_json(MISSION / f"receipts/FIT_STAGE_L{layer:03d}.json", receipt)
    return out


def stage_plane(layer: int) -> Path:
    source, expected_bytes, expected_sha = PLANE[layer]
    out = MISSION / f"inputs/planes/vq3u_layer_{layer:03d}.pt"
    if out.is_file() and out.stat().st_size == expected_bytes and sha256_file(out) == expected_sha:
        print(f"SKIP verified plane L{layer:03d}", flush=True)
        return out
    claim()
    out.parent.mkdir(parents=True, exist_ok=True)
    rsync([f"{S7}:{source}", str(out)])
    if out.stat().st_size != expected_bytes or sha256_file(out) != expected_sha:
        raise RuntimeError(f"plane hash/size mismatch L{layer:03d}")
    atomic_json(MISSION / f"receipts/PLANE_STAGE_L{layer:03d}.json", {
        "schema": "p541-plane-stage-v1", "status": "PASS", "task": TASK, "layer": layer,
        "source_host": "compute-node-7", "source_qsfp": "203.0.113.8", "source_path": source,
        "destination": str(out), "bytes": expected_bytes, "sha256": expected_sha,
        "rsync_copy_links": True, "source_read_only": True, "heldout_used": False,
    })
    return out


def unit_paths(layer: int, expert: int) -> list[tuple[Path, Path]]:
    rows = []
    for projection in ("fused13", "down"):
        artifact = MISSION / f"artifacts/L{layer:03d}_E{expert:03d}_{projection}_L16_K2_V2.pt"
        rows.append((artifact, artifact.with_suffix(".DONE.json")))
    return rows


def valid_done_pair(artifact: Path, done_path: Path, layer: int, expert: int, projection: str) -> bool:
    try:
        done = json.loads(done_path.read_text())
        return (
            done.get("status") == "PASS"
            and done.get("task") == TASK
            and done.get("identity") == {"layer": layer, "expert": expert, "projection": projection, "target_bpw": 2.0}
            and artifact.is_file()
            and int((done.get("artifact") or {}).get("bytes", -1)) == artifact.stat().st_size
            and (done.get("artifact") or {}).get("sha256") == sha256_file(artifact)
            and bool(done.get("gates"))
            and all(bool(v) for v in done["gates"].values())
        )
    except Exception:
        return False


def build_layer(layer: int, fit: Path, plane: Path) -> dict[str, Any]:
    layer_started = time.time()
    progress("layer_running", "RUNNING", current_layer=layer, current_expert=0, layer_started_unix=layer_started)
    for expert in range(256):
        claim()
        pairs = unit_paths(layer, expert)
        projections = ("fused13", "down")
        states = [valid_done_pair(a, d, layer, expert, p) for (a, d), p in zip(pairs, projections)]
        if all(states):
            print(f"SKIP verified DONE L{layer:03d} E{expert:03d} both projections", flush=True)
            progress("unit_resume_skipped", "RUNNING", current_layer=layer, current_expert=expert)
            continue
        if any(d.is_file() and not ok for (_, d), ok in zip(pairs, states)):
            raise RuntimeError(f"invalid existing DONE; refusing replay L{layer:03d} E{expert:03d}")
        progress("expert_running", "RUNNING", current_layer=layer, current_expert=expert)
        command = [
            PYTHON, str(MISSION / "code/qtip_rate_unit_p541.py"),
            "--mission", str(MISSION),
            "--qtip-root", str(MISSION / "qtip-canonical"),
            "--tlut-source", str(MISSION / "inputs/L017_E005_fused13_QTIP_HYB_L16_K3_V2.pt"),
            "--fit-receipt", str(fit),
            "--current-plane", str(plane),
            "--s3-model", S3_MODEL,
            "--layer", str(layer), "--expert", str(expert),
            "--L", "16", "--K", "2", "--V", "2", "--target-bpw", "2.0",
        ]
        for attempt in range(1, 6):
            try:
                run(command, check=False)
                break
            except RuntimeError:
                if attempt == 5:
                    raise
                claim()
                progress(
                    "unit_transient_retry",
                    "RUNNING",
                    current_layer=layer,
                    current_expert=expert,
                    retry_attempt=attempt,
                    retry_limit=5,
                )
                delay = 5 * attempt
                print(
                    f"TRANSIENT unit retry {attempt}/5 after {delay}s "
                    f"L{layer:03d} E{expert:03d}",
                    flush=True,
                )
                time.sleep(delay)
        progress("expert_done", "RUNNING", current_layer=layer, current_expert=expert)
    progress("layer_units_built", "RUNNING", current_layer=layer, current_expert=255)
    cp = run([
        PYTHON, str(MISSION / "code/p541_finalize_layer.py"),
        "--layer", str(layer), "--layer-started-unix", repr(layer_started),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    print(cp.stdout, end="", flush=True)
    result = json.loads(cp.stdout.strip().splitlines()[-1])
    progress("layer_sealed", "RUNNING", current_layer=layer, layer_manifest=result)
    return result


def gpu_apps() -> list[str]:
    cp = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20,
    )
    if cp.returncode:
        raise RuntimeError(f"nvidia-smi failed at release: {cp.stderr}")
    return [line for line in cp.stdout.splitlines() if line.strip()]


def exact_release(final_manifest: Path, final_sha: str) -> dict[str, Any]:
    lock_path = CLAIM.with_suffix(".lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before, before_sha, obj = claim()
        apps = gpu_apps()
        if apps:
            raise RuntimeError(f"GPU apps remain before release: {apps}")
        now = time.time()
        st = os.statvfs("/")
        available = st.f_bavail * st.f_frsize
        release_obj = {
            "schema": "host-claim-v1", "owner": "UNCLAIMED", "task_id": "UNCLAIMED",
            "host": "compute-node-5-work", "mission": "UNCLAIMED", "output_root": "UNCLAIMED",
            "progress": "UNCLAIMED", "pidfile": "UNCLAIMED", "log": "UNCLAIMED",
            "claim_nonce": "UNCLAIMED", "claimed_unix": 0, "gpu_empty": True,
            "capacity_available_bytes": available, "previous_claim_sha256": before_sha,
            "released_by": TASK, "released_unix": now,
            "release_reason": "P541 layers [0,2,4,6] terminal seals complete; GPU empty; driver exiting",
            "final_manifest": str(final_manifest), "final_manifest_sha256": final_sha,
        }
        after = (json.dumps(release_obj, indent=2, sort_keys=True) + "\n").encode()
        if CLAIM.read_bytes() != before:
            raise RuntimeError("claim changed during exact release")
        atomic_bytes(CLAIM, after)
        if CLAIM.read_bytes() != after:
            raise RuntimeError("release postimage readback mismatch")
        receipt = {
            "schema": "p541-exact-release-v1", "status": "PASS_UNCLAIMED", "task": TASK,
            "host": socket.gethostname(), "claim_preimage_sha256": before_sha,
            "claim_postimage_sha256": sha256_bytes(after), "new_owner": "UNCLAIMED",
            "gpu_apps": [], "driver_pid_expected_to_exit": os.getpid(),
            "available_bytes_at_release": available, "released_unix": now,
            "final_manifest": str(final_manifest), "final_manifest_sha256": final_sha,
        }
        path = MISSION / "receipts/HOST_EXACT_RELEASE.json"
        atomic_json(path, receipt)
        receipt["path"] = str(path)
        receipt["sha256"] = sha256_file(path)
        return receipt


def main() -> int:
    started = time.time()
    if socket.gethostname() != "compute-node-5-work":
        raise RuntimeError("hard host violation")
    lock_path = MISSION / "status/P541_SINGLE_ACTOR.lock"
    with lock_path.open("a+b") as actor_lock:
        try:
            fcntl.flock(actor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another P541 driver actor already holds the lock")
        raw, claim_sha, _ = claim()
        atomic_json(MISSION / "receipts/DRIVER_START.json", {
            "schema": "p541-driver-start-v1", "status": "PASS", "task": TASK,
            "host": socket.gethostname(), "pid": os.getpid(), "started_unix": started,
            "claim_sha256": claim_sha, "layers": LAYERS, "heldout_used": False,
        })
        progress("staging_inputs", "RUNNING", started_unix=started)
        staged: dict[int, tuple[Path, Path]] = {}
        for layer in LAYERS:
            progress("staging_layer_inputs", "RUNNING", current_layer=layer)
            staged[layer] = (stage_fit(layer), stage_plane(layer))
            progress("layer_inputs_staged", "RUNNING", current_layer=layer)
        manifests = []
        for layer in LAYERS:
            fit, plane = staged[layer]
            manifests.append(build_layer(layer, fit, plane))
        total, per = completed_counts()
        if total != 2048 or any(per[layer] != 512 for layer in LAYERS):
            raise RuntimeError(f"terminal coverage invalid total={total} per={per}")
        forbidden = list((MISSION / "artifacts").glob("L016_*"))
        if forbidden:
            raise RuntimeError(f"forbidden L016 artifact exists in task mission: {forbidden[:3]}")
        aggregate = {
            "schema": "p541-sharda-manifest-v1", "status": "PASS_4_LAYERS_2048_OF_2048",
            "task": TASK, "host": "compute-node-5-work", "mission": str(MISSION),
            "layers": LAYERS, "forbidden_layers": [16, 11, 14, 19],
            "coverage": {"complete_units": total, "expected_units": 2048, "per_layer": {str(k): v for k, v in per.items()}},
            "geometry": {"L": 16, "K": 2, "V": 2, "target_bpw": 2.0},
            "layer_manifests": manifests,
            "logical_bytes_total": sum(int(m["logical_bytes_total"]) for m in manifests),
            "physical_serialized_bytes_total": sum(int(m["physical_serialized_bytes_total"]) for m in manifests),
            "actual_wall_seconds": time.time() - started,
            "heldout_used": False, "eval_used": False, "solve_used": False,
            "l016_replayed": False, "sealed_unix": time.time(),
        }
        aggregate_path = MISSION / "results/P541_SHARDA_MANIFEST.json"
        atomic_json(aggregate_path, aggregate)
        aggregate_sha = sha256_file(aggregate_path)
        progress("terminal_manifest_sealed", "PASS", layer_manifests=manifests,
                 final_manifest=str(aggregate_path), final_manifest_sha256=aggregate_sha,
                 finished_unix=time.time())
        release = exact_release(aggregate_path, aggregate_sha)
        done = {
            "schema": "p541-driver-done-v1", "status": "PASS", "task": TASK,
            "host": socket.gethostname(), "pid": os.getpid(), "started_unix": started,
            "finished_unix": time.time(), "actual_wall_seconds": time.time() - started,
            "final_manifest": str(aggregate_path), "final_manifest_sha256": aggregate_sha,
            "release": release,
        }
        atomic_json(MISSION / "status/DRIVER_DONE.json", done)
        print(json.dumps(done, sort_keys=True), flush=True)
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BaseException as exc:
        try:
            progress("driver_failed", "FAIL", error=repr(exc), traceback=traceback.format_exc(), failed_unix=time.time())
            atomic_json(MISSION / "status/DRIVER_FAILED.json", {
                "schema": "p541-driver-failed-v1", "status": "FAIL", "task": TASK,
                "host": socket.gethostname(), "pid": os.getpid(), "failed_unix": time.time(),
                "error": repr(exc), "traceback": traceback.format_exc(),
                "claim_left_held_fail_closed": True,
            })
        except Exception:
            traceback.print_exc()
        raise
