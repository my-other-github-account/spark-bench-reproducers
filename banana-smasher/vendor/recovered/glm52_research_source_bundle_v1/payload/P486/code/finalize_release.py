#!/usr/bin/env python3
"""Seal the measured wall, exact-CAS release spark-2, then write DONE."""
from __future__ import annotations
import hashlib, json, os, socket, subprocess, time
from pathlib import Path
from typing import Any
TASK = "task-redacted"
M = Path("${SPARK_HOME}/missions/VISIBLE_EVAL_FULL164_t_872fd554_s2")
CLAIM = Path("${SPARK_HOME}/HOST_CLAIM.json")
FINAL = M / "out/FINAL_ACTUAL_FULL164_RECEIPT.json"
BUDGET = 5400.0

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def main() -> None:
    if socket.gethostname() != "spark-2":
        raise RuntimeError("hard-host violation")
    gpu = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], text=True).strip().splitlines()
    if gpu:
        raise RuntimeError(f"GPU not empty at release: {gpu}")
    claim_bytes = CLAIM.read_bytes(); claim_sha = hashlib.sha256(claim_bytes).hexdigest(); claim = json.loads(claim_bytes)
    if claim.get("owner") != TASK or claim.get("mission") != str(M):
        raise RuntimeError(f"claim lost before release: {claim.get('owner')} {claim.get('mission')}")
    final = json.loads(FINAL.read_text())
    if final.get("coverage", {}).get("completed") != 164 or not final.get("coverage", {}).get("exact_canonical_order"):
        raise RuntimeError("cannot release without exact 164 coverage")
    released_epoch = time.time()
    total_wall = released_epoch - float(final["timing"]["pipeline_start_epoch"])
    speed_pass = total_wall <= BUDGET
    graph_pass = bool(final.get("graph_proof", {}).get("decode_T_le_4_hit"))
    status = "PASS" if speed_pass and graph_pass else ("FAIL_SPEED" if not speed_pass else "FAIL_GRAPH_SENTINEL")
    release = {
        "schema": "host-release-v1", "state": "RELEASED", "status": "UNCLAIMED", "host": "spark-2",
        "owner": "UNCLAIMED", "task": None, "task_id": None, "mission": None,
        "released_from": TASK, "released_at_epoch": released_epoch,
        "reason": f"actual full164 graph-on visible-eval sealed status={status}; GPU/process-empty",
        "previous_claim_sha256": claim_sha, "previous_claim": claim,
        "gpu_apps": [], "task_processes": [],
        "final_receipt": str(FINAL),
    }
    release_raw = (json.dumps(release, indent=2, sort_keys=True) + "\n").encode()
    tmp = CLAIM.with_name(CLAIM.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(release_raw); os.chmod(tmp, 0o600); os.replace(tmp, CLAIM)
    (M / "receipts/HOST_RELEASE.json").write_bytes(release_raw)
    final["status"] = status
    final["timing"].update({
        "sealed_epoch": released_epoch, "actual_total_wall_seconds": total_wall,
        "seconds_per_task": total_wall / 164, "budget_seconds": BUDGET,
        "headroom_seconds": BUDGET - total_wall, "pass_under_1p5h": speed_pass,
    })
    final["graph_proof"]["pass"] = graph_pass
    final["release"] = {
        "claim_sha256_before_release": claim_sha,
        "host_claim_post_sha256": hashlib.sha256(release_raw).hexdigest(),
        "host_release_receipt": str(M / "receipts/HOST_RELEASE.json"),
        "gpu_apps": [], "task_processes": [], "owner": "UNCLAIMED",
    }
    atomic_json(FINAL, final)
    done = {
        "schema": "visible-eval-full164-done-v1", "status": status, "task_id": TASK,
        "final_receipt": str(FINAL), "final_receipt_sha256": sha256(FINAL),
        "coverage": 164, "actual_total_wall_seconds": total_wall,
        "pass_under_1p5h": speed_pass, "released_epoch": released_epoch,
    }
    atomic_json(M / "run/DONE.json", done)
    atomic_json(M / "run/STATUS.json", {"status": status, "detail": "actual full164 receipt sealed and spark-2 exact-released", "epoch": time.time()})
    print(json.dumps(done, sort_keys=True))

if __name__ == "__main__":
    main()
