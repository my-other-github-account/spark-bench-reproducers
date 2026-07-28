#!/usr/bin/env python3
"""Score one newly generated batch with the pinned EvalPlus image APIs."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from typing import Any

EXPECTED_DATASET_SHA = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
DATASET = Path(os.environ.get(
    "HUMANEVAL_OVERRIDE_PATH",
    str(Path(__file__).resolve().with_name("HumanEvalPlus-v0.1.10.jsonl")),
)).resolve()
os.environ.setdefault("HUMANEVAL_OVERRIDE_PATH", str(DATASET))

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import PASS, check_correctness, get_groundtruth
from evalplus.sanitize import sanitize

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if sha256(DATASET) != EXPECTED_DATASET_SHA:
        raise RuntimeError("pinned HumanEvalPlus dataset hash drift")
    payload = json.loads(args.batch.read_text())
    rows = payload["rows"]
    if not 1 <= len(rows) <= 4:
        raise RuntimeError(f"invalid batch size {len(rows)}")
    ids = [r["task_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate task IDs in batch")
    problems = get_human_eval_plus()
    dataset_hash = get_human_eval_plus_hash()
    expected = get_groundtruth(problems, dataset_hash, [])
    started = time.perf_counter()

    def score_one(row: dict[str, Any]) -> dict[str, Any]:
        tid = row["task_id"]
        problem = problems[tid]
        clean = sanitize(str(row["solution"]), entrypoint=problem["entry_point"])
        t0 = time.perf_counter()
        result = check_correctness(
            "humaneval", 0, problem, clean, expected[tid],
            base_only=False, fast_check=False, identifier=tid,
        )
        score_wall = time.perf_counter() - t0
        base_status, base_details = result["base"]
        plus_status, plus_details = result["plus"]
        return {
            "task_id": tid,
            "base_status": base_status,
            "plus_status": plus_status,
            "base_score": int(base_status == PASS),
            "plus_score": int(plus_status == PASS),
            "evalplus_score": int(base_status == PASS and plus_status == PASS),
            "base_tests_completed": len(base_details),
            "plus_tests_completed": len(plus_details),
            "sanitized_solution": clean,
            "sanitized_solution_sha256": hashlib.sha256(clean.encode()).hexdigest(),
            "score_wall_seconds": score_wall,
        }

    # EvalPlus check_correctness forks a sandbox process. Run the four rows
    # serially: forking from ThreadPoolExecutor threads is rejected by the
    # pinned image's filelock fork-safety hook.
    scored = [score_one(row) for row in rows]
    receipt = {
        "schema": "visible-eval-stream-batch-score-v1",
        "status": "PASS",
        "scorer": "evalplus:26d6d00 --network none",
        "dataset_sha256": EXPECTED_DATASET_SHA,
        "evalplus_dataset_hash": dataset_hash,
        "batch_sha256": sha256(args.batch),
        "batch_ids": ids,
        "rows": scored,
        "batch_score_wall_seconds": time.perf_counter() - started,
        "created_epoch": time.time(),
    }
    atomic_json(args.out, receipt)
    print(json.dumps({"status": "PASS", "ids": ids, "base": sum(r["base_score"] for r in scored), "plus": sum(r["plus_score"] for r in scored), "wall": receipt["batch_score_wall_seconds"]}, sort_keys=True), flush=True)

if __name__ == "__main__":
    main()
