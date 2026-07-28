#!/usr/bin/env python3
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import socket
import time
import urllib.error
import urllib.request

TASK = "task-redacted"
MISSION = pathlib.Path("${SPARK_HOME}/missions/CLEAN_HE164_TRANSFER8_t_93420eec_s8")
DATASET_SHA256 = "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f"
CHECKPOINT_SHA256 = "4086e9d8be9ece067ce3b713c22654e59bcad614af9444bdfacd2e66e0a02fd5"
FINGERPRINT = "vllm-0.24.0-3f34bf12"
MODEL = "deepseek-v4-flash-bq3-step8transfer"
PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"


def sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def post(url: str, payload: dict, timeout: int) -> tuple[dict, float, int]:
    request = urllib.request.Request(url, data=json.dumps(payload, separators=(",", ":")).encode(), headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(1, 4):
        start = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), time.time() - start, attempt
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            last = RuntimeError(f"HTTP {exc.code}: {body}")
        except Exception as exc:
            last = exc
        if attempt < 3:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"request failed after 3 attempts: {last}")


def valid_raw(path: pathlib.Path, task_id: str, max_tokens: int, top_p: float) -> bool:
    try:
        row = json.loads(path.read_text())
        req = row["request"]
        response = row["response"]
        choice = response["choices"][0]
        return (
            row["task_id"] == task_id
            and row["checkpoint_sha256"] == CHECKPOINT_SHA256
            and req["model"] == MODEL
            and req["temperature"] == 0.0
            and req["top_p"] == top_p
            and req["max_tokens"] == max_tokens
            and req["n"] == 1
            and response.get("system_fingerprint") == FINGERPRINT
            and isinstance(choice.get("token_ids"), list)
            and isinstance(response.get("prompt_token_ids"), list)
            and isinstance((choice.get("message") or {}).get("content"), (str, type(None)))
        )
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=pathlib.Path, default=MISSION / "assets/data/HumanEvalPlus-v0.1.10.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8013/v1")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    claim = json.loads(pathlib.Path("${SPARK_HOME}/HOST_CLAIM.json").read_text())
    assert claim.get("owner") == TASK and claim.get("mission") == str(MISSION), claim
    assert socket.gethostname() == "spark-8"
    assert sha(args.dataset) == DATASET_SHA256
    dataset_rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    expected_ids = [f"HumanEval/{index}" for index in range(164)]
    assert [row["task_id"] for row in dataset_rows] == expected_ids
    dataset = {row["task_id"]: row for row in dataset_rows}
    raw_dir = MISSION / "results/generation/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    status_path = MISSION / "run/GENERATION_STATUS.json"
    endpoint = args.base_url.rstrip("/") + "/chat/completions"

    completed = {task_id for task_id in expected_ids if valid_raw(raw_dir / (task_id.replace("/", "_") + ".json"), task_id, args.max_tokens, args.top_p)}
    pending = [task_id for task_id in expected_ids if task_id not in completed]

    def one(task_id: str) -> dict:
        prompt = dataset[task_id]["prompt"].strip() + "\n"
        user_prompt = PREFIX + f"\n```python\n{prompt.strip()}\n```"
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": user_prompt}],
            "temperature": 0.0,
            "top_p": args.top_p,
            "n": 1,
            "max_tokens": args.max_tokens,
            "seed": 0,
            "return_token_ids": True,
        }
        response, wall_seconds, attempts = post(endpoint, payload, args.timeout)
        if response.get("system_fingerprint") != FINGERPRINT:
            raise RuntimeError(f"{task_id}: fingerprint drift: {response.get('system_fingerprint')!r}")
        choice = response["choices"][0]
        token_ids = choice.get("token_ids")
        prompt_ids = response.get("prompt_token_ids")
        if not isinstance(token_ids, list) or not isinstance(prompt_ids, list):
            raise RuntimeError(f"{task_id}: exact token IDs omitted")
        usage = response.get("usage") or {}
        if int(usage.get("completion_tokens", len(token_ids))) != len(token_ids):
            raise RuntimeError(f"{task_id}: completion token count drift")
        row = {
            "schema": "clean-he164-transfer8-raw-v1",
            "task": TASK,
            "task_id": task_id,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "dataset_sha256": DATASET_SHA256,
            "request": payload,
            "response": response,
            "wall_seconds": wall_seconds,
            "http_attempts": attempts,
            "sealed_epoch": time.time(),
        }
        path = raw_dir / (task_id.replace("/", "_") + ".json")
        atomic_json(path, row)
        return {
            "task_id": task_id,
            "raw_path": str(path),
            "raw_sha256": sha(path),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": len(token_ids),
            "content_is_null": (choice.get("message") or {}).get("content") is None,
            "wall_seconds": wall_seconds,
        }

    atomic_json(status_path, {"state": "running", "total": 164, "sealed": len(completed), "pending": len(pending), "started_or_resumed_epoch": time.time()})
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(one, task_id): task_id for task_id in pending}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed.add(result["task_id"])
            atomic_json(status_path, {"state": "running", "total": 164, "sealed": len(completed), "pending": 164 - len(completed), "last": result, "updated_epoch": time.time()})
            print(json.dumps({"event": "sealed", "sealed": len(completed), **result}, sort_keys=True), flush=True)

    rows = []
    for task_id in expected_ids:
        path = raw_dir / (task_id.replace("/", "_") + ".json")
        assert valid_raw(path, task_id, args.max_tokens, args.top_p), task_id
        raw = json.loads(path.read_text())
        choice = raw["response"]["choices"][0]
        rows.append({
            "task_id": task_id,
            "raw_path": str(path),
            "raw_sha256": sha(path),
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": len(choice["token_ids"]),
            "content_is_null": (choice.get("message") or {}).get("content") is None,
        })
    manifest = {
        "schema": "clean-he164-transfer8-generation-manifest-v1",
        "status": "PASS",
        "task": TASK,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "model": MODEL,
        "system_fingerprint": FINGERPRINT,
        "generation": {"temperature": 0.0, "top_p": args.top_p, "max_tokens": args.max_tokens, "seed": 0, "n": 1, "concurrency": args.concurrency, "prompt_contract": "EvalPlus 26d6d00 OpenAIChatDecoder"},
        "sealed_rows": len(rows),
        "rows": rows,
        "created_epoch": time.time(),
    }
    manifest_path = MISSION / "results/generation/GENERATION_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    atomic_json(status_path, {"state": "sealed", "total": 164, "sealed": 164, "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path), "updated_epoch": time.time()})
    print(json.dumps({"status": "PASS", "sealed": 164, "manifest": str(manifest_path), "manifest_sha256": sha(manifest_path)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
