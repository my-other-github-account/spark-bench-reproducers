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
from typing import Any, Dict, List, Optional, Tuple

from p968_common import ARMS, MAX_TOKENS, atomic_json, conversation, load_dataset, row_path, sha256, sha256_json, sha_i64, summarize_openai_logprobs, valid_row


def post_json(url: str, payload: Dict[str, Any], timeout: int, attempts: int = 3) -> Tuple[Dict[str, Any], float, int]:
    last_error = None
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, data=json.dumps(payload, separators=(",", ":")).encode(), headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response), time.perf_counter() - started, attempt
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            last_error = RuntimeError("HTTP %d: %s" % (exc.code, body))
            if exc.code == 400:
                raise last_error
        except Exception as exc:
            last_error = exc
        if attempt < attempts:
            time.sleep(2 ** attempt)
    raise RuntimeError("POST failed after %d attempts: %r" % (attempts, last_error))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mission", type=pathlib.Path, required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=/absolute/path")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model-arm", default="iq4")
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--server-root-url", required=True)
    parser.add_argument("--model-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--server-receipt", type=pathlib.Path, required=True)
    parser.add_argument("--arms", default="sampled,greedy")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--smoke-tasks", type=int, default=0)
    args = parser.parse_args()

    if socket.gethostname() != args.expected_hostname:
        raise RuntimeError("wrong host")
    claim_path = pathlib.Path(os.environ.get("BANANA_SMASHER_CLAIM_PATH", "/run/banana-smasher/HOST_CLAIM.json"))
    claim = json.loads(claim_path.read_text())
    if claim.get("owner") != args.task_id or claim.get("mission") != str(args.mission):
        raise RuntimeError("claim mismatch: %r" % claim)
    for receipt in (args.model_receipt, args.server_receipt):
        if json.loads(receipt.read_text()).get("status") != "PASS":
            raise RuntimeError("non-PASS receipt %s" % receipt)
    model_receipt_sha = sha256(args.model_receipt)
    server_receipt_sha = sha256(args.server_receipt)
    datasets = {}
    dataset_receipts = {}
    for spec in args.dataset:
        name, raw_path = spec.split("=", 1)
        rows, receipt = load_dataset(name, pathlib.Path(raw_path))
        if args.smoke_tasks:
            rows = rows[: args.smoke_tasks]
        datasets[name] = rows
        dataset_receipts[name] = receipt

    chat_url = args.base_url.rstrip("/") + "/chat/completions"
    tokenize_url = args.server_root_url.rstrip("/") + "/tokenize"
    arms = [name for name in args.arms.split(",") if name]
    status_path = args.mission / "run/GENERATION_STATUS.json"
    sealed = 0

    def generate_one(source: Dict[str, Any], dataset_name: str, arm_name: str, sample_index: int) -> Dict[str, Any]:
        arm = ARMS[arm_name]
        seed = int(arm["seed_start"]) + sample_index
        messages = conversation(source)
        payload = {"model": args.model, "messages": messages, "temperature": arm["temperature"], "top_p": arm["top_p"], "n": 1, "max_tokens": MAX_TOKENS, "seed": seed, "stream": False, "logprobs": True, "top_logprobs": 5}
        logprob_request = "REQUESTED"
        try:
            response, wall, attempts = post_json(chat_url, payload, args.timeout)
        except RuntimeError as exc:
            if not str(exc).startswith("HTTP 400"):
                raise
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
            response, wall, attempts = post_json(chat_url, payload, args.timeout)
            logprob_request = "UNAVAILABLE_HTTP_400"
        choice = response["choices"][0]
        text = (choice.get("message") or {}).get("content")
        if text is None:
            text = ""
        if not isinstance(text, str):
            raise RuntimeError("non-string response")
        token_ids: List[int] = []
        tokenize_status = "UNAVAILABLE"
        try:
            tokenized, _, _ = post_json(tokenize_url, {"content": text, "add_special": False}, min(args.timeout, 600))
            values = tokenized.get("tokens")
            if isinstance(values, list) and all(isinstance(value, int) for value in values):
                token_ids = values
                tokenize_status = "AVAILABLE"
        except Exception:
            pass
        uncertainty = summarize_openai_logprobs(choice)
        if uncertainty["status"] == "UNAVAILABLE" and logprob_request != "REQUESTED":
            uncertainty["reason"] = logprob_request
        return {
            "schema": "p968-generation-row-v1",
            "task_id": source["task_id"],
            "dataset": dataset_name,
            "decode_arm": arm_name,
            "sample_index": sample_index,
            "seed": seed,
            "model_arm": args.model_arm,
            "task_owner": args.task_id,
            "claim_sha256": sha256(claim_path),
            "dataset_sha256": dataset_receipts[dataset_name]["sha256"],
            "model_receipt_sha256": model_receipt_sha,
            "server_receipt_sha256": server_receipt_sha,
            "prompt_sha256": sha256_json(messages),
            "request": payload,
            "output": {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_count": len(token_ids), "tokenize_status": tokenize_status, "token_ids_sha256_i64le": sha_i64(token_ids), "finish_reason": choice.get("finish_reason"), "completion_tokens": int((response.get("usage") or {}).get("completion_tokens", len(token_ids)))},
            "uncertainty": uncertainty,
            "wall_seconds": wall,
            "http_attempts": attempts,
            "created_epoch": time.time(),
        }

    for dataset_name, rows in datasets.items():
        for arm_name in arms:
            arm = ARMS[arm_name]
            n_samples = 1 if args.smoke_tasks else int(arm["n"])
            for sample_index in range(n_samples):
                seed = int(arm["seed_start"]) + sample_index
                pending = []
                for row in rows:
                    expected = {"schema": "p968-generation-row-v1", "task_id": row["task_id"], "dataset": dataset_name, "decode_arm": arm_name, "sample_index": sample_index, "seed": seed, "model_arm": args.model_arm}
                    path = row_path(args.out, args.model_arm, dataset_name, arm_name, sample_index, row["task_id"])
                    if not valid_row(path, expected):
                        pending.append(row)
                for offset in range(0, len(pending), args.concurrency):
                    current_claim = json.loads(claim_path.read_text())
                    if current_claim.get("owner") != args.task_id or current_claim.get("mission") != str(args.mission):
                        raise RuntimeError("claim lost")
                    batch = pending[offset : offset + args.concurrency]
                    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                        futures = [pool.submit(generate_one, row, dataset_name, arm_name, sample_index) for row in batch]
                        values = [future.result() for future in futures]
                    batch_rows = []
                    for value in values:
                        value["batch_task_ids"] = [row["task_id"] for row in batch]
                        path = row_path(args.out, args.model_arm, dataset_name, arm_name, sample_index, value["task_id"])
                        atomic_json(path, value)
                        batch_rows.append({"task_id": value["task_id"], "path": str(path), "sha256": sha256(path)})
                        sealed += 1
                    batch_path = args.mission / "generation_batches" / dataset_name / arm_name / ("s%03d_b%04d.json" % (sample_index, offset // args.concurrency))
                    atomic_json(batch_path, {"schema": "p968-generation-batch-v1", "status": "PASS", "dataset": dataset_name, "decode_arm": arm_name, "sample_index": sample_index, "seed": seed, "rows": batch_rows, "created_epoch": time.time()})
                    atomic_json(status_path, {"status": "RUNNING", "sealed_this_process": sealed, "last_batch": str(batch_path), "updated_epoch": time.time()})
                    print(json.dumps({"event": "sealed_batch", "dataset": dataset_name, "arm": arm_name, "sample": sample_index, "tasks": [row["task_id"] for row in batch]}, sort_keys=True), flush=True)

    manifest = {"schema": "p968-generation-manifest-v1", "status": "PASS", "task_id": args.task_id, "host": socket.gethostname(), "model_arm": args.model_arm, "model": args.model, "claim_sha256": sha256(claim_path), "model_receipt_sha256": model_receipt_sha, "server_receipt_sha256": server_receipt_sha, "datasets": dataset_receipts, "arms": {name: ARMS[name] for name in arms}, "max_tokens": MAX_TOKENS, "completed_epoch": time.time()}
    manifest_path = args.mission / "GENERATION_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    atomic_json(status_path, {"status": "PASS", "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "updated_epoch": time.time()})
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
