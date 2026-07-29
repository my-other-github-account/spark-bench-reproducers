#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import platform
import socket
import time
from typing import Any, Dict, List

from p968_common import ARMS, MAX_TOKENS, atomic_json, conversation, load_dataset, row_path, sha256, sha256_json, sha_i64, summarize_vllm_logprobs, valid_row

PUBLIC_SOURCE_ROOT = pathlib.Path(os.environ.get("P968_TRUE_C_SOURCE_ROOT", "/PUBLIC_SOURCE_ROOT"))
MODEL_ROOT = pathlib.Path(os.environ.get("P968_TRUE_C_MODEL_ROOT", str(PUBLIC_SOURCE_ROOT / "model_view")))
CLAIM_PATH = pathlib.Path(os.environ.get("P968_HOST_CLAIM", "/run/p968/HOST_CLAIM.json"))
EXPECTED_HOSTNAME = os.environ.get("P968_TRUE_C_HOSTNAME", "compute-node-b")

EXPECTED_ARTIFACTS = {
    "model_config": (str(MODEL_ROOT / "config.json"), "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818"),
    "model_index": (str(MODEL_ROOT / "model.safetensors.index.json"), "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a"),
    "tokenizer": (str(MODEL_ROOT / "tokenizer.json"), "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf"),
    "tokenizer_config": (str(MODEL_ROOT / "tokenizer_config.json"), "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547"),
    "dense_patch": (str(PUBLIC_SOURCE_ROOT / "overlay/dense_patch.combo_v4_step0.safetensors"), "efef5162b4f26d6c7319f7cd2b5123fd6023493d1e693d7a8fdc615ab3f73dee"),
    "wire_manifest": (str(PUBLIC_SOURCE_ROOT / "wire/combo_v4_step0/PACK_MANIFEST.json"), "de88c68c221e6e6af7bc93d2b1b3f7af64fad6bea0c598738b7d09f80b8223df"),
    "wire_complete": (str(PUBLIC_SOURCE_ROOT / "wire/combo_v4_step0/PACK_COMPLETE"), "e6be6796bc2f91fa60defe2e93e4bde096f2ab2f200dc3d6bd4c96a579b797d7"),
}


def artifact_hashes() -> Dict[str, str]:
    observed = {}
    for label, item in EXPECTED_ARTIFACTS.items():
        path, expected = item
        value = sha256(pathlib.Path(path))
        if value != expected:
            raise RuntimeError("artifact drift %s: %s" % (label, value))
        observed[label] = value
    return observed


def assert_claim(task_id: str, mission: pathlib.Path) -> str:
    path = CLAIM_PATH
    value = json.loads(path.read_text())
    if value.get("owner") != task_id or value.get("mission") != str(mission) or value.get("status") != "CLAIMED":
        raise RuntimeError("claim mismatch: %r" % value)
    return sha256(path)


def graph_evidence(log_path: pathlib.Path) -> Dict[str, Any]:
    lines = []
    if log_path.exists():
        lines = [line.strip() for line in log_path.read_text(errors="replace").splitlines() if "DECODE-GRAPH ON-PATH sentinel" in line]
    t_values = []
    for line in lines:
        marker = "T="
        if marker in line:
            tail = line.split(marker, 1)[1].split()[0].rstrip(",")
            try:
                t_values.append(int(tail))
            except ValueError:
                pass
    return {"sentinel_count": len(lines), "sentinel_T_values": t_values, "decode_T_le_4_hit": any(value <= 4 for value in t_values), "line_sha256": [hashlib.sha256(line.encode()).hexdigest() for line in lines]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mission", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", action="append", required=True, help="name=/absolute/path")
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--model-arm", default="true_c")
    parser.add_argument("--arms", default="sampled,greedy")
    parser.add_argument("--smoke-tasks", type=int, default=0)
    parser.add_argument("--logprobs", type=int, default=5)
    args = parser.parse_args()

    if socket.gethostname() != EXPECTED_HOSTNAME:
        raise RuntimeError("true-C runner is pinned to %s" % EXPECTED_HOSTNAME)
    args.mission.mkdir(parents=True, exist_ok=True)
    claim_sha = assert_claim(args.task_id, args.mission)
    before = artifact_hashes()
    datasets = {}
    dataset_receipts = {}
    for spec in args.dataset:
        name, raw_path = spec.split("=", 1)
        rows, receipt = load_dataset(name, pathlib.Path(raw_path))
        if args.smoke_tasks:
            rows = rows[: args.smoke_tasks]
        datasets[name] = rows
        dataset_receipts[name] = receipt

    engine_contract = {
        "model": str(MODEL_ROOT),
        "trust_remote_code": True,
        "kv_cache_dtype": "fp8",
        "block_size": 256,
        "max_model_len": 5120,
        "gpu_memory_utilization": 0.80,
        "kv_cache_memory_bytes": 2415919104,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 4,
        "scheduler_reserve_full_isl": False,
        "enable_prefix_caching": False,
        "enforce_eager": False,
        "kernel_config": {"enable_flashinfer_autotune": False},
        "compilation_config": {"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
    }
    if os.getenv("VLLM_MOE_W2_DECODE_GRAPH") != "1" or os.getenv("VLLM_MOE_W2_DECODE_GRAPH_MAX_T") != "4":
        raise RuntimeError("graph-on environment drift")

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    load_started = time.perf_counter()
    llm = LLM(**engine_contract)
    load_seconds = time.perf_counter() - load_started
    first_rows = next(iter(datasets.values()))[:4]
    warm = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=12, min_tokens=12)
    warm_started = time.perf_counter()
    llm.chat([conversation(row) for row in first_rows], warm, use_tqdm=False)
    warm_seconds = time.perf_counter() - warm_started

    arms = [name for name in args.arms.split(",") if name]
    status_path = args.mission / "run/GENERATION_STATUS.json"
    sealed = 0
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
                for offset in range(0, len(pending), 4):
                    assert_claim(args.task_id, args.mission)
                    batch = pending[offset : offset + 4]
                    params = SamplingParams(temperature=float(arm["temperature"]), top_p=float(arm["top_p"]), seed=seed, max_tokens=MAX_TOKENS, min_tokens=1, logprobs=args.logprobs)
                    started = time.perf_counter()
                    outputs = llm.chat([conversation(row) for row in batch], params, use_tqdm=False)
                    wall = time.perf_counter() - started
                    if len(outputs) != len(batch):
                        raise RuntimeError("output count drift")
                    batch_rows = []
                    for source, output in zip(batch, outputs):
                        if len(output.outputs) != 1:
                            raise RuntimeError("choice count drift")
                        choice = output.outputs[0]
                        text = choice.text
                        token_ids = [int(value) for value in choice.token_ids]
                        if not isinstance(text, str):
                            raise RuntimeError("non-string output")
                        value = {
                            "schema": "p968-generation-row-v1",
                            "task_id": source["task_id"],
                            "dataset": dataset_name,
                            "decode_arm": arm_name,
                            "sample_index": sample_index,
                            "seed": seed,
                            "model_arm": args.model_arm,
                            "task_owner": args.task_id,
                            "claim_sha256": claim_sha,
                            "dataset_sha256": dataset_receipts[dataset_name]["sha256"],
                            "prompt_sha256": sha256_json(conversation(source)),
                            "request": {"temperature": arm["temperature"], "top_p": arm["top_p"], "max_tokens": MAX_TOKENS, "logprobs": args.logprobs},
                            "output": {"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_count": len(token_ids), "token_ids_sha256_i64le": sha_i64(token_ids), "finish_reason": choice.finish_reason, "stop_reason": getattr(choice, "stop_reason", None)},
                            "uncertainty": summarize_vllm_logprobs(token_ids, getattr(choice, "logprobs", None)),
                            "batch_task_ids": [item["task_id"] for item in batch],
                            "batch_wall_seconds": wall,
                            "created_epoch": time.time(),
                        }
                        path = row_path(args.out, args.model_arm, dataset_name, arm_name, sample_index, source["task_id"])
                        atomic_json(path, value)
                        batch_rows.append({"task_id": source["task_id"], "path": str(path), "sha256": sha256(path)})
                        sealed += 1
                    batch_receipt_path = args.mission / "generation_batches" / dataset_name / arm_name / ("s%03d_b%04d.json" % (sample_index, offset // 4))
                    atomic_json(batch_receipt_path, {"schema": "p968-generation-batch-v1", "status": "PASS", "dataset": dataset_name, "decode_arm": arm_name, "sample_index": sample_index, "seed": seed, "rows": batch_rows, "wall_seconds": wall, "graph": graph_evidence(pathlib.Path("/tmp/vllm_debug_tp0.log")), "created_epoch": time.time()})
                    atomic_json(status_path, {"status": "RUNNING", "sealed_this_process": sealed, "last_batch": str(batch_receipt_path), "updated_epoch": time.time()})
                    print(json.dumps({"event": "sealed_batch", "dataset": dataset_name, "arm": arm_name, "sample": sample_index, "tasks": [item["task_id"] for item in batch], "wall_seconds": wall}, sort_keys=True), flush=True)

    after = artifact_hashes()
    if before != after:
        raise RuntimeError("artifact TOCTOU drift")
    manifest = {
        "schema": "p968-generation-manifest-v1",
        "status": "PASS",
        "task_id": args.task_id,
        "host": socket.gethostname(),
        "model_arm": args.model_arm,
        "claim_sha256": claim_sha,
        "engine_contract": engine_contract,
        "engine_contract_sha256": sha256_json(engine_contract),
        "datasets": dataset_receipts,
        "arms": {name: ARMS[name] for name in arms},
        "max_tokens": MAX_TOKENS,
        "vllm_version": vllm_version,
        "python": platform.python_version(),
        "load_seconds": load_seconds,
        "warm_seconds": warm_seconds,
        "graph": graph_evidence(pathlib.Path("/tmp/vllm_debug_tp0.log")),
        "artifact_hashes": after,
        "completed_epoch": time.time(),
    }
    manifest_path = args.mission / "GENERATION_MANIFEST.json"
    atomic_json(manifest_path, manifest)
    atomic_json(status_path, {"status": "PASS", "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "updated_epoch": time.time()})
    print(json.dumps({"status": "PASS", "manifest": str(manifest_path), "sha256": sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
