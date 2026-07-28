#!/usr/bin/env python3
"""One resumable, canonical HumanEval-164 graph-on visible-eval run."""
from __future__ import annotations
import gc, hashlib, json, os, platform, re, socket, struct, subprocess, time
from pathlib import Path
from typing import Any

TASK = "task-redacted"
M = Path("${SPARK_HOME}/missions/VISIBLE_EVAL_FULL164_t_872fd554_s2")
CODE = M / "task_local"
MODEL = Path("${SPARK_HOME}/missions/TAILFIX_BASELINE_OWNGEN_t_614cf545/model_view")
WIRE = Path("${SPARK_HOME}/missions/TAILFIX_BASELINE_OWNGEN_t_614cf545/wire/combo_v4_step0")
DENSE = Path("${SPARK_HOME}/missions/TAILFIX_BASELINE_OWNGEN_t_614cf545/overlay/dense_patch.combo_v4_step0.safetensors")
DATASET = Path("${SPARK_HOME}/missions/EVAL_SERVE_P4_t_53e6a555_s2/inputs/HumanEvalPlus-v0.1.10.jsonl")
PARENT_RECEIPT = Path("${SPARK_HOME}/missions/VISIBLE_EVAL_GRAPH_t_11b414f5_s2/out/FINAL_GRAPH_VISIBLE_EVAL_RECEIPT.json")
EXPECTED_PARENT_SHA = "90d64910458b87ff64d69b9fbb524b0a322c99e94dbb2fb88e596a28c478ea2d"
EXPECTED = {
    "dataset": "42526ec0e7d5f3ee0b06d6ced98f8c8bae3d76519151bfb3d36f79010645bd7f",
    "model_config": "b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818",
    "model_index": "7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a",
    "tokenizer": "8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf",
    "tokenizer_config": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "wire_manifest": "de88c68c221e6e6af7bc93d2b1b3f7af64fad6bea0c598738b7d09f80b8223df",
    "wire_complete": "e6be6796bc2f91fa60defe2e93e4bde096f2ab2f200dc3d6bd4c96a579b797d7",
    "dense_patch": "efef5162b4f26d6c7319f7cd2b5123fd6023493d1e693d7a8fdc615ab3f73dee",
    "checkpoint": "fae41d519193269aec4b2221c97a1dc00e0b00d3d66074d917a78489fac2149c",
}
IDS = [f"HumanEval/{i}" for i in range(164)]
WARMUP_IDS = ["HumanEval/4", "HumanEval/5", "HumanEval/6", "HumanEval/7"]
PREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
PROGRESS = M / "out/progress.jsonl"
LOG = M / "logs/pipeline.log"
BUDGET_SECONDS = 5400.0

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def sha_i64(values: list[int]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(struct.pack("<q", int(value)))
    return h.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def append_jsonl(path: Path, value: Any) -> None:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, raw.encode()); os.fsync(fd)
    finally:
        os.close(fd)

def load_progress() -> list[dict[str, Any]]:
    if not PROGRESS.exists():
        return []
    rows = []
    for line_no, line in enumerate(PROGRESS.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(f"invalid sealed progress line {line_no}: {exc}")
    got = [r["task_id"] for r in rows]
    if got != IDS[:len(got)] or len(got) != len(set(got)):
        raise RuntimeError(f"sealed progress is not a unique canonical prefix: {got[-5:]}")
    return rows

def assert_claim() -> str:
    path = Path("${SPARK_HOME}/HOST_CLAIM.json")
    data = json.loads(path.read_text())
    if socket.gethostname() != "spark-2" or data.get("owner") != TASK or data.get("mission") != str(M):
        raise RuntimeError(f"claim mismatch host={socket.gethostname()} owner={data.get('owner')} mission={data.get('mission')}")
    return sha256(path)

def artifact_hashes() -> dict[str, str]:
    paths = {
        "dataset": DATASET,
        "model_config": MODEL / "config.json",
        "model_index": MODEL / "model.safetensors.index.json",
        "tokenizer": MODEL / "tokenizer.json",
        "tokenizer_config": MODEL / "tokenizer_config.json",
        "wire_manifest": WIRE / "PACK_MANIFEST.json",
        "wire_complete": WIRE / "PACK_COMPLETE",
        "dense_patch": DENSE,
    }
    got = {key: sha256(path) for key, path in paths.items()}
    for key, value in got.items():
        if value != EXPECTED[key]:
            raise RuntimeError(f"{key} hash drift: {value} != {EXPECTED[key]}")
    if json.loads((WIRE / "PACK_MANIFEST.json").read_text()).get("checkpoint_sha256") != EXPECTED["checkpoint"]:
        raise RuntimeError("wire checkpoint hash drift")
    if sha256(PARENT_RECEIPT) != EXPECTED_PARENT_SHA:
        raise RuntimeError("completed parent gate receipt drift")
    return got

def graph_evidence() -> dict[str, Any]:
    text = LOG.read_text(errors="ignore") if LOG.exists() else ""
    lines = [line for line in text.splitlines() if "DECODE-GRAPH ON-PATH sentinel" in line]
    t_values = []
    for line in lines:
        match = re.search(r"\bT=(\d+)", line)
        if match:
            t_values.append(int(match.group(1)))
    return {
        "required_sentinel": "DECODE-GRAPH ON-PATH sentinel",
        "sentinel_count": len(lines),
        "sentinel_T_values": t_values,
        "decode_T_le_4_hit": bool(lines) and bool(t_values) and all(1 <= value <= 4 for value in t_values),
        "sentinel_line_sha256": [hashlib.sha256(line.encode()).hexdigest() for line in lines],
    }

def conv(dataset: dict[str, dict[str, Any]], task_id: str) -> list[dict[str, str]]:
    prompt = str(dataset[task_id]["prompt"]).strip()
    return [{"role": "user", "content": PREFIX + f"\n```python\n{prompt}\n```"}]

def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return repr(value)

def main() -> None:
    start_path = M / "run/START.json"
    if start_path.exists():
        start = json.loads(start_path.read_text())
    else:
        start = {"pipeline_start_epoch": time.time(), "task_id": TASK, "host": socket.gethostname(), "pid": os.getpid()}
        atomic_json(start_path, start)
    pipeline_start = float(start["pipeline_start_epoch"])
    claim_sha = assert_claim()
    before = artifact_hashes()
    dataset_rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    dataset = {row["task_id"]: row for row in dataset_rows}
    if [row["task_id"] for row in dataset_rows] != IDS:
        raise RuntimeError("HumanEval dataset is not canonical 0..163 order")
    existing = load_progress()
    if len(existing) == 164:
        print(json.dumps({"status": "ALREADY_COMPLETE", "rows": 164}), flush=True)
        return
    engine_contract = {
        "model": str(MODEL), "trust_remote_code": True, "kv_cache_dtype": "fp8", "block_size": 256,
        "max_model_len": 5120, "gpu_memory_utilization": 0.80, "kv_cache_memory_bytes": 2415919104,
        "max_num_batched_tokens": 8192, "max_num_seqs": 4, "scheduler_reserve_full_isl": False,
        "enable_prefix_caching": False, "enforce_eager": False,
        "kernel_config": {"enable_flashinfer_autotune": False},
        "compilation_config": {"mode": "VLLM_COMPILE", "cudagraph_mode": "NONE"},
        "VLLM_MOE_W2_DECODE_GRAPH": os.getenv("VLLM_MOE_W2_DECODE_GRAPH"),
        "VLLM_MOE_W2_DECODE_GRAPH_MAX_T": os.getenv("VLLM_MOE_W2_DECODE_GRAPH_MAX_T"),
    }
    if engine_contract["VLLM_MOE_W2_DECODE_GRAPH"] != "1" or engine_contract["VLLM_MOE_W2_DECODE_GRAPH_MAX_T"] != "4":
        raise RuntimeError("graph-on environment drift")
    task_contract = {
        "dataset": str(DATASET), "dataset_sha256": EXPECTED["dataset"], "task_ids": IDS,
        "prompt_prefix": PREFIX, "prompt_format": "one user message; vLLM DeepSeek-V4 renderer",
        "temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 4096, "output_count": 164,
        "batch_size": 4,
    }
    scorer_image_id = subprocess.check_output(["docker", "image", "inspect", "evalplus:26d6d00", "--format", "{{.Id}}"], text=True).strip()
    from vllm import LLM, SamplingParams, __version__ as vllm_version
    llm_kw = {key: value for key, value in engine_contract.items() if key not in {"VLLM_MOE_W2_DECODE_GRAPH", "VLLM_MOE_W2_DECODE_GRAPH_MAX_T"}}
    load_t0 = time.perf_counter(); llm = LLM(**llm_kw); load_seconds = time.perf_counter() - load_t0
    warm = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=12, min_tokens=12)
    measured = SamplingParams(temperature=0.0, top_p=1.0, seed=0, max_tokens=4096, min_tokens=1)
    warm_convs = [conv(dataset, task_id) for task_id in WARMUP_IDS]
    warm_t0 = time.perf_counter()
    llm.chat(warm_convs[0], warm, use_tqdm=False)
    llm.chat(warm_convs, warm, use_tqdm=False)
    llm.chat(warm_convs, warm, use_tqdm=False)
    warm_seconds = time.perf_counter() - warm_t0
    completed = {row["task_id"] for row in existing}
    remaining = [task_id for task_id in IDS if task_id not in completed]
    batches_dir = M / "out/batches"; scores_dir = M / "out/scores"
    batches_dir.mkdir(parents=True, exist_ok=True); scores_dir.mkdir(parents=True, exist_ok=True)
    scorer_log = M / "logs/scorer.log"
    next_sequence = len(existing)
    for offset in range(0, len(remaining), 4):
        assert_claim()
        batch_ids = remaining[offset:offset + 4]
        batch_index = next_sequence // 4
        t0 = time.perf_counter()
        outputs = llm.chat([conv(dataset, task_id) for task_id in batch_ids], measured, use_tqdm=False)
        batch_generation_wall = time.perf_counter() - t0
        if len(outputs) != len(batch_ids):
            raise RuntimeError(f"batch {batch_index} output count {len(outputs)} != {len(batch_ids)}")
        raw_rows = []
        for task_id, output in zip(batch_ids, outputs, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError(f"choice count drift {task_id}")
            choice = output.outputs[0]
            text = choice.text
            if not isinstance(text, str) or not text:
                raise RuntimeError(f"empty output {task_id}")
            pids = [int(v) for v in output.prompt_token_ids]
            oids = [int(v) for v in choice.token_ids]
            raw_rows.append({
                "task_id": task_id, "solution": text,
                "prompt_tokens": len(pids), "output_tokens": len(oids),
                "prompt_token_ids_sha256_i64le": sha_i64(pids),
                "output_token_ids_sha256_i64le": sha_i64(oids),
                "output_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "finish_reason": choice.finish_reason,
                "stop_reason": jsonable(getattr(choice, "stop_reason", None)),
            })
        batch_path = batches_dir / f"batch_{batch_index:03d}.json"
        batch_receipt = {
            "schema": "visible-eval-generation-batch-v1", "status": "PASS", "task_id": TASK,
            "batch_index": batch_index, "batch_ids": batch_ids, "batch_generation_wall_seconds": batch_generation_wall,
            "rows": raw_rows, "created_epoch": time.time(),
        }
        atomic_json(batch_path, batch_receipt)
        score_path = scores_dir / f"batch_{batch_index:03d}.scores.json"
        batch_inside = "/work/" + str(batch_path.relative_to(M))
        score_inside = "/work/" + str(score_path.relative_to(M))
        cmd = [
            "docker", "run", "--rm", "--network", "none", "--cpus", "4", "--memory", "6g", "--pids-limit", "256",
            "-v", f"{M}:/work",
            "-v", f"{M / 'cache/evalplus/HumanEvalPlus-v0.1.10.jsonl'}:/work/HumanEvalPlus-v0.1.10.jsonl:ro",
            "-e", "HUMANEVAL_OVERRIDE_PATH=/work/HumanEvalPlus-v0.1.10.jsonl",
            "evalplus:26d6d00", "python", "/work/task_local/score_batch.py", "--batch", batch_inside, "--out", score_inside,
        ]
        score_t0 = time.perf_counter()
        with scorer_log.open("a") as logf:
            logf.write(f"BATCH {batch_index} START {time.time()} {batch_ids}\n"); logf.flush()
            subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT, timeout=600)
            logf.write(f"BATCH {batch_index} END {time.time()}\n"); logf.flush(); os.fsync(logf.fileno())
        score_invoke_wall = time.perf_counter() - score_t0
        score_receipt = json.loads(score_path.read_text())
        if score_receipt.get("batch_sha256") != sha256(batch_path) or score_receipt.get("batch_ids") != batch_ids:
            raise RuntimeError(f"score receipt identity drift batch {batch_index}")
        score_by_id = {row["task_id"]: row for row in score_receipt["rows"]}
        evidence = graph_evidence()
        for raw in raw_rows:
            task_id = raw["task_id"]
            scored = score_by_id[task_id]
            row = {
                "schema": "visible-eval-full164-progress-row-v1",
                "sequence_index": next_sequence, "task_id": task_id,
                "generation_batch_index": batch_index,
                "generation_batch_ids": batch_ids,
                "generation_batch_wall_seconds": batch_generation_wall,
                "generation_wall_seconds": batch_generation_wall / len(batch_ids),
                "prompt_tokens": raw["prompt_tokens"], "tokens": raw["output_tokens"], "output_tokens": raw["output_tokens"],
                "finish_reason": raw["finish_reason"],
                "base_status": scored["base_status"], "plus_status": scored["plus_status"],
                "base_score": scored["base_score"], "plus_score": scored["plus_score"], "evalplus_score": scored["evalplus_score"],
                "score_wall_seconds": scored["score_wall_seconds"], "score_batch_invoke_wall_seconds": score_invoke_wall,
                "graph_sentinel_count": evidence["sentinel_count"], "graph_sentinel_T_values": evidence["sentinel_T_values"],
                "cumulative_elapsed_seconds": time.time() - pipeline_start,
                "output_text_sha256": raw["output_text_sha256"], "sanitized_solution_sha256": scored["sanitized_solution_sha256"],
                "generation_batch_sha256": sha256(batch_path), "score_batch_sha256": sha256(score_path),
                "created_epoch": time.time(),
            }
            append_jsonl(PROGRESS, row)
            next_sequence += 1
            progress_rows = load_progress()
            atomic_json(M / "out/PROGRESS.json", {
                "schema": "visible-eval-full164-progress-v1", "status": "RUNNING", "completed": len(progress_rows),
                "total": 164, "last_task_id": task_id, "last_row_epoch": row["created_epoch"],
                "cumulative_elapsed_seconds": row["cumulative_elapsed_seconds"], "progress_jsonl_sha256": sha256(PROGRESS),
            })
            print(json.dumps({"ROW": next_sequence, "task_id": task_id, "base": row["base_score"], "plus": row["plus_score"], "tokens": row["tokens"], "elapsed": row["cumulative_elapsed_seconds"]}, sort_keys=True), flush=True)
    rows = load_progress()
    if len(rows) != 164 or [row["task_id"] for row in rows] != IDS:
        raise RuntimeError(f"coverage failure {len(rows)}/164")
    after = artifact_hashes()
    if before != after:
        raise RuntimeError("model/input artifact TOCTOU drift")
    evidence = graph_evidence()
    raw_aggregate = []
    sanitized_aggregate = []
    for batch_path in sorted(batches_dir.glob("batch_*.json")):
        batch = json.loads(batch_path.read_text())
        score_path = scores_dir / (batch_path.stem + ".scores.json")
        scores = {row["task_id"]: row for row in json.loads(score_path.read_text())["rows"]}
        for raw in batch["rows"]:
            raw_aggregate.append({"task_id": raw["task_id"], "solution": raw["solution"]})
            sanitized_aggregate.append({"task_id": raw["task_id"], "solution": scores[raw["task_id"]]["sanitized_solution"]})
    if [row["task_id"] for row in raw_aggregate] != IDS or [row["task_id"] for row in sanitized_aggregate] != IDS:
        raise RuntimeError("aggregate sample coverage/order drift")
    raw_path = M / "out/raw_solutions.jsonl"; samples_path = M / "out/samples.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in raw_aggregate))
    samples_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in sanitized_aggregate))
    total_now = time.time() - pipeline_start
    base_failures = [row["task_id"] for row in rows if not row["base_score"]]
    plus_failures = [row["task_id"] for row in rows if not row["plus_score"]]
    evalplus_failures = [row["task_id"] for row in rows if not row["evalplus_score"]]
    cfg = json.loads((MODEL / "config.json").read_text())
    final = {
        "schema": "visible-eval-actual-full164-graph-on-v1", "status": "PRE_RELEASE", "task_id": TASK, "host": "spark-2",
        "parent_gate": {"task_id": "task-redacted", "receipt": str(PARENT_RECEIPT), "receipt_sha256": EXPECTED_PARENT_SHA, "status": "CONSUMED_NO_RERUN"},
        "coverage": {"completed": 164, "total": 164, "exact_canonical_order": True, "task_ids_sha256": sha_json(IDS)},
        "scores": {
            "base_passes": 164 - len(base_failures), "base_pass_rate": (164 - len(base_failures)) / 164,
            "plus_passes": 164 - len(plus_failures), "plus_pass_rate": (164 - len(plus_failures)) / 164,
            "evalplus_passes": 164 - len(evalplus_failures), "evalplus_pass_rate": (164 - len(evalplus_failures)) / 164,
            "base_failure_ids": base_failures, "plus_failure_ids": plus_failures, "evalplus_failure_ids": evalplus_failures,
        },
        "timing": {
            "pipeline_start_epoch": pipeline_start, "pre_release_epoch": time.time(),
            "actual_total_wall_seconds_pre_release": total_now, "seconds_per_task_pre_release": total_now / 164,
            "model_load_seconds": load_seconds, "warmup_seconds": warm_seconds,
            "sum_generation_batch_wall_seconds": sum(json.loads(path.read_text())["batch_generation_wall_seconds"] for path in sorted(batches_dir.glob("batch_*.json"))),
            "budget_seconds": BUDGET_SECONDS,
        },
        "graph_proof": evidence,
        "official_graph": {"framework": "vLLM official HF DeepseekV4ForCausalLM graph", "architecture": cfg.get("architectures"), "model_type": cfg.get("model_type"), "num_hidden_layers": cfg.get("num_hidden_layers"), "vllm_version": vllm_version, "python": platform.python_version(), "plain_torch_expert_adapter_used": False},
        "engine_contract": engine_contract, "engine_contract_sha256": sha_json(engine_contract),
        "task_contract": task_contract, "task_contract_sha256": sha_json(task_contract),
        "model_config_input_hashes": {**after, "wire_checkpoint_sha256": EXPECTED["checkpoint"]},
        "scorer": {"image": "evalplus:26d6d00", "image_id": scorer_image_id, "network": "none", "dataset_sha256": EXPECTED["dataset"], "test_details": True},
        "artifacts": {
            "progress_jsonl": str(PROGRESS), "progress_jsonl_sha256": sha256(PROGRESS),
            "raw_solutions": str(raw_path), "raw_solutions_sha256": sha256(raw_path),
            "sanitized_samples": str(samples_path), "sanitized_samples_sha256": sha256(samples_path),
            "generation_batches": str(batches_dir), "score_batches": str(scores_dir), "pipeline_log": str(LOG),
        },
        "claim_sha256_at_start": claim_sha, "created_epoch": time.time(),
    }
    atomic_json(M / "out/FINAL_ACTUAL_FULL164_RECEIPT.json", final)
    atomic_json(M / "out/PROGRESS.json", {"schema": "visible-eval-full164-progress-v1", "status": "COMPLETE_PENDING_RELEASE", "completed": 164, "total": 164, "last_task_id": IDS[-1], "progress_jsonl_sha256": sha256(PROGRESS), "cumulative_elapsed_seconds": total_now})
    print(json.dumps({"status": "PRE_RELEASE", "coverage": 164, "base": final["scores"]["base_passes"], "plus": final["scores"]["plus_passes"], "wall": total_now, "graph": evidence["decode_T_le_4_hit"]}, sort_keys=True), flush=True)
    del llm; gc.collect()

if __name__ == "__main__":
    main()
