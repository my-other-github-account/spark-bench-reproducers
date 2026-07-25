#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any
from urllib.parse import urlparse

TARGETS = (512, 2048, 8192, 16384)
PREFIXES = (
    "Atlas cache-cold row one",
    "Beacon cache-cold row two",
    "Cedar cache-cold row three",
)
WARM_PREFIX = "Kernel-shape warmup excluded from measured rows"
CORPUS = (
    "The quick brown fox crosses the quiet valley while a field researcher records "
    "weather, soil, and river conditions in a careful notebook. Each observation is "
    "written in complete sentences so the passage remains ordinary natural language. "
    "Later, the research team compares the notes, checks the measurements, and prepares "
    "a concise account for colleagues who were not present during the survey. "
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


class MemAvailableMonitor:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._lock = threading.Lock()
        self._minimum: int | None = None
        threading.Thread(target=self._run, name="memavailable-monitor", daemon=True).start()

    @staticmethod
    def _read() -> int:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
        raise RuntimeError("MemAvailable unavailable")

    def _run(self) -> None:
        while True:
            value = self._read()
            with self._lock:
                if self._minimum is None or value < self._minimum:
                    self._minimum = value
            time.sleep(self.interval)

    def reset(self) -> None:
        with self._lock:
            self._minimum = self._read()

    def minimum(self) -> int:
        with self._lock:
            return self._minimum if self._minimum is not None else self._read()


def make_exact_prompt(tokenizer: Any, prefix: str, target: int) -> tuple[str, list[int]]:
    text = prefix + ": " + CORPUS
    while len(tokenizer.encode(text, add_special_tokens=False).ids) < target + 64:
        text += CORPUS
    ids = list(tokenizer.encode(text, add_special_tokens=False).ids[:target])
    prompt = tokenizer.decode(ids)
    for _ in range(8):
        check = list(tokenizer.encode(prompt, add_special_tokens=False).ids)
        if len(check) == target:
            return prompt, check
        if len(check) > target:
            prompt = tokenizer.decode(check[:target])
        else:
            prompt += CORPUS
            prompt = tokenizer.decode(
                tokenizer.encode(prompt, add_special_tokens=False).ids[:target])
    raise RuntimeError(
        f"could not make exact prompt target={target}, got={len(tokenizer.encode(prompt, add_special_tokens=False).ids)}")


def get_json(base: str, path: str, timeout: float = 30.0) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    conn.request("GET", path)
    response = conn.getresponse()
    body = response.read()
    status = response.status
    conn.close()
    return status, json.loads(body)


def post_stream(base: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    parsed = urlparse(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    body = json.dumps(payload).encode()
    started = time.perf_counter()
    conn.request("POST", "/v1/completions", body=body,
                 headers={"Content-Type": "application/json"})
    response = conn.getresponse()
    events: list[dict[str, Any]] = []
    first_client_seconds = None
    while True:
        line = response.readline()
        if not line:
            break
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith(b"data:"):
            stripped = stripped[5:].lstrip()
        if stripped == b"[DONE]":
            continue
        event = json.loads(stripped)
        if "event" not in event:
            if "token_text" in event:
                event["event"] = "first_token"
            elif "choices" in event:
                event["event"] = "done"
        events.append(event)
        if event.get("event") == "error":
            raise RuntimeError(f"server stream error: {event}")
        if event.get("event") == "first_token" and first_client_seconds is None:
            first_client_seconds = time.perf_counter() - started
    wall = time.perf_counter() - started
    status = response.status
    conn.close()
    first = next((event for event in events if event.get("event") == "first_token"), None)
    done = next((event for event in events if event.get("event") == "done"), None)
    if status != 200 or first is None or done is None or first_client_seconds is None:
        raise RuntimeError(
            f"incomplete stream status={status} first={first is not None} done={done is not None} events={events[-3:]}")
    return {
        "http_status": status,
        "client_ttft_seconds": first_client_seconds,
        "client_total_wall_seconds": wall,
        "first": first,
        "done": done,
    }


def all_positive(values: dict[str, Any]) -> bool:
    return len(values) == 4 and all(int(value) > 0 for value in values.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8128")
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--targets", default=",".join(str(value) for value in TARGETS))
    parser.add_argument("--rows", type=int, default=len(PREFIXES))
    parser.add_argument("--task", default="public-validation")
    parser.add_argument("--variant", default="mixed_backpack")
    args = parser.parse_args()
    targets = tuple(int(value) for value in args.targets.split(",") if value)
    prefixes = PREFIXES[:args.rows]
    if not targets or not prefixes:
        raise ValueError("targets and rows must be nonempty")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    health_status, health = get_json(args.base, "/health")
    if health_status != 200 or health.get("status") != "ok":
        raise RuntimeError(f"server not healthy: http={health_status} body={health}")
    if health.get("cotenant_guard", {}).get("allowed") is not True:
        raise RuntimeError(f"cotenant fence not in exact wait: {health.get('cotenant_guard')}")
    monitor = MemAvailableMonitor()

    out = args.out.resolve()
    rows_dir = out / "rows"
    warmups_dir = out / "warmups"
    rows: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    ledger_path = out / "ROW_LEDGER.json"

    for target in targets:
        warm_prompt, warm_ids = make_exact_prompt(tokenizer, WARM_PREFIX, target)
        warm = post_stream(args.base, {
            "model": health["model"], "prompt": warm_prompt,
            "expected_prompt_tokens": target, "max_tokens": 1, "stream": True,
        }, args.timeout)
        warm_record = {
            "schema": "mixed-prefill-shape-warmup-v1",
            "task": args.task, "prompt_tokens": target,
            "prompt_sha256": hashlib.sha256(warm_prompt.encode()).hexdigest(),
            "token_id_sha256": hashlib.sha256(
                json.dumps(warm_ids, separators=(",", ":")).encode()).hexdigest(),
            "excluded_from_measurements": True,
            "result": warm, "finished_unix": time.time(),
        }
        atomic_json(warmups_dir / f"warmup_{target:05d}.json", warm_record)
        warmups.append(warm_record)

        for row_index, prefix in enumerate(prefixes, start=1):
            health_now_status, health_now = get_json(args.base, "/health")
            if health_now_status != 200 or health_now.get("cotenant_guard", {}).get("allowed") is not True:
                raise RuntimeError(f"cotenant fence changed before target={target} row={row_index}")
            prompt, ids = make_exact_prompt(tokenizer, prefix, target)
            monitor.reset()
            result = post_stream(args.base, {
                "model": health["model"], "prompt": prompt,
                "expected_prompt_tokens": target,
                "max_tokens": args.decode_tokens, "stream": True,
            }, args.timeout)
            request_mem_available_min = monitor.minimum()
            first = result["first"]
            done = result["done"]
            usage = done["usage"]
            mixed = done["mixed_tier"]
            client_ttft = float(result["client_ttft_seconds"])
            row = {
                "schema": "mixed-prefill-ladder-row-v1",
                "task": args.task, "variant": args.variant,
                "host": health["host"], "prompt_target_tokens": target,
                "prompt_tokens": int(usage["prompt_tokens"]),
                "prompt_prefix": prefix, "cache_cold_row": row_index,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "token_id_sha256": hashlib.sha256(
                    json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
                "natural_text_prompt": True,
                "client_ttft_seconds": client_ttft,
                "server_ttft_seconds": float(mixed["ttft_seconds"]),
                "server_prefill_seconds": float(mixed["prefill_seconds"]),
                "prefill_tok_s_prompt_over_client_ttft": target / client_ttft,
                "prefill_tok_s_server_prefill_only": float(mixed["prefill_tok_s_server"]),
                "decode_tokens": int(usage["completion_tokens"]),
                "decode_tok_s": float(mixed["decode_tok_s"]),
                "prefill_tier_kernel_launches": mixed["prefill_tier_kernel_launches"],
                "decode_tier_kernel_launches": mixed["decode_tier_kernel_launches"],
                "prefill_actual_triton_kernel_launches": mixed.get(
                    "prefill_actual_triton_kernel_launches"),
                "decode_actual_triton_kernel_launches": mixed.get(
                    "decode_actual_triton_kernel_launches"),
                "prefill_mbatched_dispatch_calls": mixed.get("prefill_mbatched_dispatch_calls", 0),
                "prefill_mbatched_rows": mixed.get("prefill_mbatched_rows", 0),
                "prefill_dense_dispatch_calls": mixed.get("prefill_dense_dispatch_calls", 0),
                "prefill_dequantizations": mixed.get("prefill_dequantizations", 0),
                "prefill_dense_gemm_chunks": mixed.get("prefill_dense_gemm_chunks", 0),
                "decode_dense_dispatch_calls": mixed.get("decode_dense_dispatch_calls", 0),
                "mem_available_bytes_after_request": mixed.get("mem_available_bytes"),
                "mem_available_bytes_min_during_request": request_mem_available_min,
                "request_vmhwm_bytes": mixed.get("vmhwm_bytes"),
                "request_vmrss_bytes": mixed.get("vmrss_bytes"),
                "residency_mode": mixed.get("residency_mode", health["residency"].get("mode")),
                "transient_scratch_bytes": mixed.get("transient_scratch_bytes_declared"),
                "resident_product_bytes": mixed.get("resident_product_bytes"),
                "kv_cache_bytes": mixed.get("kv_cache_bytes"),
                "kv_cache_note": mixed.get("kv_cache_note"),
                "prefill_tier_counters": mixed.get("prefill_tier_counters"),
                "decode_tier_counters": mixed.get("decode_tier_counters"),
                "prefill_tier_expert_projection_operations": mixed[
                    "prefill_tier_expert_projection_operations"],
                "decode_tier_expert_projection_operations": mixed[
                    "decode_tier_expert_projection_operations"],
                "prefill_physical_logical_ratio": mixed[
                    "prefill_physical_logical_ratio"],
                "decode_physical_logical_ratio": mixed[
                    "decode_physical_logical_ratio"],
                "configured_layers": mixed["configured_layers"],
                "active_layers": mixed["active_layers"],
                "dedup_factor": mixed["placeholder_exact_dedup_factor"],
                "prefix_cache_enabled": mixed["prefix_cache_enabled"],
                "mtp_enabled": mixed["mtp_enabled"],
                "max_model_len": health["max_model_len"],
                "model_len_margin_tokens": (
                    int(health["max_model_len"]) - target - args.decode_tokens),
                "http_status": result["http_status"],
                "client_total_wall_seconds": result["client_total_wall_seconds"],
                "server_request_id": done["id"],
                "residency": {
                    "mode": health["residency"].get("mode", "anonymous_exact"),
                    "mem_available_drop_bytes": health["residency"][
                        "mem_available_drop_bytes"],
                    "vmhwm_bytes": health["residency"]["process_after"]["vmhwm_bytes"],
                    "resident_product_bytes": health["residency"]["resident_product_bytes"],
                    "target_product_bytes": health["residency"]["target_product_bytes"],
                },
                "finished_unix": time.time(),
            }
            row["gates"] = {
                "http_200": row["http_status"] == 200,
                "prompt_tokens_exact": row["prompt_tokens"] == target,
                "decode_tokens_128": row["decode_tokens"] == args.decode_tokens == 128,
                "layers_43": row["configured_layers"] == row["active_layers"] == 43,
                "dedup_1": row["dedup_factor"] == 1,
                "prefix_cache_off": row["prefix_cache_enabled"] is False,
                "mtp_off": row["mtp_enabled"] is False,
                "prefill_physical_logical_ratio_1": abs(
                    float(row["prefill_physical_logical_ratio"]) - 1.0) <= 1e-12,
                "decode_physical_logical_ratio_1": abs(
                    float(row["decode_physical_logical_ratio"]) - 1.0) <= 1e-12,
                "all_prefill_tier_launches_nonzero": all_positive(
                    row["prefill_tier_kernel_launches"]),
                "all_decode_tier_launches_nonzero": all_positive(
                    row["decode_tier_kernel_launches"]),
                "memavailable_drop_ge_90gb": (
                    row["residency"]["mode"] == "file_backed_mincore" or
                    row["residency"]["mem_available_drop_bytes"] >= 90_000_000_000),
                "vmhwm_ge_90gb": (
                    row["residency"]["mode"] == "file_backed_mincore" or
                    row["residency"]["vmhwm_bytes"] >= 90_000_000_000),
                "resident_product_exact": (
                    row["residency"]["resident_product_bytes"]
                    == row["residency"]["target_product_bytes"]),
                "resident_product_dynamic_exact": (
                    row["resident_product_bytes"]
                    == row["residency"]["target_product_bytes"]),
                "model_len_margin_nonnegative": row["model_len_margin_tokens"] >= 0,
            }
            if args.variant in {"rung1_dequant_dense", "rung1_p526_hybrid"}:
                row["gates"].update({
                    "prefill_dense_dispatch_nonzero": row["prefill_dense_dispatch_calls"] > 0,
                    "dequant_once_per_dense_dispatch": (
                        row["prefill_dequantizations"]
                        == row["prefill_dense_dispatch_calls"]),
                    "decode_retains_triton": (
                        int(row["decode_actual_triton_kernel_launches"] or 0) > 0),
                    "decode_has_no_dense_dispatch": row["decode_dense_dispatch_calls"] == 0,
                    "memavailable_min_during_request_ge_8gib": (
                        row["mem_available_bytes_min_during_request"] >= (8 << 30)),
                })
            if args.variant == "rung1_p526_hybrid":
                row["gates"]["prefill_mbatched_dispatch_nonzero"] = (
                    row["prefill_mbatched_dispatch_calls"] > 0)
            row["status"] = "PASS" if all(row["gates"].values()) else "FAIL"
            row_path = rows_dir / f"mixed_pp{target:05d}_cold{row_index}.json"
            atomic_json(row_path, row)
            rows.append(row)
            atomic_json(ledger_path, {
                "schema": "mixed-prefill-row-ledger-v1", "task": args.task,
                "status": "RUNNING", "completed_rows": len(rows),
                "expected_rows": len(targets) * len(prefixes),
                "row_files": [
                    str(rows_dir / f"mixed_pp{r['prompt_target_tokens']:05d}_cold{r['cache_cold_row']}.json")
                    for r in rows],
                "updated_unix": time.time(),
            })
            print(json.dumps({
                "event": "banked_row", "target": target, "row": row_index,
                "status": row["status"], "ttft": row["client_ttft_seconds"],
                "prefill_tok_s": row["prefill_tok_s_prompt_over_client_ttft"],
                "decode_tok_s": row["decode_tok_s"], "path": str(row_path),
            }, sort_keys=True), flush=True)
            if row["status"] != "PASS":
                raise RuntimeError(f"row failed closed: {row_path} gates={row['gates']}")

    summary_rows = []
    for target in targets:
        group = [row for row in rows if row["prompt_target_tokens"] == target]
        launch_shapes = [row["prefill_tier_kernel_launches"] for row in group]
        summary_rows.append({
            "prompt_tokens": target,
            "rows": len(group),
            "client_ttft_seconds_raw": [row["client_ttft_seconds"] for row in group],
            "client_ttft_seconds_median": statistics.median(
                row["client_ttft_seconds"] for row in group),
            "prefill_tok_s_raw": [
                row["prefill_tok_s_prompt_over_client_ttft"] for row in group],
            "prefill_tok_s_median": statistics.median(
                row["prefill_tok_s_prompt_over_client_ttft"] for row in group),
            "decode_tok_s_raw": [row["decode_tok_s"] for row in group],
            "decode_tok_s_median": statistics.median(
                row["decode_tok_s"] for row in group),
            "prefill_tier_kernel_launches": launch_shapes[0],
            "prefill_launch_shapes_identical_across_cold_rows": all(
                shape == launch_shapes[0] for shape in launch_shapes),
            "mtp_enabled": False,
        })
    summary = {
        "schema": "mixed-prefill-ladder-result-v1",
        "task": args.task, "status": "PASS",
        "host": health["host"], "variant": args.variant,
        "targets": list(targets), "cache_cold_rows_per_target": len(prefixes),
        "measured_rows": len(rows), "warmup_rows_excluded": len(warmups),
        "rows": summary_rows,
        "raw_row_files": [
            str(rows_dir / f"mixed_pp{row['prompt_target_tokens']:05d}_cold{row['cache_cold_row']}.json")
            for row in rows],
        "residency": rows[0]["residency"],
        "health_pre_bench_gates": health["pre_bench_gates"],
        "artifact_sha256": health["artifact_sha256"],
        "artifact_manifest": health["artifact_manifest"],
        "uniform_qtip_comparison": {
            "run": False,
            "reason": "single-host residency fence permits only one >=90GB resident server; the mixed ladder is the systems-serving validation priority",
        },
        "prefix_cache_enabled": False, "mtp_enabled": False,
        "finished_unix": time.time(),
    }
    atomic_json(out / "MIXED_PREFILL_LADDER_RESULT.json", summary)
    atomic_json(ledger_path, {
        "schema": "mixed-prefill-row-ledger-v1", "task": args.task,
        "status": "PASS", "completed_rows": len(rows),
        "expected_rows": len(targets) * len(prefixes),
        "result": str(out / "MIXED_PREFILL_LADDER_RESULT.json"),
        "updated_unix": time.time(),
    })
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
