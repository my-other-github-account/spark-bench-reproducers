#!/usr/bin/env python3
"""Run a fixed-prompt Atlas benchmark and emit JSON + CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True, choices=("ar", "dflash"))
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--model-id")
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--min-prompts", type=int, default=64)
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def read_json(url: str, timeout: float, payload: dict | None = None) -> dict:
    if payload is None:
        req = urllib.request.Request(url, method="GET")
    else:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def resolve_model_id(base_url: str, timeout: float) -> str:
    models = read_json(f"{base_url}/v1/models", timeout)
    data = models.get("data", [])
    if not data:
        raise RuntimeError(f"{base_url}/v1/models returned no models")
    model_id = data[0].get("id")
    if not model_id:
        raise RuntimeError(f"{base_url}/v1/models returned malformed payload: {models}")
    return str(model_id)


def load_prompts(path: pathlib.Path, min_prompts: int) -> list[dict]:
    prompts = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("id", "category", "prompt"):
            if key not in row:
                raise RuntimeError(f"{path}:{lineno} missing key {key}")
        prompts.append(row)
    if len(prompts) < min_prompts:
        raise RuntimeError(f"{path} has {len(prompts)} prompts; need at least {min_prompts}")
    return prompts


def run_one(
    prompt_row: dict,
    base_url: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    timeout: float,
    extra_body: dict,
) -> dict:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_row["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    payload.update(extra_body)
    start_unix = time.time()
    start_perf = time.perf_counter()
    try:
        response = read_json(f"{base_url}/v1/chat/completions", timeout, payload)
        elapsed = time.perf_counter() - start_perf
        end_unix = time.time()
        choice = response["choices"][0]
        usage = response.get("usage", {})
        text = choice.get("message", {}).get("content", "")
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        completion_tokens = int(usage.get("completion_tokens", 0))
        return {
            "id": prompt_row["id"],
            "category": prompt_row["category"],
            "prompt": prompt_row["prompt"],
            "ok": True,
            "http_status": 200,
            "started_unix": start_unix,
            "finished_unix": end_unix,
            "elapsed_seconds": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "output_tps": completion_tokens / elapsed if elapsed > 0 else 0.0,
            "finish_reason": choice.get("finish_reason", ""),
            "response_text": text,
        }
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - start_perf
        end_unix = time.time()
        error_body = exc.read().decode("utf-8", errors="replace")
        return {
            "id": prompt_row["id"],
            "category": prompt_row["category"],
            "prompt": prompt_row["prompt"],
            "ok": False,
            "http_status": exc.code,
            "started_unix": start_unix,
            "finished_unix": end_unix,
            "elapsed_seconds": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "output_tps": 0.0,
            "finish_reason": "",
            "response_text": "",
            "error": error_body,
        }
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - start_perf
        end_unix = time.time()
        return {
            "id": prompt_row["id"],
            "category": prompt_row["category"],
            "prompt": prompt_row["prompt"],
            "ok": False,
            "http_status": 0,
            "started_unix": start_unix,
            "finished_unix": end_unix,
            "elapsed_seconds": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "output_tps": 0.0,
            "finish_reason": "",
            "response_text": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def summarize(results: list[dict], args: argparse.Namespace, model_id: str) -> dict:
    started = min(row["started_unix"] for row in results)
    finished = max(row["finished_unix"] for row in results)
    wall_seconds = finished - started
    total_request_seconds = sum(row["elapsed_seconds"] for row in results)
    total_prompt_tokens = sum(row["prompt_tokens"] for row in results)
    total_completion_tokens = sum(row["completion_tokens"] for row in results)
    ok_results = [row for row in results if row["ok"]]
    failed_results = [row for row in results if not row["ok"]]
    prompt_tps_values = [row["output_tps"] for row in ok_results]
    return {
        "label": args.label,
        "mode": args.mode,
        "base_url": args.base_url,
        "model_id": model_id,
        "prompt_file": str(pathlib.Path(args.prompts).resolve()),
        "prompt_count": len(results),
        "success_count": len(ok_results),
        "failure_count": len(failed_results),
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "concurrency": args.concurrency,
        "token_accounting": "Atlas OpenAI response usage.completion_tokens divided by aggregate benchmark wall-clock seconds; per-prompt tps uses completion_tokens / request elapsed_seconds.",
        "started_at_utc": dt.datetime.fromtimestamp(started, tz=dt.timezone.utc).isoformat(),
        "finished_at_utc": dt.datetime.fromtimestamp(finished, tz=dt.timezone.utc).isoformat(),
        "wall_seconds": wall_seconds,
        "total_request_seconds": total_request_seconds,
        "prompt_tokens_total": total_prompt_tokens,
        "completion_tokens_total": total_completion_tokens,
        "aggregate_output_tps": total_completion_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "aggregate_prompt_tps": total_prompt_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        "mean_per_prompt_output_tps": statistics.mean(prompt_tps_values) if prompt_tps_values else 0.0,
        "median_per_prompt_output_tps": statistics.median(prompt_tps_values) if prompt_tps_values else 0.0,
    }


def write_csv(path: pathlib.Path, results: list[dict]) -> None:
    fieldnames = [
        "id",
        "category",
        "ok",
        "http_status",
        "elapsed_seconds",
        "prompt_tokens",
        "completion_tokens",
        "output_tps",
        "finish_reason",
        "response_text",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    prompt_path = pathlib.Path(args.prompts)
    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(prompt_path, args.min_prompts)
    extra_body = json.loads(args.extra_body_json)
    model_id = args.model_id or resolve_model_id(args.base_url, args.timeout)
    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "time": utc_now(),
                "label": args.label,
                "mode": args.mode,
                "model_id": model_id,
                "prompt_count": len(prompts),
                "concurrency": args.concurrency,
            }
        ),
        flush=True,
    )
    results: list[dict] = [None] * len(prompts)  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_map = {
            pool.submit(
                run_one,
                prompt_row,
                args.base_url,
                model_id,
                args.max_tokens,
                args.temperature,
                args.timeout,
                extra_body,
            ): idx
            for idx, prompt_row in enumerate(prompts)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            results[idx] = future.result()
            row = results[idx]
            print(
                json.dumps(
                    {
                        "event": "prompt_done",
                        "id": row["id"],
                        "ok": row["ok"],
                        "elapsed_seconds": round(row["elapsed_seconds"], 3),
                        "completion_tokens": row["completion_tokens"],
                        "output_tps": round(row["output_tps"], 3),
                    }
                ),
                flush=True,
            )
    summary = summarize(results, args, model_id)
    artifact = {
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), results)
    print(json.dumps({"event": "benchmark_complete", "time": utc_now(), "summary": summary}), flush=True)
    if summary["failure_count"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
