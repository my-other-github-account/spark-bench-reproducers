#!/usr/bin/env python3
import datetime as dt
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "deepseek-v4-flash"
MISSION_ROOT = os.environ.get("MISSION_ROOT", os.path.expanduser("~/missions"))
MODEL_ROOT = os.environ.get("MODEL_ROOT", "$MODEL_ROOT")
ROOT = os.path.join(MISSION_ROOT, "SERVED_AB")
ROW = os.path.join(ROOT, "SPARK8_TPS_GPU_ROW.json")
ROWS = os.path.join(ROOT, "SPARK8_TPS_GPU_ROWS.jsonl")
CLAIM = os.path.join(MISSION_ROOT, "LP4_BLOCKWISE", "HOST_CLAIM.json")
TASK = "t_d70c837a"

# Public-domain prose from Alice's Adventures in Wonderland; repeated to form
# a realistic long prompt rather than a synthetic repeated-token string.
SNIPPET = (
    "Alice was beginning to get very tired of sitting by her sister on the bank, "
    "and of having nothing to do: once or twice she had peeped into the book her "
    "sister was reading, but it had no pictures or conversations in it, and what "
    "is the use of a book, thought Alice, without pictures or conversation? "
)


def log(msg):
    print(f"[{dt.datetime.now(dt.timezone.utc).isoformat()}] {msg}", flush=True)


def valid_row(obj):
    required = ("prefill_tps_median", "decode_tps_median", "prefill_trials", "decode_trials", "config", "ts", "host")
    return (
        isinstance(obj, dict)
        and all(k in obj for k in required)
        and isinstance(obj.get("prefill_tps_median"), (int, float))
        and isinstance(obj.get("decode_tps_median"), (int, float))
    )


def get_json(path, timeout=30):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.status, json.load(r)


def post_json(path, payload, timeout=600):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data, time.perf_counter() - start


def tokenize(prompt):
    data, _ = post_json("/tokenize", {"model": MODEL, "prompt": prompt}, timeout=60)
    return int(data["count"])


def make_prompt(marker, repeats):
    return f"Benchmark trial marker {marker}. Read the following passage carefully.\n\n" + SNIPPET * repeats


def choose_repeats(target=5200):
    lo, hi = 1, 128
    while tokenize(make_prompt("calibration", hi)) < target:
        hi *= 2
        if hi > 2048:
            raise RuntimeError("failed to bracket target prompt length")
    while lo < hi:
        mid = (lo + hi) // 2
        if tokenize(make_prompt("calibration", mid)) < target:
            lo = mid + 1
        else:
            hi = mid
    candidates = [max(1, lo - 1), lo]
    return min(candidates, key=lambda n: abs(tokenize(make_prompt("calibration", n)) - target))


def completion_nonstream(prompt, max_tokens):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    data, elapsed = post_json("/v1/completions", payload, timeout=600)
    usage = data.get("usage") or {}
    return data, elapsed, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


def completion_stream_decode(prompt, max_tokens=256):
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first_text_at = None
    usage = None
    text_chars = 0
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            datum = line[5:].strip()
            if datum == "[DONE]":
                break
            chunk = json.loads(datum)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                text = choice.get("text") or ""
                if text:
                    if first_text_at is None:
                        first_text_at = time.perf_counter()
                    text_chars += len(text)
    end = time.perf_counter()
    if first_text_at is None:
        raise RuntimeError("decode stream returned no text")
    if not usage:
        raise RuntimeError("decode stream returned no usage; cannot ground token count")
    completion_tokens = int(usage["completion_tokens"])
    prompt_tokens = int(usage["prompt_tokens"])
    ttft = first_text_at - start
    decode_seconds = end - first_text_at
    if completion_tokens <= 0 or decode_seconds <= 0:
        raise RuntimeError(f"invalid decode measurement tokens={completion_tokens} seconds={decode_seconds}")
    # Requested definition: accepted completion tokens divided by elapsed time
    # after first-token arrival. With MTP k2, completion_tokens are accepted tokens.
    tps = completion_tokens / decode_seconds
    return {
        "tps": tps,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "ttft_seconds": ttft,
        "decode_seconds": decode_seconds,
        "total_latency_seconds": end - start,
        "text_chars": text_chars,
    }


def main():
    os.makedirs(ROOT, exist_ok=True)
    if os.path.exists(ROW):
        try:
            existing = json.load(open(ROW))
        except Exception:
            existing = None
        if valid_row(existing):
            log(f"resume-safe exit: complete row already exists at {ROW}")
            print(json.dumps(existing, indent=2, sort_keys=True))
            return

    claim = json.load(open(CLAIM))
    if claim.get("owner") != TASK or claim.get("no_services") is not True:
        raise RuntimeError(f"HOST_CLAIM not owned by {TASK}: {claim}")

    health_status, _ = get_json("/v1/models", timeout=15)
    _, models = get_json("/v1/models", timeout=15)
    model_ids = [x.get("id") for x in models.get("data", [])]
    with urllib.request.urlopen(BASE + "/health", timeout=15) as r:
        health_code = r.status
    if health_code != 200 or MODEL not in model_ids:
        raise RuntimeError(f"serve not healthy: health={health_code} models={model_ids}")
    log(f"serve health=200 model={MODEL} verified")

    repeats = choose_repeats(5200)
    probe_len = tokenize(make_prompt("probe", repeats))
    if not (4000 <= probe_len <= 6000):
        raise RuntimeError(f"calibrated prompt length outside target: {probe_len}")
    log(f"calibrated long prompt repeats={repeats} tokens={probe_len}")

    warm_prompt = make_prompt("warmup-sacrificial", repeats)
    _, warm_elapsed, warm_prompt_tokens, warm_completion_tokens = completion_nonstream(warm_prompt, 1)
    log(f"discarded exactly one JIT warmup: prompt_tokens={warm_prompt_tokens} completion_tokens={warm_completion_tokens} latency={warm_elapsed:.4f}s")

    prefill_details = []
    for i in range(1, 4):
        prompt = make_prompt(f"prefill-{i}-{time.time_ns()}", repeats)
        _, elapsed, prompt_tokens, completion_tokens = completion_nonstream(prompt, 1)
        if not (4000 <= prompt_tokens <= 6000):
            raise RuntimeError(f"prefill trial {i} prompt_tokens={prompt_tokens} outside target")
        tps = prompt_tokens / elapsed
        detail = {
            "trial": i,
            "tps": tps,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_seconds": elapsed,
        }
        prefill_details.append(detail)
        log(f"prefill trial {i}/3: {prompt_tokens} tokens / {elapsed:.4f}s = {tps:.4f} tok/s")

    decode_details = []
    for i in range(1, 4):
        prompt = f"Decode benchmark trial {i}. Continue with a detailed explanation of why careful measurement matters."
        detail = completion_stream_decode(prompt, max_tokens=256)
        detail["trial"] = i
        decode_details.append(detail)
        log(f"decode trial {i}/3 (MTP k2): accepted={detail['completion_tokens']} ttft={detail['ttft_seconds']:.4f}s decode={detail['decode_seconds']:.4f}s tps={detail['tps']:.4f}")

    prefill_trials = [x["tps"] for x in prefill_details]
    decode_trials = [x["tps"] for x in decode_details]
    row = {
        "host": "spark-8",
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(),
        "prefill_tps_median": statistics.median(prefill_trials),
        "decode_tps_median": statistics.median(decode_trials),
        "prefill_trials": prefill_trials,
        "decode_trials": decode_trials,
        "prefill_trial_details": prefill_details,
        "decode_trial_details": decode_details,
        "config": {
            "model": MODEL,
            "model_path": MODEL_ROOT,
            "gpu_offloaded": True,
            "accelerator": "NVIDIA GB10 single-host",
            "single_host": True,
            "kv_cache_dtype": "fp8",
            "block_size": 256,
            "gpu_memory_utilization": 0.78,
            "max_model_len": 8192,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": 4,
            "enforce_eager": True,
            "speculative_method": "deepseek_mtp",
            "mtp_num_speculative_tokens": 2,
            "prefill_method": "usage.prompt_tokens / end-to-end latency; max_tokens=1",
            "decode_method": "stream usage.completion_tokens / time after first accepted text chunk",
            "decode_label": "decode TPS (with MTP k2)",
            "prompt_len_used": [x["prompt_tokens"] for x in prefill_details],
            "decode_len_used": [x["completion_tokens"] for x in decode_details],
            "decode_target_tokens": 256,
            "ignore_eos": True,
            "temperature": 0,
            "warmup_requests_discarded": 1,
            "prompt_source": "repeated public-domain prose from Alice's Adventures in Wonderland",
        },
    }
    tmp = ROW + ".tmp"
    with open(tmp, "w") as f:
        json.dump(row, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, ROW)
    with open(ROWS, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    log(f"wrote {ROW}")
    log(f"appended {ROWS}")
    log(f"RESULT prefill_tps_median={row['prefill_tps_median']:.6f} decode_tps_median={row['decode_tps_median']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        log(f"ERROR {type(exc).__name__}: {exc}")
        raise
