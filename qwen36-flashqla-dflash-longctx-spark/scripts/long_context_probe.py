#!/usr/bin/env python3
import json, os, pathlib, time, traceback, urllib.request
from transformers import AutoTokenizer
models_dir = os.environ.get("MODELS_DIR", "/models")
model_dir = os.environ.get("MODEL_DIR", f"{models_dir}/Qwen3.6-27B-NVFP4")
model = os.environ.get("MODEL", "qwen36-27b")
host = os.environ.get("HOST", "127.0.0.1")
port = os.environ.get("PORT", "8000")
target = int(os.environ.get("TARGET_PROMPT_TOKENS", "258048"))
out = pathlib.Path(os.environ.get("OUTDIR", "/repro/results/longctx-live"))
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
base = " Sherlock Holmes examined the evidence carefully, numbered each observation, and wrote a concise note for Watson.\n"
base_ids = tok.encode(base, add_special_tokens=False)
ids = (base_ids * ((target // len(base_ids)) + 1))[:target]
prompt = tok.decode(ids, skip_special_tokens=True)
actual = len(tok.encode(prompt, add_special_tokens=False))
(out / "request-meta.json").write_text(json.dumps({"target_prompt_tokens": target, "actual_prompt_tokens_local": actual, "chars": len(prompt), "max_tokens": 1, "endpoint": "/v1/completions"}, indent=2))
print(json.dumps({"event": "sending", "actual_prompt_tokens_local": actual, "chars": len(prompt), "time": time.time()}), flush=True)
req = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0, "stream": False}
t0 = time.time()
try:
    r = urllib.request.Request(f"http://{host}:{port}/v1/completions", data=json.dumps(req).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=int(os.environ.get("LONGCTX_TIMEOUT", "1800"))) as resp:
        body = resp.read().decode(); status = resp.status
except Exception as e:
    (out / "error.txt").write_text(repr(e) + "\n" + traceback.format_exc())
    raise
t1 = time.time()
(out / "response.json").write_text(body)
obj = json.loads(body)
summary = {"status": status, "elapsed_sec": t1 - t0, "prompt_tokens_api": obj.get("usage", {}).get("prompt_tokens"), "completion_tokens_api": obj.get("usage", {}).get("completion_tokens"), "total_tokens_api": obj.get("usage", {}).get("total_tokens"), "local_prompt_tokens": actual, "throughput_prompt_tokens_per_sec_wall": actual / (t1 - t0), "finish_reason": obj.get("choices", [{}])[0].get("finish_reason")}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
