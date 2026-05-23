#!/usr/bin/env bash
set -euo pipefail

# Skeleton only: fill ATLAS_BIN/MODEL/MODEL_NAME before use.
ATLAS_BIN=${ATLAS_BIN:-atlas}
MODEL=${MODEL:-/path/to/model}
MODEL_NAME=${MODEL_NAME:-test-model}
PORT=${PORT:-18080}
OUT=${OUT:-/tmp/atlas_fixed_length_repro}
mkdir -p "$OUT"

"$ATLAS_BIN" serve \
  --model-from-path "$MODEL" \
  --model-name "$MODEL_NAME" \
  --port "$PORT" \
  --max-num-seqs 1 \
  --max-batch-size 1 \
  >"$OUT/server.log" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null || true' EXIT

for i in $(seq 1 300); do
  curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  sleep 1
done

python3 - <<'PY' "$PORT" "$MODEL_NAME" "$OUT/result.json"
import json, sys, urllib.request
port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
body = {
    "model": model,
    "messages": [{"role": "user", "content": "Answer briefly: hello"}],
    "temperature": 0,
    "max_tokens": 128,
    "min_tokens": 128,
    "stream": True,
    "stream_options": {"include_usage": True},
}
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type":"application/json"},
)
usage = None
chunks = []
with urllib.request.urlopen(req, timeout=300) as r:
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            continue
        obj = json.loads(data)
        chunks.append(obj)
        if obj.get("usage"):
            usage = obj["usage"]
result = {"usage": usage, "num_chunks": len(chunks)}
open(out, "w").write(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
assert usage and usage.get("completion_tokens") == 128, usage
PY
