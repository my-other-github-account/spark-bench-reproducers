#!/usr/bin/env bash
# bench.sh — measure decode tok/s against a bound GLM-5.2 serve, and stamp the served checkpoint.
# Usage: HEAD=http://<rank0-ip>:8000 CKPT_MD5=<index.json md5> bash bench.sh
# Run AFTER :8000 binds. The first call pays the one-time first-forward Triton JIT (can take minutes);
# this script probes until a 1-token gen returns 200, then runs the real timed 160-token benchmark.
set -u
HEAD="${HEAD:-http://127.0.0.1:8000}"
URL="$HEAD/v1/completions"
CKPT_MD5="${CKPT_MD5:-unknown}"
PROMPT='Write a short paragraph explaining what a transformer neural network is.'
OUT="${OUT:-glm52_decode_result.json}"

echo "phase 1: probing until JIT done (1-token gen returns 200)..."
while true; do
  code=$(curl -s -m 600 -o /tmp/probe.json -w '%{http_code}' "$URL" \
    -H 'Content-Type: application/json' \
    -d '{"model":"glm52","prompt":"2+2=","max_tokens":1,"temperature":0}' 2>/dev/null)
  [ "$code" = "200" ] && break
  echo "  not ready (http $code), retrying in 20s..."; sleep 20
done
echo "phase 2: warm — running timed 160-token benchmark..."
START=$(date +%s.%N)
curl -s -m 600 -o /tmp/bench_resp.json "$URL" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"glm52\",\"prompt\":\"${PROMPT}\",\"max_tokens\":160,\"temperature\":0}" 2>/dev/null
END=$(date +%s.%N)

python3 - "$START" "$END" "$CKPT_MD5" "$OUT" <<'PY'
import json, sys, datetime
start, end, md5, out = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
wall = end - start
try:
    d = json.load(open("/tmp/bench_resp.json"))
    ct = d["usage"]["completion_tokens"]; txt = d["choices"][0]["text"]
except Exception as e:
    ct, txt = 0, "PARSE_FAIL: " + repr(e)
tps = round(ct / wall, 3) if wall > 0 and ct else 0.0
res = {
  "decode_tok_s": tps, "completion_tokens": ct, "wall_s": round(wall, 3), "text": txt,
  "checkpoint_index_md5": md5, "model": "nvidia/GLM-5.2-NVFP4",
  "quant": "modelopt_fp4 cutlass NOREPACK TP=4 dense index_topk=0",
  "captured_utc": datetime.datetime.utcnow().isoformat() + "Z",
}
json.dump(res, open(out, "w"), indent=2)
print(json.dumps(res, indent=2))
print(f"\nWROTE {out}: decode_tok_s={tps} completion_tokens={ct} wall_s={wall:.1f}")
PY
