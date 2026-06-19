#!/usr/bin/env bash
# bench.sh — single-stream tg256 c=1 decode benchmark + coherence check.
# Usage: SERVER=<head_ip> bash scripts/bench.sh   (writes results/decode-tg256-n5.json by default)
# Pure stdlib; no llama-benchy needed for this headline number.
set -u
SERVER="${SERVER:-127.0.0.1}"
PORT="${PORT:-8000}"
OUT="${OUT:-$(cd "$(dirname "$0")/.." && pwd)/results/decode-tg256-n5.json}"
MAXTOK="${MAXTOK:-256}"
RUNS="${RUNS:-5}"

BASE="http://$SERVER:$PORT" OUT="$OUT" MAXTOK="$MAXTOK" RUNS="$RUNS" python3 - <<'PY'
import json, os, time, urllib.request
BASE=os.environ["BASE"]; OUT=os.environ["OUT"]
MAXTOK=int(os.environ["MAXTOK"]); RUNS=int(os.environ["RUNS"])

def gen(prompt, mt):
    body=json.dumps({"model":"glm5","prompt":prompt,"max_tokens":mt,"temperature":0}).encode()
    req=urllib.request.Request(BASE+"/v1/completions", body, {"Content-Type":"application/json"})
    t0=time.time(); d=json.load(urllib.request.urlopen(req, timeout=180)); dt=time.time()-t0
    return d["usage"]["completion_tokens"], dt, d["choices"][0]["text"]

# real decode check first (health=200 is NOT enough)
c22=gen("What is 2+2? Answer in one word.",6)[2]
cfr=gen("The capital of France is",6)[2]
print("coherence  2+2 ->", repr(c22), "| France ->", repr(cfr))

gen("Hello",8); gen("Hello",8)  # warmups
runs=[]
for i in range(RUNS):
    tok,dt,_=gen("Write a detailed paragraph about the ocean and its many creatures.",MAXTOK)
    tps=round(tok/dt,2); runs.append({"run":i+1,"completion_tokens":tok,"wall_s":round(dt,3),"tok_per_s":tps})
    print(f"  run{i+1}: {tok} tok / {dt:.3f}s -> {tps} tok/s")
t8,d8,_=gen("Write about the ocean.",8)
tps=sorted(r["tok_per_s"] for r in runs); med=tps[len(tps)//2]
out={"model":"nvidia/GLM-5-NVFP4","served_name":"glm5",
     "config":"TP=4, 4x DGX Spark GB10, enforce-eager, max-model-len 2048, kv fp8_e4m3, moe cutlass",
     "metric":f"single-stream tg{MAXTOK} c=1 end-to-end (HTTP+prefill+decode)","runs":runs,
     "median_tok_per_s":med,"mean_tok_per_s":round(sum(r['tok_per_s'] for r in runs)/len(runs),2),
     "short_gen_8tok_wall_s":round(d8,3),
     "coherence":{"2+2":c22,"France":cfr}}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT,"w"), indent=2)
print("median", med, "tok/s | mean", out["mean_tok_per_s"], "tok/s ->", OUT)
PY
