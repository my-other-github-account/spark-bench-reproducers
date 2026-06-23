#!/usr/bin/env python3
# ablate_bench.py — poll a head node, classify the outcome, and (if coherent) bench.
# Never throws; always writes a JSON verdict. Statuses:
#   NO_BIND          : /health never reached 200 within timeout (load/bind/finalize wall)
#   DECODE_ERROR     : health 200 but a generation call errored / 5xx (e.g. Triton SMEM)
#   SERVED_INCOHERENT: decodes return text but it's salad (dense-MLA disabled, etc.)
#   SERVED_COHERENT  : 2+2 contains '4' AND France contains 'Paris' -> real serve; benched
import json, os, sys, time, urllib.request, urllib.error

SERVER = os.environ.get("SERVER", "127.0.0.1")
PORT = os.environ.get("PORT", "8000")
OUT = os.environ["OUT"]
LABEL = os.environ.get("LABEL", "?")
HEALTH_TIMEOUT = int(os.environ.get("HEALTH_TIMEOUT", "780"))  # 13 min default
RUNS = int(os.environ.get("RUNS", "5"))
MAXTOK = int(os.environ.get("MAXTOK", "256"))
BASE = f"http://{SERVER}:{PORT}"

def health():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=5) as r:
            return r.status
    except Exception:
        return None

def gen(prompt, mt):
    body = json.dumps({"model": "glm5", "prompt": prompt, "max_tokens": mt, "temperature": 0}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.load(r)
    dt = time.time() - t0
    return d["usage"]["completion_tokens"], dt, d["choices"][0]["text"]

def write(obj):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(obj, open(OUT, "w"), indent=2)
    print(f"[{LABEL}] status={obj['status']} -> {OUT}")
    if obj.get("coherence"):
        print(f"[{LABEL}] coherence 2+2={obj['coherence'].get('2+2')!r} France={obj['coherence'].get('France')!r}")
    if obj.get("median_tok_per_s") is not None:
        print(f"[{LABEL}] median {obj['median_tok_per_s']} tok/s | mean {obj.get('mean_tok_per_s')}")

res = {"label": LABEL, "server": SERVER, "ts": int(time.time()), "status": None,
       "coherence": None, "runs": None, "median_tok_per_s": None, "mean_tok_per_s": None, "error": None}

# 1) wait for bind
t_start = time.time()
bound = False
while time.time() - t_start < HEALTH_TIMEOUT:
    if health() == 200:
        bound = True
        res["health_after_s"] = round(time.time() - t_start, 1)
        break
    time.sleep(10)
if not bound:
    res["status"] = "NO_BIND"
    res["error"] = f"/health != 200 within {HEALTH_TIMEOUT}s"
    write(res); sys.exit(0)

# 2) coherence probe (use a prompt that reliably emits the digit when coherent)
try:
    _, _, c22 = gen("What is 2+2? Answer with just the number and nothing else.", 4)
    _, _, cfr = gen("The capital of France is", 6)
    _, _, cpr = gen("List the first five prime numbers:", 16)
    res["coherence"] = {"2+2": c22, "France": cfr, "primes": cpr}
except urllib.error.HTTPError as e:
    res["status"] = "DECODE_ERROR"; res["error"] = f"HTTP {e.code}: {e.read()[:300].decode('utf-8','replace')}"
    write(res); sys.exit(0)
except Exception as e:
    res["status"] = "DECODE_ERROR"; res["error"] = f"{type(e).__name__}: {e}"
    write(res); sys.exit(0)

# coherent if 2/3 anchors pass: 2+2 has '4', France has 'Paris', primes has '2, 3, 5, 7'
checks = [("4" in c22), ("Paris" in cfr), ("2, 3, 5, 7" in cpr or "2,3,5,7" in cpr.replace(" ", ""))]
coherent = sum(checks) >= 2
if not coherent:
    res["status"] = "SERVED_INCOHERENT"
    write(res); sys.exit(0)

# 3) bench
try:
    gen("Hello", 8); gen("Hello", 8)  # warmups
    runs = []
    for i in range(RUNS):
        tok, dt, _ = gen("Write a detailed paragraph about the ocean and its many creatures.", MAXTOK)
        runs.append({"run": i+1, "completion_tokens": tok, "wall_s": round(dt, 3), "tok_per_s": round(tok/dt, 2)})
        print(f"[{LABEL}]   run{i+1}: {tok} tok / {dt:.3f}s -> {round(tok/dt,2)} tok/s")
    tps = sorted(r["tok_per_s"] for r in runs)
    res["runs"] = runs
    res["median_tok_per_s"] = tps[len(tps)//2]
    res["mean_tok_per_s"] = round(sum(r["tok_per_s"] for r in runs)/len(runs), 2)
    res["status"] = "SERVED_COHERENT"
except Exception as e:
    res["status"] = "SERVED_COHERENT_BENCHFAIL"; res["error"] = f"{type(e).__name__}: {e}"
write(res)
