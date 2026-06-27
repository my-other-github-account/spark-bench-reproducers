#!/usr/bin/env python3
# GLM-5.2-NVFP4 TP=4 QUALITY/DEGRADATION battery (NOT a speed test).
# Part A: discriminating reasoning probes at shallow context (logit-integrity check).
# Part B: needle-in-haystack retrieval at depth (deep-prefill attention-rot check).
# Greedy decode (temp 0). Reports PASS/FAIL per probe against an expected substring.
import json, time, http.client, re
HOST, PORT, MODEL = "127.0.0.1", 8000, "glm52"
RATIO = 3.6  # tokens per filler-word, measured on this build

def complete(prompt, max_tokens, stop=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "stream": False}
    if stop: body["stop"] = stop
    c = http.client.HTTPConnection(HOST, PORT, timeout=900)
    t0 = time.time()
    c.request("POST", "/v1/completions", json.dumps(body),
              {"Content-Type": "application/json"})
    r = c.getresponse(); raw = r.read(); st = r.status; c.close()
    dt = time.time() - t0
    if st != 200:
        return {"status": st, "err": raw[:200].decode("utf-8", "ignore"), "dt": round(dt, 1)}
    o = json.loads(raw)
    txt = o["choices"][0]["text"]
    u = o.get("usage", {})
    return {"status": 200, "text": txt, "prompt_tokens": u.get("prompt_tokens"),
            "completion_tokens": u.get("completion_tokens"), "dt": round(dt, 1)}

def filler(nwords, salt):
    # neutral junk tokens, deterministic, no semantic content that could leak the needle
    return " ".join(f"{salt}{i}" for i in range(nwords))

# ---------- PART A: reasoning / logit-integrity probes (shallow context) ----------
PROBES = [
    ("arith_mult",  "Q: What is 17 times 23? Answer with only the number.\nA:", ["391"]),
    ("arith_word",  "Q: A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost in cents? Answer with only the number of cents.\nA:", ["5"]),
    ("syllogism",   "Q: All roses are flowers. Some flowers fade quickly. Does it logically follow that some roses fade quickly? Answer yes or no.\nA:", ["no"]),
    ("trick_false", "Q: Is it true that 2 + 2 = 5? Answer yes or no.\nA:", ["no"]),
    ("capital",     "Q: What is the capital of France? One word.\nA:", ["paris"]),
    ("compare_num", "Q: Which is larger, 9.9 or 9.11? Answer with just the number.\nA:", ["9.9"]),
]

# ---------- PART B: needle-in-haystack at depth ----------
NEEDLE_CODE = "MAGENTA-7431"
NEEDLE = f"IMPORTANT: The secret passcode for this session is {NEEDLE_CODE}. Remember it."
QUESTION = "\n\nBased only on the text above, what is the secret passcode for this session? Answer with only the passcode and nothing else.\n"
# (target_real_tokens, needle_position_fraction)
DEPTHS = [
    (200,   0.5, "ctrl"),     # control: tiny context
    (5000,  0.5, "5k_mid"),
    (15000, 0.5, "15k_mid"),
    (30000, 0.1, "30k_early"),
    (30000, 0.5, "30k_mid"),
    (30000, 0.9, "30k_late"),
    (50000, 0.5, "50k_mid"),
]

def build_haystack(real_tokens, frac, salt):
    nwords = max(2, int(real_tokens / RATIO))
    before = int(nwords * frac)
    after = nwords - before
    parts = [filler(before, salt), NEEDLE, filler(after, salt + "z")]
    return " ".join(parts)

def passed(text, expected):
    t = text.lower()
    return any(e.lower() in t for e in expected)

out = {"reasoning": [], "needle": []}
print("=== PART A: reasoning / logit-integrity (shallow) ===", flush=True)
a_pass = 0
for name, prompt, exp in PROBES:
    r = complete(prompt, 24, stop=["\n"])
    ok = r.get("status") == 200 and passed(r.get("text", ""), exp)
    a_pass += ok
    rec = {"probe": name, "expected": exp, "got": (r.get("text", "") or "").strip()[:80],
           "pass": ok, "status": r.get("status"), "dt": r.get("dt")}
    out["reasoning"].append(rec)
    print(json.dumps(rec), flush=True)

print(f"\n=== PART B: needle-in-haystack at depth (needle={NEEDLE_CODE}) ===", flush=True)
b_pass = 0
for real, frac, label in DEPTHS:
    prompt = build_haystack(real, frac, "w") + QUESTION
    r = complete(prompt, 24, stop=["\n"])
    found = r.get("status") == 200 and (NEEDLE_CODE.lower() in (r.get("text", "") or "").lower())
    b_pass += found
    rec = {"depth": label, "target_tokens": real, "needle_frac": frac,
           "prompt_tokens": r.get("prompt_tokens"), "found": found,
           "got": (r.get("text", "") or "").strip()[:80], "status": r.get("status"),
           "dt": r.get("dt"), "err": r.get("err")}
    out["needle"].append(rec)
    print(json.dumps(rec), flush=True)

out["summary"] = {"reasoning_pass": f"{a_pass}/{len(PROBES)}",
                  "needle_pass": f"{b_pass}/{len(DEPTHS)}"}
print("\n=== SUMMARY ===", flush=True)
print(json.dumps(out["summary"]), flush=True)
with open("/tmp/glm52_degradation_result.json", "w") as f:
    json.dump(out, f, indent=2)
print("written /tmp/glm52_degradation_result.json", flush=True)
