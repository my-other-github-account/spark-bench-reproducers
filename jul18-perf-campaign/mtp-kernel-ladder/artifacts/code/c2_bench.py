#!/usr/bin/env python3
"""c2 concurrency bench: N simultaneous streaming requests, aggregate wall tok/s.

Usage: c2_bench.py --n 2 --max-tokens 256 --out /path/out.json --label c2_round1
"""
import argparse, json, time, threading, urllib.request

URL = "http://127.0.0.1:8001/v1/chat/completions"
PROMPT = "Write a Python module that implements an LRU cache with TTL support, unit tests included."

def run_stream(idx, max_tokens, results):
    body = {
        "model": "deepseek-v4-flash-iq3-combo-v4-step32",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    n_tok = 0
    chunks = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            ch = d.get("choices", [])
            if not ch:
                continue
            delta = ch[0].get("delta", {})
            txt = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if txt:
                chunks += 1
                if first is None:
                    first = time.monotonic()
            u = d.get("usage")
            if u and u.get("completion_tokens"):
                n_tok = u["completion_tokens"]
    t1 = time.monotonic()
    results[idx] = {"start": t0, "first": first, "end": t1,
                    "completion_tokens": n_tok or max_tokens, "chunks": chunks}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="c2")
    a = ap.parse_args()
    results = {}
    threads = [threading.Thread(target=run_stream, args=(i, a.max_tokens, results))
               for i in range(a.n)]
    wall0 = time.monotonic()
    for t in threads: t.start()
    for t in threads: t.join()
    wall1 = time.monotonic()
    total_tok = sum(r["completion_tokens"] for r in results.values())
    firsts = [r["first"] for r in results.values() if r["first"]]
    # aggregate decode rate after last first-token (both streams decoding)
    t_start = max(firsts) if firsts else wall0
    agg_wall = wall1 - wall0
    agg_tok_s_wall = total_tok / agg_wall
    # after-first aggregate (excludes prefill of the later request)
    decode_span = wall1 - t_start
    agg_tok_s_after_first = (total_tok - len(firsts)) / decode_span if decode_span > 0 else 0
    per_req = []
    for i, r in sorted(results.items()):
        span = r["end"] - (r["first"] or r["start"])
        per_req.append({"idx": i, "completion_tokens": r["completion_tokens"],
                        "decode_tok_s_after_first": (r["completion_tokens"] - 1) / span if span > 0 else 0,
                        "chunks": r["chunks"]})
    out = {"format": "c2-bench-v1", "label": a.label, "n": a.n,
           "max_tokens": a.max_tokens, "prompt": PROMPT,
           "agg_wall_s": agg_wall, "total_completion_tokens": total_tok,
           "agg_tok_s_wall": agg_tok_s_wall,
           "agg_tok_s_after_first": agg_tok_s_after_first,
           "per_request": per_req}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("label", "agg_tok_s_wall", "agg_tok_s_after_first")}))
    for p in per_req:
        print(f'  req{p["idx"]}: {p["decode_tok_s_after_first"]:.3f} tok/s')

if __name__ == "__main__":
    main()
