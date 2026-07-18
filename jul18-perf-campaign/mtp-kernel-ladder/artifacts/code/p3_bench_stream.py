#!/usr/bin/env python3
"""<source-task> P3 minimal streaming bench row (P1/P2 style, stdlib only).

One streaming chat completion against the local combo server; measures
decode tok/s after first token; captures /metrics spec-decode counter
deltas so acceptance-per-draft is computed from THIS row's counters.
"""
import argparse, json, re, sys, time, urllib.request

BASE = "http://127.0.0.1:8001"

def get_metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        txt = r.read().decode()
    keep = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        if re.search(r"spec_decode|draft", line, re.I):
            m = re.match(r"^(\S+?)(\{[^}]*\})?\s+([0-9eE+.\-naif]+)$", line)
            if m:
                try:
                    keep[m.group(1) + (m.group(2) or "")] = float(m.group(3))
                except ValueError:
                    pass
    return keep

def run_row(model, max_tokens, prompt, out_path, label):
    before = get_metrics()
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        BASE + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    t_first = None
    t_last = None
    n_chunks = 0
    usage = None
    text_parts = []
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            ch = obj.get("choices") or []
            if ch:
                delta = ch[0].get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning") or delta.get("reasoning_content") or "")
                now = time.monotonic()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
                if piece:
                    text_parts.append(piece)
    after = get_metrics()
    completion_tokens = (usage or {}).get("completion_tokens")
    decode_s = (t_last - t_first) if (t_first and t_last) else None
    tps = None
    if completion_tokens and decode_s and decode_s > 0:
        tps = (completion_tokens - 1) / decode_s
    delta = {}
    for k, v in after.items():
        b = before.get(k, 0.0)
        d = v - b
        if d != 0:
            delta[k] = {"before": b, "after": v, "delta": d}
    # acceptance per draft from this row's counters (exact vllm counter names,
    # excluding per_pos family to avoid double counting)
    def find_exact(counter_name):
        tot = 0.0
        hit = False
        for k, v in delta.items():
            base = k.split("{")[0]
            if base == counter_name:
                tot += v["delta"]
                hit = True
        return tot if hit else None
    accepted = find_exact("vllm:spec_decode_num_accepted_tokens_total")
    drafts = find_exact("vllm:spec_decode_num_drafts_total")
    draft_tokens = find_exact("vllm:spec_decode_num_draft_tokens_total")
    accepted_per_draft = None
    if accepted is not None and drafts:
        accepted_per_draft = accepted / drafts
    cycle_ms = None
    if tps and accepted_per_draft is not None:
        cycle_ms = 1000.0 * (1.0 + accepted_per_draft) / tps
    row = {
        "format": "p3-row-v1",
        "label": label,
        "request": body,
        "wall_total_s": time.monotonic() - t0,
        "ttft_s": (t_first - t0) if t_first else None,
        "decode_s_after_first": decode_s,
        "completion_tokens": completion_tokens,
        "n_stream_chunks": n_chunks,
        "decode_tok_s_after_first": tps,
        "usage": usage,
        "output_head": "".join(text_parts)[:200],
        "output_nonempty": bool(text_parts),
        "spec_metrics_delta": delta,
        "row_accepted_per_draft": accepted_per_draft,
        "row_accepted_total": accepted,
        "row_num_drafts": drafts,
        "row_draft_tokens": draft_tokens,
        "row_cycle_ms": cycle_ms,
        "unix": time.time(),
    }
    with open(out_path, "w") as f:
        json.dump(row, f, indent=1, sort_keys=True)
    print(json.dumps({"label": label, "tps": tps, "tokens": completion_tokens,
                      "accepted_per_draft": accepted_per_draft, "cycle_ms": cycle_ms,
                      "nonempty": bool(text_parts)}))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash-iq3-combo-v4-step32")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="row")
    ap.add_argument("--prompt", default="Write an endless comma-separated sequence of positive integers starting at 1. Output only the sequence and continue until the response token limit.")
    a = ap.parse_args()
    run_row(a.model, a.max_tokens, a.prompt, a.out, a.label)
