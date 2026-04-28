#!/usr/bin/env python3
"""summarize_results.py — turn the 8 result JSONs into RESULTS.md.

Layout matches the 27B Qwen3.6 reproducer's RESULTS table:
  Server | Corpus | Think | tg/s median (warm) | tg/s mean | tg/s std | ttfr ms (median) | pp tok/s | n
"""
import json
import os
import statistics as st
import sys

CELLS = [
    ("spec", "sherlock", "ON"),
    ("spec", "sherlock", "OFF"),
    ("spec", "codegen",  "ON"),
    ("spec", "codegen",  "OFF"),
    ("ar",   "sherlock", "ON"),
    ("ar",   "sherlock", "OFF"),
    ("ar",   "codegen",  "ON"),
    ("ar",   "codegen",  "OFF"),
]

def load(rd, server, corpus, think):
    p = os.path.join(rd, f"result-{server}-{corpus}-think{think}.json")
    if not os.path.exists(p):
        return None, p
    d = json.load(open(p))
    b = d["benchmarks"][0]
    tg = b["tg_throughput"]["values"]
    ttfr_key = "e2e_ttft" if "e2e_ttft" in b else "ttfr"
    ttfr = b[ttfr_key]["values"]
    tg_warm = tg[1:] if len(tg) > 1 else tg
    ttfr_warm = ttfr[1:] if len(ttfr) > 1 else ttfr
    return {
        "tg_mean":   st.mean(tg_warm),
        "tg_median": st.median(tg_warm),
        "tg_std":    st.pstdev(tg_warm),
        "tg_n":      len(tg_warm),
        "ttfr_med":  st.median(ttfr_warm),
        "pp_mean":   b["pp_throughput"]["mean"],
        "prompt":    b.get("prompt_size"),
        "response":  b.get("response_size"),
    }, p

def main():
    if len(sys.argv) < 2:
        print("usage: summarize_results.py RESULTS_DIR", file=sys.stderr)
        sys.exit(2)
    rd = sys.argv[1]

    print("# MiniMax-M2.7-UD-IQ4_XS + n-gram spec decode — measured cells\n")
    print("All cells: llama.cpp commit `45cac7ca7`, ctx=32768, q8_0 KV, c=1, depth=0,")
    print("pp=128, tg=128, n=30 trials, no warmup, sherlock prose corpus by default,")
    print("codegen corpus = vllm/v1/worker/gpu_model_runner.py (~7065 lines).")
    print("Warm-pass values (cold-start sample dropped). DGX Spark GB10, 128 GiB UMA.\n")
    print("**Headline (localmaxxing submission)**: spec / sherlock / thinkON / median tg/s.\n")
    print("| Server | Corpus | Think | tg/s median (warm) | tg/s mean | tg/s std | ttfr ms (median) | pp tok/s | n |")
    print("|---|---|---|---|---|---|---|---|---|")
    missing = []
    for srv, c, t in CELLS:
        cell, path = load(rd, srv, c, t)
        if cell is None:
            missing.append(path)
            print(f"| {srv} | {c} | {t} | — | — | — | — | — | — |")
            continue
        print(f"| {srv} | {c} | {t} | **{cell['tg_median']:.2f}** | {cell['tg_mean']:.2f} | {cell['tg_std']:.2f} | {cell['ttfr_med']:.0f} | {cell['pp_mean']:.1f} | {cell['tg_n']} |")
    if missing:
        print()
        print("> Missing result files (fill in by running the corresponding bench-all.sh cell):")
        for m in missing:
            print(f"> - `{m}`")

    # Speedup summary if we have both AR and spec sherlock-ON
    spec_cell, _ = load(rd, "spec", "sherlock", "ON")
    ar_cell, _   = load(rd, "ar",   "sherlock", "ON")
    if spec_cell and ar_cell:
        speedup = spec_cell["tg_median"] / ar_cell["tg_median"]
        print()
        print("## Speedup (sherlock thinkON, median over warm n)\n")
        print(f"- spec (ngram-simple): **{spec_cell['tg_median']:.2f}** tok/s")
        print(f"- AR baseline:         {ar_cell['tg_median']:.2f} tok/s")
        print(f"- speedup:             **{speedup:.2f}×**")

if __name__ == "__main__":
    main()
