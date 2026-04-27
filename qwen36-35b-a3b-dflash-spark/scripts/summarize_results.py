#!/usr/bin/env python3
# summarize_results.py — read all 4 result JSONs and emit a markdown table
# matching the format used in the 27B README.
#
# Usage:
#   python3 scripts/summarize_results.py /path/to/results-dir > RESULTS.md
import json
import statistics as st
import sys
import os

CELLS = [
    ("sherlock", "thinkON",  "Sherlock prose"),
    ("sherlock", "thinkOFF", "Sherlock prose"),
    ("codegen",  "thinkON",  "vLLM gpu_model_runner.py"),
    ("codegen",  "thinkOFF", "vLLM gpu_model_runner.py"),
]

def load_cell(results_dir, corpus, think_label):
    path = os.path.join(results_dir, f"result-{corpus}-{think_label}.json")
    if not os.path.exists(path):
        return None, path
    with open(path) as f:
        d = json.load(f)
    b = d["benchmarks"][0]
    tg_v = b["tg_throughput"]["values"]
    ttfr_v = b["ttfr"]["values"]
    tg_warm   = tg_v[1:] if len(tg_v) > 1 else tg_v
    ttfr_warm = ttfr_v[1:] if len(ttfr_v) > 1 else ttfr_v
    return {
        "tg_mean":     st.mean(tg_warm),
        "tg_median":   st.median(tg_warm),
        "tg_std":      st.pstdev(tg_warm),
        "tg_n":        len(tg_warm),
        "ttfr_median": st.median(ttfr_warm),
        "ttfr_mean":   st.mean(ttfr_warm),
        "pp":          b["pp_throughput"]["mean"],
        "prompt_size": b.get("prompt_size"),
        "response_size": b.get("response_size"),
    }, path

def main():
    if len(sys.argv) < 2:
        print("usage: summarize_results.py RESULTS_DIR", file=sys.stderr); sys.exit(2)
    rd = sys.argv[1]

    print("# Qwen3.6-35B-A3B-NVFP4 + DFlash on DGX Spark — measured cells")
    print()
    print("All cells: llama-benchy 0.3.7+, c=1, depth=0, pp=128, tg=128, n=30,")
    print("warm-pass values (cold-start sample dropped). Same hardware (DGX Spark GB10),")
    print("same engine, same drafter (z-lab/Qwen3.6-35B-A3B-DFlash, num_speculative_tokens=15).")
    print()
    print("| Corpus | Think | tg/s median (warm) | tg/s mean | tg/s std | ttfr ms (median) | pp tok/s | n |")
    print("|---|---|---|---|---|---|---|---|")
    missing = []
    for corpus, think, _label in CELLS:
        c, path = load_cell(rd, corpus, think)
        if c is None:
            missing.append(path)
            print(f"| {corpus} | {think} | _missing_ | — | — | — | — | — |")
            continue
        print(f"| {corpus} | {think} | **{c['tg_median']:.2f}** | {c['tg_mean']:.2f} | {c['tg_std']:.2f} | {c['ttfr_median']:.0f} | {c['pp']:.1f} | {c['tg_n']} |")
    if missing:
        print()
        print("> Missing result files (will appear after the corresponding bench cell completes):")
        for m in missing:
            print(f"> - `{m}`")
    print()
    print("## Per-cell prompt sizing")
    print()
    print("| Corpus | Think | prompt_size (tokens) | response_size (tokens) |")
    print("|---|---|---|---|")
    for corpus, think, _label in CELLS:
        c, _ = load_cell(rd, corpus, think)
        if c is None:
            continue
        print(f"| {corpus} | {think} | {c['prompt_size']} | {c['response_size']} |")

if __name__ == "__main__":
    main()
