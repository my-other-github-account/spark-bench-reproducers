#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    ROOT / "results/native_atlas_dflash_gate_20260522T1134Z_decodefixed_fullgate_r3_benchy3",
    ROOT / "results/native_atlas_dflash_gate_20260522T1157Z_decodefixed_benchy_codegen_cell_r3",
]

def load(path):
    with open(path) as f:
        return json.load(f)

def show_gate(label, path):
    if not path.exists():
        return
    data = load(path)
    print(f"\n{label}: {path.relative_to(ROOT)}")
    for key in [
        "gate_pass",
        "aggregate_ar_tps",
        "aggregate_dflash_tps",
        "aggregate_ratio",
        "all_exact_token_accounting_ok",
        "all_rows_ratio_ok",
        "all_usage_completion_tokens_ok",
        "server_markers_ok",
        "target_generation_tokens",
    ]:
        if key in data:
            print(f"  {key}: {data[key]}")
    rows = data.get("rows") or {}
    for name, row in rows.items():
        ar = row.get("ar_tps")
        df = row.get("dflash_tps")
        ratio = row.get("ratio")
        usage = row.get("dflash_usage_completion_tokens") or row.get("dflash_completion_tokens_total")
        print(f"  row {name}: ar={ar} dflash={df} ratio={ratio} dflash_usage={usage}")

for run in RUNS:
    print(f"# {run.name}")
    show_gate("diverse", run / "diverse_gate_summary.json")
    show_gate("llama_benchy", run / "llama_benchy_gate_summary.json")
