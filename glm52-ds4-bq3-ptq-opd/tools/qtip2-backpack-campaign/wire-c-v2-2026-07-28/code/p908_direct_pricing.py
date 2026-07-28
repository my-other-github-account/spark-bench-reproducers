#!/usr/bin/env python3
"""Inspect the sealed P908 direct-pricing receipt without fitting a transfer model."""
import argparse, json
from pathlib import Path

CLASSES = ("agentic", "chat", "code", "multilingual", "prose", "reasoning")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts/DIRECT_SELECTED_CODE_PRICES.public.json")
    args = parser.parse_args()
    doc = json.loads(args.receipt.read_text())
    surfaces = doc["surfaces"]
    out = {
        "schema": "p908-public-direct-pricing-view-v1",
        "control_mean_kld_vector": {c: doc["canonical_sanity"]["control_mean_kld_vector"][c] for c in CLASSES},
        "qtip2_direct_price_kld_per_logical_gib": {c: surfaces["q2"]["direct_price_kld_per_logical_gib_vector"][c] for c in CLASSES},
        "qtip3_direct_price_kld_per_logical_gib": {c: surfaces["q3"]["direct_price_kld_per_logical_gib_vector"][c] for c in CLASSES},
        "policy": "direct measured anchor prices only; no contaminated transfer fit",
    }
    if set(out["control_mean_kld_vector"]) != set(CLASSES):
        raise SystemExit("P908 receipt class coverage failed")
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
