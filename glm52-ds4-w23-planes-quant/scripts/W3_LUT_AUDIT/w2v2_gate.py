#!/usr/bin/env python3
"""Audit emitted W2v2 bytes against source and shipped W2 for t_bd7728ee."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.home() / "missions/W3_LUT_AUDIT"
sys.path.insert(0, str(ROOT))
import w3v2_gate as base

V2 = ROOT / "moe_w2_planes_v2"
CKPT = Path.home() / "models/hf/DeepSeek-V4-Flash"
OUT = ROOT / "GATE_W2V2.json"
LAYERS = [1, 9, 15, 21, 27, 33, 39, 41]
EXPERTS = [3, 77, 201]
W2LV = [-4.0, -1.0, 1.0, 4.0]


def main():
    rows = []
    for L in LAYERS:
        lut = json.loads((V2 / f"layer_{L:03d}.meta.json").read_text())["lut"]
        for e in EXPERTS:
            for tier, names, N, K in (("13", ("w1", "w3"), 4096, 4096), ("2", ("w2",), 4096, 2048)):
                w = base.src(L, e, names)
                v2 = base.deq(str(V2), L, e, tier, N, K, lut)
                cur = base.deq(str(CKPT / "moe_w2_planes"), L, e, tier, N, K, W2LV)
                rows.append({"L": L, "e": e, "tier": tier,
                             "rel_v2": base.relrms(v2, w), "rel_current": base.relrms(cur, w),
                             "rms_ratio_v2": (v2.pow(2).mean().sqrt()/w.pow(2).mean().sqrt()).item()})
        print(f"L{L:03d} audited", flush=True)
    rv2 = float(np.mean([r["rel_v2"] for r in rows]))
    rcur = float(np.mean([r["rel_current"] for r in rows]))
    rms = float(np.mean([r["rms_ratio_v2"] for r in rows]))
    gates = {"relrms_ratio_lt_0.95": [rv2/rcur < 0.95, rv2/rcur],
             "rms_ratio_0.90_1.10": [0.90 <= rms <= 1.10, rms]}
    result = {"pass": all(v[0] for v in gates.values()), "gates": gates,
              "means": {"rel_v2": rv2, "rel_current": rcur, "rms_ratio_v2": rms},
              "sample": {"layers": LAYERS, "experts": EXPERTS}, "rows": rows}
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("pass", "gates", "means")}, indent=2))
    sys.exit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
