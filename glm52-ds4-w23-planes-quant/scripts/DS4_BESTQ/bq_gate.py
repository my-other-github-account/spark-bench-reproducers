#!/usr/bin/env python3
"""t_3d6e422d weight-space gate for moe_w2_planes_v2e43 (RMS gate, card
step 1). Sampled layers/experts DIFFERENT from the shootout fit set;
dequant EXACTLY as the rail loader (planes_unpack + meta lut + alphas)
vs the f64 ckpt source, compared against the shipped W2 planes.

  gate 1: rms ratio in [0.90, 1.10]
  gate 2: relRMS(v2e43+alpha) / relRMS(ship_w2) < 0.95
          (shootout said 0.9198x on SSE-refit alone)

Writes GATE_W2V2.json; exit 0 iff all pass.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_BESTQ"))
import bq_common as bq  # noqa: E402
import planes_unpack as pu  # noqa: E402

V2 = f"{bq.BQ}/moe_w2_planes_v2e43"
SHIP = f"{bq.CKPT}/moe_w2_planes"
LAYERS = [1, 9, 15, 21, 27, 33, 39, 41]
EXPERTS = [3, 77, 201]
W2LV = torch.tensor([-4.0, -1.0, 1.0, 4.0])


def deq(dirp, L, e, tier, N, K, levels, alpha=None):
    p = torch.from_numpy(np.load(
        f"{dirp}/layer_{L:03d}.planes{tier}.npy", mmap_mode="r")[e].copy())
    sc = torch.from_numpy(np.load(
        f"{dirp}/layer_{L:03d}.sc{tier}.npy", mmap_mode="r")[e].copy())
    codes = pu.unpack_fragment_major(p, N, K)
    sb = pu.unpack_scales(sc, N, K // 32)
    vals = torch.as_tensor(levels, dtype=torch.float64)[codes.long()]
    dq = vals * torch.exp2(sb.double() - 127.0).repeat_interleave(32, dim=1)
    if alpha is not None:
        dq = dq * alpha.view(-1, 1).double()
    return dq


def main():
    rows = []
    for L in LAYERS:
        meta = json.load(open(f"{V2}/layer_{L:03d}.meta.json"))
        lut = meta["lut"]
        ap = f"{V2}/layer_{L:03d}.alphas.npy"
        alphas = np.load(ap) if os.path.exists(ap) else None
        for e in EXPERTS:
            for tier, names, N, K in ((
                    "13", ("w1", "w3"), bq.N13, bq.K13),
                    ("2", ("w2",), bq.N2, bq.K2)):
                w = bq.src_dense(L, e, names, dev="cpu",
                                 dtype=torch.float64)
                a_row = None
                if alphas is not None:
                    a = alphas[e]
                    if tier == "13":
                        a_row = torch.cat([
                            torch.full((2048,), float(a[0])),
                            torch.full((2048,), float(a[1]))])
                    else:
                        a_row = torch.full((N,), float(a[2]))
                v2 = deq(V2, L, e, tier, N, K, lut, a_row)
                cur = deq(SHIP, L, e, tier, N, K, W2LV)
                rows.append({
                    "L": L, "e": e, "tier": tier,
                    "rel_v2": bq.relrms(v2, w),
                    "rel_ship": bq.relrms(cur, w),
                    "rms_ratio_v2": (v2.pow(2).mean().sqrt()
                                     / w.pow(2).mean().sqrt()).item()})
        print(f"L{L:03d} audited", flush=True)
    rv2 = float(np.mean([r["rel_v2"] for r in rows]))
    rcur = float(np.mean([r["rel_ship"] for r in rows]))
    rms = float(np.mean([r["rms_ratio_v2"] for r in rows]))
    gates = {"rms_ratio_0.90_1.10": [bool(0.90 <= rms <= 1.10), rms],
             "relrms_ratio_lt_0.95": [bool(rv2 / rcur < 0.95), rv2 / rcur]}
    res = {"pass": all(v[0] for v in gates.values()), "gates": gates,
           "means": {"rel_v2": rv2, "rel_ship_w2": rcur,
                     "rms_ratio_v2": rms},
           "sample": {"layers": LAYERS, "experts": EXPERTS}, "rows": rows}
    with open(f"{bq.BQ}/GATE_W2V2.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("pass", "gates", "means")},
                     indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
