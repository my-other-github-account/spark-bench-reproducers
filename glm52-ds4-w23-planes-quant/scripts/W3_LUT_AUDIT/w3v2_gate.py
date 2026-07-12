#!/usr/bin/env python3
"""Weight-space gate for moe_w3_planes_v2 (t_eee6b0cc SEQ-3 GATE).

On sampled experts (DIFFERENT from the shootout fit set), dequant the
EMITTED v2 bytes (through planes_unpack + meta lut, i.e. exactly what the
rail loader does) and compare vs the f64 source:

  gate 1: rms ratio in [0.95, 1.05]           (scale convention correct)
  gate 2: relRMS(v2) / relRMS(ship_w2) < 0.55 (the '~0.5x of W2' order)
  gate 3: relRMS(v2) < relRMS(ship_w3)        (strictly better than old)

Writes GATE_W3V2.json {pass: bool, rows: [...]}; exit 0 iff all pass.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_TEACHER"))
import planes_unpack as pu  # noqa
from safetensors import safe_open  # noqa

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
V2 = os.path.expanduser("~/missions/W3_LUT_AUDIT/moe_w3_planes_v2")
OUT = os.path.expanduser("~/missions/W3_LUT_AUDIT")

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)
WM = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_h = {}


def get(name):
    sh = WM[name]
    if sh not in _h:
        _h[sh] = safe_open(os.path.join(CKPT, sh), framework="pt")
    return _h[sh].get_tensor(name)


def src(L, e, names):
    ws = []
    for wname in names:
        k = f"layers.{L}.ffn.experts.{e}.{wname}"
        wp = get(k + ".weight").view(torch.uint8)
        sb = get(k + ".scale").view(torch.uint8)
        nib = torch.stack((wp & 0xF, wp >> 4), dim=-1).flatten(-2)
        w = _E2M1[nib.long()]
        ws.append(w * torch.exp2(sb.double() - 127.0)
                  .repeat_interleave(32, dim=1))
    return torch.cat(ws, 0)


def deq(dirp, L, e, tier, N, K, levels):
    p = torch.from_numpy(np.load(
        f"{dirp}/layer_{L:03d}.planes{tier}.npy", mmap_mode="r")[e].copy())
    sc = torch.from_numpy(np.load(
        f"{dirp}/layer_{L:03d}.sc{tier}.npy", mmap_mode="r")[e].copy())
    unpack = pu.unpack_fragment_major if len(levels) == 4 \
        else pu.unpack_w3_plane
    codes = unpack(p, N, K)
    sb = pu.unpack_scales(sc, N, K // 32)
    vals = torch.tensor(levels, dtype=torch.float64)[codes.long()]
    return vals * torch.exp2(sb.double() - 127.0) \
        .repeat_interleave(32, dim=1)


LAYERS = [1, 9, 15, 21, 27, 33, 39, 41]
EXPERTS = [3, 77, 201]
W2LV = [-4.0, -1.0, 1.0, 4.0]
W3LV = [-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0]


def relrms(dq, w):
    return ((dq - w).pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()


def main():
    rows = []
    for L in LAYERS:
        v2lut = json.load(open(f"{V2}/layer_{L:03d}.meta.json"))["lut"]
        for e in EXPERTS:
            for tier, names, N, K in (("13", ("w1", "w3"), 4096, 4096),
                                      ("2", ("w2",), 4096, 2048)):
                w = src(L, e, names)
                d_v2 = deq(V2, L, e, tier, N, K, v2lut)
                d_w2 = deq(f"{CKPT}/moe_w2_planes", L, e, tier, N, K, W2LV)
                d_w3 = deq(f"{CKPT}/moe_w3_planes", L, e, tier, N, K, W3LV)
                rows.append({
                    "L": L, "e": e, "tier": tier,
                    "rel_v2": relrms(d_v2, w), "rel_w2": relrms(d_w2, w),
                    "rel_w3ship": relrms(d_w3, w),
                    "ratio_v2": (d_v2.pow(2).mean().sqrt()
                                 / w.pow(2).mean().sqrt()).item()})
        print(f"L{L:03d} audited", flush=True)

    mv2 = float(np.mean([r["rel_v2"] for r in rows]))
    mw2 = float(np.mean([r["rel_w2"] for r in rows]))
    mw3 = float(np.mean([r["rel_w3ship"] for r in rows]))
    mratio = float(np.mean([r["ratio_v2"] for r in rows]))
    g1 = 0.95 <= mratio <= 1.05
    g2 = (mv2 / mw2) < 0.55
    g3 = mv2 < mw3
    res = {"pass": bool(g1 and g2 and g3),
           "gates": {"g1_rms_ratio_0.95_1.05": [g1, mratio],
                     "g2_relrms_vs_w2_lt_0.55": [g2, mv2 / mw2],
                     "g3_better_than_ship_w3": [g3, mv2, mw3]},
           "means": {"rel_v2": mv2, "rel_w2": mw2, "rel_w3ship": mw3,
                     "ratio_v2": mratio},
           "sample": {"layers": LAYERS, "experts": EXPERTS},
           "rows": rows}
    with open(f"{OUT}/GATE_W3V2.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"pass": res["pass"], "gates": res["gates"],
                      "means": res["means"]}, indent=1))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
