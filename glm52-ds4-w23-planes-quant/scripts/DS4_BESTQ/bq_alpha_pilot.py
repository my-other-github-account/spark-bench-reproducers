#!/usr/bin/env python3
"""t_3d6e422d card-step-4 pilot: per-tensor fractional-scale (alpha).

UE8M0 scales are power-of-2-only; a per-tensor fp32 alpha recovers the
fractional gap as a trivial serve epilogue (per-output-row-group multiply
on the GEMM result). Pilot on 3 layers x 6 experts x {w1,w3,down}:

  ref_full : W2v2 full-precision winner LUT (dp_asym4_round2), SSE offsets
  a_e43    : same, LUT rounded e4m3-representable [-3.5,-1.125,0.625,2.75]
  b_post   : a_e43 codes+scales FIXED, closed-form per-tensor alpha
  c_joint  : per-tensor alpha grid (17 pts, 2^-0.5..2^0.5) jointly with
             offset search + code snap refit

And on the EXISTING sealed W3v2-GPTQ e43 tier (codes+scales untouched):
  d_base / d_post : closed-form per-tensor alpha on planes_w3v2_e43

Decisions written to PILOT_ALPHA.json:
  w2v2_alpha_mode: 'joint' if mean gain(c vs a_e43) >= 3%,
                   else 'posthoc' if gain(b vs a_e43) >= 1%, else 'none'
  w3_tier_alphas:  true if mean gain(d_post vs d_base) >= 1%
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_BESTQ"))
import bq_common as bq  # noqa: E402
import planes_unpack as pu  # noqa: E402

W3E43_DIR = os.path.expanduser("~/missions/DS4_R6/planes_w3v2_e43")
LAYERS = [0, 21, 42]
EXPERTS = [9, 50, 100, 150, 200, 254]
PROJS = [("w1", 2048, 4096), ("w3", 2048, 4096), ("w2", 4096, 2048)]
ALPHAS = [2.0 ** (k / 16.0) for k in range(-8, 9)]

LUT_FULL = torch.tensor(bq.W2V2_LUT_FULL, dtype=torch.float32, device=bq.DEV)
LUT_E43 = torch.tensor(bq.W2V2_LUT_E43, dtype=torch.float32, device=bq.DEV)


def main():
    t0 = time.time()
    rows = []
    for L in LAYERS:
        for e in EXPERTS:
            for proj, N, K in PROJS:
                w = bq.src_dense(L, e, (proj,))
                sb = bq.src_scales(L, e, (proj,))
                row = {"L": L, "e": e, "proj": proj}
                # ref: full-precision LUT
                c, sc = bq.requant(w, sb, LUT_FULL)
                row["ref_full"] = bq.relrms(bq.deq_codes(c, sc, LUT_FULL), w)
                # a: e43 LUT
                c, sc = bq.requant(w, sb, LUT_E43)
                dq = bq.deq_codes(c, sc, LUT_E43)
                row["a_e43"] = bq.relrms(dq, w)
                # b: closed-form alpha on fixed codes
                al = bq.closed_alpha(dq, w)
                row["b_post"] = bq.relrms(dq * al, w)
                row["b_alpha"] = al
                # c: joint grid
                best = (None, None)
                for a in ALPHAS:
                    sse = bq.requant_sse(w, sb, LUT_E43, a)
                    if best[0] is None or sse < best[0]:
                        best = (sse, a)
                a_star = best[1]
                ar = torch.full((N,), a_star, device=bq.DEV)
                c, sc = bq.requant(w, sb, LUT_E43, ar)
                row["c_joint"] = bq.relrms(
                    bq.deq_codes(c, sc, LUT_E43, ar), w)
                row["c_alpha"] = a_star
                rows.append(row)
            bq.log(f"L{L:03d} e{e:03d} w2v2 arms done "
                   f"({time.time()-t0:.0f}s)")

    # ---- d arm: sealed W3v2-GPTQ e43 tier, codes fixed, post-hoc alpha
    d_rows = []
    for L in LAYERS:
        meta = json.load(open(f"{W3E43_DIR}/layer_{L:03d}.meta.json"))
        lut3 = torch.tensor(meta["lut"], dtype=torch.float32, device=bq.DEV)
        p13 = np.load(f"{W3E43_DIR}/layer_{L:03d}.planes13.npy", mmap_mode="r")
        p2 = np.load(f"{W3E43_DIR}/layer_{L:03d}.planes2.npy", mmap_mode="r")
        s13 = np.load(f"{W3E43_DIR}/layer_{L:03d}.sc13.npy", mmap_mode="r")
        s2 = np.load(f"{W3E43_DIR}/layer_{L:03d}.sc2.npy", mmap_mode="r")
        for e in EXPERTS:
            c13 = pu.unpack_w3_plane(
                torch.from_numpy(np.asarray(p13[e])).to(bq.DEV),
                bq.N13, bq.K13)
            sb13 = pu.unpack_scales(
                torch.from_numpy(np.asarray(s13[e])), bq.N13,
                bq.K13 // 32).to(bq.DEV)
            c2 = pu.unpack_w3_plane(
                torch.from_numpy(np.asarray(p2[e])).to(bq.DEV),
                bq.N2, bq.K2)
            sb2 = pu.unpack_scales(
                torch.from_numpy(np.asarray(s2[e])), bq.N2,
                bq.K2 // 32).to(bq.DEV)
            w13 = bq.src_dense(L, e, ("w1", "w3"))
            wd = bq.src_dense(L, e, ("w2",))
            dq13 = bq.deq_codes(c13, sb13, lut3)
            dq2 = bq.deq_codes(c2, sb2, lut3)
            for tag, dq, w, r0, r1 in (("w1", dq13, w13, 0, 2048),
                                       ("w3", dq13, w13, 2048, 4096),
                                       ("w2", dq2, wd, 0, 4096)):
                d, s = dq[r0:r1], w[r0:r1]
                base = bq.relrms(d, s)
                al = bq.closed_alpha(d, s)
                d_rows.append({"L": L, "e": e, "proj": tag,
                               "d_base": base,
                               "d_post": bq.relrms(d * al, s),
                               "d_alpha": al})
        bq.log(f"L{L:03d} w3e43 d-arm done")

    def mean(k, rr):
        return float(np.mean([r[k] for r in rr]))

    m = {k: mean(k, rows) for k in
         ("ref_full", "a_e43", "b_post", "c_joint")}
    md = {k: mean(k, d_rows) for k in ("d_base", "d_post")}
    gains = {
        "e43_cost_vs_full": m["a_e43"] / m["ref_full"] - 1.0,
        "b_vs_a": 1.0 - m["b_post"] / m["a_e43"],
        "c_vs_a": 1.0 - m["c_joint"] / m["a_e43"],
        "d_vs_base": 1.0 - md["d_post"] / md["d_base"],
    }
    if gains["c_vs_a"] >= 0.03:
        mode = "joint"
    elif gains["b_vs_a"] >= 0.01:
        mode = "posthoc"
    else:
        mode = "none"
    out = {
        "task": "t_3d6e422d", "step": "card-4 fractional-scale pilot",
        "protocol": f"{len(LAYERS)} layers x {len(EXPERTS)} experts x "
                    f"w1/w3/down; alpha grid 17 pts 2^+-0.5",
        "lut_full": bq.W2V2_LUT_FULL, "lut_e43": bq.W2V2_LUT_E43,
        "means": {**m, **md}, "gains": gains,
        "decisions": {"w2v2_alpha_mode": mode,
                      "w3_tier_alphas": bool(gains["d_vs_base"] >= 0.01)},
        "thresholds": {"joint": 0.03, "posthoc": 0.01, "w3_post": 0.01},
        "rows": rows, "d_rows": d_rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(f"{bq.BQ}/PILOT_ALPHA.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"means": out["means"], "gains": gains,
                      "decisions": out["decisions"]}, indent=1))


if __name__ == "__main__":
    main()
