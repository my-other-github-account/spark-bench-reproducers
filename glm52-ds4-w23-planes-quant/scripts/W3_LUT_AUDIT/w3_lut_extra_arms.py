#!/usr/bin/env python3
"""Extra shootout arms (t_eee6b0cc): isolate the sign-sym cost + freeze
the winner LUT on held-out experts.

Arms (scMSE per-block UE8M0 exponent search for all):
  dp_sym8       sign-symmetric OPTIMAL: DP 4 magnitudes on folded |u| hist
                (fit experts), mirrored -> 8 sym levels. Isolates the cost
                of the sign-sym constraint vs dp_asym8.
  e2m1_sub      {+-1,+-2,+-4,+-6}: the natural e2m1-subset ladder.
  dp_asym8_fit  DP 8 levels fit ONLY on fit experts (alternation round 2
                with scale search), evaluated on eval experts = the
                held-out-validated winner candidate.

Prints the same relRMS/ratio table on the SAME eval experts as round 1.
"""
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(12)
sys.path.insert(0, os.path.expanduser("~/missions/W3_LUT_AUDIT"))
from w3_lut_shootout import (EVAL_E, FIT_E, LAYERS, MATS, OUT, dp_lloyd,
                             dequant_scmse, load_matrix, merge_hist,
                             relrms, rmsratio, u_hist)


def fold_sym(vals, mass):
    av = vals.abs()
    v, inv = torch.unique(av, return_inverse=True)
    m = torch.zeros_like(v)
    m.scatter_add_(0, inv, mass)
    # drop exact zero from magnitude fit? no: zero mass pulls the lowest
    # magnitude level toward 0, exactly the effect we want to measure.
    return v, m


def main():
    t0 = time.time()
    # ---------- fit-expert histograms (ckpt scales, round 1)
    hist = None
    for L in LAYERS:
        for e in FIT_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                h = u_hist(w, sb)
                hist = h if hist is None else merge_hist(hist, h)

    # sym: DP on folded magnitudes, 4 levels, mirror (no zero level)
    fv, fm = fold_sym(*hist)
    mag4 = dp_lloyd(fv, fm, 4)
    dp_sym8 = torch.cat([-mag4.flip(0), mag4])
    print(f"dp_sym8 LUT:  {[round(x, 4) for x in dp_sym8.tolist()]}",
          flush=True)

    e2m1_sub = torch.tensor([-6., -4., -2., -1., 1., 2., 4., 6.],
                            dtype=torch.float64)

    # asym8 alternation on FIT experts only (round1: ckpt-scale DP,
    # round2: DP on scale-search-chosen scales)
    lut1 = dp_lloyd(*hist, 8)
    h2 = None
    for L in LAYERS:
        for e in FIT_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                _, off = dequant_scmse(w, sb, lut1)
                s = torch.exp2(sb.double() - 127.0) * \
                    torch.exp2(off.double())
                h = u_hist(w, sb, s_override=s)
                h2 = h if h2 is None else merge_hist(h2, h)
    dp_asym8_fit = dp_lloyd(*h2, 8)
    print(f"dp_asym8_fit LUT (held-out winner cand): "
          f"{[round(x, 4) for x in dp_asym8_fit.tolist()]}", flush=True)

    arms = {"dp_sym8": dp_sym8, "e2m1_sub": e2m1_sub,
            "dp_asym8_fit": dp_asym8_fit}
    res = {n: {m: {"rel": [], "ratio": []} for m in MATS} for n in arms}
    for L in LAYERS:
        for e in EVAL_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                for name, lut in arms.items():
                    dq, _ = dequant_scmse(w, sb, lut)
                    res[name][mat]["rel"].append(relrms(dq, w))
                    res[name][mat]["ratio"].append(rmsratio(dq, w))
        print(f"L{L:03d} eval done ({time.time()-t0:.0f}s)", flush=True)

    out = {"luts": {n: a.tolist() for n, a in arms.items()}, "arms": {}}
    print(f"\n{'arm':16s}" + "".join(f"{m:>12s}{'ratio':>8s}" for m in MATS))
    for name in arms:
        line = f"{name:16s}"
        out["arms"][name] = {}
        for m in MATS:
            rel = float(np.mean(res[name][m]["rel"]))
            rat = float(np.mean(res[name][m]["ratio"]))
            out["arms"][name][m] = {"relrms_mean": rel,
                                    "rms_ratio_mean": rat}
            line += f"{rel:12.5f}{rat:8.4f}"
        print(line, flush=True)
    with open(f"{OUT}/SHOOTOUT_EXTRA.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {OUT}/SHOOTOUT_EXTRA.json ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
