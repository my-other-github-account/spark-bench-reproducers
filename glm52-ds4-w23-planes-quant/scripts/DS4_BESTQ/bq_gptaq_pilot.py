#!/usr/bin/env python3
"""t_3d6e422d card-step-5 pilot: GPTQv2/GPTAQ asymmetric-error solve
(arXiv 2504.02692) vs standard GPTQ on the SAME W2v2 grid.

Where it applies here: calib X for fused13 is teacher-side x_moe (identical
for both models), so the asymmetric correction is identity there. It bites
on the DOWN projection, whose quantized-model input A~ (activations of the
SHIPPED quantized fused13) differs from the teacher input A_fp. Standard
(sealed R5v2 protocol) solves ||A~ (Q - W)^T||; GPTAQ solves the true
objective ||A~ Q^T - A_fp W^T|| via the ~20-line delta:

    W*^T = (A~^T A~ + damp)^{-1} (A~^T A_fp) W^T
    Q    = GPTQ(W*, H = A~^T A~)     # same grid, same scales, same loop

Pilot: 5 layers x 6 experts. Both arms evaluated on the TRUE val objective
err = ||Av~ Q^T - Av_fp W^T|| / ||Av_fp W^T||. Adopt iff mass-weighted mean
gain > 3% (linear damage model: tier KLD scales with mass x relerr, so a
>3% error gain ~ >3% tier-KLD gain; documented proxy).
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_BESTQ"))
import bq_common as bq  # noqa: E402

LAYERS = [0, 10, 21, 32, 42]
EXPERTS = [9, 50, 100, 150, 200, 254]
LUT = torch.tensor(bq.W2V2_LUT_E43, dtype=torch.float32, device=bq.DEV)
LIMIT = bq.swiglu_limit()


def main():
    t0 = time.time()
    sel = bq.calib_selection()
    rows = []
    for L in LAYERS:
        xf, hitf = bq.load_caps(L, sel["fit_ids"])
        xv, hitv = bq.load_caps(L, sel["val_ids"])
        for e in EXPERTS:
            Xf = xf[hitf[:, e]].float()
            Xv = xv[hitv[:, e]].float()
            if Xf.shape[0] < 64 or Xv.shape[0] < 32:
                continue
            W13 = bq.src_dense(L, e, ("w1", "w3"))
            Wd = bq.src_dense(L, e, ("w2",))
            sb13 = bq.src_scales(L, e, ("w1", "w3"))
            sb2 = bq.src_scales(L, e, ("w2",))

            # fused13: standard GPTQ on the W2v2 grid (asym = identity)
            c13, sc13 = bq.requant(W13, sb13, LUT)  # RTN arm
            s13_col = torch.exp2(sc13.to(torch.float32) - 127.0) \
                .repeat_interleave(32, dim=1)
            H13 = Xf.t() @ Xf
            perm13 = bq.weight_perm(W13)
            Hinv13 = bq.gptq_prepare(H13, perm13)
            c13g = bq.gptq_loop(W13, s13_col, Hinv13, perm13, LUT)
            del H13, Hinv13
            dq13g = bq.deq_codes(c13g, sc13, LUT)
            dq13r = bq.deq_codes(c13, sc13, LUT)
            use13 = dq13g if (bq.proxy_err(Xv, dq13g, W13) or 9) <= \
                (bq.proxy_err(Xv, dq13r, W13) or 9) else dq13r
            gI, uI = use13[:2048], use13[2048:]
            Af = bq.act(Xf, gI, uI, LIMIT)
            Av = bq.act(Xv, gI, uI, LIMIT)
            Af_fp = bq.act(Xf, W13[:2048], W13[2048:], LIMIT)
            Av_fp = bq.act(Xv, W13[:2048], W13[2048:], LIMIT)

            # down grid (RTN codes for scales convention)
            c2r, sc2 = bq.requant(Wd, sb2, LUT)
            s2_col = torch.exp2(sc2.to(torch.float32) - 127.0) \
                .repeat_interleave(32, dim=1)
            H2 = Af.t() @ Af
            perm2 = bq.weight_perm(Wd)
            Hinv2 = bq.gptq_prepare(H2, perm2)

            # arm STD
            c2_std = bq.gptq_loop(Wd, s2_col, Hinv2, perm2, LUT)
            # arm GPTAQ: W* then same loop
            Wstar = bq.damp_solve(H2, (Af.t() @ Af_fp) @ Wd.t().float()).t()
            c2_aq = bq.gptq_loop(Wstar, s2_col, Hinv2, perm2, LUT)
            del H2, Hinv2

            ref = (Av_fp @ Wd.t().float())
            den = ref.norm() + 1e-30

            def true_err(codes):
                dq = bq.deq_codes(codes, sc2, LUT)
                return ((Av @ dq.t().float()) - ref).norm().item() / \
                    den.item()

            r = {"L": L, "e": e,
                 "mass": int(Xf.shape[0] + Xv.shape[0]),
                 "err_rtn": true_err(c2r),
                 "err_std": true_err(c2_std),
                 "err_gptaq": true_err(c2_aq)}
            r["gain_vs_std"] = 1.0 - r["err_gptaq"] / r["err_std"]
            rows.append(r)
            bq.log(f"L{L:03d} e{e:03d} rtn={r['err_rtn']:.4f} "
                   f"std={r['err_std']:.4f} aq={r['err_gptaq']:.4f} "
                   f"gain={r['gain_vs_std']*100:+.2f}%")
            del Xf, Xv, W13, Wd, Af, Av, Af_fp, Av_fp, Wstar
            torch.cuda.empty_cache()
        del xf, xv, hitf, hitv
        torch.cuda.empty_cache()

    tot = sum(r["mass"] for r in rows)
    wstd = sum(r["err_std"] * r["mass"] for r in rows) / tot
    waq = sum(r["err_gptaq"] * r["mass"] for r in rows) / tot
    wrtn = sum(r["err_rtn"] * r["mass"] for r in rows) / tot
    gain = 1.0 - waq / wstd
    out = {"task": "t_3d6e422d", "step": "card-5 GPTQv2/GPTAQ pilot",
           "protocol": "5 layers x 6 experts, down proj, W2v2-e43 grid, "
                       "true objective ||Av~ Q^T - Av_fp W^T||",
           "weighted_means": {"rtn": wrtn, "std": wstd, "gptaq": waq},
           "mass_weighted_gain_vs_std": gain,
           "adopt": bool(gain > 0.03),
           "threshold": 0.03, "n_rows": len(rows), "rows": rows,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(f"{bq.BQ}/PILOT_GPTAQ.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in
                      ("weighted_means", "mass_weighted_gain_vs_std",
                       "adopt")}, indent=1))


if __name__ == "__main__":
    main()
