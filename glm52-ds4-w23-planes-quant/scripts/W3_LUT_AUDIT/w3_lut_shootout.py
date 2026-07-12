#!/usr/bin/env python3
"""W3 LUT design shootout (t_eee6b0cc, Banana Bae's design question).

Weight-space relRMS on sampled DS4-Flash experts for candidate 8-level
W3 codebooks, ALL with correct per-grid scale fits.  CPU-only (the GPU
eval rail must not be touched).

Source lattice fact (verified): ckpt experts are mxfp4/e2m1 -- per
block-32 UE8M0 scale s, values u*s with u in {0,+-.5,+-1,+-1.5,+-2,
+-3,+-4,+-6}.  Any candidate LUT is judged by how it covers THAT
discrete lattice, not a Gaussian.

Arms (per matrix fused13 [w1;w3] and down w2):
  ship_w2         W2 4-level {-4,-1,1,4}, ckpt scales verbatim   (Q2 ref)
  ship_w3         current 8-level +-{.5,1.5,3,6}, amax->6 UE8M0  (the R2 arm)
  w3lut_ckpt_sc   current LUT, ckpt scales verbatim
  w3lut_scMSE     current LUT, per-block MSE-optimal p2 scale
  uniform8        sign-sym uniform no-zero +-{1,3,5,7}*(6/7), p2-MSE scale
  nf3_8           NF3 quantile placement (asym, has 0), p2-MSE scale
  lloyd_global    DP-optimal global 8-LUT on s^2-weighted u-atom hist
                  (fit on held-out experts), ckpt scales
  lloyd_scaleopt  lloyd LUT + per-block p2-MSE scale, one LUT refit round
  lloyd_perexp    DP-optimal 8-LUT per expert (headroom bound, not one LUT)

Output: JSON + printed table of mean relRMS ( ||dq-w||/||w|| ) and rms
ratio per arm, plus the u-atom mass histogram.
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_num_threads(12)

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
OUT = os.path.expanduser("~/missions/W3_LUT_AUDIT")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.expanduser("~/missions/DS4_TEACHER"))
from safetensors import safe_open  # noqa

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float64)
W2_LUT = torch.tensor([-4.0, -1.0, 1.0, 4.0], dtype=torch.float64)
W3_LUT = torch.tensor([-6.0, -3.0, -1.5, -0.5, 0.5, 1.5, 3.0, 6.0],
                      dtype=torch.float64)

WM = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_handles = {}


def get(name):
    sh = WM[name]
    if sh not in _handles:
        _handles[sh] = safe_open(os.path.join(CKPT, sh), framework="pt")
    return _handles[sh].get_tensor(name)


def load_matrix(L, e, which):
    """-> (w f64 [N,K], ckpt scale bytes u8 [N,K/32])"""
    names = ("w1", "w3") if which == "fused13" else ("w2",)
    ws, ss = [], []
    for wname in names:
        k = f"layers.{L}.ffn.experts.{e}.{wname}"
        wp = get(k + ".weight").view(torch.uint8)
        sb = get(k + ".scale").view(torch.uint8)
        nib = torch.stack((wp & 0xF, wp >> 4), dim=-1).flatten(-2)
        ws.append(_E2M1[nib.long()])
        ss.append(sb)
    w_u = torch.cat(ws, 0)
    sb = torch.cat(ss, 0)
    s = torch.exp2(sb.double() - 127.0)
    return w_u * s.repeat_interleave(32, dim=1), sb


def snap(u, lut):
    """nearest level of sorted lut; ties -> lower (bucketize right=False)."""
    mids = (lut[1:] + lut[:-1]) / 2
    idx = torch.bucketize(u.contiguous(), mids)
    return lut[idx]


def sse_blocks(w, s, lut):
    """w [N,K] f64, s [N,KB] f64 -> per-block SSE [N,KB] and dq."""
    N, K = w.shape
    sf = s.repeat_interleave(32, dim=1)
    u = w / sf
    q = snap(u, lut)
    dq = q * sf
    err2 = (dq - w).pow(2).view(N, -1, 32).sum(dim=2)
    return err2, dq


def dequant_fixed(w, sbytes, lut):
    s = torch.exp2(sbytes.double() - 127.0)
    return sse_blocks(w, s, lut)[1]


def dequant_scmse(w, sbytes_base, lut, offsets=range(-4, 3)):
    """per-block exponent search around ckpt exponent, min weight-SSE."""
    s0 = torch.exp2(sbytes_base.double() - 127.0)
    best_err = None
    best_dq = None
    for off in offsets:
        s = s0 * (2.0 ** off)
        err2, dq = sse_blocks(w, s, lut)
        if best_err is None:
            best_err, best_dq, best_off = err2, dq, torch.zeros_like(err2)
            best_off += off
        else:
            m = err2 < best_err
            best_err = torch.where(m, err2, best_err)
            mf = m.repeat_interleave(32, dim=1)
            best_dq = torch.where(mf, dq, best_dq)
            best_off = torch.where(m, torch.full_like(best_off, off), best_off)
    return best_dq, best_off


def amax6_sbytes(w):
    wb = w.view(w.shape[0], -1, 32)
    amax = wb.abs().amax(dim=2)
    exp = torch.where(amax > 0, torch.round(torch.log2(amax / 6.0 + 1e-30)),
                      torch.full_like(amax, -127.0)).clamp_(-127.0, 127.0)
    return (exp + 127.0).to(torch.uint8)


# ------------------------------------------------------------------ DP
def dp_lloyd(vals, mass, k=8):
    """MSE-optimal k-level quantizer of discrete dist (vals sorted, mass>0).
    Interval DP O(n^2 k). Returns levels tensor f64."""
    n = len(vals)
    if n <= k:
        return vals.clone()
    pm = torch.cat([torch.zeros(1, dtype=torch.float64), torch.cumsum(mass, 0)])
    pmv = torch.cat([torch.zeros(1, dtype=torch.float64),
                     torch.cumsum(mass * vals, 0)])
    pmv2 = torch.cat([torch.zeros(1, dtype=torch.float64),
                      torch.cumsum(mass * vals * vals, 0)])

    def cost(i, j):  # atoms i..j inclusive
        m = pm[j + 1] - pm[i]
        mv = pmv[j + 1] - pmv[i]
        mv2 = pmv2[j + 1] - pmv2[i]
        return (mv2 - mv * mv / m).item(), (mv / m).item()

    INF = float("inf")
    D = [[INF] * n for _ in range(k + 1)]
    P = [[-1] * n for _ in range(k + 1)]
    for j in range(n):
        D[1][j], _ = cost(0, j)
    for g in range(2, k + 1):
        for j in range(g - 1, n):
            for i in range(g - 1, j + 1):
                c = D[g - 1][i - 1] + cost(i, j)[0]
                if c < D[g][j]:
                    D[g][j] = c
                    P[g][j] = i
    # backtrack
    cuts = []
    j = n - 1
    for g in range(k, 1, -1):
        i = P[g][j]
        cuts.append(i)
        j = i - 1
    cuts = sorted(cuts)
    levels, lo = [], 0
    for c in cuts + [n]:
        levels.append(cost(lo, c - 1)[1])
        lo = c
    return torch.tensor(levels, dtype=torch.float64)


def u_hist(w, sbytes, s_override=None):
    """s^2-weighted histogram of u = w/s (exact discrete values)."""
    s = (torch.exp2(sbytes.double() - 127.0) if s_override is None
         else s_override)
    u = (w / s.repeat_interleave(32, dim=1)).flatten()
    wt = (s * s).repeat_interleave(32, dim=1).flatten()
    vals, inv = torch.unique(u, return_inverse=True)
    mass = torch.zeros_like(vals)
    mass.scatter_add_(0, inv, wt)
    return vals, mass


def merge_hist(h1, h2):
    v = torch.cat([h1[0], h2[0]])
    m = torch.cat([h1[1], h2[1]])
    vals, inv = torch.unique(v, return_inverse=True)
    mass = torch.zeros_like(vals)
    mass.scatter_add_(0, inv, m)
    return vals, mass


def nf3_levels():
    """bnb-style NF3: asym quantile placement with exact zero, 8 levels."""
    def ppf(p):
        return torch.erfinv(2 * torch.tensor(p, dtype=torch.float64) - 1) \
            * (2 ** 0.5)
    offset = 0.9677083
    pos = ppf(torch.linspace(offset, 0.5, 5).tolist())[:4]
    neg = -ppf(torch.linspace(offset, 0.5, 4).tolist())[:3]
    lv = torch.cat([neg, torch.zeros(1, dtype=torch.float64), pos])
    lv = lv / lv.abs().max() * 6.0
    return torch.sort(lv).values


LAYERS = [0, 6, 12, 18, 24, 30, 36, 42]
EVAL_E = [9, 100, 254]
FIT_E = [50, 150, 200]
MATS = ["fused13", "down"]


def relrms(dq, w):
    return ((dq - w).pow(2).mean().sqrt()
            / w.pow(2).mean().sqrt()).item()


def rmsratio(dq, w):
    return (dq.pow(2).mean().sqrt() / w.pow(2).mean().sqrt()).item()


def main():
    t0 = time.time()
    uni8 = torch.tensor([-6, -30 / 7, -18 / 7, -6 / 7, 6 / 7, 18 / 7, 30 / 7,
                         6.0], dtype=torch.float64)
    nf3 = nf3_levels()
    print(f"nf3 levels: {[round(x, 4) for x in nf3.tolist()]}", flush=True)
    print(f"uniform8:   {[round(x, 4) for x in uni8.tolist()]}", flush=True)

    # ---- pass 1: fit histogram (held-out experts, ckpt scales)
    hist = None
    for L in LAYERS:
        for e in FIT_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                h = u_hist(w, sb)
                hist = h if hist is None else merge_hist(hist, h)
    vals, mass = hist
    lloyd = dp_lloyd(vals, mass, 8)
    frac = (mass / mass.sum())
    print("\nu-atom histogram (ckpt scales, s^2-weighted mass):", flush=True)
    for v, f in zip(vals.tolist(), frac.tolist()):
        print(f"  u={v:+.3f}  mass={f:.5f}")
    print(f"\nlloyd_global LUT: {[round(x, 4) for x in lloyd.tolist()]}",
          flush=True)

    arms = {}   # name -> dict(mat -> [relrms...], ratio -> [...])

    def rec(name, mat, dq, w):
        a = arms.setdefault(name, {m: {"rel": [], "ratio": []} for m in MATS})
        a[mat]["rel"].append(relrms(dq, w))
        a[mat]["ratio"].append(rmsratio(dq, w))

    # W2 codes via direct nibble map for ship_w2 (provenance-proven equal)
    def ship_w2_dq(w, sb):
        return dequant_fixed(w, sb, W2_LUT)

    per_expert = []
    # for lloyd_scaleopt refit
    refit_hist = None

    for L in LAYERS:
        for e in EVAL_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                sb6 = amax6_sbytes(w)

                row = {"L": L, "e": e, "mat": mat}
                dq = ship_w2_dq(w, sb)
                rec("ship_w2", mat, dq, w); row["ship_w2"] = relrms(dq, w)
                dq = dequant_fixed(w, sb6, W3_LUT)
                rec("ship_w3", mat, dq, w); row["ship_w3"] = relrms(dq, w)
                dq = dequant_fixed(w, sb, W3_LUT)
                rec("w3lut_ckpt_sc", mat, dq, w)
                dq, _ = dequant_scmse(w, sb, W3_LUT)
                rec("w3lut_scMSE", mat, dq, w)
                dq, _ = dequant_scmse(w, sb, uni8)
                rec("uniform8", mat, dq, w)
                dq, _ = dequant_scmse(w, sb, nf3)
                rec("nf3_8", mat, dq, w)
                dq = dequant_fixed(w, sb, lloyd)
                rec("lloyd_global", mat, dq, w)
                row["lloyd_global"] = relrms(dq, w)
                dq, off = dequant_scmse(w, sb, lloyd)
                rec("lloyd_scaleopt", mat, dq, w)
                # collect refit hist under chosen scales
                s0 = torch.exp2(sb.double() - 127.0) * torch.exp2(off.double())
                h = u_hist(w, sb, s_override=s0)
                refit_hist = h if refit_hist is None else \
                    merge_hist(refit_hist, h)
                # per-expert lloyd (headroom)
                hv, hm = u_hist(w, sb)
                lut_pe = dp_lloyd(hv, hm, 8)
                dq = dequant_fixed(w, sb, lut_pe)
                rec("lloyd_perexp", mat, dq, w)
                per_expert.append(row)
            print(f"L{L:03d}_e{e:03d} done "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # one refit round for lloyd_scaleopt
    lloyd2 = dp_lloyd(refit_hist[0], refit_hist[1], 8)
    print(f"\nlloyd refit LUT (round 2): "
          f"{[round(x, 4) for x in lloyd2.tolist()]}", flush=True)
    for L in LAYERS:
        for e in EVAL_E:
            for mat in MATS:
                w, sb = load_matrix(L, e, mat)
                dq, _ = dequant_scmse(w, sb, lloyd2)
                rec("lloyd_scaleopt_r2", mat, dq, w)

    # ------------------------------------------------------------ report
    print(f"\n=== relRMS table (mean over {len(LAYERS) * len(EVAL_E)} experts"
          f", weight-space) ===")
    hdr = f"{'arm':20s}" + "".join(f"{m:>12s}{'ratio':>8s}" for m in MATS)
    print(hdr)
    result = {"luts": {"lloyd_global": lloyd.tolist(),
                       "lloyd_refit": lloyd2.tolist(),
                       "nf3": nf3.tolist(), "uniform8": uni8.tolist(),
                       "w3_current": W3_LUT.tolist()},
              "hist": {"vals": vals.tolist(), "mass_frac": frac.tolist()},
              "arms": {}, "per_expert": per_expert,
              "sample": {"layers": LAYERS, "eval_experts": EVAL_E,
                         "fit_experts": FIT_E}}
    for name, a in arms.items():
        line = f"{name:20s}"
        result["arms"][name] = {}
        for m in MATS:
            rel = float(np.mean(a[m]["rel"]))
            rat = float(np.mean(a[m]["ratio"]))
            result["arms"][name][m] = {
                "relrms_mean": rel, "relrms_max": float(np.max(a[m]["rel"])),
                "rms_ratio_mean": rat}
            line += f"{rel:12.5f}{rat:8.4f}"
        print(line, flush=True)

    with open(f"{OUT}/SHOOTOUT_RESULT.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {OUT}/SHOOTOUT_RESULT.json  "
          f"({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
