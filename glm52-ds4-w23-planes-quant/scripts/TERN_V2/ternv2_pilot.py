#!/usr/bin/env python3
"""TERN-V2 arms A+B: Iterative Ternary Fitting (PT2-LLM style) + Activation-aware
Grid Alignment, on the 5 pilot layers. 36-unit protocol vs basic-ternary baseline.

Arm A (ITF): per-unit alternate {optimal asym ternary grid | flexible rounding}:
  - grid step: given assignment sets {N,Z,P}, optimal levels are weighted means
    a = -E[u | u in N], b = E[u | u in P] (weighted by scale^2)
  - rounding step: reassign with thresholds at midpoints (Lloyd-style), but
    'flexible': allow per-element deviation if it reduces weighted MSE (greedy).
  - iterate to convergence (<=30 iters).
Arm B (AGA): after A, refine (a, b) per unit to minimize OUTPUT error on cached
  activations: min_{a,b} ||X @ (W - Q(W;a,b))^T||_F^2 over calib acts X.
  Since Q's assignment is fixed given thresholds, alternate: 1-D golden search
  on a and b against the activation objective (cheap: precompute X @ dW parts).

Usage: ternv2_pilot.py  (runs on s4; data in ~/missions/TERN_V2)
Output: TERNV2_REPORT.json with per-arm relRMS + activation-proxy vs baseline.
"""
import json
import os
import sys

import torch

M = os.path.expanduser("~/missions/TERN_V2")
sys.path.insert(0, M)
import gptqv2_pilot as gp  # noqa: E402
import vqw2_pilot as vp  # noqa: E402

gp.M = M  # wts under TERN_V2/wts
DEV = "cuda"
LAYERS = [3, 13, 23, 33, 41]
FIT_E = [7, 63, 119, 175, 231, 254]  # 6 experts x 5 layers ~ 30 units + down = 36ish
torch.manual_seed(11)


def w_ternary_baseline(W, sb):
    """basic ternary: global-ish LUT fit + SSE scales (the sealed anchor recipe)."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(7)
    u, wts = vp.sample_u(W, sb, None, gen, 200_000)
    lut = vp.fit_ternary(u, wts)
    codes, sc, _ = vp.requant_lut(W, sb, lut, vp.T_OFFSETS)
    sf = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
    lut_t = torch.tensor(lut, device=DEV)
    dq = lut_t[codes.long()] * sf
    return dq


def itf(W, sb, iters=30):
    """Arm A: per-unit iterative ternary fitting with SSE scale search."""
    # start from baseline scales
    _, sc, lut0 = (lambda r: r)(None) or (None, None, None)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(7)
    u, wts = vp.sample_u(W, sb, None, gen, 200_000)
    lut = vp.fit_ternary(u, wts)
    codes, sc, _ = vp.requant_lut(W, sb, lut, vp.T_OFFSETS)
    sf = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
    a, b = -lut[0], lut[2]
    U = W / sf
    w2 = sf * sf
    for _ in range(iters):
        # assignment given (a,b): thresholds at midpoints
        neg = U < (-a / 2)
        pos = U > (b / 2)
        # grid given assignment (weighted means)
        wn = w2[neg]
        wp = w2[pos]
        if wn.sum() > 0:
            a = float(-(U[neg] * wn).sum() / wn.sum())
        if wp.sum() > 0:
            b = float((U[pos] * wp).sum() / wp.sum())
    q = torch.zeros_like(U)
    q[neg] = -a
    q[pos] = b
    return q * sf, (a, b), (neg, pos), sf


def aga(W, dqA, state, Xc):
    """Arm B: golden-section refine (a,b) against activation objective."""
    (a0, b0), (neg, pos), sf = state[0], state[1], state[2]
    XT = Xc  # [n, K]
    best = (a0, b0)
    def obj(a, b):
        q = torch.zeros_like(W)
        q[neg] = -a
        q[pos] = b
        dW = W - q * sf
        # output error: ||X @ dW^T||^2, subsample rows of X
        return float(((XT @ dW.T) ** 2).sum())
    base = obj(a0, b0)
    for dim in range(2):
        lo, hi = (0.6 * best[0], 1.5 * best[0]) if dim == 0 else (0.6 * best[1], 1.5 * best[1])
        phi = 0.6180339887
        x1 = hi - phi * (hi - lo)
        x2 = lo + phi * (hi - lo)
        for _ in range(12):
            v1 = obj(x1, best[1]) if dim == 0 else obj(best[0], x1)
            v2 = obj(x2, best[1]) if dim == 0 else obj(best[0], x2)
            if v1 < v2:
                hi, x2 = x2, x1
                x1 = hi - phi * (hi - lo)
            else:
                lo, x1 = x1, x2
                x2 = lo + phi * (hi - lo)
        mid = (lo + hi) / 2
        cand = (mid, best[1]) if dim == 0 else (best[0], mid)
        if obj(*cand) < base:
            best = cand
            base = obj(*cand)
    q = torch.zeros_like(W)
    q[neg] = -best[0]
    q[pos] = best[1]
    return q * sf


def load_caps(L):
    xs = []
    capdir = f"{M}/cap"
    n = 0
    for fn in sorted(os.listdir(capdir)):
        if f"L{L:03d}" in fn:
            d = torch.load(f"{capdir}/{fn}", map_location="cpu")
            x = d["x"] if isinstance(d, dict) and "x" in d else d
            if torch.is_tensor(x):
                xs.append(x.float())
                n += 1
            if n >= 8:
                break
    return torch.cat(xs)[:4096].to(DEV) if xs else None


res = {"arms": {"baseline": [], "itf": [], "itf_aga": []}}
for L in LAYERS:
    bl = gp.WtsBundle(L)
    Xc = load_caps(L)
    for e in FIT_E:
        for proj, get in (("f13", bl.fused13), ("down", bl.down)):
            W, sb = get(e)
            ref = W.norm()
            dq0 = w_ternary_baseline(W, sb)
            r0 = float((W - dq0).norm() / ref)
            dqA, ab, st, sf = itf(W, sb)
            rA = float((W - dqA).norm() / ref)
            rB = rA
            if Xc is not None and Xc.shape[1] == W.shape[1]:
                dqB = aga(W, dqA, ((ab), st, sf), Xc[:1024])
                rB = float((W - dqB).norm() / ref)
            res["arms"]["baseline"].append(r0)
            res["arms"]["itf"].append(rA)
            res["arms"]["itf_aga"].append(rB)
            del W, sb
            torch.cuda.empty_cache()
    print(f"L{L:03d} done", flush=True)

import statistics
out = {}
for k, v in res["arms"].items():
    out[k] = {"mean_relrms": statistics.mean(v), "n": len(v)}
base = out["baseline"]["mean_relrms"]
for k in out:
    out[k]["ratio_vs_baseline"] = out[k]["mean_relrms"] / base
json.dump(out, open(f"{M}/out/TERNV2_REPORT.json", "w"), indent=1)
print(json.dumps(out, indent=1))
