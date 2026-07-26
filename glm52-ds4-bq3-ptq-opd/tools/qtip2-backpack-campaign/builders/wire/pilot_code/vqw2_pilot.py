#!/usr/bin/env python3
"""VQ-W2 + ternary stage-1 weight-space pilot (PUBLIC_TASK) -- runs on compute-node-3.

Protocol: 3 layers {3,23,41} x 6 eval experts {9,50,100,150,200,254} x
{fused13, down}. Codebooks are fit PER-EXPERT PER-PROJECTION (codebook bytes
are counted in effective bpw -- they are tiny). Scales = the W2v2 SSE-refit
convention (per-block-32 UE8M0, offsets -4..+2 vs ckpt mxfp4 exponent), so VQ
operates in u-space on the same scale grid as the scalar anchor.

Arms per (layer, expert, projection):
  w2v2       : scalar 4-level LUT (dp_asym4_round2) + SSE scales  [ANCHOR 2.25]
  w2v2_gptq  : scalar GPTQ (actorder perm) on the W2v2 grid       [context 2.25]
  vqA_km     : VQ d=4 k=256, s^2-weighted k-means fit, nearest assign
  vqA_hgptq  : same codebook, Hessian-aware group-GPTQ assignment
  vqA_hkm_hg : diag(H)-weighted k-means fit + Hessian-aware assign (card arm A)
  vqB_rvq    : residual-VQ d=8, 2x256 stages + 2 alternating refinement rounds
  vqB_hgptq  : same codebooks, Hessian-aware group-GPTQ greedy 2-stage assign
  ternary    : global per-projection asym ternary {-a,0,+b} + SSE scales [1.85]
  ternary_pe : per-expert asym ternary levels + SSE scales             [1.85]

Down-proj Hessians/activations use A_fp = act(X, W13_fp) (teacher-side) for
ALL arms consistently -- documented pilot simplification vs the production
sequential (A_q) convention; comparisons between arms remain apples-to-apples.

Gate (card): any VQ arm at <= 2.3 bpw effective with relRMS <= 0.80x the
w2v2 anchor (recomputed on this same eval set) proceeds to stage 2.

Ledger: out/VQW2_LEDGER.jsonl (one row per unit; resume-safe).
Summary: out/VQW2_PILOT.json
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

MISSION = Path(os.environ.get("VQ_MISSION", str(Path.home() / "run-bundles/VQ_W2_PILOT")))
sys.path.insert(0, str(MISSION))
import gptqv2_pilot as gp  # noqa: E402  (reuses WtsBundle/w2v2_requant/gptq machinery)

gp.M = str(MISSION)  # WtsBundle and friends resolve gp.M at call time

DEV = "cuda"
SMOKE = os.environ.get("VQ_SMOKE", "0") == "1"
LAYERS = [3] if SMOKE else [3, 23, 41]
EVAL_E = [9] if SMOKE else [9, 50, 100, 150, 200, 254]
FIT_E = [17, 77, 177]
E = 256
LLOYD_ITERS = 6 if SMOKE else 15
KPP_SUB = 1 << 17
SEED = 0
T_OFFSETS = list(range(-4, 5))
MIN_FIT_ROWS = 64
OUT = MISSION / "out"
LEDGER = OUT / ("VQW2_LEDGER_SMOKE.jsonl" if SMOKE else "VQW2_LEDGER.jsonl")
SUMMARY = OUT / ("VQW2_PILOT_SMOKE.json" if SMOKE else "VQW2_PILOT.json")

ARMS = ["w2v2", "w2v2_gptq", "vqA_km", "vqA_hgptq", "vqA_hkm_hg",
        "vqA_sh", "vqA_sh_hg", "vqB_rvq", "vqB_hgptq",
        "vqB_flat", "vqB_flat_hg", "ternary", "ternary_pe", "ternary_gptq"]
VQ_ARMS = ["vqA_km", "vqA_hgptq", "vqA_hkm_hg", "vqA_sh", "vqA_sh_hg",
           "vqB_rvq", "vqB_hgptq", "vqB_flat", "vqB_flat_hg"]
H_ARMS = {"w2v2_gptq", "vqA_hgptq", "vqA_hkm_hg", "vqA_sh_hg",
          "vqB_hgptq", "vqB_flat_hg", "ternary_gptq"}
FLAT_K = 65536
FLAT_ITERS = 6 if SMOKE else 8
FLAT_SUB = 2_000_000

BUNDLES = {}
SHARED = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jrow(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def relrms(dq, W):
    return ((dq.float() - W.float()).norm() / (W.float().norm() + 1e-30)).item()


def nweights(proj):
    return 4096 * 4096 if proj == "fused13" else 4096 * 2048


def bpw_vq(proj, d, ks):
    code_bits = sum(math.log2(k) for k in ks) / d
    cb_bytes = sum(k * d * 2 for k in ks)  # fp16 codebook entries
    return code_bits + 0.25 + cb_bytes * 8.0 / nweights(proj)


def bpw_ternary(proj, cb_bytes):
    return 8.0 / 5.0 + 0.25 + cb_bytes * 8.0 / nweights(proj)  # 5 trits/byte


def bpw_vq_shared(proj, d, ks, share=256):
    code_bits = sum(math.log2(k) for k in ks) / d
    cb_bytes = sum(k * d * 2 for k in ks)
    return code_bits + 0.25 + cb_bytes * 8.0 / (share * nweights(proj))


def shared_cbs(L, proj, gen):
    """Layer-shared codebooks fit on FIT_E experts' u-vectors:
    (A) d=4 k=256 full-lloyd, (B) d=8 k=65536 flat on FLAT_SUB subsample."""
    key = (L, proj)
    if key in SHARED:
        return SHARED[key]
    t0 = time.time()
    us, ss = [], []
    for e in FIT_E:
        W, sb = get_mat(L, e, proj)
        _, sc = gp.w2v2_requant(W, sb)
        s_col = gp.sbytes_to_scol(sc)
        us.append(W / s_col)
        ss.append(s_col)
        del W, sb
    Uall = torch.cat(us, 0)
    Sall = torch.cat(ss, 0)
    del us, ss
    V4 = Uall.view(-1, 4)
    w4 = Sall.view(-1, 4)[:, 0].reshape(-1) ** 2
    cbA = lloyd(V4, w4, kmeanspp(V4, w4, 256, gen), LLOYD_ITERS, gen)
    del V4, w4
    V8 = Uall.view(-1, 8)
    w8 = Sall.view(-1, 8)[:, 0].reshape(-1) ** 2
    if V8.shape[0] > FLAT_SUB:
        ii = torch.randint(0, V8.shape[0], (FLAT_SUB,), device=DEV,
                           generator=gen)
        V8s, w8s = V8[ii], w8[ii]
    else:
        V8s, w8s = V8, w8
    cbB = lloyd(V8s, w8s, sample_init(V8s, w8s, FLAT_K, gen), FLAT_ITERS, gen)
    del V8, w8, V8s, w8s, Uall, Sall
    torch.cuda.empty_cache()
    SHARED[key] = (cbA, cbB)
    log(f"shared cbs L{L:03d} {proj} fit in {time.time() - t0:.0f}s")
    return SHARED[key]


def arm_metrics(dq, W, Xv, bpw):
    pv = gp.proxy_sym(Xv, dq, W) if (Xv is not None and Xv.shape[0] > 0) else None
    return {"relrms": round(relrms(dq, W), 6),
            "proxy_val": (round(pv, 6) if pv is not None else None),
            "bpw": round(bpw, 4)}


def load_caps(L, wins):
    xs, tks = [], []
    for gid in wins:
        d = torch.load(f"{gp.M}/cap/xmoe_L{L:03d}_win{gid:04d}.pt",
                       map_location="cpu")
        xs.append(d["x"])
        tks.append(d["topk"].to(torch.int64))
    x = torch.cat(xs, 0).to(DEV)
    tk = torch.cat(tks, 0).to(DEV)
    hit = torch.zeros(x.shape[0], E, dtype=torch.bool, device=DEV)
    hit.scatter_(1, tk, True)
    return x, hit


def get_mat(L, e, proj):
    b = BUNDLES[L]
    return b.fused13(e) if proj == "fused13" else b.down(e)


# ------------------------------------------------------------------ k-means
def assign_chunk(V, C, chunk=None):
    if chunk is None:
        chunk = max(2048, (1 << 27) // C.shape[0])
    out = torch.empty(V.shape[0], dtype=torch.long, device=V.device)
    c2 = (C * C).sum(1)
    for i in range(0, V.shape[0], chunk):
        v = V[i:i + chunk]
        d = v @ C.t()
        d.mul_(-2).add_(c2.unsqueeze(0))
        out[i:i + chunk] = d.argmin(1)
    return out


def chunked_err(V, C, asg, wts, chunk=1 << 19):
    out = torch.empty(V.shape[0], device=V.device)
    for i in range(0, V.shape[0], chunk):
        out[i:i + chunk] = ((V[i:i + chunk] - C[asg[i:i + chunk]]) ** 2
                            ).sum(1) * wts[i:i + chunk]
    return out


def wmeans(V, asg, wts, k, Cold):
    d = V.shape[1]
    Vs = torch.zeros(k, d, device=V.device)
    Ws = torch.zeros(k, device=V.device)
    Vs.index_add_(0, asg, V * wts.unsqueeze(1))
    Ws.index_add_(0, asg, wts)
    return torch.where(Ws.unsqueeze(1) > 0,
                       Vs / Ws.clamp_min(1e-12).unsqueeze(1), Cold)


def kmeanspp(V, wts, k, gen, sub=KPP_SUB):
    n = V.shape[0]
    ii = torch.multinomial(wts, min(sub, n), replacement=True, generator=gen)
    Vs, ws = V[ii], wts[ii]
    C = torch.empty(k, V.shape[1], device=V.device)
    j = torch.multinomial(ws, 1, generator=gen)
    C[0] = Vs[j[0]]
    d2 = ((Vs - C[0]) ** 2).sum(1)
    for i in range(1, k):
        p = (ws * d2).clamp_min(1e-30)
        j = torch.multinomial(p, 1, generator=gen)
        C[i] = Vs[j[0]]
        d2 = torch.minimum(d2, ((Vs - C[i]) ** 2).sum(1))
    return C


def sample_init(V, wts, k, gen):
    ii = torch.multinomial(wts.clamp_min(1e-30), k, replacement=False,
                           generator=gen)
    return V[ii].clone()


def lloyd(V, wts, C, iters, gen):
    k = C.shape[0]
    for _ in range(iters):
        asg = assign_chunk(V, C)
        Ws = torch.zeros(k, device=V.device)
        Ws.index_add_(0, asg, wts)
        C = wmeans(V, asg, wts, k, C)
        dead = Ws <= 0
        nd = int(dead.sum())
        if nd:
            err = chunked_err(V, C, asg, wts)
            C[dead] = V[err.topk(nd).indices]
    return C


def rvq_fit(V, wts, k, gen, iters=None, refine=2):
    iters = LLOYD_ITERS if iters is None else iters
    C1 = lloyd(V, wts, kmeanspp(V, wts, k, gen), iters, gen)
    a1 = assign_chunk(V, C1)
    R = V - C1[a1]
    C2 = lloyd(R, wts, kmeanspp(R, wts, k, gen), iters, gen)
    a2 = assign_chunk(R, C2)
    del R
    for _ in range(refine):
        t1 = V - C2[a2]
        a1 = assign_chunk(t1, C1)
        C1 = wmeans(t1, a1, wts, k, C1)
        del t1
        t2 = V - C1[a1]
        a2 = assign_chunk(t2, C2)
        C2 = wmeans(t2, a2, wts, k, C2)
        del t2
    return C1, C2, a1, a2


# ------------------------------------------------- Hessian-aware VQ (GPTQ)
def _best_codes(target, sg, CB, m2, chunk=None):
    """argmin_k  sg^2*|CB_k|^2 - 2*sg*<target, CB_k>  (row-chunked)."""
    if chunk is None:
        chunk = max(256, (1 << 26) // CB.shape[0])
    N = target.shape[0]
    out = torch.empty(N, dtype=torch.long, device=target.device)
    for i in range(0, N, chunk):
        t = target[i:i + chunk]
        s = sg[i:i + chunk].unsqueeze(1)
        score = (s * s) * m2.unsqueeze(0) - 2.0 * s * (t @ CB.t())
        out[i:i + chunk] = score.argmin(1)
    return out


def vq_gptq(W, s_col, cbs, U, d, blocksize=128):
    """Group-GPTQ with vector codes: no perm (groups must stay contiguous).
    cbs = [C1] or [C1, C2] (greedy residual stages). Code choice minimizes
    the U-transformed local error ||(wg - q) @ inv(U_gg)||; error propagates
    at group granularity exactly like scalar GPTQ's lazy-batch form.
    Returns (dq [N,K] original order, codes_stage1 [N, K//d])."""
    N, K = W.shape
    Wc = W.float().clone()
    s_all = s_col.float().clamp_min(1e-38)
    dq = torch.empty_like(Wc)
    codes0 = torch.empty(N, K // d, dtype=torch.long, device=W.device)
    eye = torch.eye(d, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = Wc[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        for g0 in range(0, i2 - i1, d):
            gsl = slice(g0, g0 + d)
            B = torch.linalg.solve_triangular(U1[gsl, gsl], eye, upper=True)
            wg = W1[:, gsl]
            sg = s_all[:, i1 + g0]
            R = wg @ B
            q_u = torch.zeros(N, d, device=W.device)
            for si, C in enumerate(cbs):
                CB = C @ B
                m2 = (CB * CB).sum(1)
                target = R if si == 0 else R - sg.unsqueeze(1) * (q_u @ B)
                kk = _best_codes(target, sg, CB, m2)
                if si == 0:
                    codes0[:, (i1 + g0) // d] = kk
                q_u = q_u + C[kk]
            q = sg.unsqueeze(1) * q_u
            dq[:, i1 + g0:i1 + g0 + d] = q
            Eg = (wg - q) @ B
            if g0 + d < i2 - i1:
                W1[:, g0 + d:] -= Eg @ U1[gsl, g0 + d:]
            E1[:, gsl] = Eg
        if i2 < K:
            Wc[:, i2:] -= E1 @ U[i1:i2, i2:]
    return dq, codes0


# ---------------------------------------------------------------- ternary
def requant_lut(w, sb, lut, offsets):
    lut_t = lut.float() if torch.is_tensor(lut) else torch.tensor(
        lut, dtype=torch.float32, device=DEV)
    mids = (lut_t[1:] + lut_t[:-1]) / 2
    N, K = w.shape
    KB = K // 32
    best_err = None
    best_off = None
    for off in offsets:
        s = torch.exp2(sb.float() - 127.0 + off)
        sf = s.repeat_interleave(32, dim=1)
        q = lut_t[torch.bucketize((w / sf).contiguous(), mids)]
        err2 = (q * sf - w).pow_(2).view(N, KB, 32).sum(2)
        if best_err is None:
            best_err = err2
            best_off = torch.full_like(err2, off, dtype=torch.int16)
        else:
            m = err2 < best_err
            best_err = torch.where(m, err2, best_err)
            best_off = torch.where(m, torch.full_like(best_off, off), best_off)
    sc = (sb.to(torch.int16) + best_off).clamp_(0, 254).to(torch.uint8)
    sf = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
    codes = torch.bucketize((w / sf).contiguous(), mids).to(torch.uint8)
    return codes, sc, lut_t


def fit_ternary(u, wts, iters=30):
    m = float((u.abs() * wts).sum() / wts.sum().clamp_min(1e-30))
    a = b = max(2.0 * m, 1e-3)
    for _ in range(iters):
        neg = u < (-a / 2.0)
        pos = u > (b / 2.0)
        wn = wts[neg]
        wp = wts[pos]
        if float(wn.sum()) > 0:
            a = float(-(u[neg] * wn).sum() / wn.sum())
        if float(wp.sum()) > 0:
            b = float((u[pos] * wp).sum() / wp.sum())
    return [-a, 0.0, b]


def sample_u(W, sb, lut, gen, cap):
    if lut is None:
        sc = sb
    else:
        _, sc, _ = requant_lut(W, sb, lut, T_OFFSETS)
    s_col = gp.sbytes_to_scol(sc)
    u = (W / s_col).reshape(-1)
    wts = (s_col * s_col).reshape(-1)
    if u.numel() > cap:
        ii = torch.randint(0, u.numel(), (cap,), device=DEV, generator=gen)
        u, wts = u[ii], wts[ii]
    return u, wts


def fit_ternary_global(proj, gen, cap=1_500_000):
    lut = None
    for rnd in range(2):
        us, ws = [], []
        for L in LAYERS:
            for e in FIT_E:
                W, sb = get_mat(L, e, proj)
                u, wts = sample_u(W, sb, lut, gen, cap)
                us.append(u)
                ws.append(wts)
                del W, sb
        lut = fit_ternary(torch.cat(us), torch.cat(ws))
        log(f"ternary {proj} round{rnd + 1} lut={[round(x, 5) for x in lut]}")
        del us, ws
    return lut


def fit_ternary_expert(W, sb, gen, cap=2_000_000):
    lut = None
    for _ in range(2):
        u, wts = sample_u(W, sb, lut, gen, cap)
        lut = fit_ternary(u, wts)
        del u, wts
    return lut


# ------------------------------------------------------------- unit runner
def scalar_gptq_lut(W, s_col, U_p, perm, lut):
    """gp.gptq_loop with a temporarily swapped LUT (gp.project reads module
    globals at call time)."""
    old_lut, old_mids = gp._LUT, gp._MIDS
    try:
        gp._LUT = lut
        gp._MIDS = (lut[1:] + lut[:-1]) / 2
        codes = gp.gptq_loop(W, s_col, U_p, perm)
        return lut[codes.long()]
    finally:
        gp._LUT, gp._MIDS = old_lut, old_mids


def run_unit(L, e, proj, W, sb, Xf, Xv, tluts, gen):
    te = time.time()
    N, K = W.shape
    res = {}
    c0, sc = gp.w2v2_requant(W, sb)
    s_col = gp.sbytes_to_scol(sc)
    dq = gp.deq(c0, s_col)
    res["w2v2"] = arm_metrics(dq, W, Xv, 2.25)
    del c0, dq

    have_h = Xf is not None and Xf.shape[0] >= MIN_FIT_ROWS
    U = None
    diagH = None
    perm = None
    U_p = None
    if have_h:
        H = Xf.t() @ Xf
        diagH = H.diagonal().clone()
        perm = gp.weight_perm(W)
        U_p = gp.gptq_prepare(H, perm)
        codes = gp.gptq_loop(W, s_col, U_p, perm)
        dq = gp.deq(codes, s_col)
        res["w2v2_gptq"] = arm_metrics(dq, W, Xv, 2.25)
        del codes, dq
        U = gp.gptq_prepare(H, torch.arange(K, device=DEV))
        del H
    else:
        res["w2v2_gptq"] = None

    u = W / s_col
    cbA_sh, cbB_sh = shared_cbs(L, proj, gen)

    # ---- arm A (per-expert): d=4 k=256
    V4 = u.view(N, K // 4, 4).reshape(-1, 4)
    s4 = s_col.view(N, K // 4, 4)[:, :, 0].reshape(-1)
    w4 = s4 * s4
    cbA = lloyd(V4, w4, kmeanspp(V4, w4, 256, gen), LLOYD_ITERS, gen)
    a = assign_chunk(V4, cbA)
    dq = cbA[a].view(N, K // 4, 4).reshape(N, K) * s_col
    res["vqA_km"] = arm_metrics(dq, W, Xv, bpw_vq(proj, 4, [256]))
    del a, dq
    # ---- arm A (layer-shared codebook)
    a = assign_chunk(V4, cbA_sh)
    dq = cbA_sh[a].view(N, K // 4, 4).reshape(N, K) * s_col
    res["vqA_sh"] = arm_metrics(dq, W, Xv, bpw_vq_shared(proj, 4, [256]))
    del a, dq
    if have_h:
        dq, _ = vq_gptq(W, s_col, [cbA], U, 4)
        res["vqA_hgptq"] = arm_metrics(dq, W, Xv, bpw_vq(proj, 4, [256]))
        del dq
        hg = diagH.view(K // 4, 4).mean(1).clamp_min(1e-12)
        wh4 = w4 * hg.unsqueeze(0).expand(N, -1).reshape(-1)
        cbAh = lloyd(V4, wh4, kmeanspp(V4, wh4, 256, gen), LLOYD_ITERS, gen)
        dq, _ = vq_gptq(W, s_col, [cbAh], U, 4)
        res["vqA_hkm_hg"] = arm_metrics(dq, W, Xv, bpw_vq(proj, 4, [256]))
        del dq, cbAh, wh4, hg
        dq, _ = vq_gptq(W, s_col, [cbA_sh], U, 4)
        res["vqA_sh_hg"] = arm_metrics(dq, W, Xv, bpw_vq_shared(proj, 4, [256]))
        del dq
    else:
        res["vqA_hgptq"] = None
        res["vqA_hkm_hg"] = None
        res["vqA_sh_hg"] = None
    del V4, s4, w4, cbA

    # ---- arm B (per-expert): RVQ d=8, 2x256
    V8 = u.view(N, K // 8, 8).reshape(-1, 8)
    s8v = s_col.view(N, K // 8, 8)[:, :, 0].reshape(-1)
    w8 = s8v * s8v
    C1, C2, a1, a2 = rvq_fit(V8, w8, 256, gen)
    dq = (C1[a1] + C2[a2]).view(N, K // 8, 8).reshape(N, K) * s_col
    res["vqB_rvq"] = arm_metrics(dq, W, Xv, bpw_vq(proj, 8, [256, 256]))
    del dq, a1, a2
    if have_h:
        dq, _ = vq_gptq(W, s_col, [C1, C2], U, 8)
        res["vqB_hgptq"] = arm_metrics(dq, W, Xv, bpw_vq(proj, 8, [256, 256]))
        del dq
    else:
        res["vqB_hgptq"] = None
    del C1, C2
    # ---- arm B (layer-shared FLAT d=8 k=65536)
    a = assign_chunk(V8, cbB_sh)
    dq = cbB_sh[a].view(N, K // 8, 8).reshape(N, K) * s_col
    res["vqB_flat"] = arm_metrics(dq, W, Xv,
                                  bpw_vq_shared(proj, 8, [FLAT_K]))
    del a, dq
    if have_h:
        dq, _ = vq_gptq(W, s_col, [cbB_sh], U, 8)
        res["vqB_flat_hg"] = arm_metrics(dq, W, Xv,
                                         bpw_vq_shared(proj, 8, [FLAT_K]))
        del dq
    else:
        res["vqB_flat_hg"] = None
    del V8, s8v, w8, u
    if U is not None:
        del U

    # ---- arm C: ternary (1.85 bpw rung)
    codes, sc3, lut_t = requant_lut(W, sb, tluts[proj], T_OFFSETS)
    s3_col = gp.sbytes_to_scol(sc3)
    dq = lut_t[codes.long()] * s3_col
    res["ternary"] = arm_metrics(dq, W, Xv, bpw_ternary(proj, 0))
    del codes, dq
    lut_pe = fit_ternary_expert(W, sb, gen)
    codes, scpe, lut_tpe = requant_lut(W, sb, lut_pe, T_OFFSETS)
    dq = lut_tpe[codes.long()] * gp.sbytes_to_scol(scpe)
    res["ternary_pe"] = arm_metrics(dq, W, Xv, bpw_ternary(proj, 8))
    res["ternary_pe"]["lut"] = [round(x, 5) for x in lut_pe]
    del codes, scpe, dq
    if have_h:
        dq = scalar_gptq_lut(W, s3_col, U_p, perm, lut_t)
        dq = dq * s3_col
        res["ternary_gptq"] = arm_metrics(dq, W, Xv, bpw_ternary(proj, 0))
        del dq, U_p, perm
    else:
        res["ternary_gptq"] = None
    del s3_col

    row = {"unit": f"L{L:03d}_e{e:03d}_{proj}", "layer": L, "expert": e,
           "proj": proj,
           "n_fit_rows": int(Xf.shape[0]) if Xf is not None else 0,
           "n_val_rows": int(Xv.shape[0]) if Xv is not None else 0,
           "arms": res, "secs": round(time.time() - te, 1)}
    jrow(LEDGER, row)
    msg = " ".join(f"{k}={v['relrms']:.4f}" for k, v in res.items() if v)
    log(f"{row['unit']} {row['secs']}s {msg}")


# ------------------------------------------------------------ identity gate
def identity_gate():
    L, e = LAYERS[0], FIT_E[0]
    W, sb = get_mat(L, e, "down")
    _, sc = gp.w2v2_requant(W, sb)
    s_col = gp.sbytes_to_scol(sc)
    g = torch.Generator(device=DEV)
    g.manual_seed(1)
    X = torch.randn(512, W.shape[1], device=DEV, generator=g)
    H = X.t() @ X
    ar = torch.arange(W.shape[1], device=DEV)
    U = gp.gptq_prepare(H, ar)
    codes_ref = gp.gptq_loop(W, s_col, U, ar)
    cb = gp._LUT.unsqueeze(1)
    _, codes_mine = vq_gptq(W, s_col, [cb], U, 1)
    mism = (codes_mine.reshape(-1) != codes_ref.reshape(-1).long()
            ).float().mean().item()
    log(f"identity gate d=1 vs scalar gptq_loop: code mismatch frac={mism:.2e}")
    if mism > 1e-4:
        raise SystemExit(f"IDENTITY GATE FAIL mism={mism}")
    del W, sb, X, H, U, codes_ref, codes_mine


# ----------------------------------------------------------------- summary
def summarize(elapsed):
    rows = [json.loads(x) for x in open(LEDGER)]
    per = {}
    for arm in ARMS:
        vals = {"fused13": [], "down": []}
        bp = {"fused13": [], "down": []}
        pv = []
        for r in rows:
            a = r["arms"].get(arm)
            if a and a.get("relrms") is not None:
                vals[r["proj"]].append(a["relrms"])
                bp[r["proj"]].append(a["bpw"])
                if a.get("proxy_val") is not None:
                    pv.append(a["proxy_val"])
        allv = vals["fused13"] + vals["down"]
        if not allv:
            continue
        per[arm] = {
            "fused13_mean": round(float(np.mean(vals["fused13"])), 6) if vals["fused13"] else None,
            "down_mean": round(float(np.mean(vals["down"])), 6) if vals["down"] else None,
            "all_mean": round(float(np.mean(allv)), 6),
            "proxy_val_mean": round(float(np.mean(pv)), 6) if pv else None,
            "n": len(allv),
            "bpw_fused13": round(float(np.mean(bp["fused13"])), 4) if bp["fused13"] else None,
            "bpw_down": round(float(np.mean(bp["down"])), 4) if bp["down"] else None,
        }
    anchor = per["w2v2"]["all_mean"]
    anchor_g = per.get("w2v2_gptq", {}).get("proxy_val_mean")
    for arm in per:
        per[arm]["ratio_vs_w2v2"] = round(per[arm]["all_mean"] / anchor, 6)
        pv = per[arm].get("proxy_val_mean")
        per[arm]["proxy_ratio_vs_gptq"] = (
            round(pv / anchor_g, 6) if (pv is not None and anchor_g) else None)
    gate = {}
    for arm in VQ_ARMS:
        if arm not in per:
            continue
        mx = max(x for x in (per[arm]["bpw_fused13"], per[arm]["bpw_down"])
                 if x is not None)
        if arm in H_ARMS:
            r = per[arm]["proxy_ratio_vs_gptq"]
            gate[arm] = {"rail": "gptq_proxy_val", "ratio": r,
                         "bpw_max": mx,
                         "pass": bool(r is not None and r <= 0.80
                                      and mx <= 2.3)}
        else:
            r = per[arm]["ratio_vs_w2v2"]
            gate[arm] = {"rail": "rtn_relrms", "ratio": r, "bpw_max": mx,
                         "pass": bool(r <= 0.80 and mx <= 2.3)}
    result = {
        "task": "PUBLIC_TASK",
        "protocol": f"{len(LAYERS)} layers {LAYERS} x {len(EVAL_E)} eval experts "
                    f"{EVAL_E} x fused13/down; per-expert + layer-shared "
                    f"codebooks; W2v2 SSE scales; fit experts {FIT_E}",
        "anchor_note": "w2v2 = dp_asym4_round2 LUT + SSE scales recomputed on this "
                       "eval set (PUBLIC_TASK shootout: 0.349736 on L0/21/42)",
        "down_H_note": "down-proj H/activations use A_fp=act(X,W13_fp) for all arms",
        "rail_note": "GPTQ-class arms (incl. Hessian-aware VQ) optimize activation "
                     "error, not weight recon -- they gate on val-proxy ratio vs "
                     "w2v2_gptq; RTN-class arms gate on relRMS ratio vs w2v2",
        "seed": SEED, "lloyd_iters": LLOYD_ITERS, "flat_k": FLAT_K,
        "arms": per,
        "gate_rule": "ratio <= 0.80 on own rail at <= 2.3 bpw effective",
        "gate": gate,
        "gate_pass_any": bool(any(g["pass"] for g in gate.values())),
        "ternary_note": "ternary arms are 1.85 bpw knapsack-cold-tail rungs; "
                        "compare vs the W2v2-to-zero line, not the 0.80 gate",
        "elapsed_s": round(elapsed, 1),
    }
    tmp = SUMMARY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, SUMMARY)
    log(f"wrote {SUMMARY}")
    for arm in ARMS:
        if arm in per:
            p = per[arm]
            log(f"{arm:12s} all={p['all_mean']:.6f} ratio={p['ratio_vs_w2v2']:.4f} "
                f"pvr={p['proxy_ratio_vs_gptq']} "
                f"f13={p['fused13_mean']} down={p['down_mean']} "
                f"bpw={p['bpw_fused13']}/{p['bpw_down']}")
    log(f"GATE: { {k: v['pass'] for k, v in gate.items()} } "
        f"pass_any={result['gate_pass_any']}")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)
    sel = json.load(open(f"{gp.M}/static/CALIB_SELECTION.json"))
    done = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            try:
                done.add(json.loads(line)["unit"])
            except Exception:
                pass
    log(f"smoke={SMOKE} resume: {len(done)} units already in ledger")

    for L in LAYERS:
        BUNDLES[L] = gp.WtsBundle(L)
        log(f"bundle L{L:03d} loaded")

    identity_gate()

    tpath = OUT / "TERNARY_LUTS.json"
    if tpath.exists():
        tluts = json.loads(tpath.read_text())
        log(f"ternary luts loaded: {tluts}")
    else:
        tluts = {p: fit_ternary_global(p, gen) for p in ("fused13", "down")}
        tpath.write_text(json.dumps(tluts, indent=1))

    for L in LAYERS:
        need = [e for e in EVAL_E
                if not {f"L{L:03d}_e{e:03d}_fused13",
                        f"L{L:03d}_e{e:03d}_down"} <= done]
        if not need:
            log(f"L{L:03d} all units done, skip caps")
            continue
        xf, hitf = load_caps(L, sel["fit_ids"])
        xv, hitv = load_caps(L, sel["val_ids"])
        log(f"L{L:03d} caps: fit {tuple(xf.shape)} val {tuple(xv.shape)}")
        for e in need:
            if gp.STOP.is_set():
                log("graceful stop")
                return 1
            Xf = xf[hitf[:, e]].float()
            Xv = xv[hitv[:, e]].float()
            W13, sb13 = BUNDLES[L].fused13(e)
            if f"L{L:03d}_e{e:03d}_fused13" not in done:
                run_unit(L, e, "fused13", W13, sb13, Xf, Xv, tluts, gen)
            if f"L{L:03d}_e{e:03d}_down" not in done:
                Wd, sb2 = BUNDLES[L].down(e)
                Af = gp.act(Xf, W13[:2048], W13[2048:])
                Av = gp.act(Xv, W13[:2048], W13[2048:])
                run_unit(L, e, "down", Wd, sb2, Af, Av, tluts, gen)
                del Wd, sb2, Af, Av
            del W13, sb13, Xf, Xv
            torch.cuda.empty_cache()
        del xf, hitf, xv, hitv
        torch.cuda.empty_cache()

    summarize(time.time() - t0)
    (OUT / "PILOT_DONE").write_text("done\n")
    log(f"ALL DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
