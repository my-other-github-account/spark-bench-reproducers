#!/usr/bin/env python3
"""GPTQv2/GPTAQ asymmetric-error pilot vs standard GPTQ (PUBLIC_TASK, compute-node-1).

Card: implement the GPTQv2 asymmetric-error delta (arXiv 2504.02692, GPTAQ)
in a COPY of the G4X/ds4_gptq column loop; pilot on layers 3/13/23/33/41 on
the W2v2 grid vs the standard solve, identical fit/val split (PUBLIC_TASK
CALIB_SELECTION.json), metric = val relRMS + weight-space recon.

Grid (W2v2, PUBLIC_TASK shootout winner, gate PASS 0.9198x):
  LUT dp_asym4_round2 [-3.5111107, -1.1800192, 0.6510809, 2.7868641]
  scales = per-block-32 UE8M0, exact SSE search offsets -4..+2 vs the ckpt
  mxfp4 exponent (w3v2_rebuild.requant_chunk convention, recomputed here
  per expert since the W2v2 plane bank is still being built on s8/s2).
  Scales are STATIC during GPTQ (wire stays g_idx-free) -- same protocol
  as ds4_gptq.py / ds4_gptq_w3v2.py.

GPTQv2 delta (GPTAQ fasterquant, mirrored exactly):
  dXXT = (X_fp - X_q)^T X_q        [K,K], same normalization as H = X_q^T X_q
  P    = alpha * triu(dXXT_perm @ U^T, 1) @ U   (U = upper-chol of Hinv)
  column update:  W1[:, j+1:] -= err (x) U1[j, j+1:]  -  w_cur (x) P1[j, j+1:]
  block update:   W[:, i2:]  -= E1 @ U[b, tail]      -  Q1 @ P[b, tail]
  (Q1 = quantized values of the finished block, matching the reference where
   in-block self-updates leave W1 == quantized values.)

ASYMMETRY SURFACE (documented deviation): the calib caps are TEACHER-graph
MoE inputs, so for fused13 X_fp == X_q => dXXT = 0 => v2 == v1 identically.
The pilot delta therefore lives in down_proj: A_fp = act(X, W13_fp) vs
A_q = act(X, W13_shipped-quant). v1 solves down on (A_q, A_q); v2 matches
A_q @ Wd_q^T to A_fp @ Wd_fp^T -- exactly GPTAQ's sequential asymmetric
calibration, using only single-pass teacher caps.

Arms per expert (down_proj): rtn (W2v2 requant), gptq_v1, gptq_v2@alpha for
alpha in GV2_ALPHAS. Metrics per arm: weight recon relRMS, sym val/fit
proxy (|| A_q_val (dq-W)^T || / || A_q_val W^T ||, the historical form) and
asym val/fit proxy (|| A_q_val dq^T - A_fp_val W^T || / || A_fp_val W^T ||,
the GPTAQ objective). Plus adrift = ||A_q - A_fp||/||A_fp|| (surface size).

Identity gate: expert 0 of every layer also runs the v2 loop with P=0 and
asserts codes are bitwise-equal to the v1 loop (the ~20-line delta is
provably neutral at alpha=0).

No plane emission -- offline metrics only. Ledger: out/PILOT_LEDGER.jsonl.
"""
import gc
import json
import os
import signal
import sys
import threading
import time

import numpy as np
import torch

M = os.path.expanduser("$HOME/run-bundles/DS4_GPTQV2_PILOT")
DEV = "cuda"
LAYER = int(os.environ.get("PL_LAYER", "3"))
BLOCKSIZE = int(os.environ.get("PL_BLOCKSIZE", "128"))
PERCDAMP = float(os.environ.get("PL_PERCDAMP", "0.01"))
VAL_MARGIN = float(os.environ.get("PL_VAL_MARGIN", "0.02"))
MIN_FIT_ROWS = int(os.environ.get("PL_MIN_FIT_ROWS", "64"))
MIN_VAL_ROWS = int(os.environ.get("PL_MIN_VAL_ROWS", "32"))
MAX_EXPERTS = int(os.environ.get("PL_MAX_EXPERTS", "0"))  # smoke; 0=all
ALPHAS = [float(a) for a in
          os.environ.get("GV2_ALPHAS", "0.25,1.0").split(",")]

CAP = f"{M}/cap"
OUT = f"{M}/out"
LEDGER = f"{OUT}/PILOT_LEDGER.jsonl"
E, N13, K13, N2, K2 = 256, 4096, 4096, 4096, 2048
TOPK = 6

# W2v2 winner grid (W2V2_SHOOTOUT.json luts.dp_asym4_round2, PUBLIC_TASK)
W2V2_LUT = [-3.5111107379486137, -1.1800192351581362,
            0.6510809470728273, 2.7868641002011136]
_LUT = torch.tensor(W2V2_LUT, dtype=torch.float32, device=DEV)
_MIDS = (_LUT[1:] + _LUT[:-1]) / 2
OFFSETS = list(range(-4, 3))  # SSE scale-refit search, w3v2_rebuild conv.

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32, device=DEV)

SWIGLU_LIMIT = 10.0  # DeepSeek-V4-Flash config.json swiglu_limit (verified)

STOP = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: STOP.set())
signal.signal(signal.SIGINT, lambda *a: STOP.set())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] L{LAYER:03d} {m}", flush=True)


def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ------------------------------------------------------------ weight access
def nibbles(packed):
    lo = packed & 0xF
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


class WtsBundle:
    """experts_L{N}.pt: {w1,w3,w2}_{w,s} u8 stacks from stage_extract_wts."""

    def __init__(self, L):
        self.b = torch.load(f"{M}/wts/experts_L{L:03d}.pt",
                            map_location="cpu")

    def fused13(self, e):
        """-> (W_fp f32 [4096,4096] DEV, sb_ckpt u8 [4096,128] DEV)."""
        ws, ss = [], []
        for wname in ("w1", "w3"):
            wp = self.b[wname + "_w"][e].to(DEV)
            sb = self.b[wname + "_s"][e].to(DEV)
            nib = nibbles(wp)
            w = _E2M1[nib.long()]
            w = w * torch.exp2(sb.float() - 127.0).repeat_interleave(
                32, dim=1)
            ws.append(w)
            ss.append(sb)
        return torch.cat(ws, 0), torch.cat(ss, 0)

    def down(self, e):
        wp = self.b["w2_w"][e].to(DEV)
        sb = self.b["w2_s"][e].to(DEV)
        nib = nibbles(wp)
        w = _E2M1[nib.long()]
        w = w * torch.exp2(sb.float() - 127.0).repeat_interleave(32, dim=1)
        return w, sb


# --------------------------------------------- W2v2 RTN requant (SSE refit)
def w2v2_requant(w, sb):
    """w f32 [N,K] DEV, sb u8 [N,KB] DEV -> codes u8 [N,K], sc u8 [N,KB].
    Exact w3v2_rebuild.requant_chunk convention (n_levels=4)."""
    N, K = w.shape
    KB = K // 32
    best_err = None
    best_off = None
    for off in OFFSETS:
        s = torch.exp2(sb.float() - 127.0 + off)
        sf = s.repeat_interleave(32, dim=1)
        u = w / sf
        q = _LUT[torch.bucketize(u.contiguous(), _MIDS)]
        err2 = (q * sf - w).pow_(2).view(N, KB, 32).sum(dim=2)
        if best_err is None:
            best_err = err2
            best_off = torch.full_like(err2, off, dtype=torch.int16)
        else:
            m = err2 < best_err
            best_err = torch.where(m, err2, best_err)
            best_off = torch.where(
                m, torch.full_like(best_off, off), best_off)
    sc = (sb.to(torch.int16) + best_off).clamp_(0, 254).to(torch.uint8)
    sf = torch.exp2(sc.float() - 127.0).repeat_interleave(32, dim=1)
    codes = torch.bucketize((w / sf).contiguous(), _MIDS).to(torch.uint8)
    return codes, sc


def sbytes_to_scol(sbytes):
    return torch.exp2(sbytes.to(DEV).float() - 127.0).repeat_interleave(
        32, dim=1).clamp_min(1e-38)


def deq(codes, s_col):
    return _LUT[codes.to(DEV).long()] * s_col


def project(u):
    codes = torch.bucketize(u.contiguous(), _MIDS)
    return codes.to(torch.uint8), _LUT[codes]


# ----------------------------------------------------------------- GPTQ
def weight_perm(W):
    return torch.argsort(W.float().pow(2).sum(0), descending=True)


def gptq_prepare(H, perm, percdamp=PERCDAMP):
    """ds4_gptq.gptq_prepare verbatim: -> U = upper-chol of Hinv (permuted)."""
    K = H.shape[0]
    dev = H.device
    H = H.clone()
    diag = torch.arange(K, device=dev)
    dead = H[diag, diag] <= 0
    H[diag[dead], diag[dead]] = 1.0
    damp = percdamp * H[diag, diag].mean()
    H[diag, diag] += damp
    H = H[perm][:, perm]
    for boost in range(6):
        try:
            L_ = torch.linalg.cholesky(H)
            break
        except Exception:
            extra = (percdamp * (2 ** (boost + 1))) * H[diag, diag].mean()
            H[diag, diag] += extra
            if boost == 5:
                raise
    Hinv = torch.cholesky_inverse(L_)
    del L_
    return torch.linalg.cholesky(Hinv, upper=True)


def gv2_P(dXXT, U, perm, alpha):
    """GPTAQ P-matrix: alpha * triu(dXXT_perm @ U^T, 1) @ U (strictly upper).
    dXXT in ORIGINAL column order; permuted here to match U."""
    dp = dXXT[perm][:, perm]
    return alpha * ((dp @ U.t()).triu_(diagonal=1) @ U)


def gptq_loop(W, s_col, Hinv, perm, P=None, blocksize=BLOCKSIZE):
    """ds4_gptq.gptq_loop column loop verbatim + optional GPTQv2 P-term.
    P=None => bitwise-standard GPTQ. P (permuted order, strictly upper)
    adds the GPTAQ asymmetric-error correction. Returns codes u8 [N,K]
    in ORIGINAL column order."""
    N, K = W.shape
    W = W.float().clone()[:, perm]
    s_col = s_col.float().clamp_min(1e-38)[:, perm]
    C = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        Q1 = torch.zeros_like(W1)                        # v2: quantized vals
        U1 = Hinv[i1:i2, i1:i2]
        P1 = P[i1:i2, i1:i2] if P is not None else None  # v2
        for j in range(cnt):
            w = W1[:, j]
            d = U1[j, j]
            s = s_col[:, i1 + j]
            cj, vj = project(w / s)
            C[:, i1 + j] = cj
            q = vj * s
            Q1[:, j] = q                                 # v2
            err = (w - q) / d
            if j + 1 < cnt:
                W1[:, j + 1:] -= err.unsqueeze(1) * U1[j, j + 1:].unsqueeze(0)
                if P1 is not None:                       # v2 (GPTAQ line 123)
                    W1[:, j + 1:] += w.unsqueeze(1) * P1[j, j + 1:].unsqueeze(0)
            E1[:, j] = err
        if i2 < K:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
            if P is not None:                            # v2 (GPTAQ line 129)
                W[:, i2:] += Q1 @ P[i1:i2, i2:]
    del W1, E1, Q1
    inv = torch.argsort(perm)
    return C[:, inv]


# ----------------------------------------------------------------- metrics
def proxy_sym(X, dq, W):
    """|| X (dq-W)^T ||_F / || X W^T ||_F (historical val-gate form)."""
    if X is None or X.shape[0] == 0:
        return None
    d = dq.float() - W.float()
    num = (X.float() @ d.t()).norm()
    den = (X.float() @ W.float().t()).norm() + 1e-30
    return (num / den).item()


def proxy_asym(Xq, Xfp, dq, W):
    """|| Xq dq^T - Xfp W^T ||_F / || Xfp W^T ||_F (GPTAQ objective)."""
    if Xq is None or Xq.shape[0] == 0:
        return None
    ref = Xfp.float() @ W.float().t()
    num = (Xq.float() @ dq.float().t() - ref).norm()
    den = ref.norm() + 1e-30
    return (num / den).item()


def wrec(dq, W):
    return ((dq.float() - W.float()).norm() / (W.float().norm() + 1e-30)
            ).item()


def act(X, Wg, Wu):
    g = (X @ Wg.t()).clamp(max=SWIGLU_LIMIT)
    u = (X @ Wu.t()).clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    return torch.nn.functional.silu(g) * u


# ----------------------------------------------------------------- solve
def load_caps(wins):
    xs, tks = [], []
    for gid in wins:
        d = torch.load(f"{CAP}/xmoe_L{LAYER:03d}_win{gid:04d}.pt",
                       map_location="cpu")
        xs.append(d["x"])
        tks.append(d["topk"].to(torch.int64))
    x = torch.cat(xs, 0).to(DEV)
    tk = torch.cat(tks, 0).to(DEV)
    hit = torch.zeros(x.shape[0], E, dtype=torch.bool, device=DEV)
    hit.scatter_(1, tk, True)
    return x, hit


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    done_marker = f"{OUT}/LAYER_{LAYER:03d}_DONE"
    if os.path.exists(done_marker):
        log("done marker exists, skip layer")
        return 0

    sel = json.load(open(f"{M}/static/CALIB_SELECTION.json"))
    xf, hitf = load_caps(sel["fit_ids"])
    xv, hitv = load_caps(sel["val_ids"])
    log(f"caps loaded: fit {tuple(xf.shape)} val {tuple(xv.shape)} "
        f"alphas={ALPHAS}")
    src = WtsBundle(LAYER)
    EE = E if not MAX_EXPERTS else min(E, MAX_EXPERTS)

    n_done = 0
    for e in range(EE):
        if STOP.is_set():
            log(f"graceful stop at expert {e}")
            return 1
        te = time.time()
        Xf = xf[hitf[:, e]].float()
        Xv = xv[hitv[:, e]].float()
        n_fit, n_val = int(Xf.shape[0]), int(Xv.shape[0])
        row = dict(unit=f"L{LAYER:03d}_e{e:03d}", layer=LAYER, expert=e,
                   n_fit=n_fit, n_val=n_val)
        if n_fit < MIN_FIT_ROWS or n_val < MIN_VAL_ROWS:
            row["skip"] = "floor"
            jrow(LEDGER, **row)
            del Xf, Xv
            continue

        W13, _sb13_ckpt = src.fused13(e)
        Wd, sb2_ckpt = src.down(e)

        # W2v2 RTN arm (SSE scale refit; scales then STATIC for GPTQ)
        c13_rtn, sc13 = w2v2_requant(W13, _sb13_ckpt)
        c2_rtn, sc2 = w2v2_requant(Wd, sb2_ckpt)
        s13_col = sbytes_to_scol(sc13)
        s2_col = sbytes_to_scol(sc2)

        # ---- fused13: standard GPTQ (v2 == v1 here: X_fp == X_q by constr.)
        H13 = Xf.t() @ Xf
        perm13 = weight_perm(W13)
        U13 = gptq_prepare(H13, perm13)
        del H13
        c13_g = gptq_loop(W13, s13_col, U13, perm13)
        del U13
        dq13_g = deq(c13_g, s13_col)
        dq13_r = deq(c13_rtn, s13_col)
        f13_vg = proxy_sym(Xv, dq13_g, W13)
        f13_vr = proxy_sym(Xv, dq13_r, W13)
        dec13 = "gptq" if f13_vg <= (1.0 - VAL_MARGIN) * f13_vr else "rtn"
        dq13_ship = dq13_g if dec13 == "gptq" else dq13_r
        row["f13"] = dict(val_gptq=round(f13_vg, 6), val_rtn=round(f13_vr, 6),
                          ship=dec13)

        # ---- down inputs: A_q (quantized net) and A_fp (teacher net)
        gq, uq = dq13_ship[:2048], dq13_ship[2048:]
        gf, uf = W13[:2048], W13[2048:]
        Af_q = act(Xf, gq, uq)
        Av_q = act(Xv, gq, uq)
        Af_fp = act(Xf, gf, uf)
        Av_fp = act(Xv, gf, uf)
        adrift_f = (Af_q - Af_fp).norm().item() / (Af_fp.norm().item() + 1e-30)
        adrift_v = (Av_q - Av_fp).norm().item() / (Av_fp.norm().item() + 1e-30)
        row["adrift"] = dict(fit=round(adrift_f, 6), val=round(adrift_v, 6))
        del dq13_g, dq13_r, dq13_ship, gq, uq, c13_g

        # ---- down solves: v1 (P=None) and v2 per alpha
        H2 = Af_q.t() @ Af_q
        perm2 = weight_perm(Wd)
        U2 = gptq_prepare(H2, perm2)
        del H2
        dXXT = (Af_fp - Af_q).t() @ Af_q                 # [K2,K2] orig order

        arms = {}
        c2_v1 = gptq_loop(Wd, s2_col, U2, perm2)
        arms["v1"] = c2_v1
        if e == 0:
            zeroP = torch.zeros(K2, K2, device=DEV)
            c2_id = gptq_loop(Wd, s2_col, U2, perm2, P=zeroP)
            assert torch.equal(c2_id, c2_v1), \
                "IDENTITY GATE FAIL: v2 loop with P=0 != v1 loop"
            log("identity gate PASS (P=0 == v1, bitwise)")
            del c2_id, zeroP
        for a in ALPHAS:
            Pm = gv2_P(dXXT, U2, perm2, a)
            arms[f"v2_a{a:g}"] = gptq_loop(Wd, s2_col, U2, perm2, P=Pm)
            del Pm
        del U2, dXXT

        dstat = {}
        dq_r = deq(c2_rtn, s2_col)
        dstat["rtn"] = dict(
            wrec=round(wrec(dq_r, Wd), 6),
            sym_val=round(proxy_sym(Av_q, dq_r, Wd), 6),
            asym_val=round(proxy_asym(Av_q, Av_fp, dq_r, Wd), 6),
            sym_fit=round(proxy_sym(Af_q, dq_r, Wd), 6),
            asym_fit=round(proxy_asym(Af_q, Af_fp, dq_r, Wd), 6))
        del dq_r
        for name, cc in arms.items():
            dq = deq(cc, s2_col)
            dstat[name] = dict(
                wrec=round(wrec(dq, Wd), 6),
                sym_val=round(proxy_sym(Av_q, dq, Wd), 6),
                asym_val=round(proxy_asym(Av_q, Av_fp, dq, Wd), 6),
                sym_fit=round(proxy_sym(Af_q, dq, Wd), 6),
                asym_fit=round(proxy_asym(Af_q, Af_fp, dq, Wd), 6))
            del dq
        row["down"] = dstat
        row["secs"] = round(time.time() - te, 1)
        jrow(LEDGER, **row)
        n_done += 1
        del Xf, Xv, W13, Wd, c13_rtn, c2_rtn, s13_col, s2_col, arms
        del Af_q, Av_q, Af_fp, Av_fp
        if e % 8 == 0:
            log(f"e{e}/{EE} n_fit={n_fit} n_val={n_val} "
                f"v1_asym={dstat['v1']['asym_val']} "
                f"v2a25_asym={dstat.get('v2_a0.25', {}).get('asym_val')} "
                f"{row['secs']}s")
            gc.collect()
            torch.cuda.empty_cache()
            with open(f"{OUT}/HEARTBEAT_PILOT", "w") as f:
                json.dump({"layer": LAYER, "expert": e,
                           "ts": time.time()}, f)

    if not MAX_EXPERTS:
        with open(done_marker, "w") as f:
            f.write(f"{n_done} experts solved\n")
    mins = round((time.time() - t0) / 60, 1)
    log(f"DONE in {mins}m ({n_done} experts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
