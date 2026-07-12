#!/usr/bin/env python3
"""Shared helpers for t_3d6e422d DS4-BESTQ (s8).

Conventions pinned from the sealed lanes:
- ckpt mxfp4 access + per-block-32 UE8M0 scale handling: w3v2_rebuild.py
  (t_eee6b0cc) load/requant conventions, bucketize nearest-level snap,
  lo tie-break.
- GPTQ machinery: ds4_gptq_w3v2.py (t_26055bf3) verbatim skeleton
  (blocksize 128, percdamp 0.01, colnorm-desc perm, static scales).
- W2v2 LUT: dp_asym4_round2 winner (W2V2_SHOOTOUT.json, t_bd7728ee,
  gate 0.9198x PASS), rounded to e4m3-representable levels from the start
  (serve-anchor convention of the w3 e43 lane, t_14f51254: SASS immediate
  pools hold e4m3 bytes; measured R6e43 0.1415 <= R6 0.1475 at 2.729bpw
  says the rounding costs nothing behaviorally).
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

TEACH = os.path.expanduser("~/missions/DS4_TEACHER")
if TEACH not in sys.path:
    sys.path.insert(0, TEACH)
import planes_unpack as pu  # noqa: E402

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
BQ = os.path.expanduser("~/missions/DS4_BESTQ")
GPTQM = os.path.expanduser("~/missions/DS4_GPTQ")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
E, N13, K13, N2, K2 = 256, 4096, 4096, 4096, 2048
OFFSETS = list(range(-4, 3))
CAP = f"{GPTQM}/cap"

# dp_asym4_round2 winner (W2V2_SHOOTOUT.json t_bd7728ee)
W2V2_LUT_FULL = [-3.5111107379486137, -1.1800192351581362,
                 0.6510809470728273, 2.7868641002011136]


def e43_round(v):
    s = math.copysign(1.0, v)
    a = abs(v)
    e = math.floor(math.log2(a))
    m = round(a / (2.0 ** e) * 8) - 8
    if m == 8:
        e += 1
        m = 0
    return s * (1 + m / 8.0) * (2.0 ** e)


W2V2_LUT_E43 = [e43_round(v) for v in W2V2_LUT_FULL]
assert W2V2_LUT_E43 == [-3.5, -1.125, 0.625, 2.75], W2V2_LUT_E43

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)
_WM = None
_handles = {}


def wm():
    global _WM
    if _WM is None:
        _WM = json.load(
            open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    return _WM


def get_tensor(name):
    from safetensors import safe_open
    sh = wm()[name]
    if sh not in _handles:
        _handles[sh] = safe_open(os.path.join(CKPT, sh), framework="pt")
    return _handles[sh].get_tensor(name)


def src_dense(L, e, names, dev=DEV, dtype=torch.float32):
    """mxfp4 dequant of ckpt expert tensors, rows concat over names."""
    ws = []
    for wname in names:
        k = f"layers.{L}.ffn.experts.{e}.{wname}"
        wp = get_tensor(k + ".weight").view(torch.uint8).to(dev)
        sb = get_tensor(k + ".scale").view(torch.uint8).to(dev)
        nib = torch.stack((wp & 0xF, wp >> 4), dim=-1).flatten(-2)
        w = _E2M1.to(dev)[nib.long()]
        w = w * torch.exp2(sb.to(torch.float32) - 127.0) \
            .repeat_interleave(32, dim=1)
        ws.append(w.to(dtype))
    return torch.cat(ws, 0)


def src_scales(L, e, names, dev=DEV):
    ss = []
    for wname in names:
        k = f"layers.{L}.ffn.experts.{e}.{wname}.scale"
        ss.append(get_tensor(k).view(torch.uint8).to(dev))
    return torch.cat(ss, 0)


def requant(w, sb, lut, alpha_row=None):
    """w [N,K] f32 DEV, sb [N,KB] u8 -> codes u8 [N,K], sc u8 [N,KB].
    Per-block-32 SSE exponent-offset search (OFFSETS) with optional
    per-row fractional alpha (per-tensor scale epilogue)."""
    N, K = w.shape
    KB = K // 32
    lut = lut.to(w.device)
    mids = (lut[1:] + lut[:-1]) / 2
    a = None if alpha_row is None else alpha_row.view(N, 1).to(w.device)
    best_err = None
    best_off = None
    for off in OFFSETS:
        s = torch.exp2(sb.to(torch.float32) - 127.0 + off)
        sf = s.repeat_interleave(32, dim=1)
        if a is not None:
            sf = sf * a
        u = w / sf
        q = lut[torch.bucketize(u.contiguous(), mids)]
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
    sf = torch.exp2(sc.to(torch.float32) - 127.0).repeat_interleave(32, dim=1)
    if a is not None:
        sf = sf * a
    codes = torch.bucketize((w / sf).contiguous(), mids).to(torch.uint8)
    return codes, sc


def requant_sse(w, sb, lut, alpha):
    """Total SSE of requant at a single scalar alpha (for grid fits)."""
    N, K = w.shape
    KB = K // 32
    lut = lut.to(w.device)
    mids = (lut[1:] + lut[:-1]) / 2
    best_err = None
    for off in OFFSETS:
        sf = torch.exp2(sb.to(torch.float32) - 127.0 + off) \
            .repeat_interleave(32, dim=1) * alpha
        u = w / sf
        q = lut[torch.bucketize(u.contiguous(), mids)]
        err2 = (q * sf - w).pow_(2).view(N, KB, 32).sum(dim=2)
        best_err = err2 if best_err is None else torch.minimum(best_err, err2)
    return best_err.sum().item()


def deq_codes(codes, sc, lut, alpha_row=None):
    lut = lut.to(codes.device)
    sf = torch.exp2(sc.to(torch.float32) - 127.0).repeat_interleave(32, dim=1)
    dq = lut[codes.long()] * sf
    if alpha_row is not None:
        dq = dq * alpha_row.view(-1, 1).to(dq.device)
    return dq


def relrms(dq, w):
    return ((dq.float() - w.float()).pow(2).mean().sqrt()
            / w.float().pow(2).mean().sqrt()).item()


def closed_alpha(dq, w):
    num = (dq.double() * w.double()).sum()
    den = (dq.double() * dq.double()).sum() + 1e-30
    return float(num / den)


# --------------------------------------------------------------- GPTQ core
def weight_perm(W):
    return torch.argsort(W.float().pow(2).sum(0), descending=True)


def gptq_prepare(H, perm, percdamp=0.01):
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


def gptq_loop(W, s_col, Hinv, perm, lut, blocksize=128):
    """g4_driver.gptq_quantize_lut column loop verbatim; LUT projection.
    Returns codes u8 [N,K] in ORIGINAL column order."""
    lut = lut.to(W.device)
    mids = (lut[1:] + lut[:-1]) / 2
    N, K = W.shape
    W = W.float().clone()[:, perm]
    s_col = s_col.float().clamp_min(1e-38)[:, perm]
    C = torch.zeros(N, K, dtype=torch.uint8, device=W.device)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = Hinv[i1:i2, i1:i2]
        for j in range(cnt):
            w = W1[:, j]
            d = U1[j, j]
            s = s_col[:, i1 + j]
            cj = torch.bucketize((w / s).contiguous(), mids)
            vj = lut[cj]
            C[:, i1 + j] = cj.to(torch.uint8)
            err = (w - vj * s) / d
            if j + 1 < cnt:
                W1[:, j + 1:] -= err.unsqueeze(1) * U1[j, j + 1:].unsqueeze(0)
            E1[:, j] = err
        if i2 < K:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    del W1, E1
    inv = torch.argsort(perm)
    return C[:, inv]


def damp_solve(H, B, percdamp=0.01):
    """solve (H + damp) X = B with boost-on-failure (for the GPTAQ W*)."""
    K = H.shape[0]
    diag = torch.arange(K, device=H.device)
    Hd = H.clone()
    dead = Hd[diag, diag] <= 0
    Hd[diag[dead], diag[dead]] = 1.0
    base = Hd[diag, diag].mean()
    Hd[diag, diag] += percdamp * base
    for boost in range(6):
        try:
            return torch.linalg.solve(Hd, B)
        except Exception:
            Hd[diag, diag] += (percdamp * (2 ** (boost + 1))) * base
    raise RuntimeError("damp_solve failed after boosts")


def proxy_err(X, dq, W):
    """|| X (dq-W)^T ||_F / || X W^T ||_F (nf3_gptq form, symmetric)."""
    if X is None or X.shape[0] == 0:
        return None
    d = dq.float() - W.float()
    num = (X.float() @ d.t()).norm()
    den = (X.float() @ W.float().t()).norm() + 1e-30
    return (num / den).item()


def act(X, Wg, Wu, limit):
    """DS4 expert activation: silu(clamp(X Wg^T, max=lim)) * clamp(X Wu^T)."""
    g = (X @ Wg.t()).clamp(max=limit)
    u = (X @ Wu.t()).clamp(min=-limit, max=limit)
    return torch.nn.functional.silu(g) * u


def swiglu_limit():
    return json.load(open(f"{CKPT}/config.json"))["swiglu_limit"]


def load_caps(L, wins, dev=DEV):
    """captured x_moe/topk for layer L over window ids -> x, hit mask."""
    xs, tks = [], []
    for gid in wins:
        d = torch.load(f"{CAP}/xmoe_L{L:03d}_win{gid:04d}.pt",
                       map_location="cpu")
        xs.append(d["x"])
        tks.append(d["topk"].to(torch.int64))
    x = torch.cat(xs, 0).to(dev)
    tk = torch.cat(tks, 0).to(dev)
    hit = torch.zeros(x.shape[0], E, dtype=torch.bool, device=dev)
    hit.scatter_(1, tk, True)
    return x, hit


def calib_selection():
    return json.load(open(f"{GPTQM}/static/CALIB_SELECTION.json"))


def md5f(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jrow(path, **kw):
    kw["ts"] = round(time.time(), 3)
    with open(path, "a") as f:
        f.write(json.dumps(kw, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
