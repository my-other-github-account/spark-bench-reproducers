#!/usr/bin/env python3
"""Extracted VERBATIM from the sealed vqw2_pilot.py (t_fa2eafed, s3) --
codebook fit + VQ-GPTQ machinery only, no module-level side effects.
Identity gate (vq_gptq d=1 == scalar gptq_loop bitwise) re-run by the
consumer before use."""
import torch

KPP_SUB = 1 << 17


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
