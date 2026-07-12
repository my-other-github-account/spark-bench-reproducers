#!/usr/bin/env python3
"""t_3d6e422d card-step-1: full W2v2 plane build (e43 LUT + SSE scale
refit + optional per-tensor fractional alpha per PILOT_ALPHA.json).

Wire format identical to shipped moe_w2_planes (2 bpc fragment-major
planes13/planes2 + packed UE8M0 sc13/sc2); LUT rides in meta.json (the
sealed v3 builder + MMLU harness read it with zero code changes).
Alpha (when adopted) rides in layer_NNN.alphas.npy [E,3] f32 (w1,w3,down)
— consumed by the bq_sources.py rail wrapper; serve needs the epilogue
kernel (follow-up card).

Resume-safe: layer skipped when all files (incl. alphas when mode!=none)
exist.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_BESTQ"))
import bq_common as bq  # noqa: E402
import planes_unpack as pu  # noqa: E402

ALPHAS = [2.0 ** (k / 16.0) for k in range(-8, 9)]
LUT = torch.tensor(bq.W2V2_LUT_E43, dtype=torch.float32, device=bq.DEV)
TIERS = {"13": (("w1", "w3"), bq.N13, bq.K13,
                [(0, 2048, 0), (2048, 4096, 1)]),
         "2": (("w2",), bq.N2, bq.K2, [(0, 4096, 2)])}


def load_chunk(L, es, names, dev):
    ws, ss = [], []
    for e in es:
        ws.append(bq.src_dense(L, e, names, dev=dev))
        ss.append(bq.src_scales(L, e, names, dev=dev))
    return torch.stack(ws), torch.stack(ss)


def requant_b(w, sb, lut, alpha_row=None):
    """batched: w [B,N,K], sb [B,N,KB], alpha_row [B,N] or None."""
    B, N, K = w.shape
    KB = K // 32
    mids = (lut[1:] + lut[:-1]) / 2
    a = None if alpha_row is None else alpha_row.view(B, N, 1)
    best_err = None
    best_off = None
    for off in bq.OFFSETS:
        sf = torch.exp2(sb.to(torch.float32) - 127.0 + off) \
            .repeat_interleave(32, dim=2)
        if a is not None:
            sf = sf * a
        u = w / sf
        q = lut[torch.bucketize(u.contiguous(), mids)]
        err2 = (q * sf - w).pow_(2).view(B, N, KB, 32).sum(dim=3)
        del sf, u, q
        if best_err is None:
            best_err = err2
            best_off = torch.full_like(err2, off, dtype=torch.int16)
        else:
            m = err2 < best_err
            best_err = torch.where(m, err2, best_err)
            best_off = torch.where(
                m, torch.full_like(best_off, off), best_off)
    sc = (sb.to(torch.int16) + best_off).clamp_(0, 254).to(torch.uint8)
    sf = torch.exp2(sc.to(torch.float32) - 127.0).repeat_interleave(32, dim=2)
    if a is not None:
        sf = sf * a
    codes = torch.bucketize((w / sf).contiguous(), mids).to(torch.uint8)
    return codes, sc


def fit_alpha_b(w_sub, sb_sub, lut):
    """[B,n,K] subsample -> best alpha per B from the grid."""
    B = w_sub.shape[0]
    KB = w_sub.shape[2] // 32
    mids = (lut[1:] + lut[:-1]) / 2
    sses = torch.empty(B, len(ALPHAS), device=w_sub.device)
    for ai, a in enumerate(ALPHAS):
        best_err = None
        for off in bq.OFFSETS:
            sf = torch.exp2(sb_sub.to(torch.float32) - 127.0 + off) \
                .repeat_interleave(32, dim=2) * a
            u = w_sub / sf
            q = lut[torch.bucketize(u.contiguous(), mids)]
            err2 = (q * sf - w_sub).pow_(2) \
                .view(B, -1, KB, 32).sum(dim=3)
            best_err = err2 if best_err is None \
                else torch.minimum(best_err, err2)
            del sf, u, q
        sses[:, ai] = best_err.sum(dim=(1, 2))
    idx = sses.argmin(dim=1)
    return torch.tensor([ALPHAS[i] for i in idx.tolist()],
                        device=w_sub.device), sses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{bq.BQ}/moe_w2_planes_v2e43")
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    outdir = os.path.expanduser(a.out)
    os.makedirs(outdir, exist_ok=True)
    pilot = json.load(open(f"{bq.BQ}/PILOT_ALPHA.json"))
    mode = pilot["decisions"]["w2v2_alpha_mode"]
    bq.log(f"alpha mode = {mode}")
    lo, hi = a.layers.split("-") if "-" in a.layers else (None, None)
    layers = list(range(int(lo), int(hi) + 1)) if lo is not None \
        else [int(x) for x in a.layers.split(",")]

    for L in layers:
        base = f"{outdir}/layer_{L:03d}"
        need = [f"{base}.{x}.npy" for x in
                ("planes13", "planes2", "sc13", "sc2")] + \
               [f"{base}.meta.json"]
        if mode != "none":
            need.append(f"{base}.alphas.npy")
        if all(os.path.exists(p) for p in need):
            bq.log(f"L{L:03d} exists, skip")
            continue
        t0 = time.time()
        planes = {t: np.empty((bq.E, N * K // 4), dtype=np.uint8)
                  for t, (_, N, K, _) in TIERS.items()}
        scs = {t: np.empty((bq.E, N * (K // 32)), dtype=np.uint8)
               for t, (_, N, K, _) in TIERS.items()}
        alphas = np.ones((bq.E, 3), dtype=np.float32)
        for tier, (names, N, K, groups) in TIERS.items():
            for c0 in range(0, bq.E, a.chunk):
                es = list(range(c0, min(c0 + a.chunk, bq.E)))
                w, sb = load_chunk(L, es, names, bq.DEV)
                alpha_row = None
                if mode == "joint":
                    alpha_row = torch.ones(len(es), N, device=bq.DEV)
                    for (r0, r1, ai) in groups:
                        ba, _ = fit_alpha_b(
                            w[:, r0:r1:8].contiguous(),
                            sb[:, r0:r1:8].contiguous(), LUT)
                        alpha_row[:, r0:r1] = ba.view(-1, 1)
                        for i, e in enumerate(es):
                            alphas[e, ai] = float(ba[i])
                codes, sc = requant_b(w, sb, LUT, alpha_row)
                if mode == "posthoc":
                    for i, e in enumerate(es):
                        for (r0, r1, ai) in groups:
                            dq = bq.deq_codes(codes[i, r0:r1],
                                              sc[i, r0:r1], LUT)
                            al = bq.closed_alpha(dq, w[i, r0:r1])
                            alphas[e, ai] = al
                for i, e in enumerate(es):
                    planes[tier][e] = pu.pack_fragment_major(
                        codes[i]).cpu().numpy()
                    scs[tier][e] = pu.pack_scales(sc[i]).cpu().numpy()
                del w, sb, codes, sc, alpha_row
        for tier in TIERS:
            for tag, arr in ((f"planes{tier}", planes[tier]),
                             (f"sc{tier}", scs[tier])):
                np.save(f"{base}.{tag}.npy.tmp.npy", arr)
                os.replace(f"{base}.{tag}.npy.tmp.npy", f"{base}.{tag}.npy")
        if mode != "none":
            np.save(f"{base}.alphas.npy.tmp.npy", alphas)
            os.replace(f"{base}.alphas.npy.tmp.npy", f"{base}.alphas.npy")
        meta = {"E": bq.E, "N13": bq.N13, "K13": bq.K13,
                "N2": bq.N2, "K2": bq.K2,
                "codebook": "w2", "bpw": 2.25, "lut": bq.W2V2_LUT_E43,
                "lut_provenance":
                    "dp_asym4_round2 (W2V2_SHOOTOUT.json t_bd7728ee, gate "
                    "0.9198x) rounded to e4m3-representable levels "
                    "(serve-anchor convention, w3 e43 lane t_14f51254)",
                "scale_fit": "per-block-32 UE8M0 exponent SSE search "
                             f"offsets {bq.OFFSETS} vs ckpt mxfp4 exponent",
                "alpha_mode": mode,
                "alpha_note": "per-tensor fractional scale (w1/w3/down) in "
                              "layer_NNN.alphas.npy; dequant = alpha * "
                              "lut[codes] * 2^(sc-127); serve epilogue "
                              "pending (t_3d6e422d card step 4)",
                "task": "t_3d6e422d",
                "md5_planes13": bq.md5f(f"{base}.planes13.npy"),
                "md5_planes2": bq.md5f(f"{base}.planes2.npy")}
        tmp = f"{base}.meta.json.tmp"
        json.dump(meta, open(tmp, "w"))
        os.replace(tmp, f"{base}.meta.json")
        bq.log(f"L{L:03d} built in {time.time()-t0:.0f}s "
               f"(alpha mode {mode})")
    bq.log("build complete")


if __name__ == "__main__":
    main()
