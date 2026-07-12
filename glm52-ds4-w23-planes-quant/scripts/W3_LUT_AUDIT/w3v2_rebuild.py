#!/usr/bin/env python3
"""moe_w3_planes_v2 rebuild (t_eee6b0cc) — winner LUT + per-block refit
scales, emitted in the exact shipped vLLM-Moet wire format.

LUT = dp_asym8_fit from the W3 LUT shootout (~/missions/W3_LUT_AUDIT/
SHOOTOUT_EXTRA.json): exact interval-DP MSE-optimal 8-level quantizer of
the s^2-weighted e2m1 u-atom histogram, fit on HELD-OUT experts (8 layers
x 3 fit experts x both tiers), one scale-search alternation round.
Held-out eval relRMS 0.1537 vs current-LUT 0.2002 (-23%) and rms ratio
0.983 vs 0.893.

Scales: per block-32 UE8M0, chosen by exact SSE search over exponent
offsets [-4..2] relative to the ckpt mxfp4 exponent (same search that the
shootout arms used).  Codes: nearest-level snap (ties -> lower, exact
same bucketize convention as the shootout/planes_unpack).

Output: layer_NNN.{planes13,planes2,sc13,sc2}.npy + layer_NNN.meta.json
with the new "lut" entry -> t8192_ds4_build_v3.py --mode planes and
mmlu_ds4_offline.py --mode planes read it with ZERO code changes
(both take LUT from meta.json when present).

Resume-safe: a layer with all 5 files present is skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/missions/DS4_TEACHER"))
import planes_unpack as pu  # noqa  (import self-tests run here)

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
from safetensors import safe_open  # noqa

# dp_asym8_fit, full precision (SHOOTOUT_EXTRA.json luts.dp_asym8_fit)
W3V2_LUT = [-6.379047481233444, -3.472263410189838, -1.871824735139655,
            -0.8547080566996477, 0.1369991715083642, 1.4651236544119166,
            3.479577102951586, 6.379153893385449]
OFFSETS = list(range(-4, 3))

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)

WM = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
_handles = {}


def get(name):
    sh = WM[name]
    if sh not in _handles:
        _handles[sh] = safe_open(os.path.join(CKPT, sh), framework="pt")
    return _handles[sh].get_tensor(name)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_chunk(L, es, names, dev):
    """experts es, projections names (('w1','w3') or ('w2',))
    -> w f32 [B,N,K] on dev, sbytes u8 [B,N,KB] on dev."""
    ws, ss = [], []
    for e in es:
        pw, ps = [], []
        for wname in names:
            k = f"layers.{L}.ffn.experts.{e}.{wname}"
            wp = get(k + ".weight").view(torch.uint8).to(dev)
            sb = get(k + ".scale").view(torch.uint8).to(dev)
            nib = torch.stack((wp & 0xF, wp >> 4), dim=-1).flatten(-2)
            w = _E2M1.to(dev)[nib.long()]
            w *= torch.exp2(sb.to(torch.float32) - 127.0) \
                .repeat_interleave(32, dim=1)
            pw.append(w)
            ps.append(sb)
        ws.append(torch.cat(pw, 0))
        ss.append(torch.cat(ps, 0))
    return torch.stack(ws), torch.stack(ss)


def requant_chunk(w, sb, lut, mids):
    """w [B,N,K] f32, sb [B,N,KB] u8 -> codes u8 [B,N,K], sc u8 [B,N,KB]"""
    B, N, K = w.shape
    KB = K // 32
    best_err = None
    best_off = None
    for off in OFFSETS:
        s = torch.exp2(sb.to(torch.float32) - 127.0 + off)
        sf = s.repeat_interleave(32, dim=2)
        u = w / sf
        q = lut[torch.bucketize(u.contiguous(), mids)]
        err2 = (q * sf - w).pow_(2).view(B, N, KB, 32).sum(dim=3)
        if best_err is None:
            best_err = err2
            best_off = torch.full_like(err2, off, dtype=torch.int16)
        else:
            m = err2 < best_err
            best_err = torch.where(m, err2, best_err)
            best_off = torch.where(m, torch.full_like(best_off, off),
                                   best_off)
    sc = (sb.to(torch.int16) + best_off).clamp_(0, 254).to(torch.uint8)
    sf = torch.exp2(sc.to(torch.float32) - 127.0).repeat_interleave(32, dim=2)
    codes = torch.bucketize((w / sf).contiguous(), mids).to(torch.uint8)
    return codes, sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser(
        "~/missions/W3_LUT_AUDIT/moe_w3_planes_v2"))
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--device", default="cuda"
                    if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    dev = a.device
    os.makedirs(a.out, exist_ok=True)
    if "-" in a.layers:
        lo, hi = a.layers.split("-")
        layers = list(range(int(lo), int(hi) + 1))
    else:
        layers = [int(x) for x in a.layers.split(",")]

    lut = torch.tensor(W3V2_LUT, dtype=torch.float32, device=dev)
    mids = (lut[1:] + lut[:-1]) / 2
    E = 256
    tiers = {"13": (("w1", "w3"), 4096, 4096),
             "2": (("w2",), 4096, 2048)}

    for L in layers:
        files = [f"{a.out}/layer_{L:03d}.{x}.npy"
                 for x in ("planes13", "planes2", "sc13", "sc2")] \
            + [f"{a.out}/layer_{L:03d}.meta.json"]
        if all(os.path.exists(f) for f in files):
            log(f"L{L:03d} exists, skip")
            continue
        t0 = time.time()
        out = {}
        for tier, (names, N, K) in tiers.items():
            KB = K // 32
            pbytes = np.empty((E, N * K * 3 // 8), dtype=np.uint8)
            sbytes = np.empty((E, N * KB), dtype=np.uint8)
            for c0 in range(0, E, a.chunk):
                es = list(range(c0, min(c0 + a.chunk, E)))
                w, sb = load_chunk(L, es, names, dev)
                codes, sc = requant_chunk(w, sb, lut, mids)
                for i, e in enumerate(es):
                    pbytes[e] = pu.pack_w3_plane(codes[i]).cpu().numpy()
                    sbytes[e] = pu.pack_scales(sc[i]).cpu().numpy()
                del w, sb, codes, sc
            np.save(f"{a.out}/layer_{L:03d}.planes{tier}.npy.tmp.npy", pbytes)
            os.replace(f"{a.out}/layer_{L:03d}.planes{tier}.npy.tmp.npy",
                       f"{a.out}/layer_{L:03d}.planes{tier}.npy")
            np.save(f"{a.out}/layer_{L:03d}.sc{tier}.npy.tmp.npy", sbytes)
            os.replace(f"{a.out}/layer_{L:03d}.sc{tier}.npy.tmp.npy",
                       f"{a.out}/layer_{L:03d}.sc{tier}.npy")
        meta = {"E": E, "N13": 4096, "K13": 4096, "N2": 4096, "K2": 2048,
                "codebook": "w3", "bpw": 3.25, "lut": W3V2_LUT,
                "lut_provenance":
                    "dp_asym8_fit t_eee6b0cc W3 LUT shootout (held-out DP "
                    "MSE fit on e2m1 u-atom hist, 1 scale-alt round)",
                "scale_fit": "per-block-32 UE8M0 exponent SSE search "
                             f"offsets {OFFSETS} vs ckpt mxfp4 exponent"}
        with open(f"{a.out}/layer_{L:03d}.meta.json", "w") as f:
            json.dump(meta, f)
        log(f"L{L:03d} rebuilt in {time.time()-t0:.0f}s")
    log("rebuild complete")


if __name__ == "__main__":
    main()
