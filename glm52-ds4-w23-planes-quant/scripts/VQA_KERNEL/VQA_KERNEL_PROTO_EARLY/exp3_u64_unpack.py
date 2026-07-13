#!/usr/bin/env python3
"""t_bd0c0ac6 exp3: single-gather u64 codebook unpack + num_warps sweep.

The plain vqA gather does 4 dependent fp16 gathers per group
(cb[c,0..3]). A cb row is exactly 8 bytes -> ONE u64 gather + register
unpack (shift/mask/bitcast) replaces 4 gathers. w2lut arm gets the same
num_warps sweep so the comparison stays fair.

Correctness: python dequant reference, relL2 <= 1e-3, 3 layers x 4
experts x both projs. Perf: loop8 decode-like, 5 reps, min/median.
"""
import json
import os
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "cuda"
PLANES = Path(os.path.expanduser("~/missions/VQA_KERNEL_PROTO_EARLY/planes"))
OUT = Path(os.path.expanduser("~/missions/VQA_KERNEL_PROTO_EARLY"))
W2V2_LUT = [-3.5111107379486137, -1.1800192351581362,
            0.6510809470728273, 2.7868641002011136]
REL_TOL = 1e-3
LAYERS = [3, 13, 23]
EXPERTS = [0, 17, 128, 255]


@triton.jit
def w2lut_gemv(y_ptr, x_ptr, codes_ptr, sc_ptr, N, K,
               L0: tl.constexpr, L1: tl.constexpr,
               L2: tl.constexpr, L3: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr):
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n_offs < N
    G = K // 4
    SG = K // 32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g0 in range(0, tl.cdiv(G, BLOCK_G)):
        g_offs = g0 * BLOCK_G + tl.arange(0, BLOCK_G)
        gm = g_offs < G
        m2 = nm[:, None] & gm[None, :]
        codes = tl.load(codes_ptr + n_offs[:, None] * G + g_offs[None, :],
                        mask=m2, other=0)
        scb = tl.load(sc_ptr + n_offs[:, None] * SG + (g_offs[None, :] // 8),
                      mask=m2, other=127).to(tl.float32)
        sc = tl.exp2(scb - 127.0)
        part = tl.zeros((BLOCK_N, BLOCK_G), dtype=tl.float32)
        for j in tl.static_range(4):
            cj = (codes >> (2 * j)) & 3
            w = tl.where(cj == 0, L0,
                         tl.where(cj == 1, L1,
                                  tl.where(cj == 2, L2, L3)))
            xv = tl.load(x_ptr + g_offs * 4 + j, mask=gm,
                         other=0.0).to(tl.float32)
            part += w * xv[None, :]
        acc += tl.sum(part * sc, axis=1)
    tl.store(y_ptr + n_offs, acc, mask=nm)


@triton.jit
def vqa_gemv_u64(y_ptr, x_ptr, codes_ptr, sc_ptr, cb64_ptr, N, K,
                 BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr):
    """cb64: [256] u64 (one row = 4 packed fp16). One gather per group,
    unpack in registers via shift/mask/bitcast."""
    pid = tl.program_id(0)
    n_offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nm = n_offs < N
    G = K // 4
    SG = K // 32
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for g0 in range(0, tl.cdiv(G, BLOCK_G)):
        g_offs = g0 * BLOCK_G + tl.arange(0, BLOCK_G)
        gm = g_offs < G
        m2 = nm[:, None] & gm[None, :]
        codes = tl.load(codes_ptr + n_offs[:, None] * G + g_offs[None, :],
                        mask=m2, other=0).to(tl.int32)
        scb = tl.load(sc_ptr + n_offs[:, None] * SG + (g_offs[None, :] // 8),
                      mask=m2, other=127).to(tl.float32)
        sc = tl.exp2(scb - 127.0)
        packed = tl.load(cb64_ptr + codes).to(tl.uint64)
        part = tl.zeros((BLOCK_N, BLOCK_G), dtype=tl.float32)
        for j in tl.static_range(4):
            h = ((packed >> (16 * j)) & 0xFFFF).to(tl.uint16)
            w = h.to(tl.float16, bitcast=True).to(tl.float32)
            xv = tl.load(x_ptr + g_offs * 4 + j, mask=gm,
                         other=0.0).to(tl.float32)
            part += w * xv[None, :]
        acc += tl.sum(part * sc, axis=1)
    tl.store(y_ptr + n_offs, acc, mask=nm)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def scol_ref(sc_u8, K):
    return torch.exp2(sc_u8.to(DEV).float() - 127.0).repeat_interleave(
        32, dim=1).clamp_min(1e-38)


def ref_vqa_y(codes, sc_u8, cb, x, N, K):
    W = cb.to(DEV).float()[codes.to(DEV).long()].reshape(N, K)
    W = W * scol_ref(sc_u8, K)
    return W @ x.float()


def main():
    out = {"task": "t_bd0c0ac6", "stage": "exp3_u64_unpack",
           "device": torch.cuda.get_device_name(0),
           "rel_tol": REL_TOL, "correctness": [], "perf": {}}
    xs = {}
    for K in (4096, 2048):
        g = torch.Generator(device="cpu")
        g.manual_seed(K)
        xs[K] = torch.randn(K, generator=g).to(DEV).to(torch.float16)

    worst = 0.0
    n_pass = 0
    n_tot = 0
    perf_data = None
    for L in LAYERS:
        d = torch.load(PLANES / f"vqa_layer_{L:03d}.pt",
                       map_location="cpu", weights_only=False)
        for proj, ck, sk, cbk, N, K in (
                ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
                ("down", "codes2", "sc2", "cb2", 4096, 2048)):
            cb = d[cbk]
            cb64 = cb.contiguous().view(torch.int64).reshape(256).to(DEV)
            x = xs[K]
            for e in EXPERTS:
                codes = d[ck][e].to(DEV)
                sc = d[sk][e].to(DEV)
                yref = ref_vqa_y(codes, sc, cb, x, N, K)
                y = torch.empty(N, device=DEV, dtype=torch.float32)
                grid = ((N + 63) // 64,)
                vqa_gemv_u64[grid](y, x, codes, sc, cb64, N, K,
                                   BLOCK_N=64, BLOCK_G=128)
                rel = (torch.norm(y - yref) / torch.norm(yref)).item()
                ok = rel <= REL_TOL
                n_pass += ok
                n_tot += 1
                worst = max(worst, rel)
                out["correctness"].append(
                    {"layer": L, "expert": e, "proj": proj,
                     "relL2": rel, "pass": bool(ok)})
        if L == LAYERS[0]:
            perf_data = d
        else:
            del d
    print(f"correctness u64: {n_pass}/{n_tot} worst {worst:.3e}")
    out["correctness_gate"] = {"n_pass": n_pass, "n_total": n_tot,
                               "worst_relL2": worst,
                               "pass": bool(n_pass == n_tot)}

    d = perf_data
    es = [0, 17, 63, 101, 128, 177, 200, 255]
    cfgs = [(bn, bg, nw) for bn in (32, 64) for bg in (64, 128)
            for nw in (1, 2, 4, 8)]
    for proj, ck, sk, cbk, N, K in (
            ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
            ("down", "codes2", "sc2", "cb2", 4096, 2048)):
        cb = d[cbk]
        cb64 = cb.contiguous().view(torch.int64).reshape(256).to(DEV)
        x = xs[K]
        G = K // 4
        vq_cases = [(d[ck][e].to(DEV), d[sk][e].to(DEV)) for e in es]
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(1234 + K)
        w2_cases = [(torch.randint(0, 256, (N, G), generator=g2,
                                   dtype=torch.uint8).to(DEV), scb)
                    for _, scb in vq_cases]
        y = torch.empty(N, device=DEV, dtype=torch.float32)
        res = {}
        for arm in ("w2lut", "vqa_u64"):
            sweep = {}
            for bn, bg, nw in cfgs:
                grid = ((N + bn - 1) // bn,)
                if arm == "w2lut":
                    def loop(bn=bn, bg=bg, nw=nw, grid=grid):
                        for cc in w2_cases:
                            w2lut_gemv[grid](y, x, cc[0], cc[1], N, K,
                                             *W2V2_LUT, BLOCK_N=bn,
                                             BLOCK_G=bg, num_warps=nw)
                else:
                    def loop(bn=bn, bg=bg, nw=nw, grid=grid):
                        for cc in vq_cases:
                            vqa_gemv_u64[grid](y, x, cc[0], cc[1], cb64,
                                               N, K, BLOCK_N=bn,
                                               BLOCK_G=bg, num_warps=nw)
                t = bench(loop) / 8
                if t < 400:
                    sweep[f"{bn}x{bg}w{nw}"] = round(t, 2)
            bc = min(sweep, key=sweep.get)
            parts = bc.replace("w", "x").split("x")
            bn, bg, nw = int(parts[0]), int(parts[1]), int(parts[2])
            grid = ((N + bn - 1) // bn,)
            if arm == "w2lut":
                def loop(bn=bn, bg=bg, nw=nw, grid=grid):
                    for cc in w2_cases:
                        w2lut_gemv[grid](y, x, cc[0], cc[1], N, K,
                                         *W2V2_LUT, BLOCK_N=bn,
                                         BLOCK_G=bg, num_warps=nw)
            else:
                def loop(bn=bn, bg=bg, nw=nw, grid=grid):
                    for cc in vq_cases:
                        vqa_gemv_u64[grid](y, x, cc[0], cc[1], cb64, N, K,
                                           BLOCK_N=bn, BLOCK_G=bg,
                                           num_warps=nw)
            reps = [round(bench(loop) / 8, 2) for _ in range(5)]
            res[arm] = {"best_cfg": bc, "sweep": sweep, "reps": reps,
                        "min": min(reps), "median": round(median(reps), 2)}
            print(f"[{proj}] {arm} loop8 best {bc} min={min(reps)} "
                  f"reps={reps}")
        rmin = res["vqa_u64"]["min"] / res["w2lut"]["min"]
        rmed = res["vqa_u64"]["median"] / res["w2lut"]["median"]
        res["ratio_loop8_min"] = round(rmin, 4)
        res["ratio_loop8_median"] = round(rmed, 4)
        print(f"[{proj}] u64 loop8 ratio: min {rmin:.3f}x med {rmed:.3f}x")
        out["perf"][proj] = res

    with open(OUT / "EXP3_U64_UNPACK.json", "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE EXP3_U64_UNPACK.json")


if __name__ == "__main__":
    main()
