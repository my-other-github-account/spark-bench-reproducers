#!/usr/bin/env python3
"""t_bd0c0ac6 exp2: vqA dot-precompute GEMV variant.

Motivation: plain d=4 gather GEMV fails the 1.4x budget on down/loop8
(1.46-1.52x). The vqA structure allows an optimization the scalar W2 LUT
path does NOT need but vqA gets for free: the codebook is LAYER-SHARED
per projection, so for a given token x we can precompute
    cbx[c, g] = dot(cb[c, :], x[g*4 : g*4+4])   -> [256, G]
ONCE per (token, layer, proj), then every expert GEMV is a single
gather + scale per group:
    y[n] = sum_g cbx[codes[n, g], g] * sc[n, g//8]
The precompute (a 256 x 4 x G mini-GEMM) amortizes over the 8 routed
experts of the MoE decode step. Honest accounting: loop8 time includes
the precompute, divided by 8.

Table dtype arms: fp32 (exact-ish) and fp16 (half the L2 traffic).
Correctness gate: same python dequant reference, relL2 <= 1e-3.
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
def vqa_dot_gemv(y_ptr, cbx_ptr, codes_ptr, sc_ptr, N, K,
                 BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr):
    """cbx: [256, G] table (fp32 or fp16), gather one dot per group."""
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
        dv = tl.load(cbx_ptr + codes * G + g_offs[None, :],
                     mask=m2, other=0.0).to(tl.float32)
        acc += tl.sum(dv * sc, axis=1)
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


def make_cbx(cb, x, K, dtype):
    G = K // 4
    xg = x.float().view(G, 4)                      # [G, 4]
    return (cb.to(DEV).float() @ xg.t()).to(dtype).contiguous()  # [256,G]


def main():
    out = {"task": "t_bd0c0ac6", "stage": "exp2_dot_precompute",
           "device": torch.cuda.get_device_name(0),
           "rel_tol": REL_TOL, "correctness": [], "perf": {}}
    xs = {}
    for K in (4096, 2048):
        g = torch.Generator(device="cpu")
        g.manual_seed(K)
        xs[K] = torch.randn(K, generator=g).to(DEV).to(torch.float16)

    # ---- correctness: 3 layers x 4 experts x both projs, both dtypes ----
    worst = {"fp32": 0.0, "fp16": 0.0}
    n_pass = {"fp32": 0, "fp16": 0}
    n_tot = 0
    perf_data = None
    for L in LAYERS:
        d = torch.load(PLANES / f"vqa_layer_{L:03d}.pt",
                       map_location="cpu", weights_only=False)
        for proj, ck, sk, cbk, N, K in (
                ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
                ("down", "codes2", "sc2", "cb2", 4096, 2048)):
            cb = d[cbk]
            x = xs[K]
            G = K // 4
            for e in EXPERTS:
                codes = d[ck][e].to(DEV)
                sc = d[sk][e].to(DEV)
                yref = ref_vqa_y(codes, sc, cb, x, N, K)
                n_tot += 1
                for dt_name, dt in (("fp32", torch.float32),
                                    ("fp16", torch.float16)):
                    cbx = make_cbx(cb, x, K, dt)
                    y = torch.empty(N, device=DEV, dtype=torch.float32)
                    grid = ((N + 63) // 64,)
                    vqa_dot_gemv[grid](y, cbx, codes, sc, N, K,
                                       BLOCK_N=64, BLOCK_G=128)
                    rel = (torch.norm(y - yref) / torch.norm(yref)).item()
                    ok = rel <= REL_TOL
                    n_pass[dt_name] += ok
                    worst[dt_name] = max(worst[dt_name], rel)
                    out["correctness"].append(
                        {"layer": L, "expert": e, "proj": proj,
                         "table": dt_name, "relL2": rel, "pass": bool(ok)})
        if L == LAYERS[0]:
            perf_data = d
        else:
            del d
    print(f"correctness: fp32 {n_pass['fp32']}/{n_tot} "
          f"worst {worst['fp32']:.3e} | "
          f"fp16 {n_pass['fp16']}/{n_tot} worst {worst['fp16']:.3e}")
    out["correctness_gate"] = {
        "n_total": n_tot, "n_pass": n_pass,
        "worst_relL2": worst,
        "fp32_pass": bool(n_pass["fp32"] == n_tot),
        "fp16_pass": bool(n_pass["fp16"] == n_tot)}

    # ---- perf loop8 (incl. amortized precompute) vs w2lut ----
    d = perf_data
    es = [0, 17, 63, 101, 128, 177, 200, 255]
    cfgs = [(bn, bg) for bn in (32, 64, 128) for bg in (64, 128, 256)]
    for proj, ck, sk, cbk, N, K in (
            ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
            ("down", "codes2", "sc2", "cb2", 4096, 2048)):
        cb = d[cbk].to(DEV)
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

        # w2lut loop8 baseline (re-measured this run, same conditions)
        sweep = {}
        for bn, bg in cfgs:
            grid = ((N + bn - 1) // bn,)

            def loop_w2(bn=bn, bg=bg, grid=grid):
                for cc in w2_cases:
                    w2lut_gemv[grid](y, x, cc[0], cc[1], N, K, *W2V2_LUT,
                                     BLOCK_N=bn, BLOCK_G=bg)
            t = bench(loop_w2) / 8
            if t < 400:
                sweep[f"{bn}x{bg}"] = round(t, 2)
        bc = min(sweep, key=sweep.get)
        bn, bg = map(int, bc.split("x"))
        grid = ((N + bn - 1) // bn,)

        def loop_w2(bn=bn, bg=bg, grid=grid):
            for cc in w2_cases:
                w2lut_gemv[grid](y, x, cc[0], cc[1], N, K, *W2V2_LUT,
                                 BLOCK_N=bn, BLOCK_G=bg)
        reps = [round(bench(loop_w2) / 8, 2) for _ in range(5)]
        res["w2lut"] = {"best_cfg": bc, "reps": reps, "min": min(reps),
                        "median": round(median(reps), 2)}
        print(f"[{proj}] w2lut loop8 {bc} min={min(reps)} reps={reps}")

        for dt_name, dt in (("fp32", torch.float32),
                            ("fp16", torch.float16)):
            sweep = {}
            for bn, bg in cfgs:
                grid = ((N + bn - 1) // bn,)

                def loop_vq(bn=bn, bg=bg, grid=grid, dt=dt):
                    cbx = make_cbx(cb, x, K, dt)
                    for cc in vq_cases:
                        vqa_dot_gemv[grid](y, cbx, cc[0], cc[1], N, K,
                                           BLOCK_N=bn, BLOCK_G=bg)
                t = bench(loop_vq) / 8
                if t < 400:
                    sweep[f"{bn}x{bg}"] = round(t, 2)
            bc = min(sweep, key=sweep.get)
            bn, bg = map(int, bc.split("x"))
            grid = ((N + bn - 1) // bn,)

            def loop_vq(bn=bn, bg=bg, grid=grid, dt=dt):
                cbx = make_cbx(cb, x, K, dt)
                for cc in vq_cases:
                    vqa_dot_gemv[grid](y, cbx, cc[0], cc[1], N, K,
                                       BLOCK_N=bn, BLOCK_G=bg)
            reps = [round(bench(loop_vq) / 8, 2) for _ in range(5)]
            # precompute-only cost for the record
            tpc = bench(lambda: make_cbx(cb, x, K, dt), iters=100)
            res[f"vqa_dot_{dt_name}"] = {
                "best_cfg": bc, "reps": reps, "min": min(reps),
                "median": round(median(reps), 2),
                "precompute_us": round(tpc, 2),
                "ratio_min": round(min(reps) / res["w2lut"]["min"], 4),
                "ratio_median": round(median(reps)
                                      / res["w2lut"]["median"], 4)}
            print(f"[{proj}] vqa_dot_{dt_name} loop8 {bc} min={min(reps)} "
                  f"pc={tpc:.1f}us ratio_min="
                  f"{res[f'vqa_dot_{dt_name}']['ratio_min']}")
        out["perf"][proj] = res

    with open(OUT / "EXP2_DOT_PRECOMPUTE.json", "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE EXP2_DOT_PRECOMPUTE.json")


if __name__ == "__main__":
    main()
