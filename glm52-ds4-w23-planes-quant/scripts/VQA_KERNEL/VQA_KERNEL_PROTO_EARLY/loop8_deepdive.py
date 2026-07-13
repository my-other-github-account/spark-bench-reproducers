#!/usr/bin/env python3
"""t_bd0c0ac6 follow-up: down-shape loop8 deep dive.

The main run FAILed the 1.4x budget only on down/loop8 (1.496x) using the
single-GEMV-best config for each arm. Two hypotheses to retire before a
verdict:
  H1 config: loop8 optimum != single optimum (autotune the LOOP, both arms)
  H2 noise : run-to-run variance on a 30us kernel

Method: full 9-config sweep per arm ON THE 8-EXPERT LOOP, 5 repeats of the
best config, report min/median. Also repeat fused13 loop8 for completeness.
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
def vqa_gemv(y_ptr, x_ptr, codes_ptr, sc_ptr, cb_ptr, N, K,
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
                        mask=m2, other=0).to(tl.int32)
        scb = tl.load(sc_ptr + n_offs[:, None] * SG + (g_offs[None, :] // 8),
                      mask=m2, other=127).to(tl.float32)
        sc = tl.exp2(scb - 127.0)
        part = tl.zeros((BLOCK_N, BLOCK_G), dtype=tl.float32)
        for j in tl.static_range(4):
            w = tl.load(cb_ptr + codes * 4 + j).to(tl.float32)
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


def main():
    out = {"task": "t_bd0c0ac6", "stage": "loop8_deepdive",
           "device": torch.cuda.get_device_name(0), "shapes": {}}
    d = torch.load(PLANES / "vqa_layer_003.pt", map_location="cpu",
                   weights_only=False)
    es = [0, 17, 63, 101, 128, 177, 200, 255]
    cfgs = [(bn, bg) for bn in (32, 64, 128) for bg in (64, 128, 256)]
    for proj, ck, sk, cbk, N, K in (
            ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
            ("down", "codes2", "sc2", "cb2", 4096, 2048)):
        cb = d[cbk].to(DEV)
        G = K // 4
        g = torch.Generator(device="cpu")
        g.manual_seed(K)
        x = torch.randn(K, generator=g).to(DEV).to(torch.float16)
        vq_cases = [(d[ck][e].to(DEV), d[sk][e].to(DEV)) for e in es]
        g2 = torch.Generator(device="cpu")
        g2.manual_seed(1234 + K)
        w2_cases = [(torch.randint(0, 256, (N, G), generator=g2,
                                   dtype=torch.uint8).to(DEV), scb)
                    for _, scb in vq_cases]
        y = torch.empty(N, device=DEV, dtype=torch.float32)
        res = {}
        for arm, cases in (("w2lut", w2_cases), ("vqa", vq_cases)):
            sweep = {}
            for bn, bg in cfgs:
                grid = ((N + bn - 1) // bn,)
                if arm == "w2lut":
                    def loop():
                        for cc in cases:
                            w2lut_gemv[grid](y, x, cc[0], cc[1], N, K,
                                             *W2V2_LUT,
                                             BLOCK_N=bn, BLOCK_G=bg)
                else:
                    def loop():
                        for cc in cases:
                            vqa_gemv[grid](y, x, cc[0], cc[1], cb, N, K,
                                           BLOCK_N=bn, BLOCK_G=bg)
                t = bench(loop) / 8
                if t < 400:  # skip pathological configs in repeats
                    sweep[f"{bn}x{bg}"] = round(t, 2)
            best_cfg = min(sweep, key=sweep.get)
            bn, bg = map(int, best_cfg.split("x"))
            grid = ((N + bn - 1) // bn,)
            if arm == "w2lut":
                def loop():
                    for cc in cases:
                        w2lut_gemv[grid](y, x, cc[0], cc[1], N, K,
                                         *W2V2_LUT, BLOCK_N=bn, BLOCK_G=bg)
            else:
                def loop():
                    for cc in cases:
                        vqa_gemv[grid](y, x, cc[0], cc[1], cb, N, K,
                                       BLOCK_N=bn, BLOCK_G=bg)
            reps = [round(bench(loop) / 8, 2) for _ in range(5)]
            res[arm] = {"loop8_sweep": sweep, "best_cfg": best_cfg,
                        "reps": reps, "min": min(reps),
                        "median": round(median(reps), 2)}
            print(f"[{proj}] {arm} loop8 best {best_cfg} "
                  f"reps={reps} min={min(reps)}")
        rmin = res["vqa"]["min"] / res["w2lut"]["min"]
        rmed = res["vqa"]["median"] / res["w2lut"]["median"]
        res["ratio_loop8_min"] = round(rmin, 4)
        res["ratio_loop8_median"] = round(rmed, 4)
        print(f"[{proj}] loop8 ratio: min {rmin:.3f}x median {rmed:.3f}x")
        out["shapes"][proj] = res
    with open(OUT / "LOOP8_DEEPDIVE.json", "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE LOOP8_DEEPDIVE.json")


if __name__ == "__main__":
    main()
