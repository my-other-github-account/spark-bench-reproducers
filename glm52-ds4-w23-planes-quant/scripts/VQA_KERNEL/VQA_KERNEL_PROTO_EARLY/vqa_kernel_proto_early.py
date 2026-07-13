#!/usr/bin/env python3
"""t_bd0c0ac6 VQA-KERNEL-PROTO-EARLY -- d=4/k=256 gather GEMV on the REAL
(uncalibrated) vqA planes: kernel correctness vs python dequant reference
+ decode-perf verdict vs the W2 LUT path.

Differences vs the stage-3 random-data probe (vqa_gemv_proto.py, sealed in
KERNEL_PROTO.json):
  * REAL plane data: codes/sc/cb from s8-built vqa_layer_{003,013,023}.pt
    (t_426bbc97 stage-2 build, quality anchor NOT yet calibrated -- Banana Bae
    explicitly wants uncalibrated kernel testing).
  * REAL wire scale format: sc bytes are u8 UE8M0; both kernels decode
    exp2(sc - 127) IN-KERNEL (the proto used pre-decoded fp16 scales).
  * Correctness gate per card: kernel output vs python dequant reference
    (gp.sbytes_to_scol convention) at fp32 relL2 <= 1e-3 on
    3 layers x 4 experts x {fused13, down}.
  * Perf arms at identical DRAM traffic: codes N*K/4 bytes both arms
    (2 bits/w), sc bytes identical u8; vqA adds a cached 2KB fp16 codebook.

Arms:
  w2lut : 2-bit packed codes, W2v2 winner LUT as register selects
          (strongest possible LUT baseline -- zero gather traffic,
          stands in for the SASS-baked moe_w2_mm LUT path)
  vqa   : d=4 k=256 -- one u8 code per 4 weights along K, codebook
          [256,4] fp16 gathered per group (REAL shipped codes)

Shapes: fused13 [4096 x 4096], down [4096 x 2048] (DS4 expert dims).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

torch.manual_seed(0)
DEV = "cuda"
PLANES = Path(os.path.expanduser("~/missions/VQA_KERNEL_PROTO_EARLY/planes"))
OUT = Path(os.path.expanduser("~/missions/VQA_KERNEL_PROTO_EARLY"))
LAYERS = [3, 13, 23]
EXPERTS = [0, 17, 128, 255]
REL_TOL = 1e-3
# W2v2 winner grid (W2V2_SHOOTOUT.json luts.dp_asym4_round2, t_bd7728ee)
W2V2_LUT = [-3.5111107379486137, -1.1800192351581362,
            0.6510809470728273, 2.7868641002011136]

# GPU-courteous gate (s1 convention, vqa_build.py verbatim)
GATE_PATTERNS = "bq_mmlu|bq_s1rows|llama-perplexity|r6arm_rail"


def gpu_gate():
    r = subprocess.run(["pgrep", "-af", GATE_PATTERNS],
                       capture_output=True, text=True)
    hits = [ln for ln in r.stdout.splitlines()
            if "pgrep" not in ln and ln.strip()]
    if hits:
        print(f"GPU BUSY ({hits[0][:100]}) -- refusing to launch")
        sys.exit(2)


@triton.jit
def w2lut_gemv(y_ptr, x_ptr, codes_ptr, sc_ptr, N, K,
               L0: tl.constexpr, L1: tl.constexpr,
               L2: tl.constexpr, L3: tl.constexpr,
               BLOCK_N: tl.constexpr, BLOCK_G: tl.constexpr):
    """codes: u8 packed 2-bit x4 [N, K//4]; sc: u8 UE8M0 [N, K//32]."""
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
    """codes: u8 group code [N, K//4]; sc: u8 UE8M0 [N, K//32];
    cb: fp16 [256, 4] gathered per group."""
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


def bench(fn, iters=100, warmup=25):
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
    return s.elapsed_time(e) / iters * 1e3  # us


def scol_ref(sc_u8, K):
    """gp.sbytes_to_scol convention: exp2(byte - 127), blk32 expand."""
    return torch.exp2(sc_u8.to(DEV).float() - 127.0).repeat_interleave(
        32, dim=1).clamp_min(1e-38)


def ref_vqa_y(codes, sc_u8, cb, x, N, K):
    W = cb.to(DEV).float()[codes.to(DEV).long()].reshape(N, K)
    W = W * scol_ref(sc_u8, K)
    return W @ x.float()


def ref_w2_y(c2, sc_u8, lut, x, N, K):
    idx = torch.stack([(c2.to(DEV).int() >> (2 * j)) & 3
                       for j in range(4)], dim=2).long()
    W = lut[idx].reshape(N, K) * scol_ref(sc_u8, K)
    return W @ x.float()


def main():
    gpu_gate()
    out = {"task": "t_bd0c0ac6", "stage": "3_kernel_proto_early_realplanes",
           "device": torch.cuda.get_device_name(0),
           "torch": torch.__version__, "triton": triton.__version__,
           "planes": "s8 t_426bbc97 stage-2 build (UNCALIBRATED)",
           "layers": LAYERS, "experts": EXPERTS,
           "rel_tol": REL_TOL, "correctness": [], "perf": {}}
    lut = torch.tensor(W2V2_LUT, dtype=torch.float32, device=DEV)

    # ---------------- correctness: real planes, all layers/experts -----
    worst = {"fused13": 0.0, "down": 0.0}
    xs = {}
    for K in (4096, 2048):
        g = torch.Generator(device="cpu")
        g.manual_seed(K)
        xs[K] = torch.randn(K, generator=g).to(DEV).to(torch.float16)

    perf_data = None  # keep L3 tensors for the perf section
    n_pass = 0
    for L in LAYERS:
        t0 = time.time()
        d = torch.load(PLANES / f"vqa_layer_{L:03d}.pt",
                       map_location="cpu", weights_only=False)
        print(f"[L{L:03d}] plane loaded in {time.time() - t0:.0f}s")
        for proj, ck, sk, cbk, N, K in (
                ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
                ("down", "codes2", "sc2", "cb2", 4096, 2048)):
            cb = d[cbk].to(DEV)
            x = xs[K]
            for e in EXPERTS:
                codes = d[ck][e].to(DEV)
                sc = d[sk][e].to(DEV)
                yref = ref_vqa_y(codes, sc, cb, x, N, K)
                y = torch.empty(N, device=DEV, dtype=torch.float32)
                grid = ((N + 63) // 64,)
                vqa_gemv[grid](y, x, codes, sc, cb, N, K,
                               BLOCK_N=64, BLOCK_G=128)
                rel = (torch.norm(y - yref) / torch.norm(yref)).item()
                mx = (y - yref).abs().max().item()
                ok = rel <= REL_TOL
                n_pass += ok
                worst[proj] = max(worst[proj], rel)
                out["correctness"].append(
                    {"layer": L, "expert": e, "proj": proj,
                     "relL2": rel, "max_abs": mx, "pass": bool(ok)})
                print(f"[L{L:03d} e{e:03d} {proj:7s}] relL2={rel:.3e} "
                      f"max_abs={mx:.3e} {'PASS' if ok else 'FAIL'}")
        if L == LAYERS[0]:
            perf_data = d
        else:
            del d
    n_tot = len(out["correctness"])
    out["correctness_gate"] = {
        "n_pass": n_pass, "n_total": n_tot,
        "worst_relL2": worst,
        "pass": bool(n_pass == n_tot)}
    print(f"\nCORRECTNESS GATE: {n_pass}/{n_tot} "
          f"worst relL2 f13={worst['fused13']:.3e} "
          f"down={worst['down']:.3e} "
          f"-> {'PASS' if n_pass == n_tot else 'FAIL'}")

    # ---------------- perf: vqA (real L3 codes) vs W2 LUT path ---------
    d = perf_data
    cfgs = [(bn, bg) for bn in (32, 64, 128) for bg in (64, 128, 256)]
    for proj, ck, sk, cbk, N, K in (
            ("fused13", "codes13", "sc13", "cb13", 4096, 4096),
            ("down", "codes2", "sc2", "cb2", 4096, 2048)):
        cb = d[cbk].to(DEV)
        x = xs[K]
        G = K // 4
        # 8 real experts for the decode-like loop (real gather statistics)
        es = [0, 17, 63, 101, 128, 177, 200, 255]
        vq_cases = [(d[ck][e].to(DEV), d[sk][e].to(DEV)) for e in es]
        # w2 arm: random packed 2-bit codes at identical byte count,
        # same real sc bytes (scale traffic identical by construction)
        g = torch.Generator(device="cpu")
        g.manual_seed(1234 + K)
        w2_cases = [(torch.randint(0, 256, (N, G), generator=g,
                                   dtype=torch.uint8).to(DEV), scb)
                    for _, scb in vq_cases]
        y = torch.empty(N, device=DEV, dtype=torch.float32)
        grid = lambda bn: ((N + bn - 1) // bn,)

        # w2 kernel correctness (sanity, same tolerance)
        yr2 = ref_w2_y(w2_cases[0][0], w2_cases[0][1], lut, x, N, K)
        w2lut_gemv[grid(64)](y, x, w2_cases[0][0], w2_cases[0][1], N, K,
                             *W2V2_LUT, BLOCK_N=64, BLOCK_G=128)
        e2 = (torch.norm(y - yr2) / torch.norm(yr2)).item()
        print(f"[{proj}] w2lut sanity relL2={e2:.3e}")
        assert e2 <= REL_TOL, "W2LUT ARM CORRECTNESS FAIL"

        best = {}
        for arm, fn in (
                ("w2lut", lambda bn, bg, cc: w2lut_gemv[grid(bn)](
                    y, x, cc[0], cc[1], N, K, *W2V2_LUT,
                    BLOCK_N=bn, BLOCK_G=bg)),
                ("vqa", lambda bn, bg, cc: vqa_gemv[grid(bn)](
                    y, x, cc[0], cc[1], cb, N, K,
                    BLOCK_N=bn, BLOCK_G=bg))):
            cases = w2_cases if arm == "w2lut" else vq_cases
            times = {}
            for bn, bg in cfgs:
                t = bench(lambda: fn(bn, bg, cases[0]))
                times[f"{bn}x{bg}"] = round(t, 2)
            k = min(times, key=times.get)
            best[arm] = {"cfg": k, "us": times[k], "sweep": times}
            print(f"[{proj}] {arm}: best {k} = {times[k]:.2f} us")

        bn2, bg2 = map(int, best["w2lut"]["cfg"].split("x"))
        bnv, bgv = map(int, best["vqa"]["cfg"].split("x"))

        def loop_w2():
            for cc in w2_cases:
                w2lut_gemv[grid(bn2)](y, x, cc[0], cc[1], N, K, *W2V2_LUT,
                                      BLOCK_N=bn2, BLOCK_G=bg2)

        def loop_vqa():
            for cc in vq_cases:
                vqa_gemv[grid(bnv)](y, x, cc[0], cc[1], cb, N, K,
                                    BLOCK_N=bnv, BLOCK_G=bgv)

        t8_2 = bench(loop_w2, iters=50, warmup=10) / 8
        t8_v = bench(loop_vqa, iters=50, warmup=10) / 8

        Wb = torch.randn(N, K, device=DEV, dtype=torch.bfloat16)
        tb = bench(lambda: torch.mv(Wb, x.to(torch.bfloat16)))
        del Wb

        r1 = best["vqa"]["us"] / best["w2lut"]["us"]
        r8 = t8_v / t8_2
        print(f"[{proj}] single vqa/w2lut = {r1:.3f}x ; "
              f"loop8 = {r8:.3f}x ; bf16 mv {tb:.1f} us")
        out["perf"][proj] = {
            "N": N, "K": K,
            "bytes_per_expert_proj": N * K // 4 + N * K // 32,
            "w2lut": best["w2lut"], "vqa": best["vqa"],
            "loop8_us_per_gemv": {"w2lut": round(t8_2, 2),
                                  "vqa": round(t8_v, 2)},
            "bf16_mv_us": round(tb, 2),
            "ratio_single": round(r1, 4),
            "ratio_loop8": round(r8, 4)}

    rs = [out["perf"][p][f"ratio_{m}"]
          for p in out["perf"] for m in ("single", "loop8")]
    out["max_ratio"] = max(rs)
    out["perf_gate"] = {
        "rule": "40%-track budget: vqA within 1.4x of W2-LUT GEMV",
        "max_ratio": out["max_ratio"],
        "pass": bool(out["max_ratio"] <= 1.4)}
    out["verdict"] = ("PASS" if (out["correctness_gate"]["pass"]
                                 and out["perf_gate"]["pass"]) else "FAIL")
    print(f"\nPERF GATE: max ratio {out['max_ratio']:.3f} "
          f"-> {'PASS' if out['perf_gate']['pass'] else 'FAIL'}")
    print(f"VERDICT: {out['verdict']}")
    with open(OUT / "KERNEL_PROTO_EARLY.json", "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE KERNEL_PROTO_EARLY.json")


if __name__ == "__main__":
    main()
