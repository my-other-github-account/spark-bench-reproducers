#!/usr/bin/env python3
"""t_426bbc97 STAGE 2 -- build the vqA d=4/k=256 tier planes on spark-1.

Slice: ALL 256 experts x {fused13, down} for the 5 staged layers
L{3,13,23,33,41} (the GPTQv2-pilot staging = damage-band spread across
depth). This enables the iso-bytes rail A/B: R4v2 (all W2v2-GPTQ)
baseline vs the same model with these 5 layers swapped to vqA -- vqA
bytes == w2 bytes per projection, so the swap is exactly iso-bytes
(+2KB/layer/proj codebook).

Per (layer, proj): layer-shared codebook (d=4 k=256) fit on FIT_E
experts' u-vectors, s^2-weighted lloyd, seed 0 -- the pilot's
shared_cbs(A) verbatim. Per expert: W2v2 SSE scale refit (sc bytes),
then TWO assignment arms:
  nn : nearest-neighbor codes (RTN-class)
  hg : VQ-GPTQ group assignment (vq_gptq, the pilot's vqA_sh_hg arm)
Ship = val-gated (2% margin, G4X convention): hg unless its val proxy
is >2% worse than nn (or no H rows -> nn).

Output per layer: ~/missions/VQA_TIER/planes/vqa_layer_NNN.pt
  {codes13 u8 [256,4096,1024], sc13 u8 [256,4096,128],
   codes2 u8 [256,4096,512],  sc2 u8 [256,4096,64],
   cb13 fp16 [256,4], cb2 fp16 [256,4], meta}
Ledger: out/VQA_BUILD_LEDGER.jsonl (resume-safe, one row per expert).
"""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

MISSION = Path(os.path.expanduser("~/missions/VQA_TIER"))
PILOT = Path(os.path.expanduser("~/missions/DS4_GPTQV2_PILOT"))
sys.path.insert(0, str(PILOT))
sys.path.insert(0, str(MISSION))
import gptqv2_pilot as gp  # noqa: E402

gp.M = str(PILOT)
from vqw2_pilot_lib import (assign_chunk, kmeanspp, lloyd,  # noqa: E402
                            vq_gptq)

DEV = "cuda"
SMOKE = os.environ.get("VQA_SMOKE", "0") == "1"
LAYERS = [3] if SMOKE else [3, 13, 23, 33, 41]
N_EXPERTS = 4 if SMOKE else 256
FIT_E = [17, 77, 177]
LLOYD_ITERS = 15
SEED = 0
MIN_FIT_ROWS = 64
VAL_MARGIN = 0.02
OUT = MISSION / "out"
PLANES = MISSION / "planes"
LEDGER = OUT / ("VQA_BUILD_LEDGER_SMOKE.jsonl" if SMOKE
                else "VQA_BUILD_LEDGER.jsonl")

STOP = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: STOP.set())
signal.signal(signal.SIGINT, lambda *a: STOP.set())

# GPU-courteous gate: yield to the s1 MMLU offload lane (t_9f643585,
# bq_mmlu/bq-s1rows) whenever it fires -- its rows are 16-min bursts.
GATE_PATTERNS = "bq_mmlu|bq_s1rows|llama-perplexity|r6arm_rail"


def gpu_gate():
    while not STOP.is_set():
        r = subprocess.run(["pgrep", "-af", GATE_PATTERNS],
                           capture_output=True, text=True)
        hits = [ln for ln in r.stdout.splitlines()
                if "pgrep" not in ln and ln.strip()]
        if not hits:
            return
        log(f"gpu-gate wait: {hits[0][:100]}")
        time.sleep(60)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def jrow(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def relrms(dq, W):
    return ((dq.float() - W.float()).norm()
            / (W.float().norm() + 1e-30)).item()


def load_caps(L, wins):
    xs, tks = [], []
    for gid in wins:
        d = torch.load(f"{gp.M}/cap/xmoe_L{L:03d}_win{gid:04d}.pt",
                       map_location="cpu")
        xs.append(d["x"])
        tks.append(d["topk"].to(torch.int64))
    x = torch.cat(xs, 0).to(DEV)
    tk = torch.cat(tks, 0).to(DEV)
    hit = torch.zeros(x.shape[0], 256, dtype=torch.bool, device=DEV)
    hit.scatter_(1, tk, True)
    return x, hit


def fit_shared_cb(bundle, proj, gen):
    us, ss = [], []
    for e in FIT_E:
        W, sb = (bundle.fused13(e) if proj == "fused13"
                 else bundle.down(e))
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
    cb = lloyd(V4, w4, kmeanspp(V4, w4, 256, gen), LLOYD_ITERS, gen)
    del V4, w4, Uall, Sall
    torch.cuda.empty_cache()
    return cb


def identity_gate(bundle):
    W, sb = bundle.down(FIT_E[0])
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
    mism = (codes_mine.reshape(-1)
            != codes_ref.reshape(-1).long()).float().mean().item()
    log(f"identity gate d=1 vs scalar gptq_loop: mismatch={mism:.2e}")
    if mism > 1e-4:
        raise SystemExit(f"IDENTITY GATE FAIL mism={mism}")
    del W, sb, X, H, U, codes_ref, codes_mine
    torch.cuda.empty_cache()


def build_unit(W, sb, cb, Xf, Xv):
    """-> codes u8 [N, K//4], sc u8 [N, K//32], row-metrics dict."""
    N, K = W.shape
    _, sc = gp.w2v2_requant(W, sb)
    s_col = gp.sbytes_to_scol(sc)
    V4 = (W / s_col).view(N, K // 4, 4).reshape(-1, 4)
    a_nn = assign_chunk(V4, cb).view(N, K // 4)
    dq_nn = cb[a_nn.reshape(-1)].view(N, K // 4, 4).reshape(N, K) * s_col
    pv_nn = gp.proxy_sym(Xv, dq_nn, W)
    rr_nn = relrms(dq_nn, W)
    del V4
    have_h = Xf is not None and Xf.shape[0] >= MIN_FIT_ROWS
    row = {"relrms_nn": round(rr_nn, 6),
           "pv_nn": round(pv_nn, 6) if pv_nn is not None else None,
           "n_fit": int(Xf.shape[0]) if Xf is not None else 0,
           "n_val": int(Xv.shape[0]) if Xv is not None else 0}
    if have_h:
        H = Xf.t() @ Xf
        U = gp.gptq_prepare(H, torch.arange(K, device=DEV))
        del H
        dq_hg, codes_hg = vq_gptq(W, s_col, [cb], U, 4)
        del U
        pv_hg = gp.proxy_sym(Xv, dq_hg, W)
        row["relrms_hg"] = round(relrms(dq_hg, W), 6)
        row["pv_hg"] = round(pv_hg, 6) if pv_hg is not None else None
        ship_hg = (pv_hg is not None and pv_nn is not None
                   and pv_hg <= pv_nn * (1 + VAL_MARGIN))
        row["ship"] = "hg" if ship_hg else "nn"
        codes = codes_hg.to(torch.uint8) if ship_hg else a_nn.to(torch.uint8)
        del dq_hg, codes_hg
    else:
        row["ship"] = "nn_noH"
        codes = a_nn.to(torch.uint8)
    del a_nn, dq_nn
    return codes, sc, row


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PLANES.mkdir(parents=True, exist_ok=True)
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
    log(f"smoke={SMOKE} resume: {len(done)} units in ledger")

    for L in LAYERS:
        marker = PLANES / f"vqa_layer_{L:03d}.DONE"
        if marker.exists():
            log(f"L{L:03d} plane file sealed, skip")
            continue
        bundle = gp.WtsBundle(L)
        log(f"bundle L{L:03d} loaded")
        if L == LAYERS[0]:
            identity_gate(bundle)
        cbs = {}
        for proj in ("fused13", "down"):
            tcb = time.time()
            cbs[proj] = fit_shared_cb(bundle, proj, gen)
            log(f"L{L:03d} {proj} shared cb fit in {time.time() - tcb:.0f}s")
        xf, hitf = load_caps(L, sel["fit_ids"])
        xv, hitv = load_caps(L, sel["val_ids"])
        log(f"L{L:03d} caps: fit {tuple(xf.shape)} val {tuple(xv.shape)}")

        c13 = torch.zeros(N_EXPERTS, 4096, 1024, dtype=torch.uint8)
        s13 = torch.zeros(N_EXPERTS, 4096, 128, dtype=torch.uint8)
        c2 = torch.zeros(N_EXPERTS, 4096, 512, dtype=torch.uint8)
        s2 = torch.zeros(N_EXPERTS, 4096, 64, dtype=torch.uint8)
        part = PLANES / f"vqa_layer_{L:03d}.partial.pt"
        if part.exists():
            d = torch.load(part, map_location="cpu")
            c13, s13, c2, s2 = d["codes13"], d["sc13"], d["codes2"], d["sc2"]
            log(f"L{L:03d} partial plane state loaded")
        for e in range(N_EXPERTS):
            u13 = f"L{L:03d}_e{e:03d}_fused13"
            u2 = f"L{L:03d}_e{e:03d}_down"
            if {u13, u2} <= done:
                continue
            if STOP.is_set():
                log(f"graceful stop at L{L} e{e}")
                tmp = part.with_suffix(".tmp")
                torch.save({"codes13": c13, "sc13": s13,
                            "codes2": c2, "sc2": s2}, tmp)
                os.replace(tmp, part)
                return 1
            gpu_gate()
            te = time.time()
            Xf = xf[hitf[:, e]].float()
            Xv = xv[hitv[:, e]].float()
            W13, sb13 = bundle.fused13(e)
            if u13 not in done:
                codes, sc, row = build_unit(W13, sb13, cbs["fused13"],
                                            Xf, Xv)
                c13[e] = codes.cpu()
                s13[e] = sc.cpu()
                row.update(unit=u13, layer=L, expert=e, proj="fused13",
                           secs=round(time.time() - te, 1))
                jrow(LEDGER, row)
            if u2 not in done:
                td = time.time()
                Wd, sb2 = bundle.down(e)
                Af = gp.act(Xf, W13[:2048], W13[2048:])
                Av = gp.act(Xv, W13[:2048], W13[2048:])
                codes, sc, row = build_unit(Wd, sb2, cbs["down"], Af, Av)
                c2[e] = codes.cpu()
                s2[e] = sc.cpu()
                row.update(unit=u2, layer=L, expert=e, proj="down",
                           secs=round(time.time() - td, 1))
                jrow(LEDGER, row)
                del Wd, sb2, Af, Av
            del W13, sb13, Xf, Xv
            if e % 16 == 15:
                torch.cuda.empty_cache()
                tmp = part.with_suffix(".tmp")
                torch.save({"codes13": c13, "sc13": s13,
                            "codes2": c2, "sc2": s2}, tmp)
                os.replace(tmp, part)
                log(f"L{L:03d} e{e} checkpointed "
                    f"({time.time() - t0:.0f}s elapsed)")
        meta = {"task": "t_426bbc97", "tier": "vqA", "d": 4, "k": 256,
                "layer": L, "codebook": "layer-shared per-proj, "
                f"fit_e={FIT_E}, lloyd={LLOYD_ITERS}, seed={SEED}",
                "scales": "W2v2 SSE refit (per-blk32 UE8M0)",
                "assign": "val-gated nn/vq_gptq (2% margin)",
                "down_H_note": "A_fp teacher-side (pilot convention)"}
        fn = PLANES / f"vqa_layer_{L:03d}.pt"
        tmp = fn.with_suffix(".tmp")
        torch.save({"codes13": c13, "sc13": s13, "codes2": c2, "sc2": s2,
                    "cb13": cbs["fused13"].to(torch.float16).cpu(),
                    "cb2": cbs["down"].to(torch.float16).cpu(),
                    "meta": meta}, tmp)
        os.replace(tmp, fn)
        part.unlink(missing_ok=True)
        marker.write_text("done\n")
        log(f"L{L:03d} SEALED -> {fn}")
        del bundle, xf, hitf, xv, hitv, c13, s13, c2, s2
        torch.cuda.empty_cache()

    # summary
    rows = [json.loads(x) for x in open(LEDGER)]
    import statistics as st
    summ = {"task": "t_426bbc97", "stage": "2_build",
            "layers": LAYERS, "n_units": len(rows),
            "ship_counts": {}, "elapsed_s": round(time.time() - t0, 1)}
    for proj in ("fused13", "down"):
        rs = [r for r in rows if r["proj"] == proj]
        if not rs:
            continue
        ships = {}
        for r in rs:
            ships[r["ship"]] = ships.get(r["ship"], 0) + 1
        mw = sum(r["n_val"] for r in rs)
        pv_ship = sum(r["n_val"] * (r["pv_hg"] if r["ship"] == "hg"
                                    else r["pv_nn"]) for r in rs) / mw
        summ[proj] = {
            "ship": ships,
            "relrms_nn_mean": round(st.mean(r["relrms_nn"] for r in rs), 6),
            "pv_ship_massw": round(pv_ship, 6)}
        summ["ship_counts"][proj] = ships
    with open(OUT / "VQA_BUILD_SUMMARY.json", "w") as f:
        json.dump(summ, f, indent=1)
    (OUT / "BUILD_DONE").write_text("OK\n")
    log(f"ALL DONE {time.time() - t0:.0f}s -> VQA_BUILD_SUMMARY.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
