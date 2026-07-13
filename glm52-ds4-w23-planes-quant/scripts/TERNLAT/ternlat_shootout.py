#!/usr/bin/env python3
"""t_426bbc97 STAGE-2a -- ternary lattice VQ shootout (runs on spark-3).

Banana Bae scope addition (Jul12): compare the BASIC ternary rung (pilot arm,
independent per-weight {-1,0,+1} + SSE scales, 1.85 bpw) against d=8
ternary LATTICE VQ (IQ1_S-style: codes restricted to 2,048 curated ternary
patterns, 11 bits / 8 weights = 1.375 bpw + scales):

  tern_lat_crib_nn  : iq1s_grid's 2048 patterns VERBATIM (llama.cpp
                      ggml-common.h on s8, decoded little-endian int8
                      bytes {-1,0,+1}), nearest-neighbor assignment.
  tern_lat_crib_hg  : same codebook, Hessian-aware VQ-GPTQ assignment
                      (pilot vq_gptq, identity-gated d=1 == scalar gptq).
  tern_lat_refit_nn : OUR 2048 ternary patterns -- s^2 mass-weighted
                      selection over ternary-projected d=8 groups of the
                      GOLD-CALIB fit experts (FIT_E), nearest assignment.
  tern_lat_refit_hg : refit codebook + VQ-GPTQ assignment.

Protocol: SAME 36 units as the sealed pilot (t_fa2eafed): layers {3,23,41}
x eval experts {9,50,100,150,200,254} x {fused13,down}; seed 0; W2v2 SSE
scale-refit convention (per-block-32 UE8M0, offsets -4..+4) with the
symmetric lut [-m, 0, +m], m = mean(|a|,b) of the sealed TERNARY_LUTS.json
global per-projection levels (a,b are symmetric to 3 decimals anyway).
Codebook = m * G (G in {-1,0,+1}^{2048x8}), layer+expert-shared per
projection for crib; layer-shared per projection for refit (fit on FIT_E,
the pilot shared_cbs convention -- GOLD-CALIB enters via the fit-expert
selection and the H/val eval, mass = s^2).

Rails (pilot rail_note convention):
  nn arms  (RTN-class)  -> relRMS ratio vs pilot 'ternary' arm (0.446928).
  hg arms  (GPTQ-class) -> val-proxy ratio vs pilot 'ternary_gptq'
                           (proxy_val_mean 0.420141).
Winner rule (card): lattice beats basic by >5% on its rail -> takes the
KLD-rail cold-tail slot. Bytes: lattice 1.375+0.25 bpw < basic 1.85 bpw
(strictly cheaper, so an error win is a strict domination; iso-bytes is
satisfied from below). Codebook 2048x8 fp16 = 32KiB/layer/proj amortized
over 256 experts (~6e-5 bpw, counted).

Basic-ternary per-unit rows are JOINED from the sealed pilot ledger
(VQW2_LEDGER.jsonl) -- same units, same seed, same eval windows.

Ledger: out/TERNLAT_LEDGER.jsonl (resume-safe). Summary:
out/TERNLAT_SHOOTOUT.json + out/TERNLAT_TABLE.md. Sentinel: out/TERNLAT_DONE.
"""
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

MISSION = Path(os.path.expanduser("~/missions/TERNLAT_SHOOTOUT"))
PILOT = Path(os.environ.get("VQ_MISSION",
                            str(Path.home() / "missions/VQ_W2_PILOT")))
sys.path.insert(0, str(PILOT))
import vqw2_pilot as vp  # noqa: E402  (sealed pilot machinery, verbatim)

gp = vp.gp
DEV = "cuda"
SMOKE = os.environ.get("TL_SMOKE", "0") == "1"
LAYERS = [3] if SMOKE else [3, 23, 41]
EVAL_E = [9] if SMOKE else [9, 50, 100, 150, 200, 254]
FIT_E = [17, 77, 177]
SEED = 0
MIN_FIT_ROWS = 64
T_OFFSETS = list(range(-4, 5))
NPAT = 2048
D = 8
OUT = MISSION / "out"
LEDGER = OUT / ("TERNLAT_LEDGER_SMOKE.jsonl" if SMOKE
                else "TERNLAT_LEDGER.jsonl")
SUMMARY = OUT / ("TERNLAT_SHOOTOUT_SMOKE.json" if SMOKE
                 else "TERNLAT_SHOOTOUT.json")

ARMS = ["tern_lat_crib_nn", "tern_lat_crib_hg",
        "tern_lat_refit_nn", "tern_lat_refit_hg"]

signal.signal(signal.SIGTERM, lambda *a: gp.STOP.set())
signal.signal(signal.SIGINT, lambda *a: gp.STOP.set())


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


def nweights(proj):
    return 4096 * 4096 if proj == "fused13" else 4096 * 2048


def bpw_lat(proj):
    # 11 bits / 8 weights + 0.25 scale + fp16 codebook shared across the
    # layer's 256 experts (pilot bpw_vq_shared convention, share=256).
    cb_bytes = NPAT * D * 2
    return math.log2(NPAT) / D + 0.25 + cb_bytes * 8.0 / (256 * nweights(proj))


def arm_metrics(dq, W, Xv, bpw):
    pv = gp.proxy_sym(Xv, dq, W) if (Xv is not None and Xv.shape[0] > 0) else None
    return {"relrms": round(relrms(dq, W), 6),
            "proxy_val": (round(pv, 6) if pv is not None else None),
            "bpw": round(bpw, 4)}


# ------------------------------------------------------------ codebooks
def load_crib_grid():
    pats = json.load(open(MISSION / "iq1s_grid_2048.json"))
    G = torch.tensor(pats, dtype=torch.float32, device=DEV)
    assert G.shape == (NPAT, D)
    assert set(G.unique().tolist()) <= {-1.0, 0.0, 1.0}
    return G


POW3 = None


def pat_ids(T):
    """T [n,8] in {-1,0,1} -> base-3 ids [n] int64."""
    global POW3
    if POW3 is None:
        POW3 = (3 ** torch.arange(D, device=DEV, dtype=torch.long))
    return ((T.long() + 1) * POW3).sum(1)


def ids_to_pats(ids):
    d = []
    x = ids.clone()
    for _ in range(D):
        d.append((x % 3) - 1)
        x //= 3
    return torch.stack(d, 1).float()


REFIT = {}


def refit_grid(L, proj, m):
    """Top-2048 s^2-mass ternary patterns of the FIT_E experts' projected
    d=8 groups (layer-shared per projection; scale refit via lut [-m,0,m],
    the same convention the eval units use)."""
    key = (L, proj)
    if key in REFIT:
        return REFIT[key]
    t0 = time.time()
    mass = torch.zeros(3 ** D, device=DEV)
    lut_m = torch.tensor([-m, 0.0, m], dtype=torch.float32, device=DEV)
    for e in FIT_E:
        W, sb = get_mat(L, e, proj)
        _, sc, _ = vp.requant_lut(W, sb, lut_m, T_OFFSETS)
        s_col = gp.sbytes_to_scol(sc)
        u = W / s_col
        V8 = u.view(-1, D)
        s8 = s_col.view(-1, D)[:, 0]
        T = torch.zeros_like(V8)
        T[V8 > (m / 2)] = 1.0
        T[V8 < (-m / 2)] = -1.0
        mass.index_add_(0, pat_ids(T), s8 * s8)
        del W, sb, sc, s_col, u, V8, s8, T
    torch.cuda.empty_cache()
    top = mass.topk(NPAT)
    G = ids_to_pats(top.indices)
    share = float(top.values.sum() / mass.sum().clamp_min(1e-30))
    REFIT[key] = (G, share)
    log(f"refit grid L{L:03d} {proj}: top-{NPAT} mass share {share:.4f} "
        f"({time.time() - t0:.0f}s)")
    return REFIT[key]


def grid_stats(G_crib, G_refit):
    a = set(map(tuple, G_crib.long().tolist()))
    b = set(map(tuple, G_refit.long().tolist()))
    return len(a & b)


# ------------------------------------------------------------ data access
BUNDLES = {}


def get_mat(L, e, proj):
    b = BUNDLES[L]
    return b.fused13(e) if proj == "fused13" else b.down(e)


# ------------------------------------------------------------ unit runner
def run_unit(L, e, proj, W, sb, Xf, Xv, m, G_crib, gen):
    te = time.time()
    N, K = W.shape
    res = {}
    lut_m = torch.tensor([-m, 0.0, m], dtype=torch.float32, device=DEV)
    _, sc, _ = vp.requant_lut(W, sb, lut_m, T_OFFSETS)
    s_col = gp.sbytes_to_scol(sc)
    u = W / s_col
    V8 = u.view(-1, D)
    bpw = bpw_lat(proj)

    have_h = Xf is not None and Xf.shape[0] >= MIN_FIT_ROWS
    U = None
    if have_h:
        H = Xf.t() @ Xf
        U = gp.gptq_prepare(H, torch.arange(K, device=DEV))
        del H

    G_refit, share = refit_grid(L, proj, m)
    cbs = {"crib": m * G_crib, "refit": m * G_refit}
    for name, C in cbs.items():
        a = vp.assign_chunk(V8, C)
        dq = C[a].view(N, K // D, D).reshape(N, K) * s_col
        res[f"tern_lat_{name}_nn"] = arm_metrics(dq, W, Xv, bpw)
        del a, dq
        if have_h:
            dq, _ = vp.vq_gptq(W, s_col, [C], U, D)
            res[f"tern_lat_{name}_hg"] = arm_metrics(dq, W, Xv, bpw)
            del dq
        else:
            res[f"tern_lat_{name}_hg"] = None
    if U is not None:
        del U
    del u, V8, s_col
    torch.cuda.empty_cache()

    row = {"unit": f"L{L:03d}_e{e:03d}_{proj}", "layer": L, "expert": e,
           "proj": proj, "m": round(m, 6),
           "refit_mass_share": round(share, 6),
           "n_fit_rows": int(Xf.shape[0]) if Xf is not None else 0,
           "n_val_rows": int(Xv.shape[0]) if Xv is not None else 0,
           "arms": res, "secs": round(time.time() - te, 1)}
    jrow(LEDGER, row)
    msg = " ".join(f"{k}={v['relrms']:.4f}" for k, v in res.items() if v)
    log(f"{row['unit']} {row['secs']}s {msg}")


# ------------------------------------------------------------ identity gate
def identity_gate():
    L, e = LAYERS[0], FIT_E[0]
    W, sb = get_mat(L, e, "down")
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
    _, codes_mine = vp.vq_gptq(W, s_col, [cb], U, 1)
    mism = (codes_mine.reshape(-1) != codes_ref.reshape(-1).long()
            ).float().mean().item()
    log(f"identity gate d=1 vs scalar gptq_loop: mismatch frac={mism:.2e}")
    if mism > 1e-4:
        raise SystemExit(f"IDENTITY GATE FAIL mism={mism}")
    del W, sb, X, H, U, codes_ref, codes_mine
    torch.cuda.empty_cache()


# ---------------------------------------------------------------- summary
def pilot_ternary_rows():
    """Join basic-ternary per-unit rows from the sealed pilot ledger."""
    led = PILOT / "out" / "VQW2_LEDGER.jsonl"
    out = {}
    if led.exists():
        for line in open(led):
            r = json.loads(line)
            out[r["unit"]] = {a: r["arms"].get(a)
                              for a in ("ternary", "ternary_pe",
                                        "ternary_gptq")}
    return out


def summarize(elapsed, crib_overlap):
    rows = [json.loads(x) for x in open(LEDGER)]
    pil = pilot_ternary_rows()
    per = {}
    all_arms = ARMS + ["ternary", "ternary_pe", "ternary_gptq"]
    for arm in all_arms:
        vals = {"fused13": [], "down": []}
        pv = []
        bp = []
        for r in rows:
            a = (r["arms"].get(arm) if arm in r["arms"]
                 else pil.get(r["unit"], {}).get(arm))
            if a and a.get("relrms") is not None:
                vals[r["proj"]].append(a["relrms"])
                bp.append(a["bpw"])
                if a.get("proxy_val") is not None:
                    pv.append(a["proxy_val"])
        allv = vals["fused13"] + vals["down"]
        if not allv:
            continue
        per[arm] = {
            "fused13_mean": round(float(np.mean(vals["fused13"])), 6)
            if vals["fused13"] else None,
            "down_mean": round(float(np.mean(vals["down"])), 6)
            if vals["down"] else None,
            "all_mean": round(float(np.mean(allv)), 6),
            "proxy_val_mean": round(float(np.mean(pv)), 6) if pv else None,
            "n": len(allv),
            "bpw_mean": round(float(np.mean(bp)), 4),
        }
    base_rtn = per.get("ternary", {}).get("all_mean")
    base_hg = per.get("ternary_gptq", {}).get("proxy_val_mean")
    for arm in per:
        r = per[arm]
        r["ratio_vs_basic_rtn"] = (round(r["all_mean"] / base_rtn, 6)
                                   if base_rtn else None)
        r["proxy_ratio_vs_basic_gptq"] = (
            round(r["proxy_val_mean"] / base_hg, 6)
            if (base_hg and r["proxy_val_mean"] is not None) else None)
    verdict = {}
    for name in ("crib", "refit"):
        nn = per.get(f"tern_lat_{name}_nn", {})
        hg = per.get(f"tern_lat_{name}_hg", {})
        verdict[name] = {
            "nn_rail_ratio": nn.get("ratio_vs_basic_rtn"),
            "hg_rail_ratio": hg.get("proxy_ratio_vs_basic_gptq"),
            "nn_beats_5pct": bool(nn.get("ratio_vs_basic_rtn") is not None
                                  and nn["ratio_vs_basic_rtn"] <= 0.95),
            "hg_beats_5pct": bool(
                hg.get("proxy_ratio_vs_basic_gptq") is not None
                and hg["proxy_ratio_vs_basic_gptq"] <= 0.95),
        }
    result = {
        "task": "t_426bbc97-stage2a-ternlat",
        "protocol": f"{len(LAYERS)} layers {LAYERS} x {len(EVAL_E)} eval "
                    f"experts {EVAL_E} x fused13/down; seed {SEED}; sym lut "
                    f"[-m,0,m] SSE scale refit; crib=iq1s_grid verbatim, "
                    f"refit=top-2048 s^2-mass ternary-projected d=8 patterns "
                    f"of FIT_E {FIT_E} (layer-shared)",
        "rails": "nn arms: relRMS vs pilot 'ternary'; hg arms: val-proxy vs "
                 "pilot 'ternary_gptq' (pilot rail_note convention). "
                 "Winner needs <=0.95 on its rail (card: beats basic by >5%).",
        "bytes_note": "lattice 1.375 code + 0.25 scale + ~6e-5 codebook bpw "
                      "= 1.6251 vs basic ternary 1.85 -- lattice is 12.2% "
                      "FEWER bytes, so any error win is strict domination.",
        "crib_refit_overlap_patterns": crib_overlap,
        "arms": per,
        "verdict": verdict,
        "elapsed_s": round(elapsed, 1),
    }
    tmp = SUMMARY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, SUMMARY)
    log(f"wrote {SUMMARY}")

    md = ["| arm | bpw | relRMS f13 | relRMS down | relRMS all | rtn-rail | "
          "val-proxy | gptq-rail |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("ternary", "ternary_gptq", "tern_lat_crib_nn",
                "tern_lat_crib_hg", "tern_lat_refit_nn",
                "tern_lat_refit_hg"):
        if arm not in per:
            continue
        p = per[arm]
        md.append(
            f"| {arm} | {p['bpw_mean']} | {p['fused13_mean']} | "
            f"{p['down_mean']} | {p['all_mean']} | "
            f"{p['ratio_vs_basic_rtn']} | {p['proxy_val_mean']} | "
            f"{p['proxy_ratio_vs_basic_gptq']} |")
    (OUT / "TERNLAT_TABLE.md").write_text("\n".join(md) + "\n")
    for ln in md:
        log(ln)
    log(f"VERDICT: {json.dumps(verdict)}")
    return result


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)
    tluts = json.loads((PILOT / "out" / "TERNARY_LUTS.json").read_text())
    M_PROJ = {p: (abs(tluts[p][0]) + tluts[p][2]) / 2.0
              for p in ("fused13", "down")}
    log(f"lattice magnitudes m = {M_PROJ}")
    G_crib = load_crib_grid()
    sel = json.load(open(PILOT / "static" / "CALIB_SELECTION.json"))

    done = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            try:
                done.add(json.loads(line)["unit"])
            except Exception:
                pass
    log(f"smoke={SMOKE} resume: {len(done)} units already in ledger")

    for L in LAYERS:
        BUNDLES[L] = gp.WtsBundle(L)
        log(f"bundle L{L:03d} loaded")

    identity_gate()

    for L in LAYERS:
        need = [e for e in EVAL_E
                if not {f"L{L:03d}_e{e:03d}_fused13",
                        f"L{L:03d}_e{e:03d}_down"} <= done]
        if not need:
            log(f"L{L:03d} all units done, skip caps")
            continue
        xf, hitf = vp.load_caps(L, sel["fit_ids"])
        xv, hitv = vp.load_caps(L, sel["val_ids"])
        log(f"L{L:03d} caps: fit {tuple(xf.shape)} val {tuple(xv.shape)}")
        for e in need:
            if gp.STOP.is_set():
                log("graceful stop")
                return 1
            Xf = xf[hitf[:, e]].float()
            Xv = xv[hitv[:, e]].float()
            W13, sb13 = BUNDLES[L].fused13(e)
            if f"L{L:03d}_e{e:03d}_fused13" not in done:
                run_unit(L, e, "fused13", W13, sb13, Xf, Xv,
                         M_PROJ["fused13"], G_crib, gen)
            if f"L{L:03d}_e{e:03d}_down" not in done:
                Wd, sb2 = BUNDLES[L].down(e)
                Af = gp.act(Xf, W13[:2048], W13[2048:])
                Av = gp.act(Xv, W13[:2048], W13[2048:])
                run_unit(L, e, "down", Wd, sb2, Af, Av,
                         M_PROJ["down"], G_crib, gen)
                del Wd, sb2, Af, Av
            del W13, sb13, Xf, Xv
            torch.cuda.empty_cache()
        del xf, hitf, xv, hitv
        torch.cuda.empty_cache()

    overlap = {}
    for (L, proj), (G, _s) in sorted(REFIT.items()):
        overlap[f"L{L:03d}_{proj}"] = grid_stats(G_crib, G)
    res = summarize(time.time() - t0, overlap)
    (OUT / "TERNLAT_DONE").write_text("done\n")
    log(f"ALL DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
