#!/usr/bin/env python3
"""t_426bbc97 STAGE-2a addendum -- ternary lattice k-sweep (spark-3).

Traces the d=8 ternary lattice frontier between the IQ1_S point (k=2048,
1.375 bits/w code) and TRUE iso-bytes with basic ternary: k=4096 (1.5) and
k=6561 = 3^8 FULL ternary book (1.585 bits/w ~= basic's 1.6 trit-packing).

At k=6561/nn the lattice restriction vanishes: nearest-pattern assignment
== independent per-weight nearest rounding == basic ternary (sanity anchor,
expected ratio 1.0). The only remaining lever at iso-bytes is hg (d=8
group VQ-GPTQ) vs scalar ternary GPTQ -- a pure assignment-granularity
comparison.

Refit selection only (crib==refit at k=2048 showed the grid choice is
immaterial). Ledger: out/TERNLAT_KSWEEP_LEDGER.jsonl.
Summary merged into out/TERNLAT_KSWEEP.json.
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
import vqw2_pilot as vp  # noqa: E402

gp = vp.gp
DEV = "cuda"
SMOKE = os.environ.get("TL_SMOKE", "0") == "1"
LAYERS = [3] if SMOKE else [3, 23, 41]
EVAL_E = [9] if SMOKE else [9, 50, 100, 150, 200, 254]
FIT_E = [17, 77, 177]
SEED = 0
MIN_FIT_ROWS = 64
T_OFFSETS = list(range(-4, 5))
D = 8
KS = [4096, 6561]
OUT = MISSION / "out"
LEDGER = OUT / ("TERNLAT_KSWEEP_LEDGER_SMOKE.jsonl" if SMOKE
                else "TERNLAT_KSWEEP_LEDGER.jsonl")
SUMMARY = OUT / ("TERNLAT_KSWEEP_SMOKE.json" if SMOKE
                 else "TERNLAT_KSWEEP.json")

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


def bpw_lat(proj, k):
    cb_bytes = k * D * 2
    return (math.log2(k) / D + 0.25
            + cb_bytes * 8.0 / (256 * nweights(proj)))


def arm_metrics(dq, W, Xv, bpw):
    pv = gp.proxy_sym(Xv, dq, W) if (Xv is not None and Xv.shape[0] > 0) else None
    return {"relrms": round(relrms(dq, W), 6),
            "proxy_val": (round(pv, 6) if pv is not None else None),
            "bpw": round(bpw, 4)}


POW3 = None


def pat_ids(T):
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


def refit_grid(L, proj, m, k):
    key = (L, proj, k)
    if key in REFIT:
        return REFIT[key]
    if k == 3 ** D:
        G = ids_to_pats(torch.arange(3 ** D, device=DEV))
        REFIT[key] = G
        return G
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
    G = ids_to_pats(mass.topk(k).indices)
    REFIT[key] = G
    return G


BUNDLES = {}


def get_mat(L, e, proj):
    b = BUNDLES[L]
    return b.fused13(e) if proj == "fused13" else b.down(e)


def run_unit(L, e, proj, W, sb, Xf, Xv, m, gen):
    te = time.time()
    N, K = W.shape
    res = {}
    lut_m = torch.tensor([-m, 0.0, m], dtype=torch.float32, device=DEV)
    _, sc, _ = vp.requant_lut(W, sb, lut_m, T_OFFSETS)
    s_col = gp.sbytes_to_scol(sc)
    u = W / s_col
    V8 = u.view(-1, D)

    have_h = Xf is not None and Xf.shape[0] >= MIN_FIT_ROWS
    U = None
    if have_h:
        H = Xf.t() @ Xf
        U = gp.gptq_prepare(H, torch.arange(K, device=DEV))
        del H

    for k in KS:
        G = refit_grid(L, proj, m, k)
        C = m * G
        bpw = bpw_lat(proj, k)
        a = vp.assign_chunk(V8, C)
        dq = C[a].view(N, K // D, D).reshape(N, K) * s_col
        res[f"lat_k{k}_nn"] = arm_metrics(dq, W, Xv, bpw)
        del a, dq
        if have_h:
            dq, _ = vp.vq_gptq(W, s_col, [C], U, D)
            res[f"lat_k{k}_hg"] = arm_metrics(dq, W, Xv, bpw)
            del dq
        else:
            res[f"lat_k{k}_hg"] = None
        del C
    if U is not None:
        del U
    del u, V8, s_col
    torch.cuda.empty_cache()

    row = {"unit": f"L{L:03d}_e{e:03d}_{proj}", "layer": L, "expert": e,
           "proj": proj, "m": round(m, 6),
           "n_fit_rows": int(Xf.shape[0]) if Xf is not None else 0,
           "n_val_rows": int(Xv.shape[0]) if Xv is not None else 0,
           "arms": res, "secs": round(time.time() - te, 1)}
    jrow(LEDGER, row)
    msg = " ".join(f"{n}={v['relrms']:.4f}" for n, v in res.items() if v)
    log(f"{row['unit']} {row['secs']}s {msg}")


def pilot_ternary_rows():
    led = PILOT / "out" / "VQW2_LEDGER.jsonl"
    out = {}
    if led.exists():
        for line in open(led):
            r = json.loads(line)
            out[r["unit"]] = {a: r["arms"].get(a)
                              for a in ("ternary", "ternary_gptq")}
    return out


def summarize(elapsed):
    rows = [json.loads(x) for x in open(LEDGER)]
    pil = pilot_ternary_rows()
    arms = [f"lat_k{k}_{s}" for k in KS for s in ("nn", "hg")]
    arms += ["ternary", "ternary_gptq"]
    per = {}
    for arm in arms:
        vals = {"fused13": [], "down": []}
        pv, bp = [], []
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
    result = {
        "task": "t_426bbc97-stage2a-ternlat-ksweep",
        "note": "k=6561 is the FULL 3^8 ternary book: nn == independent "
                "per-weight rounding == basic ternary (sanity anchor); its "
                "1.5849 bits/w code ~= basic's 1.6 trit-packing -> true "
                "iso-bytes. hg at k=6561 isolates d=8 group-GPTQ assignment "
                "granularity vs scalar GPTQ at iso-bytes.",
        "arms": per,
        "elapsed_s": round(elapsed, 1),
    }
    tmp = SUMMARY.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, SUMMARY)
    log(f"wrote {SUMMARY}")
    for arm in arms:
        if arm in per:
            p = per[arm]
            log(f"{arm:14s} bpw={p['bpw_mean']:.4f} all={p['all_mean']:.6f} "
                f"rtn-rail={p['ratio_vs_basic_rtn']} "
                f"pv={p['proxy_val_mean']} "
                f"gptq-rail={p['proxy_ratio_vs_basic_gptq']}")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)
    tluts = json.loads((PILOT / "out" / "TERNARY_LUTS.json").read_text())
    M_PROJ = {p: (abs(tluts[p][0]) + tluts[p][2]) / 2.0
              for p in ("fused13", "down")}
    sel = json.load(open(PILOT / "static" / "CALIB_SELECTION.json"))

    done = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            try:
                done.add(json.loads(line)["unit"])
            except Exception:
                pass
    log(f"smoke={SMOKE} ks={KS} resume: {len(done)} units in ledger")

    for L in LAYERS:
        BUNDLES[L] = gp.WtsBundle(L)
        log(f"bundle L{L:03d} loaded")

    for L in LAYERS:
        need = [e for e in EVAL_E
                if not {f"L{L:03d}_e{e:03d}_fused13",
                        f"L{L:03d}_e{e:03d}_down"} <= done]
        if not need:
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
                         M_PROJ["fused13"], gen)
            if f"L{L:03d}_e{e:03d}_down" not in done:
                Wd, sb2 = BUNDLES[L].down(e)
                Af = gp.act(Xf, W13[:2048], W13[2048:])
                Av = gp.act(Xv, W13[:2048], W13[2048:])
                run_unit(L, e, "down", Wd, sb2, Af, Av, M_PROJ["down"], gen)
                del Wd, sb2, Af, Av
            del W13, sb13, Xf, Xv
            torch.cuda.empty_cache()
        del xf, hitf, xv, hitv
        torch.cuda.empty_cache()

    summarize(time.time() - t0)
    (OUT / "TERNLAT_KSWEEP_DONE").write_text("done\n")
    log(f"ALL DONE in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
