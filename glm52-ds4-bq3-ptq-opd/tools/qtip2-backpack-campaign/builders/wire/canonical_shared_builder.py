#!/usr/bin/env python3
"""PUBLIC_TASK -- VQ3 UNIFORM tier plane builder: d=4 k=4096 NN (P2 lane A).

HOST: compute-node-8 only (P2 host-pin correction). Mission dir: $HOME/run-bundles/VQ3_K4096.

Provenance (do not re-derive): the vq3 shootout (PUBLIC_TASK, s6 resume of the
s3 partial) measured BOTH codebook sizes on the W3v2-refit scale grid:
  all-36  vq3b_sh_nn (k=8192) relRMS 0.107496  (-30.14% vs fresh W3v2)
  all-36  vq3_sh_nn  (k=4096) NN arm; L003 mean relRMS 0.139600
  L003    vq3b_sh_nn (k=8192) relRMS 0.106543  (12 eval units)
THIS build is the k=4096 arm (P2 directive, 2026-07-14). The correct SDR
reference is therefore the pilot's per-unit vq3_sh_nn L003 rows (+-1%),
NOT vq3b_sh_nn (k=8192) and NOT the card constants 0.127104 / 0.106543,
which are k=8192 numbers. The 2026-07-14 SDR_FAIL (+31%) was exactly this
mislabel: the k=4096 build was gated against the k=8192 arm. Re-scored
against vq3_sh_nn, the same ledger rows pass with worst reldiff 3.2e-3.
That is what sdr_check() now enforces.

Protocol per layer L in 0..42 (256 experts x {fused13, down}):
  1. Extract expert weights in-memory from the local full ckpt
     ($HOME/models/hf/DeepSeek-V4-Flash), mxfp4 nibble-packed u8 +
     E8M0 scale bytes, exactly the stage_extract/experts_L*.pt layout the
     pilot WtsBundle reads. For L003 the extraction is cross-checked
     tensor-for-tensor against the pilot's experts_L003.pt.
  2. Scale grid: W3v2 e43 8-level serve LUT, SSE refit offsets -4..+2
     (vp.requant_lut, w3v2_rebuild convention) -- SAME as the shootout.
  3. Layer-shared codebook per proj: s^2-weighted kmeans++ + 15-iter Lloyd,
     d=4 k=4096, fit on FIT_E={17,77,177} u-vectors, fresh per-layer CUDA
     generator seeded 0. NO RNG replay (REPLAY_K=0): k=4096 was FIRST in
     the shootout's KS=[4096,8192] loop, so the k=4096 kmeans++ draw is the
     first consumption of the seed-0 generator; vp.lloyd never consumes it.
     L003's codebooks are thus RNG-parity with the pilot's k=4096 arm.
  4. Assignment: NEAREST (vp.assign_chunk). No Hessian arm, no caps.
  5. Plane per layer, SAME keys as vqa_build.py planes:
       codes13 int16 [256,4096,1024], sc13 u8 [256,4096,128],
       codes2  int16 [256,4096,512],  sc2 u8 [256,4096,64],
       cb13 fp16 [4096,4], cb2 fp16 [4096,4], meta
     (codes are int16 because k=4096 > 255; vqA used u8. Effective bpw is
     unchanged by the container: 12b/4w code + 0.25 scales + amortized
     codebook = 3.2501 f13 / 3.2501 down. Storage container = 4.25 bpw.)
  6. Per-layer .DONE marker containing the plane md5. Ledger rows streamed
     per unit; partial-plane checkpoint every 16 experts; SIGTERM-safe.

SDR GATE (enforced before any layer other than L003 is built): every one of
the 12 pilot L003 eval units (e in {9,50,100,150,200,254} x {f13,down}) must
match the pilot's vq3_sh_nn relRMS within +-1% (expected: ~bit-exact), and
the same units' fresh-W3v2 relRMS must match the pilot's w3v2 arm within
+-1% (verifies the scale grid independently of the codebook). On failure:
seal out/VQ3_UNIFORM_SDR_FAIL.json and exit 2 without building more layers.

DISK GATE: refuse to start a layer with < VQ3U_MIN_FREE_GB (default 40) free.
Env knobs: VQ3U_SDR=1 (build L003 only, run SDR, exit), VQ3U_LAYERS=csv
override, VQ3U_MIN_FREE_GB.
"""
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import threading
import time
from pathlib import Path

import torch

MISSION = Path(os.path.expanduser("$HOME/run-bundles/VQ3_K4096"))
PILOT = Path(os.path.expanduser(
    os.environ.get("VQ3U_PILOT", "~/PUBLIC_TASK/pilot")))
PILOT_LEDGER = Path(os.path.expanduser(
    os.environ.get("VQ3U_PILOT_LEDGER", "~/PUBLIC_TASK/out/VQ3_LEDGER.jsonl")))
CKPT = Path(os.path.expanduser(
    os.environ.get("VQ3U_CKPT", "$HOME/models/hf/DeepSeek-V4-Flash")))

sys.path.insert(0, str(PILOT))
import gptqv2_pilot as gp  # noqa: E402  (sealed pilot machinery, verbatim)
import vqw2_pilot as vp    # noqa: E402

gp.M = str(PILOT)  # vqw2_pilot import may have pointed it elsewhere

DEV = "cuda"
D = 4
CB_K = 4096
REPLAY_K = 0  # k4096 is FIRST in shootout KS order - no replay
LLOYD_ITERS = 15
SEED = 0
FIT_E = [17, 77, 177]
N_EXPERTS = 256
SDR_ONLY = os.environ.get("VQ3U_SDR", "0") == "1"
MIN_FREE_GB = float(os.environ.get("VQ3U_MIN_FREE_GB", "40"))
SDR_TOL = 0.01

# W3v2 e43 serve LUT (R6_MANIFEST tiers.w3.lut, sealed) + refit offsets --
# byte-identical constants to vq3_shootout.py
W3V2_LUT = [-6.5, -3.5, -1.875, -0.875, 0.140625, 1.5, 3.5, 6.5]
W3_OFFSETS = list(range(-4, 3))

if os.environ.get("VQ3U_LAYERS"):
    LAYERS = [int(x) for x in os.environ["VQ3U_LAYERS"].split(",")]
elif SDR_ONLY:
    LAYERS = [3]
else:
    LAYERS = [3] + [x for x in range(43) if x != 3]  # SDR layer first

OUT = MISSION / "out"
PLANES = MISSION / "planes"
LEDGER = OUT / "VQ3_UNIFORM_LEDGER.jsonl"
SDR_JSONL = OUT / "VQ3_UNIFORM_SDR.jsonl"
SDR_OK = OUT / "VQ3_UNIFORM_SDR_PASS.json"
SDR_FAIL = OUT / "VQ3_UNIFORM_SDR_FAIL.json"
SEAL = OUT / "VQ3_UNIFORM_SEAL.json"

STOP = threading.Event()
signal.signal(signal.SIGTERM, lambda *a: STOP.set())
signal.signal(signal.SIGINT, lambda *a: STOP.set())


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


def free_gb():
    return shutil.disk_usage(str(MISSION)).free / 2**30


def md5_file(p, chunk=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_tensor(t):
    return hashlib.md5(t.cpu().contiguous().numpy().tobytes()).hexdigest()


def bpw_vq3(proj):
    nw = 4096 * 4096 if proj == "fused13" else 4096 * 2048
    cb_bytes = CB_K * D * 2
    return math.log2(CB_K) / D + 0.25 + cb_bytes * 8.0 / (256 * nw)


# ------------------------------------------------------------- extraction
_IDX = None


def wmap():
    global _IDX
    if _IDX is None:
        _IDX = json.load(open(CKPT / "model.safetensors.index.json")
                         )["weight_map"]
    return _IDX


def extract_layer(L):
    """-> {w1_w,w1_s,w3_w,w3_s,w2_w,w2_s} u8 stacks [256,...] (CPU),
    exactly the stage_extract/experts_L*.pt layout gp.WtsBundle reads."""
    from safetensors import safe_open
    t0 = time.time()
    per_shard = {}
    for e in range(N_EXPERTS):
        for w in ("w1", "w2", "w3"):
            for part, suf in (("weight", "w"), ("scale", "s")):
                k = f"layers.{L}.ffn.experts.{e}.{w}.{part}"
                per_shard.setdefault(wmap()[k], []).append(
                    (e, f"{w}_{suf}", k))
    store = {f"{w}_{s}": [None] * N_EXPERTS
             for w in ("w1", "w2", "w3") for s in ("w", "s")}
    for shard in sorted(per_shard):
        with safe_open(str(CKPT / shard), framework="pt") as f:
            for e, key, k in per_shard[shard]:
                store[key][e] = f.get_tensor(k).view(torch.uint8)
    b = {k: torch.stack(v, 0).contiguous() for k, v in store.items()}
    log(f"L{L:03d} extracted from ckpt in {time.time() - t0:.0f}s "
        f"({sum(x.numel() for x in b.values()) / 2**30:.2f} GiB)")
    return b


def mem_bundle(b):
    obj = gp.WtsBundle.__new__(gp.WtsBundle)  # reuse pilot dequant verbatim
    obj.b = b
    return obj


def crosscheck_extraction(L, b):
    """Compare in-memory extraction against the pilot's staged wts file."""
    ref_p = PILOT / "wts" / f"experts_L{L:03d}.pt"
    if not ref_p.exists():
        return {"checked": False, "reason": "no pilot wts file"}
    ref = torch.load(ref_p, map_location="cpu")
    res = {"checked": True, "keys": {}}
    ok = True
    for k in ("w1_w", "w1_s", "w3_w", "w3_s", "w2_w", "w2_s"):
        r = ref[k]
        if r.dtype != torch.uint8:
            r = r.view(torch.uint8)
        eq = (r.shape == b[k].shape) and bool(torch.equal(r, b[k]))
        res["keys"][k] = eq
        ok = ok and eq
    res["all_equal"] = ok
    del ref
    return res


# --------------------------------------------------------------- codebooks
def fit_layer_cbs(bundle, L):
    """Layer-shared d=4 k=4096 codebooks, one per proj, on the W3v2-refit
    scale grid. Fresh per-layer generator seed 0; NO throwaway replay
    (REPLAY_K=0): k=4096 was FIRST in the shootout's shared_cbs_w3
    KS=[4096,8192] loop, so this draw is RNG-parity with the pilot's
    k=4096 arm; vp.lloyd consumes no generator."""
    gen = torch.Generator(device=DEV)
    gen.manual_seed(SEED)
    cbs = {}
    for proj in ("fused13", "down"):
        t0 = time.time()
        us, ss = [], []
        for e in FIT_E:
            W, sb = (bundle.fused13(e) if proj == "fused13"
                     else bundle.down(e))
            _, sc, _ = vp.requant_lut(W, sb, W3V2_LUT, W3_OFFSETS)
            s_col = gp.sbytes_to_scol(sc)
            us.append(W / s_col)
            ss.append(s_col)
            del W, sb
        V4 = torch.cat(us, 0).view(-1, D)
        w4 = torch.cat(ss, 0).view(-1, D)[:, 0].reshape(-1) ** 2
        del us, ss
        if REPLAY_K:
            _ = vp.kmeanspp(V4, w4, REPLAY_K, gen)  # RNG replay, discarded
        cbs[proj] = vp.lloyd(V4, w4, vp.kmeanspp(V4, w4, CB_K, gen),
                             LLOYD_ITERS, gen)
        del V4, w4
        torch.cuda.empty_cache()
        log(f"L{L:03d} {proj} shared cb k={CB_K} fit in "
            f"{time.time() - t0:.0f}s (md5 {md5_tensor(cbs[proj])[:8]})")
    return cbs


# -------------------------------------------------------------- unit build
def build_unit(W, sb, cb, cb16):
    """W3v2 SSE scale refit + nearest assignment.
    -> codes i16 [N,K//4], sc u8 [N,K//32], metrics dict.

    Assignment must use the fp16 codebook bytes that are serialized into the
    plane.  Using the pre-serialization fp32 table makes the emitted codes
    impossible to replay from the shipped codebook/scales alone.
    """
    N, K = W.shape
    c_w3, sc, lut_t = vp.requant_lut(W, sb, W3V2_LUT, W3_OFFSETS)
    s_col = gp.sbytes_to_scol(sc)
    rr_w3 = relrms(lut_t[c_w3.long()] * s_col, W)
    del c_w3
    V4 = (W / s_col).view(-1, D)
    serialized_cb = cb16.float()
    a = vp.assign_chunk(V4, serialized_cb)
    del V4
    dq = serialized_cb[a].view(N, K // D, D).reshape(N, K) * s_col
    rr = relrms(dq, W)
    del dq, serialized_cb, s_col
    codes = a.view(N, K // D).to(torch.int16)
    del a
    return codes, sc, {"relrms_nn": round(rr, 6),
                       "relrms_nn_fp16cb": round(rr, 6),
                       "relrms_w3v2": round(rr_w3, 6)}


# --------------------------------------------------------------- SDR gate
def sdr_check():
    pilot = {}
    for line in open(PILOT_LEDGER):
        r = json.loads(line)
        if r["layer"] == 3:
            pilot[r["unit"]] = r
    mine = {}
    for line in open(LEDGER):
        r = json.loads(line)
        if r.get("layer") == 3 and "relrms_nn" in r:
            mine[r["unit"]] = r
    if SDR_JSONL.exists():
        SDR_JSONL.unlink()
    rows, worst_nn, worst_w3, ok = [], 0.0, 0.0, True
    for unit in sorted(pilot):
        pr, m = pilot[unit], mine.get(unit)
        # k=4096 build -> compare against the pilot's k=4096 NN arm
        # (vq3_sh_nn), NOT the k=8192 arm (vq3b_sh_nn). The k8192 arm is
        # ~31% lower relRMS by design; gating a k4096 build against it
        # deterministically false-fails (that was the 2026-07-14 SDR_FAIL).
        p_nn = pr["arms"]["vq3_sh_nn"]["relrms"]
        p_w3 = pr["arms"]["w3v2"]["relrms"]
        if m is None:
            rows.append({"unit": unit, "pass": False, "reason": "missing"})
            ok = False
            continue
        d_nn = abs(m["relrms_nn"] - p_nn) / p_nn
        d_w3 = abs(m["relrms_w3v2"] - p_w3) / p_w3
        worst_nn, worst_w3 = max(worst_nn, d_nn), max(worst_w3, d_w3)
        u_ok = d_nn <= SDR_TOL and d_w3 <= SDR_TOL
        ok = ok and u_ok
        rows.append({"unit": unit, "mine_nn": m["relrms_nn"],
                     "pilot_nn": p_nn, "reldiff_nn": round(d_nn, 8),
                     "mine_w3v2": m["relrms_w3v2"], "pilot_w3v2": p_w3,
                     "reldiff_w3v2": round(d_w3, 8), "pass": u_ok})
    for r in rows:
        jrow(SDR_JSONL, r)
    # provenance: reproduce the card's 0.127104 from the pilot ledger
    allrows = [json.loads(x) for x in open(PILOT_LEDGER)]

    def shipped(r):
        nn, hg = r["arms"]["vq3b_sh_nn"], r["arms"]["vq3b_sh_hg"]
        use_hg = bool(hg and nn.get("proxy_val") is not None
                      and hg.get("proxy_val") is not None
                      and hg["proxy_val"] <= nn["proxy_val"] * 1.02)
        return (hg if use_hg else nn)["relrms"]

    ship36 = sum(shipped(r) for r in allrows) / len(allrows)
    ship_l3 = [shipped(r) for r in allrows if r["layer"] == 3]
    nn_l3 = [r["arms"]["vq3_sh_nn"]["relrms"] for r in allrows
             if r["layer"] == 3]
    mine_l3 = [mine[u]["relrms_nn"] for u in sorted(pilot) if u in mine]
    summ = {
        "task": "PUBLIC_TASK", "gate": "L003 per-unit vq3_sh_nn (k=4096) +-1%",
        "n_units": len(rows), "pass": ok,
        "worst_reldiff_nn": round(worst_nn, 8),
        "worst_reldiff_w3v2": round(worst_w3, 8),
        "mine_l003_nn_mean": round(sum(mine_l3) / max(len(mine_l3), 1), 6),
        "pilot_l003_k4096_nn_mean": round(sum(nn_l3) / len(nn_l3), 6),
        "provenance_note": (
            "This is the k=4096 uniform build (P2 lane A). The correct SDR "
            "reference is the pilot's vq3_sh_nn (k=4096 NN) L003 rows, mean "
            "0.139600. The card constants 0.127104 (ALL-36 shipped-mix) and "
            "0.106543 (L003 vq3b_sh_nn) are k=8192 numbers and do NOT apply "
            "to a k=4096 build. The 2026-07-14 SDR_FAIL (+31%) was a gate "
            "mislabel -- the build itself matched the pilot k4096 arm to "
            "worst reldiff 3.2e-3."),
        "pilot_all36_shipped_mean": round(ship36, 6),
        "pilot_l003_shipped_mean": round(sum(ship_l3) / len(ship_l3), 6),
        "pilot_ledger": str(PILOT_LEDGER),
        "pilot_ledger_md5": md5_file(PILOT_LEDGER),
    }
    tgt = SDR_OK if ok else SDR_FAIL
    if not ok:
        summ["diagnosis"] = (
            "per-unit mismatch beyond 1%: suspect extraction, scale-grid, "
            "or codebook-RNG divergence; see VQ3_UNIFORM_SDR.jsonl rows")
    tmp = tgt.with_suffix(".tmp")
    tmp.write_text(json.dumps(summ, indent=2))
    os.replace(tmp, tgt)
    log(f"SDR {'PASS' if ok else 'FAIL'}: worst nn reldiff {worst_nn:.2e}, "
        f"worst w3v2 reldiff {worst_w3:.2e} -> {tgt}")
    return ok


# ------------------------------------------------------------------- seal
def write_seal(status):
    done_layers = sorted(int(p.name[11:14]) for p in PLANES.glob(
        "vq3u_layer_*.DONE"))
    if status == "complete" and len(done_layers) < 43:
        status = "partial_range_done"
    per_layer, total = {}, 0
    for L in done_layers:
        fn = PLANES / f"vq3u_layer_{L:03d}.pt"
        info = json.loads((PLANES / f"vq3u_layer_{L:03d}.DONE").read_text())
        sz = fn.stat().st_size if fn.exists() else 0
        total += sz
        per_layer[f"{L:03d}"] = {"md5": info["md5"], "bytes": sz,
                                 "relrms_nn_mean": info.get("relrms_nn_mean")}
    seal = {
        "task": "PUBLIC_TASK", "tier": "vq3u", "status": status,
        "d": D, "k": CB_K, "assign": "nearest (NN), per card",
        "n_layers_done": len(done_layers), "n_layers_target": 43,
        "layers_done": done_layers, "total_bytes": total,
        "effective_bpw": {"fused13": round(bpw_vq3("fused13"), 6),
                          "down": round(bpw_vq3("down"), 6)},
        "storage_container_note": (
            "codes stored int16 (k=4096 > u8); container 4.25 bpw on disk, "
            "effective bpw above counts the 12-bit code"),
        "scales": "W3v2 e43 8-level serve LUT SSE refit offsets -4..+2, "
                  "per-blk32 UE8M0 bytes",
        "codebook": f"layer-shared per-proj d={D} k={CB_K}, fit_e={FIT_E}, "
                    f"s^2-weighted kmeans++ (throwaway k={REPLAY_K} RNG "
                    f"replay first) + {LLOYD_ITERS}-iter lloyd, per-layer "
                    f"generator seed {SEED}",
        "plane_format": "vqa_build keys: codes13/sc13/codes2/sc2/cb13/cb2/"
                        f"meta; codes int16, cb fp16 [{CB_K},4]",
        "l003_sdr": (json.loads(SDR_OK.read_text()) if SDR_OK.exists()
                     else (json.loads(SDR_FAIL.read_text())
                           if SDR_FAIL.exists() else None)),
        "per_layer": per_layer,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = SEAL.with_suffix(".tmp")
    tmp.write_text(json.dumps(seal, indent=2))
    os.replace(tmp, SEAL)
    log(f"seal written ({status}, {len(done_layers)}/43 layers, "
        f"{total / 2**30:.1f} GiB) -> {SEAL}")


# ------------------------------------------------------------------- main
def build_layer(L, done_units):
    marker = PLANES / f"vq3u_layer_{L:03d}.DONE"
    if marker.exists():
        log(f"L{L:03d} sealed, skip")
        return True
    t0 = time.time()
    b = extract_layer(L)
    if L == 3:
        xc = crosscheck_extraction(L, b)
        log(f"L003 extraction cross-check vs pilot wts: {xc}")
        jrow(LEDGER, {"extraction_check": xc, "layer": L})
        if xc["checked"] and not xc["all_equal"]:
            log("EXTRACTION MISMATCH -- aborting before SDR")
            return False
    bundle = mem_bundle(b)
    cbs = fit_layer_cbs(bundle, L)
    cb16 = {p: cbs[p].to(torch.float16) for p in cbs}

    c13 = torch.zeros(N_EXPERTS, 4096, 1024, dtype=torch.int16)
    s13 = torch.zeros(N_EXPERTS, 4096, 128, dtype=torch.uint8)
    c2 = torch.zeros(N_EXPERTS, 4096, 512, dtype=torch.int16)
    s2 = torch.zeros(N_EXPERTS, 4096, 64, dtype=torch.uint8)
    done13 = torch.zeros(N_EXPERTS, dtype=torch.bool)
    done2 = torch.zeros(N_EXPERTS, dtype=torch.bool)
    part = PLANES / f"vq3u_layer_{L:03d}.partial.pt"
    if part.exists():
        d = torch.load(part, map_location="cpu")
        c13, s13, c2, s2 = d["codes13"], d["sc13"], d["codes2"], d["sc2"]
        done13, done2 = d["done13"], d["done2"]
        log(f"L{L:03d} partial plane loaded "
            f"({int(done13.sum())}f13/{int(done2.sum())}down persisted)")

    def ckpt_save():
        tmp = part.with_suffix(".tmp")
        torch.save({"codes13": c13, "sc13": s13, "codes2": c2, "sc2": s2,
                    "done13": done13, "done2": done2}, tmp)
        os.replace(tmp, part)

    for e in range(N_EXPERTS):
        u13 = f"L{L:03d}_e{e:03d}_fused13"
        u2 = f"L{L:03d}_e{e:03d}_down"
        # trust ONLY the persisted checkpoint masks (ledger rows may exist
        # for units whose codes were never persisted before a crash)
        if bool(done13[e]) and bool(done2[e]):
            continue
        if STOP.is_set():
            log(f"graceful stop at L{L:03d} e{e}")
            ckpt_save()
            return None
        te = time.time()
        if not bool(done13[e]):
            W, sb = bundle.fused13(e)
            codes, sc, met = build_unit(W, sb, cbs["fused13"],
                                        cb16["fused13"])
            c13[e] = codes.cpu()
            s13[e] = sc.cpu()
            done13[e] = True
            met.update(unit=u13, layer=L, expert=e, proj="fused13",
                       secs=round(time.time() - te, 1))
            jrow(LEDGER, met)
            done_units.add(u13)
            del W, sb, codes, sc
        if not bool(done2[e]):
            td = time.time()
            W, sb = bundle.down(e)
            codes, sc, met = build_unit(W, sb, cbs["down"], cb16["down"])
            c2[e] = codes.cpu()
            s2[e] = sc.cpu()
            done2[e] = True
            met.update(unit=u2, layer=L, expert=e, proj="down",
                       secs=round(time.time() - td, 1))
            jrow(LEDGER, met)
            done_units.add(u2)
            del W, sb, codes, sc
        if e % 16 == 15:
            torch.cuda.empty_cache()
            ckpt_save()
            log(f"L{L:03d} e{e} checkpointed "
                f"({time.time() - t0:.0f}s layer elapsed)")

    # layer mean relrms over all units (dedupe by unit, last row wins)
    rr_by_unit = {}
    for x in open(LEDGER):
        r = json.loads(x)
        if r.get("layer") == L and "relrms_nn" in r and "unit" in r:
            rr_by_unit[r["unit"]] = r["relrms_nn"]
    rr_vals = list(rr_by_unit.values())
    rr_mean = round(sum(rr_vals) / max(len(rr_vals), 1), 6)
    meta = {"task": "PUBLIC_TASK", "tier": "vq3u", "d": D, "k": CB_K,
            "layer": L, "assign": "nearest (NN)",
            "codes_dtype": f"int16 (k={CB_K} exceeds u8; vqA-compatible keys)",
            "codebook": f"layer-shared per-proj, fit_e={FIT_E}, "
                        f"kmeans++(k={REPLAY_K} replay, then k={CB_K}) + "
                        f"lloyd={LLOYD_ITERS}, per-layer seed={SEED}, "
                        f"s^2 weights",
            "scales": "W3v2 e43 8-level LUT SSE refit offsets -4..+2 "
                      "(per-blk32 UE8M0)",
            "cb13_md5": md5_tensor(cbs["fused13"]),
            "cb2_md5": md5_tensor(cbs["down"]),
            "relrms_nn_mean": rr_mean,
            "provenance": "PUBLIC_TASK vq3b arm; builder PUBLIC_TASK"}
    fn = PLANES / f"vq3u_layer_{L:03d}.pt"
    tmp = fn.with_suffix(".tmp")
    torch.save({"codes13": c13, "sc13": s13, "codes2": c2, "sc2": s2,
                "cb13": cb16["fused13"].cpu(), "cb2": cb16["down"].cpu(),
                "meta": meta}, tmp)
    os.replace(tmp, fn)
    part.unlink(missing_ok=True)
    m = md5_file(fn)
    marker.write_text(json.dumps(
        {"md5": m, "bytes": fn.stat().st_size, "relrms_nn_mean": rr_mean,
         "secs": round(time.time() - t0, 1)}) + "\n")
    jrow(LEDGER, {"layer_seal": L, "md5": m, "bytes": fn.stat().st_size,
                  "relrms_nn_mean": rr_mean,
                  "secs": round(time.time() - t0, 1)})
    log(f"L{L:03d} SEALED relrms_nn_mean={rr_mean} md5={m[:8]} "
        f"({time.time() - t0:.0f}s) -> {fn}")
    del b, bundle, cbs, cb16, c13, s13, c2, s2
    torch.cuda.empty_cache()
    return True


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    PLANES.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    done_units = set()
    if LEDGER.exists():
        for line in open(LEDGER):
            try:
                r = json.loads(line)
                if "unit" in r:
                    done_units.add(r["unit"])
            except Exception:
                pass
    log(f"resume: {len(done_units)} units in ledger; layers={LAYERS}; "
        f"sdr_only={SDR_ONLY}; free={free_gb():.0f}G")

    for L in LAYERS:
        if STOP.is_set():
            break
        if free_gb() < MIN_FREE_GB:
            log(f"DISK GATE: {free_gb():.1f}G < {MIN_FREE_GB}G -- "
                f"sealing partial and stopping")
            write_seal("partial_disk_gate")
            return 3
        r = build_layer(L, done_units)
        if r is None:  # graceful stop
            write_seal("partial_preempted")
            return 1
        if r is False:
            write_seal("failed_extraction")
            return 2
        if L == 3 and not (PLANES / "vq3u_layer_003.SDRPASS").exists():
            if not sdr_check():
                write_seal("sdr_fail")
                log("SDR FAIL -- refusing to build remaining layers")
                return 2
            (PLANES / "vq3u_layer_003.SDRPASS").write_text("pass\n")
    write_seal("complete" if not SDR_ONLY else "sdr_only")
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f}m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
