#!/usr/bin/env python3
"""Resume-safe full W2v2 plane rebuild for t_bd7728ee."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path.home() / "missions/W3_LUT_AUDIT"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "missions/DS4_TEACHER"))
import planes_unpack as pu
import w3v2_rebuild as base

SHOOT = json.loads((ROOT / "W2V2_SHOOTOUT.json").read_text())
W2V2_LUT = SHOOT["luts"]["dp_asym4_round2"]
OFFSETS = list(range(-4, 3))


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "moe_w2_planes_v2"))
    ap.add_argument("--layers", default="0-42")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=16)
    a = ap.parse_args()
    outdir = Path(a.out).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)
    lo, hi = a.layers.split("-") if "-" in a.layers else (None, None)
    layers = list(range(int(lo), int(hi) + 1)) if lo is not None else [int(x) for x in a.layers.split(",")]
    lut = torch.tensor(W2V2_LUT, dtype=torch.float32, device=a.device)
    mids = (lut[1:] + lut[:-1]) / 2
    E = 256
    tiers = {"13": (("w1", "w3"), 4096, 4096), "2": (("w2",), 4096, 2048)}
    for L in layers:
        files = [outdir / f"layer_{L:03d}.{x}.npy" for x in ("planes13", "planes2", "sc13", "sc2")] + [outdir / f"layer_{L:03d}.meta.json"]
        if all(p.exists() for p in files):
            log(f"L{L:03d} exists, skip")
            continue
        t0 = time.time()
        for tier, (names, N, K) in tiers.items():
            KB = K // 32
            pbytes = np.empty((E, N * K // 4), dtype=np.uint8)
            sbytes = np.empty((E, N * KB), dtype=np.uint8)
            for c0 in range(0, E, a.chunk):
                es = list(range(c0, min(c0 + a.chunk, E)))
                w, sb = base.load_chunk(L, es, names, a.device)
                codes, sc = base.requant_chunk(w, sb, lut, mids)
                for i, e in enumerate(es):
                    pbytes[e] = pu.pack_fragment_major(codes[i]).cpu().numpy()
                    sbytes[e] = pu.pack_scales(sc[i]).cpu().numpy()
                del w, sb, codes, sc
            tmp = outdir / f"layer_{L:03d}.planes{tier}.npy.tmp.npy"
            np.save(tmp, pbytes)
            os.replace(tmp, outdir / f"layer_{L:03d}.planes{tier}.npy")
            tmp = outdir / f"layer_{L:03d}.sc{tier}.npy.tmp.npy"
            np.save(tmp, sbytes)
            os.replace(tmp, outdir / f"layer_{L:03d}.sc{tier}.npy")
        meta = {"E": E, "N13": 4096, "K13": 4096, "N2": 4096, "K2": 2048,
                "codebook": "w2v2", "bpw": 2.25, "lut": W2V2_LUT,
                "lut_provenance": "dp_asym4 round2, held-out fit in W2V2_SHOOTOUT.json t_bd7728ee",
                "scale_fit": f"per-block-32 UE8M0 exponent SSE search offsets {OFFSETS} vs ckpt mxfp4 exponent"}
        (outdir / f"layer_{L:03d}.meta.json").write_text(json.dumps(meta, indent=1))
        log(f"L{L:03d} rebuilt in {time.time()-t0:.0f}s")
    log("rebuild complete")


if __name__ == "__main__":
    main()
