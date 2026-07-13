#!/usr/bin/env python3
"""Extract per-layer expert weight stacks (experts_LNNN.pt) for ALL 43 layers
from the resident DS4-Flash ckpt on s8. Format matches DS4_GPTQV2_STAGE/wts
exactly: {w1,w3,w2}_{w,s} u8 stacks over 256 experts. Resume-safe per layer."""
import json
import os
import sys

import torch
from safetensors import safe_open

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
OUT = os.path.expanduser("~/missions/VQA_WTS/wts")
os.makedirs(OUT, exist_ok=True)

idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))
wm = idx["weight_map"]

E = 256
LAYERS = list(range(43))

for L in LAYERS:
    dst = f"{OUT}/experts_L{L:03d}.pt"
    if os.path.exists(dst + ".ok"):
        print(f"L{L:03d} skip (done)", flush=True)
        continue
    # group tensors by shard file to open each shard once
    need = {}
    for e in range(E):
        for p in ("w1", "w3", "w2"):
            for kind in ("weight", "scale"):
                key = f"layers.{L}.ffn.experts.{e}.{p}.{kind}"
                shard = wm.get(key)
                if shard is None:
                    print(f"MISSING KEY {key}", flush=True)
                    sys.exit(1)
                need.setdefault(shard, []).append((key, e, p, kind))
    bufs = {}
    for shard, items in need.items():
        with safe_open(f"{CKPT}/{shard}", framework="pt", device="cpu") as f:
            for key, e, p, kind in items:
                bufs[(e, p, kind)] = f.get_tensor(key)
    out = {}
    for p in ("w1", "w3", "w2"):
        out[f"{p}_w"] = torch.stack([bufs[(e, p, "weight")] for e in range(E)])
        out[f"{p}_s"] = torch.stack([bufs[(e, p, "scale")] for e in range(E)])
    tmp = dst + ".tmp"
    torch.save(out, tmp)
    os.rename(tmp, dst)
    open(dst + ".ok", "w").write("ok")
    shp = tuple(out["w1_w"].shape)
    print(f"L{L:03d} extracted {shp}", flush=True)

open(os.path.expanduser("~/missions/VQA_WTS/EXTRACT_DONE"), "w").write("ok")
print("ALL DONE", flush=True)
