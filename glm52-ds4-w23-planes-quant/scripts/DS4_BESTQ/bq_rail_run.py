#!/usr/bin/env python3
"""t_3d6e422d rail wrapper: sealed v3 builder --mode planes with
v3.PlaneSource swapped for AlphaPlaneSource (plane dirs) or
BestqManifestSource (R7 manifests). Sealed builder file unmodified.

usage: bq_rail_run.py <planes_dir_or_manifest.json> <out_dir> <tag>
"""
import os
import sys

BQ = os.path.expanduser("~/missions/DS4_BESTQ")
TEACH = os.path.expanduser("~/missions/DS4_TEACHER")
sys.path.insert(0, TEACH)
sys.path.insert(0, BQ)

import t8192_ds4_build_v3 as v3  # noqa: E402
from bq_sources import AlphaPlaneSource, BestqManifestSource  # noqa: E402

src = sys.argv[1]
out = sys.argv[2]
tag = sys.argv[3]

v3.PlaneSource = BestqManifestSource if src.endswith(".json") \
    else AlphaPlaneSource

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
sys.argv = [
    "t8192_ds4_build_v3.py", "--mode", "planes",
    "--planes-dir", src,
    "--ref-dir", f"{TEACH}/t8192_eval",
    "--meta-dir", CKPT, "--local-dir", CKPT,
    "--out", out,
    "--cand-pos-limit", "1024", "--chunk", "64", "--mb", "4",
    "--tag", tag,
]
os.chdir(TEACH)
v3.main()
