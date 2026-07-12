#!/usr/bin/env python3
"""t_3d6e422d MMLU wrapper: sealed mmlu_ds4_offline.py (t_e3f38867,
unmodified) with PlaneSource swapped for AlphaPlaneSource /
BestqManifestSource.

usage: bq_mmlu_run.py <planes_dir_or_manifest.json> <tag>
"""
import os
import sys

BQ = os.path.expanduser("~/missions/DS4_BESTQ")
TEACH = os.path.expanduser("~/missions/DS4_TEACHER")
MM = os.path.expanduser("~/missions/DS4_MMLU")
sys.path.insert(0, TEACH)
sys.path.insert(0, MM)
sys.path.insert(0, BQ)

import mmlu_ds4_offline as mm  # noqa: E402
from bq_sources import AlphaPlaneSource, BestqManifestSource  # noqa: E402

src = sys.argv[1]
tag = sys.argv[2]

mm.PlaneSource = BestqManifestSource if src.endswith(".json") \
    else AlphaPlaneSource

CKPT = os.path.expanduser("~/models/hf/DeepSeek-V4-Flash")
sys.argv = [
    "mmlu_ds4_offline.py", "--mode", "planes",
    "--planes-dir", src,
    "--meta-dir", CKPT, "--local-dir", CKPT,
    "--questions", f"{MM}/static/mmlu_questions_ds4.json",
    "--out", f"{MM}/out", "--tag", tag,
]
os.chdir(MM)
mm.main()
