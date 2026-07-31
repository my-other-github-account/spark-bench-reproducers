#!/usr/bin/env python3
"""Seal all repository recipe sources; excludes generated receipts/manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXCLUDE = {
    "RECIPE_MANIFEST.json",
    "SOURCE_MANIFEST.json",
    "WHEEL_MANIFEST.json",
    "RUNTIME_CACHE_MANIFEST.json",
    "P1268_C1_C2_RESULT.json",
}
rows = []
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or path.name in EXCLUDE or "__pycache__" in path.parts or "receipts" in path.parts:
        continue
    data = path.read_bytes()
    rows.append({"path": str(path.relative_to(ROOT)), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
payload = {
    "schema": "banana_smasher-golden-recipe-manifest-v2",
    "task_id": "P1298-REPRO",
    "truth_label": "PUBLIC_CANON_IQ3_WIRE; NOT P943 native TRUE-C",
    "provenance": "P943 overlay 9a4b7098 / pack 3650fe7e / planes b524c5a",
    "p1268_result_sha256": "9b1d42fe3f4dcb28e7f8660b37f800fdbfdcd7f721fb4bc57ca31a0dda313860",
    "p1321_ladder_seal_sha256": "be0453e1d6081a87a0288c8611b9ee5ec33a4b2ba927cb68c358e71a10b242f7",
    "p1321_winning_boot_sha256": "091e8eb3e4caa9793454f4a529d8c1f5fc0af0fcb4fa28cc89e34c8a4c314da2",
    "p1321_freeze_sha256": "cff72b34c5cd9d29a17d9a1842005febf5402141f6709c10f85a25cd8a61d707",
    "parent_hand_ladder_seal_sha256": "ea7df6435fa0fe6e574a20d2506abb09832591bf23f45bc3ff82a5dfb1a0e3e5",
    "files": rows,
}
output = ROOT / "RECIPE_MANIFEST.json"
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"status": "PASS", "output": str(output), "files": len(rows), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))
