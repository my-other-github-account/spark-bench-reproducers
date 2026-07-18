#!/usr/bin/env python3
"""Logic-level unit test for the T1 codes-cache in fullmenu_surface_t1.py.

Runs on a CPU-only CI host: monkeypatches torch.Tensor.to so device="cuda" is
a no-op, then exercises _codes_cache_pair directly for:
  1. disabled mode returns fresh gathers (sealed behavior),
  2. enabled mode: miss -> insert, hit -> exact same tensor objects,
  3. distinct keys get distinct entries,
  4. byte accounting matches storage sizes,
  5. LRU whole-layer eviction under a tiny cap keeps the newest layer,
  6. cached values are value-equal to fresh gathers.
Not a substitute for the GPU fixed-seed loss-bitwise gate — this validates the
cache logic only.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ["R5_CODES_CACHE"] = "1"
os.environ["R5_CODES_CACHE_GIB"] = str(64 * 1024 / (1 << 30))  # 64 KiB cap

import torch

_orig_to = torch.Tensor.to


def _fake_to(self, *args, **kwargs):
    if args and args[0] == "cuda":
        return _orig_to(self, "cpu")
    if kwargs.get("device") == "cuda":
        kwargs = dict(kwargs, device="cpu")
    return _orig_to(self, *args, **kwargs)


torch.Tensor.to = _fake_to

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "t1mod", HERE.parent / "src/fullmenu_surface_t1.py"
)
t1 = importlib.util.module_from_spec(spec)
# fullmenu_surface_t1 imports torch/np only at module level; safe on CPU.
spec.loader.exec_module(t1)

results = []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name + (f" | {detail}" if detail else ""))


payload = {
    "codes13": torch.randint(0, 255, (64, 128, 8), dtype=torch.uint8),
    "sc13": torch.randint(0, 255, (64, 128 // 4), dtype=torch.uint8),
}
rows_a = torch.tensor([0, 1, 2, 3], dtype=torch.long)
rows_b = torch.tensor([4, 5, 6, 7], dtype=torch.long)

# 1. disabled mode (sealed behavior)
t1._CODES_CACHE_ENABLED = False
pair1 = t1._codes_cache_pair(0, ("13", "base", "d8_k256", (0, 1, 2, 3)),
                             payload, "codes13", "sc13", rows_a)
pair2 = t1._codes_cache_pair(0, ("13", "base", "d8_k256", (0, 1, 2, 3)),
                             payload, "codes13", "sc13", rows_a)
check("disabled_no_cache", pair1[0] is not pair2[0] and not t1._codes_cache,
      "fresh tensors each call, cache untouched")
check("disabled_value_correct",
      torch.equal(pair1[0], payload["codes13"][rows_a])
      and torch.equal(pair1[1], payload["sc13"][rows_a]))

# 2. enabled: miss then hit returns identical objects
t1._CODES_CACHE_ENABLED = True
key_a = ("13", "base", "d8_k256", (0, 1, 2, 3))
m0 = t1._codes_cache_pair(0, key_a, payload, "codes13", "sc13", rows_a)
m1 = t1._codes_cache_pair(0, key_a, payload, "codes13", "sc13", rows_a)
check("hit_same_objects", m0[0] is m1[0] and m0[1] is m1[1])
check("hit_value_equals_fresh",
      torch.equal(m0[0], payload["codes13"][rows_a])
      and torch.equal(m0[1], payload["sc13"][rows_a]))

# 3. distinct keys distinct entries
key_b = ("13", "base", "d8_k256", (4, 5, 6, 7))
n0 = t1._codes_cache_pair(0, key_b, payload, "codes13", "sc13", rows_b)
check("distinct_keys", n0[0] is not m0[0]
      and torch.equal(n0[0], payload["codes13"][rows_b]))

# 4. byte accounting
expected = sum(
    t.untyped_storage().nbytes()
    for pair in t1._codes_cache[0].values() for t in pair
)
check("byte_accounting", t1._codes_cache_bytes == expected,
      f"tracked={t1._codes_cache_bytes} expected={expected}")

# 5. LRU whole-layer eviction under the 64 KiB cap
per_pair = sum(t.untyped_storage().nbytes() for t in m0)
layers_needed = (64 * 1024) // per_pair + 3
for layer in range(1, int(layers_needed) + 1):
    t1._codes_cache_pair(layer, key_a, payload, "codes13", "sc13", rows_a)
newest = int(layers_needed)
check("lru_eviction_ran", len(t1._codes_cache) < layers_needed + 1,
      f"layers cached={len(t1._codes_cache)}")
check("lru_keeps_newest", newest in t1._codes_cache)
check("lru_cap_respected",
      t1._codes_cache_bytes <= 64 * 1024 or len(t1._codes_cache) == 1,
      f"bytes={t1._codes_cache_bytes}")
recount = sum(
    t.untyped_storage().nbytes()
    for lc in t1._codes_cache.values() for pair in lc.values() for t in pair
)
check("byte_accounting_after_eviction", t1._codes_cache_bytes == recount,
      f"tracked={t1._codes_cache_bytes} recount={recount}")

out = HERE.parent / "receipts/T1_UNIT_RECEIPT.json"
out.write_text(json.dumps(
    {"all_pass": all(r["pass"] for r in results), "results": results,
     "torch": torch.__version__}, indent=2) + "\n")
print(f"receipt: {out}")
sys.exit(0 if all(r["pass"] for r in results) else 1)
