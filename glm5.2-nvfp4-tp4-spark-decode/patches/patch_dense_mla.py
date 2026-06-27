#!/usr/bin/env python3
# patch_dense_mla.py
# Make GLM-5.0 (nvidia/GLM-5-NVFP4) with index_topk=0 take the DENSE MLA attention path
# instead of the DEEPSEEK_V32_INDEXER sparse backend (which is a no-op on GLM-5.0 and only
# adds a second KV cache + KeyErrors on orphan indexer weights).
#
# Two idempotent, indentation-safe edits to vllm/model_executor/models/deepseek_v2.py:
#   P1: gate `is_v32` on index_topk not in (None, 0)  -> index_topk=0 is treated as NOT sparse
#   P2: before each `param = params_dict[name]`, skip checkpoint "indexer" weights that have
#       no dense-path destination param.
#
# Run INSIDE the engine container (the launcher mounts this read-only and execs it).
# GLM-5.0 ONLY: on GLM-5.1 the DSA indexer is load-bearing; disabling it yields token-salad.
import sys, re
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py"
lines = open(P).read().split("\n")
out = []
n1 = n2 = 0
GUARD_RE = re.compile(r'if "indexer" in name and name not in params_dict:')

i = 0
while i < len(lines):
    line = lines[i]

    # --- Patch 1: is_v32 gating (treat index_topk==0 as NOT-sparse) ---
    if 'self.is_v32 = hasattr(config, "index_topk")' in line:
        line = line.replace(
            'self.is_v32 = hasattr(config, "index_topk")',
            'self.is_v32 = getattr(config, "index_topk", None) not in (None, 0)')
        n1 += 1

    # --- Patch 2: before each `param = params_dict[name]`, skip checkpoint indexer
    #     weights that have no dense-path param. Match the line's OWN indent. ---
    stripped = line.lstrip()
    if stripped == "param = params_dict[name]":
        indent = line[:len(line) - len(stripped)]
        # idempotency: is the guard already the (non-blank) line above?
        prev_nonblank = None
        for j in range(len(out) - 1, -1, -1):
            if out[j].strip() != "":
                prev_nonblank = out[j]
                break
        if prev_nonblank is None or not GUARD_RE.search(prev_nonblank):
            out.append(f'{indent}if "indexer" in name and name not in params_dict:')
            out.append(f'{indent}    continue')
            n2 += 1

    out.append(line)
    i += 1

new_src = "\n".join(out)
open(P, "w").write(new_src)
print(f"[P1] is_v32 gating applied x{n1}")
print(f"[P2] indexer-skip guards inserted x{n2}")

# verify it compiles
import py_compile
try:
    py_compile.compile(P, doraise=True)
    print("PY_COMPILE OK")
except py_compile.PyCompileError as e:
    print("PY_COMPILE FAILED:", e)
    sys.exit(2)

v = open(P).read()
ok1 = 'not in (None, 0)' in v
ok2 = v.count('if "indexer" in name and name not in params_dict:') >= 1
print(f"VERIFY is_v32={ok1} indexer_skip={ok2}")
sys.exit(0 if (ok1 and ok2) else 1)
