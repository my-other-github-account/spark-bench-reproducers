#!/usr/bin/env python3
# patch_sparse_gate.py — open the FlashInfer sparse-MLA compute-capability gate for GB10 (sm_121).
#
# On LATEST vLLM (dev207+gdced29076 main HEAD) the DSA sparse MLA backend
# (vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py) gates itself to
# Blackwell SM 10.x only:
#     def supports_compute_capability(cls, capability):
#         return capability.major == 10
# GB10 (DGX Spark) reports compute capability 12.1 (major=12), so the sparse
# backend is rejected -> vLLM falls back / errors for GLM-5.2's load-bearing DSA
# indexer. GLM-5.2 ships index_topk=2048 + per-layer indexer weights, so the
# sparse path is mandatory (the GLM-5.0 dense hack would token-salad).
#
# This patch opens the gate to {10, 12} so the FLASHINFER_MLA_SPARSE backend
# selects on sm_121, the same one-line gate-open used on the proven dev156 image.
# Idempotent + py_compile-verified. Anchor verified present on dev207.
import importlib.util, py_compile, sys

PATH = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py"

with open(PATH) as f:
    src = f.read()

OLD = "        return capability.major == 10"
NEW = "        return capability.major in (10, 12)  # GB10/sm_121 gate-open (T3.7)"

if NEW in src:
    print("[SPARSE_GATE] already opened (idempotent no-op)")
elif OLD in src:
    src = src.replace(OLD, NEW, 1)
    with open(PATH, "w") as f:
        f.write(src)
    py_compile.compile(PATH, doraise=True)
    print("[SPARSE_GATE] opened capability.major == 10 -> in (10, 12); PY_COMPILE OK")
else:
    print("[SPARSE_GATE] ANCHOR NOT FOUND — gate logic changed; inspect", PATH, file=sys.stderr)
    # show the actual current gate so the operator can see what changed
    import re
    for m in re.finditer(r"supports_compute_capability.*?\n(?:.*\n){0,4}", src):
        sys.stderr.write(m.group(0))
    sys.exit(2)

# verify the gate now accepts major==12
import re
m = re.search(r"def supports_compute_capability.*?return ([^\n]+)", src, re.S)
print("[SPARSE_GATE] VERIFY gate expr:", m.group(1).strip() if m else "??")
