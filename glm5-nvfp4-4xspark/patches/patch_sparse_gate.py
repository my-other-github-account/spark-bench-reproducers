#!/usr/bin/env python3
# patch_sparse_gate.py — enable the FlashInfer sparse-MLA (DSA) backend on GB10 (sm_121)
# for LATEST vLLM (dev207+gdced29076 main HEAD). GLM-5.2 ships a LOAD-BEARING DSA sparse
# indexer (config index_topk=2048), so vLLM must select a sparse MLA attention backend.
#
# Two coupled blockers on LATEST vLLM for sm_121 (both fixed here):
#
#  (A) CANDIDATE-LIST BRANCH — vllm/platforms/cuda.py::_get_backend_priorities()
#      hard-branches on `device_capability.major == 10` (Blackwell). The major==10 branch
#      offers BOTH FLASHINFER_MLA_SPARSE + FLASHMLA_SPARSE as sparse candidates; the `else`
#      branch (GB10 = sm_121, major==12) offers ONLY FLASHMLA_SPARSE — which needs the
#      uncompiled `vllm._flashmla_C` -> ImportError -> "No valid attention backend found
#      ... use_sparse=True". So on sm_121 the working FlashInfer sparse backend is never
#      even CONSIDERED. Fix: add FLASHINFER_MLA_SPARSE to the else-branch sparse candidates
#      (ahead of FLASHMLA_SPARSE).
#
#  (B) CAPABILITY GATE — vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py
#      ::supports_compute_capability() returns `capability.major == 10` (Blackwell only).
#      Once (A) makes it a candidate, this gate would still reject sm_121. Fix: open to
#      (10, 12). [Same one-line gate-open used on the proven dev156 image.]
#
# Idempotent + py_compile-verified. Both anchors verified present on dev207.
import py_compile, sys, re

CUDA = "/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py"
SPARSE = "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py"

rc = 0

# ---- (A) candidate-list branch in cuda.py ----
with open(CUDA) as f:
    src = f.read()

A_OLD = """        else:
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]"""
A_NEW = """        else:
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE,  # GB10/sm_121 DSA sparse (T3.7)
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]"""
MARKER_A = "AttentionBackendEnum.FLASHINFER_MLA_SPARSE,  # GB10/sm_121 DSA sparse (T3.7)"
if MARKER_A in src:
    print("[SPARSE_GATE A] cuda.py candidate-list already patched (idempotent)")
elif A_OLD in src:
    src = src.replace(A_OLD, A_NEW, 1)
    with open(CUDA, "w") as f:
        f.write(src)
    py_compile.compile(CUDA, doraise=True)
    print("[SPARSE_GATE A] cuda.py: added FLASHINFER_MLA_SPARSE to sm_121 sparse candidates; PY_COMPILE OK")
else:
    print("[SPARSE_GATE A] ANCHOR NOT FOUND in cuda.py — _get_backend_priorities else-branch changed", file=sys.stderr)
    rc = 2

# ---- (B) capability gate in flashinfer_mla_sparse.py ----
with open(SPARSE) as f:
    s2 = f.read()

B_OLD = "        return capability.major == 10"
B_NEW = "        return capability.major in (10, 12)  # GB10/sm_121 gate-open (T3.7)"
if B_NEW in s2:
    print("[SPARSE_GATE B] flashinfer_mla_sparse gate already opened (idempotent)")
elif B_OLD in s2:
    s2 = s2.replace(B_OLD, B_NEW, 1)
    with open(SPARSE, "w") as f:
        f.write(s2)
    py_compile.compile(SPARSE, doraise=True)
    print("[SPARSE_GATE B] flashinfer_mla_sparse: capability.major == 10 -> in (10, 12); PY_COMPILE OK")
else:
    print("[SPARSE_GATE B] ANCHOR NOT FOUND in flashinfer_mla_sparse.py", file=sys.stderr)
    rc = 2

# verify
with open(SPARSE) as f:
    m = re.search(r"def supports_compute_capability.*?return ([^\n]+)", f.read(), re.S)
print("[SPARSE_GATE] VERIFY flashinfer gate expr:", m.group(1).strip() if m else "??")
with open(CUDA) as f:
    print("[SPARSE_GATE] VERIFY cuda.py else-branch has FLASHINFER_MLA_SPARSE:", MARKER_A in f.read())
sys.exit(rc)
