# syntax=docker/dockerfile:1.7
# =============================================================================
# glm5-nvfp4-4xspark — reproducible GLM-5.0-NVFP4 TP=4 serving engine image
# =============================================================================
# Replaces the opaque hand-built `vllm-node:dsa` (sha256:f5ca01a47aa9..., built
# locally on the fleet 2026-06-19, RepoDigests EMPTY -> no registry digest) with
# a from-public-base, version-asserted, patch-baked image.
#
# PROVENANCE (every pin traceable to results/image-inventory.txt):
#   The live `vllm-node:dsa` IS the eugr "dgx-vllm" nightly build re-tagged
#   locally. Proof: its baked /workspace/build-metadata.yaml reports
#     build_script_commit: 986960209e27b3c181f72b4a30c718a3bc237f2e
#     vllm_version:        0.23.1rc1.dev156+g08985351f.d20260618
#     vllm_commit:         08985351f
#     flashinfer_commit:   9c5ed7c1
#     gpu_arch:            12.1a
#     build_date:          2026-06-18T18:33:08Z
#   and its engine stack (vLLM 0.23.1rc1.dev156+g08985351f, torch 2.11.0+cu130,
#   flashinfer-python 0.6.13 from /workspace/wheels, deep_gemm 2.5.0,
#   transformers 5.12.1, triton 3.6.0) matches the eugr nightly published at
#   ghcr.io/spark-arena/dgx-vllm-eugr-nightly (see repo-root README "Hardware
#   baseline"). That nightly itself sits on nvidia/cuda:13.0.2 ubuntu24.04 sbsa
#   (NVARCH=sbsa, CUDA_VERSION=13.0.2 — see docker history, results/image-inventory.txt §7).
#
# WHY WE DO NOT `pip install vllm==... flashinfer==...` HERE:
#   The load-bearing wheels (vLLM, flashinfer, deep_gemm) are custom aarch64 /
#   sm_121 / 12.1a builds NOT on PyPI. `pip install flashinfer-python==0.6.13`
#   would fetch a generic/x86 wheel and clobber the GB10 stack. So the correct
#   "pin" for a from-source nightly is: start FROM the nightly that already
#   carries these exact builds, then ASSERT the inventoried versions are present
#   at build time (the version-assert RUN below fails the build loudly if the
#   :latest nightly has rolled forward off the inventoried point).
# =============================================================================

# -----------------------------------------------------------------------------
# Base image. Default is the public eugr nightly documented in the repo-root
# README. It is a MOVING :latest tag, so the build-time assertions below pin the
# *content*. For a hard digest pin (recommended for T2's actual build), override:
#   docker build --build-arg BASE_IMAGE=ghcr.io/spark-arena/dgx-vllm-eugr-nightly@sha256:<DIGEST> ...
# where <DIGEST> is captured on a node that can pull from ghcr
# (`docker manifest inspect ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest`).
# We deliberately leave NO fabricated digest here (RepoDigests was empty on the
# fleet image; a real digest could not be determined from the offline inventory).
# -----------------------------------------------------------------------------
ARG BASE_IMAGE=ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
FROM ${BASE_IMAGE}

# Expected provenance of the base (asserted below). Sourced from the live image's
# /workspace/build-metadata.yaml (results/image-inventory.txt §3).
ARG EXPECT_VLLM_COMMIT=08985351f
ARG EXPECT_BUILD_SCRIPT_COMMIT=986960209e27b3c181f72b4a30c718a3bc237f2e

LABEL org.opencontainers.image.title="glm5-nvfp4-4xspark engine" \
      org.opencontainers.image.description="GLM-5.0 (nvidia/GLM-5-NVFP4) vLLM TP=4 serving engine for DGX Spark GB10/sm_121, dense-MLA path" \
      glm5.base.provenance="eugr dgx-vllm nightly; build_script_commit 986960209e27b3c181f72b4a30c718a3bc237f2e; vllm_commit 08985351f; build_date 2026-06-18" \
      glm5.target.arch="12.1a (GB10 / sm_121 / aarch64)"

# -----------------------------------------------------------------------------
# 1. Pin the engine stack by ASSERTION against the captured inventory.
#    Every expected value below is traceable to results/image-inventory.txt §4/§6.
#    If the base nightly drifted off these versions, the build FAILS here.
# -----------------------------------------------------------------------------
RUN set -eu; \
    echo "==> Asserting base build provenance (/workspace/build-metadata.yaml)"; \
    test -f /workspace/build-metadata.yaml || { echo "FATAL: base is not an eugr dgx-vllm build (no /workspace/build-metadata.yaml)"; exit 1; }; \
    grep -q "vllm_commit: ${EXPECT_VLLM_COMMIT}" /workspace/build-metadata.yaml \
        || { echo "FATAL: base vllm_commit != ${EXPECT_VLLM_COMMIT}"; cat /workspace/build-metadata.yaml; exit 1; }; \
    grep -q "build_script_commit: ${EXPECT_BUILD_SCRIPT_COMMIT}" /workspace/build-metadata.yaml \
        || echo "WARN: base build_script_commit != ${EXPECT_BUILD_SCRIPT_COMMIT} (nightly rolled; engine versions still asserted below)"; \
    echo "==> Asserting load-bearing engine versions (== results/image-inventory.txt)"; \
    python3 - <<'PYEOF'
import sys, importlib, importlib.metadata as m
# (package, accessor, expected) — every value sourced from results/image-inventory.txt
EXPECT = [
    ("torch",              lambda: __import__("torch").__version__,                 "2.11.0+cu130"),
    ("torch.version.cuda", lambda: __import__("torch").version.cuda,                "13.0"),
    ("vllm",               lambda: __import__("vllm").__version__,                  "0.23.1rc1.dev156+g08985351f.d20260618"),
    ("transformers",       lambda: __import__("transformers").__version__,          "5.12.1"),
    ("triton",             lambda: __import__("triton").__version__,                "3.6.0"),
    ("flashinfer-python",  lambda: m.version("flashinfer-python"),                  "0.6.13"),
    ("deep_gemm",          lambda: m.version("deep_gemm"),                          "2.5.0+88965b0"),
    ("ray",                lambda: m.version("ray"),                                "2.55.1"),
    ("fastsafetensors",    lambda: m.version("fastsafetensors"),                    "0.3.2"),
    ("nvidia-nccl-cu13",   lambda: m.version("nvidia-nccl-cu13"),                   "2.28.9"),
]
bad = []
for name, get, exp in EXPECT:
    try:
        got = get()
    except Exception as e:
        got = f"ABSENT({e.__class__.__name__})"
    flag = "OK " if got == exp else "MISMATCH"
    print(f"  [{flag}] {name:20s} got={got!r} expect={exp!r}")
    if got != exp:
        bad.append((name, got, exp))
if bad:
    print("FATAL: base image versions drifted from the inventoried pins:", bad)
    sys.exit(2)
print("==> All load-bearing versions match results/image-inventory.txt")
PYEOF

# -----------------------------------------------------------------------------
# 2. Re-declare the load-bearing runtime ENV so the image is self-documenting.
#    (These are already baked in the base; declaring them here makes the engine
#    contract explicit and independent of the base's ENV layer. SM121/GB10 only.)
# -----------------------------------------------------------------------------
ENV TORCH_CUDA_ARCH_LIST=12.1a \
    FLASHINFER_CUDA_ARCH_LIST=12.1a \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    VLLM_USE_FLASHINFER_MOE_FP4=0 \
    VLLM_ENABLE_V1_MULTIPROCESSING=0

# -----------------------------------------------------------------------------
# 3. Bake the two REQUIRED source patches into the image.
#    The live recipe applies these at RUNTIME (launch_node.sh mounts them ro and
#    execs them inside the container). Baking them here makes the image coherent
#    by default and removes the runtime mount as a single point of failure; the
#    patches are idempotent + self-verifying (exit!=0 on failure), so the
#    launcher's runtime re-application is a harmless no-op.
#
#    P_dense  : index_topk=0 -> dense MLA path, skip orphan indexer weights
#               (GLM-5.0 ONLY — on 5.1 the DSA indexer is load-bearing).
#    P_smem   : GB10 sm_121 101376-byte SMEM cap -> num_stages=1 for MLA decode.
#    Targets verified present in the base (results/image-inventory.txt + live probe):
#      vllm/model_executor/models/deepseek_v2.py
#      vllm/v1/attention/ops/triton_decode_attention.py
# -----------------------------------------------------------------------------
COPY patches/patch_dense_mla.py        /opt/glm5-patches/patch_dense_mla.py
COPY patches/patch_triton_decode_smem.py /opt/glm5-patches/patch_triton_decode_smem.py

RUN set -eu; \
    echo "==> Applying patch_dense_mla.py"; \
    python3 /opt/glm5-patches/patch_dense_mla.py; \
    echo "==> Applying patch_triton_decode_smem.py"; \
    python3 /opt/glm5-patches/patch_triton_decode_smem.py; \
    echo "==> Dropping stale bytecode"; \
    find /usr/local/lib/python3.12/dist-packages/vllm -name '*.pyc' -delete 2>/dev/null || true; \
    echo "==> Re-verifying patches are present (fail build if not)"; \
    grep -q 'not in (None, 0)' /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_v2.py; \
    grep -q 'GB10 sm_121 101376B SMEM cap' /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_decode_attention.py; \
    echo "==> Patches baked OK"

# -----------------------------------------------------------------------------
# 4. Entrypoint/Cmd: keep the base's nvidia entrypoint and the detached-sleep
#    pattern the launcher relies on (launch_node.sh runs the container with
#    `sleep infinity`, execs the patches [now no-ops], then execs `vllm serve`).
#    The full serve flags live in scripts/launch_node.sh — NOT baked here so the
#    config stays editable without an image rebuild (matches current behavior).
# -----------------------------------------------------------------------------
CMD ["sleep", "infinity"]
