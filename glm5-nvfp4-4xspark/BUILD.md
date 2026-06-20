# BUILD.md — building the glm5-nvfp4-4xspark engine image

This document is the build contract for the GLM-5.0-NVFP4 TP=4 serving engine
image. It replaces the opaque, locally hand-built `vllm-node:dsa` with a
reproducible `docker build` from a **public** base, with every load-bearing
version pinned and **traceable to a real command output** in
[`results/image-inventory.txt`](results/image-inventory.txt).

> **Anti-fabrication note.** Nothing in this file is invented. Every version
> below was read off the live `vllm-node:dsa` image on the fleet (rank0 / node0 /
> <node11-lan>) via read-only `docker inspect` / `docker history` /
> `docker exec … pip freeze` on **2026-06-19**, captured verbatim to
> `results/image-inventory.txt`. The base-image registry digest was resolved
> with `docker manifest inspect` (a registry API call) from build-host.

---

## 1. Build command

```bash
cd glm5-nvfp4-4xspark

# Recommended: hard-pin the base to the digest captured in the inventory (§8).
docker build \
  --build-arg BASE_IMAGE=ghcr.io/spark-arena/dgx-vllm-eugr-nightly@sha256:13a5d18f54cb5ff67fde21821557c9def17574a93365825905f57ae4f41de7a0 \
  -t vllm-node:dsa-repro \
  .

# Convenience (base defaults to ...:latest; build-time assertions still pin the
# engine *content* and will FAIL the build loudly if the nightly rolled forward):
#   docker build -t vllm-node:dsa-repro .
```

The build is **arm64 / aarch64 only** (GB10 / sm_121). Run it on a DGX Spark
node or any `linux/arm64` host that can pull the base. It is NOT buildable on an
x86 host without `--platform linux/arm64` emulation, and the engine wheels are
GB10-specific, so do the real build on the fleet (this is T2).

### What the build does
1. `FROM` the public eugr nightly (or the digest-pinned base).
2. **Asserts provenance**: checks `/workspace/build-metadata.yaml` carries
   `vllm_commit: 08985351f`, then runs a Python assertion that every pinned
   engine version (table below) is present — the build **fails** on any drift.
3. Re-declares the load-bearing SM121 runtime ENVs.
4. **Bakes** the two required source patches (`patch_dense_mla.py`,
   `patch_triton_decode_smem.py`) into the image and re-verifies them, dropping
   stale bytecode. (The live recipe applies these at runtime; baking makes the
   image coherent-by-default. The patches are idempotent + self-verifying, so
   `launch_node.sh` re-applying them at runtime is a harmless no-op.)

---

## 2. Base-image provenance

| Field | Value | Source |
|---|---|---|
| Public base (tag) | `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` | repo-root `README.md` "Hardware baseline"; matches live image stack |
| Public base (digest, arm64) | `sha256:13a5d18f54cb5ff67fde21821557c9def17574a93365825905f57ae4f41de7a0` | `docker manifest inspect --verbose …` → `results/image-inventory.txt` §8 |
| Live fleet image | `vllm-node:dsa` — id `sha256:f5ca01a47aa9…`, **RepoDigests `[]`** (built locally, never pushed → no registry digest) | `docker image inspect` → inventory §1 |
| Live image build provenance | `build_script_commit: 986960209e27b3c181f72b4a30c718a3bc237f2e`, `vllm_commit: 08985351f`, `flashinfer_commit: 9c5ed7c1`, `gpu_arch: 12.1a`, `build_date: 2026-06-18T18:33:08Z` | baked `/workspace/build-metadata.yaml` → inventory §3 |
| Base OS / CUDA | Ubuntu 24.04 sbsa, CUDA 13.0.2, `NVARCH=sbsa` | `docker history` + `Config.Env` → inventory §7/§2 |
| Host driver (fleet) | NVIDIA GB10, driver `580.126.09` | `nvidia-smi` → inventory §5 |

**Why "could not determine" applies to the live image's own digest:** the
running `vllm-node:dsa` was built locally with buildkit and never pushed to a
registry, so `RepoDigests` is empty — there is no content-addressable digest for
*that* tag. We do **not** invent one. The reproducible path instead pins the
**public** base by the real digest above (§8 of the inventory) and asserts the
engine content at build time.

---

## 3. Pinned versions (every value traceable to `results/image-inventory.txt`)

These are the load-bearing pins the Dockerfile **asserts** at build time
(`results/image-inventory.txt` §4 = python probes, §6 = full `pip freeze`):

| Component | Pinned version | Inventory source |
|---|---|---|
| Python | `3.12.3` | §4 |
| vLLM | `0.23.1rc1.dev156+g08985351f.d20260618` (commit `08985351f`) | §3 / §4 / §6 |
| torch | `2.11.0+cu130` | §4 / §6 |
| torch CUDA | `13.0` | §4 |
| flashinfer-python | `0.6.13` (commit `9c5ed7c1`, from `/workspace/wheels/`) | §3 / §4 / §6 |
| flashinfer-cubin / -jit-cache | `0.6.13` | §6 |
| deep_gemm | `2.5.0+88965b0` (aarch64 wheel) | §4 / §6 |
| transformers | `5.12.1` | §4 / §6 |
| triton | `3.6.0` | §4 / §6 |
| ray | `2.55.1` | §4 / §6 |
| fastsafetensors | `0.3.2` | §4 / §6 |
| nvidia-nccl-cu13 | `2.28.9` (image libnccl2 `2.28.3-1+cuda13.0`) | §6 / §2 |
| CUDA toolkit (apt) | `13.0.2` (`CUDA_VERSION=13.0.2`) | §2 / §7 |
| GPU arch target | `12.1a` (`TORCH_CUDA_ARCH_LIST` / `FLASHINFER_CUDA_ARCH_LIST`) | §2 / §3 |

> The Dockerfile deliberately does **not** `pip install` these from PyPI: the
> vLLM, flashinfer, and deep_gemm builds are custom aarch64 / sm_121 / 12.1a
> artifacts that are **not on PyPI** (flashinfer + the vLLM wheel come from
> `/workspace/wheels/` in the base; see `docker history` §7). Installing the
> PyPI versions would clobber the GB10 stack. The correct "pin" for a
> from-source nightly is therefore *FROM the nightly that already carries these
> exact builds* + *assert the inventoried versions at build time*.

---

## 4. Patches baked into the image

| Patch | File | Effect |
|---|---|---|
| Dense MLA | `patches/patch_dense_mla.py` | `index_topk=0` → dense MLA path (not DSA sparse indexer); skip orphan indexer weights. **GLM-5.0 only.** |
| Triton decode SMEM | `patches/patch_triton_decode_smem.py` | Relax the `BLOCK_DMODEL>=1024` guard so MLA decode uses `num_stages=1`, fitting GB10's 101376-byte per-block SMEM cap. |

Patch targets verified present in the base (inventory §9 + live probe):
`vllm/model_executor/models/deepseek_v2.py`,
`vllm/v1/attention/ops/triton_decode_attention.py`. Inventory §9 also proves the
patches are **not** pre-baked in the pristine image (original anchors intact),
so baking them here is a real change, not a no-op.

---

## 5. Build validation performed for T1 (this card)

| Check | Result |
|---|---|
| Dockerfile exists at `glm5-nvfp4-4xspark/Dockerfile` | ✅ |
| Base reference resolves/pulls publicly | ✅ `docker manifest inspect` returned a real arm64 manifest + digest `sha256:13a5d18f…` (inventory §8) |
| `hadolint` lint | ⚠️ not available on build-host (`which hadolint` → empty) |
| `docker buildx build --check` | ⚠️ buildx plugin not installed locally (`docker: unknown command: docker buildx`) |
| **`docker build` actually run** | ❌ **DEFERRED TO T2** — the local Docker **daemon is not running** on build-host (`/var/run/docker.sock` absent), and the image is arm64/sm_121-specific so it must be built on the fleet. T1 is authoring + inventory only (and the card forbids touching fleet GPUs / `vllm serve`). |

**`docker build` could not run in the T1 environment.** This is stated
explicitly per the card's anti-fabrication rule. The Dockerfile syntax is
standard (`# syntax=docker/dockerfile:1.7`, single-stage, heredoc `RUN`), the
base resolves, and the build is left for **T2** to execute on a fleet node.

---

## 6. Non-disruption confirmation

The live GLM-5.0 server was confirmed **UP and untouched** before and after this
work (read-only inventory only; no container stop/restart, no `vllm serve`):

```
ssh user@<node> "ss -ltn | grep :8000"   → LISTEN 0 2048 0.0.0.0:8000
ssh user@<node> "curl -s -o /dev/null -w '%{http_code}' localhost:8000/health" → 200
docker ps → vllm_node  vllm-node:dsa  Up 57 minutes
```

(Re-checked at end of run — see the task completion handoff.)
