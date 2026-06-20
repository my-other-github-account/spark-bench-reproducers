# BUILD-latest.md — GLM-5.0-NVFP4 on LATEST vLLM (built from source for sm_121)

**Status: ✅ VIABLE AND PROVEN.** GLM-5.0-NVFP4 TP=4 serves coherently on vLLM `main` HEAD
(`dev207`, commit `dced2907`), built from source for DGX Spark GB10 / sm_121.

> This supersedes the earlier "NOT VIABLE" note. That conclusion was wrong because it only
> checked whether the **prebuilt published nightly** had been republished — it never built vLLM
> from source. The community DGX-Spark builder compiles vLLM + FlashInfer wheels for sm_121 from
> any git ref; doing so gives us "latest vLLM" without waiting on anyone's nightly.

## Result (measured on the fleet)
- **vLLM `0.23.1rc1.dev207+gdced29076.d20260620`** (commit `dced2907693e3d6bf9eb7168d0a8fecf1cd22dca`,
  vllm-project/vllm `main` HEAD 2026-06-20) — NEWER than the pinned `dev156/g08985351f` anchor.
- FlashInfer `9c5ed7c194e7412780862491742fc655daaad6ac`, gpu_arch `12.1a`, built via eugr builder
  commit `9bac1ea09b65`. Image `glm5-repro:latest` (19 GB), distributed to all 4 nodes over QSFP.
- **Coherent:** `France → " Paris. Distance from London to Paris is 342 km"` — byte-identical to the
  dev156 anchor. `2+2=` → coherent English (not salad).
- **Bench:** ~11.1 tok/s median (tg256, n=5, c=1, temp0) vs 11.35 anchor = **−2.2%, parity.**
- **Patches required on latest:** THREE — `patch_mem_bypass` (NEW, see below) + `patch_dense_mla` +
  `patch_triton_decode_smem`. None of the three became upstream-removable on main HEAD. Patch-
  removability of `--enforce-eager` / `--moe-backend` re-tested per T3.7 — see results/LATEST-VLLM.md.

## How to build it (on the DGX Spark head node — NOT macOS/x86)

### Step 1 — build the latest vLLM base from source (eugr builder)
```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
./build-and-copy.sh \
    --rebuild-vllm --rebuild-flashinfer \
    --vllm-ref main \            # or pin a sha/tag (e.g. v0.23.1rc0) for reproducibility
    --gpu-arch 12.1a \
    -j 16 \
    -t glm5-repro:latest \
    -c                            # docker save | ssh docker load to peer nodes
```
- Compiles vLLM (`TORCH_CUDA_ARCH_LIST=12.1a`, CUDA 13.0.2-devel, NCCL v2.30u1 built
  `-gencode arch=compute_121,code=sm_121`) + FlashInfer from source. ~20-40 min; ccache makes
  subsequent builds faster. Bakes `/workspace/build-metadata.yaml` with the exact commits.
- `--vllm-ref` accepts any branch/tag/sha. Use `main` for truest "latest"; pin a sha for a
  reproducible image. If `main` fails to compile for sm_121, step back to the newest building ref
  and record it (that is a valid finding, not a wall).

### Step 2 — bake the GLM-5.0 serving patches (this repo's Dockerfile.latest)
```bash
docker build -f Dockerfile.latest \
    --build-arg BASE_IMAGE=glm5-repro:latest \
    -t glm5-repro:latest-glm50 .
```
`Dockerfile.latest` asserts the base is a from-source latest build (rejects the dev156 anchor),
then bakes + re-verifies the three patches (build fails loudly if any anchor is missing).

### Step 3 — serve
```bash
# on each node, with its rank (rank 0 = head, API on :8000):
NODE_RANK=<r> MASTER_ADDR=<rank0 fabric IP> HOST_IP=<this node fabric IP> \
IMAGE=glm5-repro:latest-glm50 \
  scripts/launch_node_latest_glm50.sh
```
Coherence-gate with a REAL decode (`2+2`, `France→Paris`), not just `/health`.

## deep_gemm note (matters for GLM-5.2, not 5.0)
The from-source main build does **not** carry `deep_gemm` (`import deep_gemm` → ModuleNotFoundError);
the runtime-NVRTC-JIT assumption did not hold for this ref. **GLM-5.0 dense does not need it** — this
image serves without it. **GLM-5.2 DSA-sparse DOES** (load-bearing `fp8_mqa_logits`): install
`github.com/deepseek-ai/DeepGEMM` (OSS, unarchived) from source into the image before the 5.2 bring-up.

## The patches (all self-verifying; build fails if an anchor is gone)
| patch | why | marker (grep) | file |
|---|---|---|---|
| patch_mem_bypass | GB10 unified-mem startup free-mem check refuses to start with resident NVFP4 weights; pinned image baked this, from-source does not | `GB10 unified mem bypass` | vllm/v1/worker/utils.py |
| patch_dense_mla | index_topk → dense MLA, skip orphan indexer weights (GLM-5.0 only) | `not in (None, 0)` | vllm/model_executor/models/deepseek_v2.py |
| patch_triton_decode_smem | GB10 sm_121 101376B SMEM cap → num_stages=1 | `GB10 sm_121 101376B SMEM cap` | vllm/v1/attention/ops/triton_decode_attention.py |

## Compat ceiling / bumping to a newer HEAD
To move to a newer main HEAD: rebuild step 1 with `--vllm-ref <newer>`, then rebuild step 2 with
`--build-arg EXPECT_VLLM_COMMIT=<newer> --build-arg EXPECT_VLLM_VERSION_PREFIX=<newer>`. Re-run the
coherence gate AND re-test patch-removability (an upstream fix can retire a patch; a refactor can move
an anchor and the build will fail loudly until the patch is updated).
