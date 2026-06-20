# BUILD-latest.md — TRACK A (latest vLLM stack) investigation

**Card:** T3.5 · **Date:** 2026-06-20 · **Result: NOT VIABLE on the currently-published nightly.**

## Question
Can we move the GLM-5.0-NVFP4 TP=4 serve to a LATEST vLLM/flashinfer/deepgemm/transformers/triton
stack that builds for GB10/sm_121 (arch 12.1a, aarch64, CUDA 13), to (a) load faster and/or (b) make
the two runtime patches (`patch_triton_decode_smem`, `patch_dense_mla`) upstream-removable?

## Finding: the public nightly is the SAME drifted build T2 already rejected — and it lacks deep_gemm

The proven stack pins (from T1 inventory / T2 build):
- base = `ghcr.io/spark-arena/dgx-vllm-eugr-nightly` @ **vLLM 0.23.1rc1.dev156+g08985351f** (build 06-18),
  with `deep_gemm 2.5.0+88965b0` present (load-bearing for the GLM-5.0 DSA-dense MLA path).

T3.5 re-checked the registry from a fleet node (`docker manifest inspect` from s2):
- Current `ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest` (arm64) digest =
  **`sha256:13a5d18f54cb5ff67fde21821557c9def17574a93365825905f57ae4f41de7a0`**.
- This digest is **UNCHANGED since T1/T2.** T2 already built from it and it FAILED the Dockerfile's
  own version-assertion: that build is **vLLM dev197/g01192139b (build 06-19T21:45) with `deep_gemm`
  ABSENT.** (CAMPAIGN_STATE.md, T2 decision log: "that digest is now vllm_commit 01192139b (dev197) and
  deep_gemm is ABSENT … faithful from-public-base repro is impossible TODAY.")
- The `vllm-node:latest` tag already present on the fleet nodes was probed (`docker run --rm
  --entrypoint python3 vllm-node:latest`): it reports vLLM **dev156** but **deep_gemm ABSENT** — so it
  is NOT a newer vLLM either, just a deep_gemm-less local tag. No newer working vLLM exists on disk.

### Why this kills TRACK A on the published nightly
- `deep_gemm` is load-bearing for GLM-5.0's DSA/dense-MLA path on this stack. A build without it does
  not serve GLM-5.0 coherently; the Dockerfile asserts its presence and FAILS the build by design.
- The exact dev156 nightly that the proven stack uses is **not republished at a stable retrievable
  digest** (the moving `:latest` tag rolled to dev197 and the registry keeps only the latest). So there
  is no newer-than-dev156 published nightly that (a) carries deep_gemm and (b) is retrievable.
- Therefore: pulling 19 GB, building, and QSFP-distributing the current `:latest` across all 4 nodes
  (multi-hour, high wedge-risk on these unified-memory boxes) is pre-determined to land on a
  deep_gemm-less, assertion-failing stack. Not a sound use of the fleet window, and the card's actual
  TOP priority (load speed) was already conclusively answered by TRACK B (no fast loader binds).

### Patch-removability re-test on latest: NOT POSSIBLE
Re-testing whether `patch_triton_decode_smem` / `patch_dense_mla` became upstream-removable requires a
*newer working* vLLM than dev156. None is published/retrievable (above). On the pinned dev156 stack the
T3 ablation already proved both patches LOAD-BEARING (remove→NO_BIND / DECODE_ERROR). No change.

## Recommendation
- **Stay on the pinned stack** (`glm5-repro:t2`, vLLM dev156 + deep_gemm) with serial load.
- **Registry action to UNBLOCK a real TRACK A later:** push the proven `glm5-repro:t2` (or the original
  `vllm-node:dsa`, which carries dev156 + deep_gemm) to a registry so it has a STABLE digest, and
  separately watch for a future eugr nightly that re-includes deep_gemm at dev>156. Only then is a
  latest-stack bump testable. This is exactly the "push the proven image to a registry for a stable
  digest" action T2 already flagged.
- If/when a newer deep_gemm-carrying nightly appears: rebuild via `Dockerfile --build-arg
  BASE_IMAGE=<new digest>`, serve TP=4 with `launch_node_minimal.sh`, coherence-gate (2+2→4,
  France→Paris), then for each patch do remove→relaunch→classify.
