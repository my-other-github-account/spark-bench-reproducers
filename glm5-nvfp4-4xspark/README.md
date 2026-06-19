# glm5-nvfp4-4xspark

Minimal from-scratch reproducer for serving **GLM-5.0 (`nvidia/GLM-5-NVFP4`, 744B MoE /
40B active)** with **vLLM tensor-parallel across 4 DGX Spark (GB10) nodes**, bound on port
8000, **coherent**, at **~11.3 tok/s single-stream decode**.

This is the *dense* path: GLM-5.0's DSA sparse indexer is a no-op, so we force the dense MLA
attention path (`index_topk=0`) and need **no** behavior-only model surgery — only two small,
generic source patches (one to take the dense path cleanly, one to fit a Triton kernel inside
the GB10 shared-memory limit).

| | |
|---|---|
| Model | `nvidia/GLM-5-NVFP4` (revision `dc54ff55a7e9e71b85db953d8bc22eca894b44c6`) |
| Quant | NVFP4 (`modelopt_fp4`) + fp8_e4m3 KV |
| Topology | TP=4 over 4× DGX Spark GB10, QSFP/RDMA fabric |
| Headline | **11.35 tok/s median** (single-stream tg256, c=1, end-to-end) |
| Engine | vLLM `0.23.1rc1.dev156+g08985351f` (torch 2.11.0+cu130) |
| Status | ✅ reproduced + coherent (2+2→4, France→Paris) |

## Why this is non-trivial

A naive `vllm serve --tensor-parallel-size 4 --nnodes 4` of this checkpoint on GB10 dies at
one of **four** distinct walls, in order. Each masks the next, so they have to be cleared in
sequence:

1. **NVFP4 post-load finalize OOM.** At `MoEPrepareAndFinalizeNoDPEPModular` (right after
   `Model loading took ~104 GiB`) a worker allocates a **non-swappable** GPU-pool chunk on the
   unified LPDDR5X. Survival is governed purely by **free physical RAM margin** at that
   instant — not KV size, not util. *Fix:* launch into freshly-booted nodes (max physical
   headroom) and keep page cache dropped during load (`scripts/cache_reaper.sh`), util `0.99`.

2. **`VLLM_HOST_IP` / ZMQ bind to the wrong node.** vLLM's `get_ip()` returns `VLLM_HOST_IP`
   **verbatim**, and `shm_broadcast.py` binds a ZMQ socket to it at engine-core init — *after*
   NCCL "Connected all trees", ~10 min in. If the per-node `VLLM_HOST_IP` doesn't match that
   node's own fabric IP, the worker tries to `bind()` a remote IP →
   `zmq.error.ZMQError: Cannot assign requested address`. *Fix:* each node must export its
   **own** fabric IP (see `launch_node.sh`; the rank→IP map is the single most error-prone
   line when copying this across fleets — always verify it against `ip -br addr`).

3. **Triton decode-attention shared-memory overflow.** The grouped MLA decode kernel
   (`triton_decode_attention.py::_decode_grouped_att_m_fwd`) requests **102400 bytes** of
   per-block SMEM at `num_stages=2`, but GB10 (sm_121) caps at **101376** — over by exactly
   1 KB. vLLM already drops `num_stages`→1 for this, but behind a `BLOCK_DMODEL>=1024` guard
   that GLM-5.0's MLA decode doesn't trip. Surfaces only on the **first real decode**, after
   bind + `/health`=200, as `triton.runtime.errors.OutOfResources: out of resource: shared
   memory` followed by a downstream `KeyError` in the scheduler and an HTTP 500. *Fix:*
   `patches/patch_triton_decode_smem.py` relaxes the guard so non-HIP always uses
   `num_stages=1` (pipelining depth only — numerically identical).

4. **`index_topk=0` must take the dense MLA path.** Without
   `patches/patch_dense_mla.py`, the `index_topk` config + indexer checkpoint weights route
   GLM-5.0 into the DSA sparse backend (a no-op on 5.0) and into KeyErrors on indexer weights.
   The patch gates `is_v32` on `index_topk not in (None, 0)` and skips the orphan indexer
   weights. (`index_topk=0` is **GLM-5.0 only** — on GLM-5.1 the indexer is load-bearing and
   this would produce token-salad.)

> **Two diagnostic rules that save hours here:** (a) the API server binds + returns
> `/health`=200 *before* it can decode, so **`health=200` is not success — you must exercise a
> real completion**; (b) the container runs `--network=host`, so the `:8000` listener is in the
> **host** netns — `docker exec … ss` won't see it; check `ss -ltn` on the host.

## Layout

```
glm5-nvfp4-4xspark/
├── README.md
├── patches/
│   ├── patch_dense_mla.py            # index_topk=0 -> dense MLA, skip orphan indexer weights
│   └── patch_triton_decode_smem.py   # GB10 sm_121 SMEM: num_stages=1 for MLA decode
├── scripts/
│   ├── download_models.sh            # pull nvidia/GLM-5-NVFP4 (pinned revision)
│   ├── cache_reaper.sh               # drop clean page cache during serial load (finalize headroom)
│   ├── launch_node.sh                # run on EACH node with its rank; applies patches + serves
│   ├── wait_for_server.sh            # poll host :8000 + /health
│   └── bench.sh                      # single-stream tg256 c=1, 5 timed runs + coherence
└── results/
    ├── decode-tg256-n5.json          # the measured run below
    └── RESULTS.md
```

## Reproduce (4 nodes)

Pick node-rank 0 as the head; set its fabric IP as `MASTER_ADDR`. On **every** node, export
that node's own fabric IP and its rank, then launch. Requires the engine base image (vLLM
0.23.1 nightly for GB10 / `sm_121` with FlashInfer for `12.1a`; see the repo root README
"Hardware baseline").

```bash
# --- on every node: stage weights once (shared cache or per-node) ---
bash scripts/download_models.sh

# --- node 0 (head) ---
NODE_RANK=0 NNODES=4 \
  MASTER_ADDR=<node0_fabric_ip> HOST_IP=<node0_fabric_ip> \
  IFACE=enp1s0f1np1 IB_HCA=rocep1s0f1 IB_GID_INDEX=3 \
  bash scripts/launch_node.sh

# --- node 1 ---
NODE_RANK=1 NNODES=4 \
  MASTER_ADDR=<node0_fabric_ip> HOST_IP=<node1_fabric_ip> \
  IFACE=enp1s0f1np1 IB_HCA=rocep1s0f1 IB_GID_INDEX=3 \
  bash scripts/launch_node.sh

# --- node 2 / node 3: same, NODE_RANK=2/3, HOST_IP=<that node's fabric ip> ---

# --- once up (run anywhere that can reach the head) ---
bash scripts/wait_for_server.sh <node0_fabric_ip>
SERVER=<node0_fabric_ip> bash scripts/bench.sh
```

`HOST_IP` **must** be the fabric IP of the node you run it on, and `MASTER_ADDR` **must** be
node 0's fabric IP on all nodes. Launch the head first, then the workers within ~30 s.

## Measured result

5 timed single-stream runs, 256 new tokens each, temperature 0, after 2 warmups
(`results/decode-tg256-n5.json`):

| run | tokens | wall (s) | tok/s |
|---|---|---|---|
| 1 | 256 | 22.637 | 11.31 |
| 2 | 256 | 22.564 | 11.35 |
| 3 | 256 | 22.606 | 11.32 |
| 4 | 256 | 22.504 | 11.38 |
| 5 | 256 | 22.426 | 11.42 |

**median 11.35 tok/s · mean 11.36 tok/s** end-to-end (HTTP + ~0.76 s prefill + decode).
Coherence: `2+2 → " - 4"`, `The capital of France is → " Paris."`

## Config (the headline `vllm serve` flags)

```
--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --moe-backend cutlass
--tensor-parallel-size 4 --nnodes 4 --node-rank $RANK
--max-model-len 2048 --gpu-memory-utilization 0.99
--enforce-eager --no-enable-flashinfer-autotune
--max-num-seqs 1 --max-num-batched-tokens 256 --num-gpu-blocks-override 128
```

`--max-model-len 2048` + tiny block/batch caps are the **minimum-footprint** serving config
used to fit alongside the NVFP4 weights on the unified-memory finalize wall; raise them once
you have headroom. `--enforce-eager` avoids CUDA-graph capture cost on `sm_121`.
