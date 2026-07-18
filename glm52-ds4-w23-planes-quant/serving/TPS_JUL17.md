# Jul 17 serving throughput

## Protocol

- one uncontended NVIDIA GB10;
- temperature 0;
- no MTP or speculative decoding;
- primary: one streaming 4,096-token output, decode-after-first over the full output;
- secondary: two warmups followed by five 64-token streams.

## Results

| Stack | Full-4K decode-after-first | 5×64 median |
|---|---:|---:|
| Clean grouped-Triton + graphs | 6.9900 tok/s | — |
| CUDA warp VQ + per-layer decode graphs | **14.1345 tok/s** | **14.9296 tok/s** |

Depth windows for the winner: 0–256 = 10.7331, 1024–1280 = 14.5462, 3840–4096 = 15.0149 tok/s. There is no late-depth sag in this run.

The serving flags were:

```text
VLLM_MOE_VQ_FAST=1
VLLM_MOE_VQ_GROUP_FAST=1
VLLM_MOE_VQ_CUDA_WARP=1
VLLM_MOE_W2_DECODE_GRAPH=1
```

## Kernel measurements and quality

Real-layer isolated medians:

- vq13: 1.6897 ms → 0.2360 ms (7.16×);
- vq2: 0.8231 ms → 0.1228 ms (6.70×);
- combined: 2.5128 ms → 0.3588 ms (7.00×).

The kernel changes reduction order. On the real-layer panel, max absolute difference was 0.015625, cosine 1.0, and equal fraction about 0.998. The matched 64-window bitwise-vs-warp served control attributed only +0.0002967% NLL to the warp kernel (PASS ≤0.3%). The separate offline-to-served drift is common to both paths and remains under investigation.

The fast path is decode-only and assumes T=1 with row stride 4. T≥2 must use the general fallback.
