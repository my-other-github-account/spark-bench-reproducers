# Results

All throughput values are real streaming raw-autoregressive measurements with temperature 0, no MTP/speculative decoding, and an uncontended single GB10.

| Run | Full 4096-token decode-after-first | Late window 3840–4096 | 5x64 median decode-after-first | Receipt |
|---|---:|---:|---:|---|
| Live winning stack | 14.1345 tok/s | 15.0149 tok/s | 14.9296 tok/s | `receipts/claim_public.json`, `receipts/short64_cudawarp_*.json` |
| Clean mounted-artifact container on GB10 | not rerun | not rerun | **14.9935 tok/s** | `receipts/container-e2e.json` |

The live baseline without the CUDA warp path measured 6.9900 tok/s over the full 4096-token decode; the winning throughput stack was 2.0221× faster. Kernel isolation stayed inside a quantified BF16 reduction-order envelope (max absolute error 0.015625, cosine 1.0, equal fraction about 0.998), and the greedy behavior gate passed 574 strictly ascending integers.

## Quality attribution

The initial post-throughput 64-window comparison measured exact offline arm4 reconstruction NLL 1.3167553125 versus served warp+graph NLL 1.3344540367054745: **+1.3441164078%**, above the original 1.0% offline gate. Served-vs-teacher top1 was 0.9062347412 versus offline-arm4-vs-teacher 0.9090881563; direct served-vs-offline argmax agreement was 0.9133911133. That result did not isolate the warp kernel.

A subsequent 64-window SOLO bitwise control on the identical serving path measured NLL 1.3344500772447927. The paired CUDA-warp increment is therefore only **+0.0002967106%**, passing the re-registered 0.3% isolated-kernel gate and superseding the earlier warp-failure framing. The common bitwise-serving-versus-offline gap is **+1.3438157095%** and cannot honestly be attributed to the warp kernel. No model, quantization, or artifact tuning was performed.

The scrubbed receipts are `receipts/quality_seal_public.json` for the original offline comparison and `receipts/attribution_control_public.json` for the paired bitwise control. The package remains research-only because the common served-path drift is unresolved, even though the CUDA warp path itself passed attribution.

The container receipt also seals coherent greedy32 output and one occurrence of each required on-path sentinel: CUDA warp GEMV, W2 decode graph, and exact VQ. Its five decode-after-first rates are 14.993464, 15.008157, 14.993205, 14.990828, and 15.004194 tok/s. The public KLD row receipts are in `receipts/kld_rows.jsonl`. Warmups are not included in either reported 5x64 median.
