## Summary

This updates Atlas's existing DFlash path so it is both correct as verified speculative decoding and fast enough to beat normal autoregressive decoding on the tested `llama-benchy` workload.

Atlas already has DFlash support. This PR does not add a new speculative decoding system. It fixes the existing DFlash path so draft blocks are verified against the target model, rejected draft rows are discarded, and the next draft is conditioned only on verified state. It also adds the proposer/cache/kernel fast paths needed for the full gamma=16 block path to be worthwhile.

DFlash is a draft head: it proposes several future tokens at once, then the main model verifies those tokens before they become part of the response. The user-visible contract is the same as normal autoregressive decoding: DFlash may make generation faster, but it should not change the target model's accepted token stream.

## What changed

Correctness/state handling:

- Use the existing DFlash proposer and verifier path for the full draft block instead of the one-token safety cap.
- Stage target hidden states for every row in the verifier block.
- Append those verifier rows into DFlash proposer context, then trim to row 0 plus the accepted draft prefix.
- Roll back rejected draft tokens from sequence state, recurrent/GDN state, and proposer context.
- Wire the target-model projection through the DFlash path for quantized verifier scoring.

Performance:

- Cache projected context rows used by the DFlash proposer.
- Cache DFlash context K/V state instead of rebuilding it every step.
- Add the fast gamma=16 FFN/GEMM path.
- Add support for the transposed output projection used by the proposer.
- Add the corresponding GB10 CUDA kernels and quantized transpose helpers.

## llama-benchy before / after

I compared normal autoregressive decoding and DFlash with the same `llama-benchy` shape:

```bash
llama-benchy \
  --base-url http://127.0.0.1:18180/v1 \
  --api-key [REDACTED] \
  --model Qwen3.6-27B-NVFP4-unsloth \
  --served-model-name Qwen3.6-27B-NVFP4-unsloth \
  --pp 2048 \
  --tg 128 \
  --depth 0 \
  --concurrency 1 \
  --runs 3 \
  --no-cache \
  --no-adapt-prompt \
  --no-warmup \
  --latency-mode none \
  --skip-coherence \
  --format json
```

Before this PR's DFlash changes, `llama-benchy` showed DFlash slightly slower than AR:

```text
AR tg_throughput:     13.53 tok/s
DFlash tg_throughput: 12.85 tok/s
Ratio:                0.95x AR
```

After this PR's DFlash changes, the same `llama-benchy` shape showed DFlash faster than AR:

```text
AR tg_throughput:     13.49 tok/s
DFlash tg_throughput: 29.31 tok/s
Ratio:                2.17x AR
```

The fixed-generation token accounting check passed for the final `llama-benchy` run:

```text
AR usage.completion_tokens:     [128, 128, 128]
DFlash usage.completion_tokens: [128, 128, 128]
```

The DFlash server logs also showed the verifier path active with mixed acceptance lengths, including zero-token, partial, and full-block accepts. That shows the verifier is deciding per block rather than blindly accepting the draft block.

I also ran:

```bash
git diff --check
```

## Notes

This PR keeps the DFlash correctness and performance changes together because the meaningful reviewer-facing before/after is the `llama-benchy` result for the combined DFlash path: the previous DFlash path was slower than AR, while the updated verified DFlash path is faster than AR with exact fixed-token accounting.
