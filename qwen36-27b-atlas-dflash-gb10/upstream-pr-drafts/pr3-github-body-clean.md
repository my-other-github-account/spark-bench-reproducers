## Summary

This updates Atlas's existing DFlash path so it is both correct as verified speculative decoding and fast enough to beat normal autoregressive decoding on the tested fixed-token workload.

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

## How I verified it

I used a deterministic 128-token OpenAI-compatible completion request and compared normal autoregressive decoding with DFlash enabled.

Request used:

```bash
qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/requests/dflash-perf-sherlock-128.json
```

Benchmark/client command shape:

```bash
python3 qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/scripts/repro_client.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --request qwen36-27b-atlas-dflash-gb10/upstream-pr-drafts/requests/dflash-perf-sherlock-128.json \
  --runs 3
```

Before these DFlash changes, the DFlash path was slower than AR on this workload:

```text
AR:     13.53 tok/s
DFlash: 12.85 tok/s
Ratio:  0.95x AR
```

After these DFlash changes, the same workload was faster with DFlash enabled:

```text
AR:     13.49 tok/s
DFlash: 29.31 tok/s
Ratio:  2.17x AR
```

Completion-token accounting stayed exact in the final run:

```text
AR:     [128, 128, 128]
DFlash: [128, 128, 128]
```

The DFlash server logs also showed the verifier path active with mixed acceptance lengths, including zero-token, partial, and full-block accepts. That shows the verifier is deciding per block rather than blindly accepting the draft block.

I also ran:

```bash
git diff --check
```

## Notes

This PR intentionally keeps the DFlash correctness and performance changes together because the useful before/after outcome is the combined DFlash path: the previous DFlash path did not produce a reviewer-friendly standalone correctness result, while the combined change demonstrates the actual user-visible outcome: verified DFlash runs and beats AR on the tested fixed-token workload.
