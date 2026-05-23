# Speed up DFlash draft block generation

## Summary

This speeds up the DFlash proposer after the verified decode path is in place.

DFlash works by proposing a block of tokens and letting the main model verify the block. PR #2 adds the correctness path for that flow. This PR reduces the cost of producing each draft block so the verified path is faster than plain autoregressive decoding.

## What changed

- Cache projected context rows used by the DFlash proposer.
- Cache DFlash context K/V state instead of rebuilding it every step.
- Add the fast gamma=16 FFN/GEMM path.
- Add support for the transposed output projection used by the proposer.
- Add the corresponding GB10 CUDA kernels and quantized transpose helpers.

## How I verified it

I ran the same deterministic 128-token completion benchmark before and after the proposer/cache/kernel changes.

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

Throughput improved from slower-than-AR to faster-than-AR:

```text
Before: DFlash 12.85 tok/s, 0.95x AR
After:  DFlash 29.31 tok/s, 2.17x AR
```

Completion lengths stayed exact in the final run:

```text
AR:     [128, 128, 128]
DFlash: [128, 128, 128]
```

The server logs also showed the DFlash verifier path and the fast gamma=16 path active during the run.

I also ran:

```bash
git diff --check
```

## Notes

This PR is performance-only relative to the verified DFlash decode path. It should not change the accept/reject rule or serving stop semantics.

