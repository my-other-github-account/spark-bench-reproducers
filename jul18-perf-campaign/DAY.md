# July 18 campaign ledger

## Headline results

| Topic | Result | Status / caveat |
|---|---:|---|
| MTP kernel ladder | sealed 14.54 tok/s → L3 16.03 → L3c 17.34; L3d code prompt 17.01 and 16.63 at 4K | Prompt changed for the L3d canonical row; raw receipts are preserved. L3c achieved 18.09 tok/s on its sustained-4K stream. |
| c=2 concurrency | c1 13.829; c2 clean median 19.886 aggregate; 1.438× | PASS versus the preregistered 1.4× ratio bar. MTP disabled for this concurrency row. |
| IQ4_XS RPC `memset_tensor` | patched build and exact-output N=1/N=4 canary PASS | Patch removed the DSV4 RPC assertion, but the old payload-copy implementation remains an upstream design risk and was not promoted for the long N=3 run. |
| 3003 MHz clock lock | c1 12.571; c2 19.625; c4 29.257 aggregate tok/s | Applied without reboot; clock persistence must be rechecked after driver reset or reboot. |
| Quality rows | R5 full-512 0.07506409375; GPTQ gate64 0.09078628125 | Same scorer family but different rail lengths; never present the GPTQ gate64 as a full-512 claim. R5 footprint receipt reports 193.063787137 GB. |
| Gate64 predictor | paired gate64/full512 rank correlation ρ=0.978; unpaired inflation 1.0365 | Campaign certainty planning put the equivalent unpaired budget at about 7,000 windows. Do not multiply already paired full-512 scores by the unpaired factor. |
| Q2 repair training | step-45 panel 0.0712644 → 0.0545097 (-23.5106%) at 95.75 GB | Training-progress receipt only; full-512 rail was still pending. |
| Compression-loop acceleration | rail profile 47% load / 53% forward; trainer 35% frozen-row gather+H2D | Bulk `fill_layer`, readahead, and trainer code/scale cache are published with 12/12 and 10/10 unit receipts. End-to-end GPU gates remain required before production adoption. |

## Chronology and negative results

1. **The apparent 64 ms host gap was refuted.** The first aggregate `nsys` read attributed about half of a 127 ms speculative cycle to launches and `cudaEventSynchronize`. A steady-state `--cuda-graph-trace=node` decomposition instead showed 126.3/126.7 ms GPU-busy time. The synchronization call was waiting for the GPU; the eager kernels belonged to warmup/prefill. Lever A—capture an allegedly eager speculative step—was therefore dead on arrival.
2. **The real MTP cost was inside the captured graph.** The T=2 graph spent about 69.4 ms/cycle in learned-VQ GEMV because the old `grid.z=valid_m` kernel decoded the same weight rows once per valid activation row. L3 decoded weights once per M-group; L3b packed activation pairs; L3c fixed lane-blocked, roughly 40–60-byte private code spans with coalesced warp-wide loads; L3d pipelined the next code/scale fetch.
3. **A second phantom eager-VQ hypothesis was rejected.** Eighty-six visible eager VQ launches all occurred during prompt/prefill, not steady-state decode. Changing the per-layer graph depth could not remove work that was not executing there. MTP k=2 was also measured and rejected: graph padding T=3→4 raised the cycle to about 165 ms and produced only 14.7 tok/s.
4. **The first concurrency host was a data-locality failure, not a kernel result.** A 95 GB wire exposed through SSHFS/FUSE remained wedged in startup for more than 26 minutes. No endpoint and no throughput row were claimed. Local placement removed the wedge; the final L3d/native-graph c=2 row passed at 1.438×.
5. **The IQ4 slowdown was a configuration pathology.** `--no-kv-offload` kept DSV4 KV on the CPU and forced remote-layer attention/KV traffic through the RPC path every token. The 2.16 tok/s row was about 6× below the patched GPU-KV configuration. PR #13601 supplied the missing RPC `memset_tensor`; removing `--no-kv-offload` restored 12.571/19.625/29.257 tok/s at c1/c2/c4.
6. **Clock labels became part of the instrument.** Reachable GB10s were observed at 2437–2541 MHz versus 3003 MHz maximum. Earlier MTP and rail rows remain valid but are labeled pre-lock; later rows must not be mixed into the same median unless all samples share the same clock policy.
7. **Compression acceleration stopped at the evidence boundary.** Production-loop wall splits were more useful than sampling for the rail; `py-spy` was useful for the trainer. The code and CPU contracts landed, but no projected 540–600 s chunk is reported as measured until the GPU bit-identity and same-chunk timing gates pass.

These failed hypotheses and blocked attempts are retained because they prevent repeating expensive but non-causal work.

## Topic packages

1. [`mtp-kernel-ladder/`](mtp-kernel-ladder/) — L3/L3b/L3c/L3d CUDA kernels, launchers, benches, isolated real-plane gates, and served rows.
2. [`concurrency-cliff/`](concurrency-cliff/) — small-M dispatch, M4 gate, c1/c2 protocol, and final c=2 receipt.
3. [`iq4-rpc-memset/`](iq4-rpc-memset/) — pinned llama.cpp build, patch, launchers, canaries, and throughput harness.
4. [`clock-lock/`](clock-lock/) — no-reboot GPU clock recipe and receipt.
5. [`quality-results/`](quality-results/) — R5/GPTQ score and provenance summary.
6. [`quality-predictor/`](quality-predictor/) — gate64/full512 calibration methodology and reusable standard-library script.
7. [`q2-training/`](q2-training/) — 95.75 GB Q2 training-progress receipt and integrity rules.
8. [`compression-loop-accel/`](compression-loop-accel/) — rail/trainer profiles, optimization source, and unit receipts.

## Measurement law

- Warmups are excluded unless a receipt explicitly labels them.
- Throughput rows report decode tokens/s after first token when that is the instrument; concurrency rows report aggregate completion tokens divided by batch wall.
- Kernel correctness receipts are tolerance-gated against the grouped dequant oracle. They are **not** bit-equal to that oracle. L3/L3b preserve per-row M=1 FMA order; L3c changes lane accumulation order and is therefore tolerance-gated.
- The two golden-preflight text files are retained as documented known deviations, not as PASS receipts: MTP needs the full checkpoint for the draft head and sets `VLLM_MOE_VQ_CUDA_WARP_MAX_M=4`.
- No forecast is promoted as a measurement. Staged accelerations are labeled staged.

## Repository lineage

This package extends the existing July 18 repository state containing the Q2 full-menu publication and Docker reproducer lineages. Every new topic is committed independently for auditability.
