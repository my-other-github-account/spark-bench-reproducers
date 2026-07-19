# Acceleration ledger

Updated 2026-07-19. Every speed claim below is measured on matched work. Projections and theoretical ceilings are labeled separately.

## Sealed acceleration wins

### Full-rail scorer: 3.447117x

A same-host, same-window, same-timer 64-window comparison measured:

| path | wall time |
|---|---:|
| scalar baseline | 2,752.213 s |
| exact accelerated candidate | 798.410 s |
| speedup | **3.4471173958x** |

All 64/64 output files were bit-identical; max, mean-absolute, and signed relative deltas were all zero. The independent 16-window gate measured 697.586 s versus 227.236 s = 3.069874x with 16/16 bit identity.

The exact candidate combines allocator-cache retention between layers, chunk-end flush, record-stream lifetime discipline, and direct FP8 block128→BF16 dequantization. It does not alter rail labels, windows, timer boundaries, or score math.

Receipt SHA-256:

- matched: `98a06d1a1ab40dfde68cee136a77059bb0a267e52bc2ac49f3f113f432e1b234`
- quality: `039cb3655fd0195491c114ed16056d443d817c2c47d311b2f3076a1b4a1dcb76`
- split: `d89a4517ed6817b24ab7bb0233e7eb841d60e41119c96f5a33ac0a590e6a14fc`
- independent verification: `0b252ec88eecc76a2664d7bd1eb372eaeb627ca9884a6fbe0a4014838050bff7`

### VQ plane builder: 15.065546x on row work

The deterministic builder losslessly groups bit-identical d=4 rows, invokes the unchanged sealed nearest-assignment calculation once per representative, and expands the inverse map.

A full 512-unit canary measured 736.187529 s → 48.865639 s = **15.065546x**. Content SHA-256 and metrics SHA-256 were identical. Holding the measured 287.8-second non-row remainder fixed gives a separately labeled reconstructed stage speedup of 1023.987529 s → 336.665639 s = **3.041556x**.

- drop-in builder SHA-256: `9c1856d673a879629a0a98a161bafce5c8c979aeb5f9419b0129af83c5966202`
- exact content SHA-256: `936a8144522e4a3bc4a8bf9f1fd903c155205d8792503fcca21d9a32861409a7`
- exact metrics SHA-256: `9172485ac091af6437bfa6414f5b5dc566e7ba41cfb833415ca0225e86adf056`
- matched canary receipt SHA-256: `d1677ac99838bb84b81fab151607ce815b05b2245988c8b31b790e7c5e7c6eb8`
- final manifest SHA-256: `66381bc7150eee77cc4a98d37123a513fb3c03f0489d8552f2b198212176688c`

## ACCEL-3: repair-training step profile and no-go

The next target is **>=3x sustained repair-step speedup** on a real 10-step segment, with bit-identical state or an explicitly accepted numerical bound below 1e-9.

An initial old-path profile was about 363 s/step: forward 166.8 s, backward 195.4 s, and optimizer below 0.5 s. A later host-valid matched 10-step profile measured 370.272563 s/step and decomposed as:

| phase | 10-step total | mean per step |
|---|---:|---:|
| forward | 1,699.804907 s | 169.980491 s |
| backward | 1,996.162576 s | 199.616258 s |
| grad norm | 0.228746 s | 0.022875 s |
| optimizer | 5.502314 s | 0.550231 s |
| checkpoint | 1.027093 s | 0.102709 s |
| total | 3,702.725635 s | **370.272563 s** |

The dominant cost is forward+backward, not optimizer work.

Two candidate profiles did not meet the gate:

| candidate | step time | speedup | exactness |
|---|---:|---:|---|
| active-expert chunked dequant | 325.061135 s | 1.139086x | 87 state tensors differed; max abs 1.1920929e-6 |
| grouped GEMM | 355.693544 s | 1.040988x | 87 state tensors differed; max abs 1.9073486e-6 |

Even an impossible zero-cost candidate forward, while retaining the best measured non-forward phases, has a **1.821503x** ceiling. Therefore these local dequant regrouping levers cannot reach 3x and are not promoted. No packageable served-path kernel was produced.

No-go receipt SHA-256: `ea0e5504ba5ddbe182f6d82ba985644595d65fe33817fe9064904ef80ab89025`.

## ACCEL-3 next architecture

The target requires structural work rather than another dequant micro-optimization:

1. batch multiple windows through each loaded layer to amortize layer materialization;
2. overlap next-layer load with current-layer compute;
3. bank or reuse invariant teacher activations within a predeclared storage budget;
4. fuse or replace the dominant forward/backward path at an exact served call site;
5. retain the matched 10-step state/optimizer/scheduler gate.

Acceptance remains: >=3.0x sustained on identical ten-step work, unchanged window order and optimizer schedule, final state bit identity or accepted <1e-9 trajectory/state bound, and a drop-in integration used by BIN T/Q2/BIN 0.

## Reproduction templates

Rail aggregation uses the published helper's actual CLI:

```bash
python3 "$CAMPAIGN_ROOT/tooling/agg_rail.py" \
  "$INPUT"/candidate-shard-*.jsonl \
  --expected-windows 64 \
  --output "$CAMPAIGN_ROOT/accel/candidate64.json"
```

The repository does not yet expose the environment-specific repair-step profiler as a standalone command. Do not infer the 370.273-second profile from `agg_rail.py`; reproduce it only with the model-specific trainer while preserving the ten-step phase timers and identity gate described above.
