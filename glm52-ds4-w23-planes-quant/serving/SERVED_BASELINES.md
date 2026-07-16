# Served baselines and wire gates

## Baselines

- Served W2 baseline: NLL `1.5191343`, PPL `4.5683`, five windows `{0,174,321,332,380}`.
- GPU-offloaded llama.cpp reference, three repetitions:
  - UD-Q2_K_XL: prefill-2048 `286.589 tok/s`, decode-128 `13.7591 tok/s`.
  - UD-IQ3_XXS: prefill-2048 `285.403 tok/s`, decode-128 `13.0359 tok/s`.
- `AB64_RESULTS.jsonl` contains nine completed baseline-A windows at package cut: mean KLD `0.08977064366141956`, mean wall time `450.6667 s/window`. It is not a completed 64-window A/B verdict.

## Export and wire result

`EXPORT_REPORT.json` proves checkpoint-to-pack conversion completed for the representable codebook/LUT fields. `WIRE_GATE_win0.json` records:

- A fp32 checkpoint: `0.25219660997390747`;
- B in-memory wire snapshot: `0.24353156983852386`;
- C reloaded exported bytes: `0.24353156983852386`;
- export gate: PASS, B equals C;
- wire gate: FAIL, relative difference from A `3.4358%` against a `0.5%` tolerance.

The failure is expected for non-wire-representable W3 LUT repair state: the serving format uses a shared per-layer LUT and resets that state. Therefore the export machinery is verified but this repaired checkpoint is not yet a valid served quality claim.

## Reproduction order

1. Export with the checked-in manifest and pack scripts.
2. Run the in-memory wire snapshot.
3. Reload the exported bytes and require B/C parity.
4. Require A/B tolerance before starting a served A/B rail.
5. Run NLL first, then throughput, then the full disjoint-window A/B ledger.

Never report a served repair win from the partial baseline file or from B/C parity alone.
