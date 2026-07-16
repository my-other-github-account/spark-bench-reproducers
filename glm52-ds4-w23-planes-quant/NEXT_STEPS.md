# Next steps

## Binding targets

- T1: KLD `<0.0927` at `<=101.95 GB`.
- T2: KLD `<0.0927` at `<=95.75 GB`.
- T3: KLD `<=0.0594` and top-1 `>=0.9301` at `101.95 GB`.
- T4: KLD `<=0.0594` and top-1 `>=0.9301` at `95.75 GB`.

## Priority order

1. **Finish and ingest the d8 K2048/K4096 rails.** They were still running at the publication cut. Add their measured rows only after seal/hash verification.
2. **Re-solve the full menu once.** Use all sealed d4 and d8 rungs, emit exact manifests for both budgets, and rail only the best Pareto candidates. Current penta/hexa values are predictions.
3. **Apply replicated repair to the exact candidate.** Reproduce the codebook and RMSNorm-gamma gains on the candidate manifest with disjoint probes and the same `2.6%` floor. Do not transfer percentages arithmetically between artifacts.
4. **Close the wire gap.** `WIRE_GATE_win0.json` proves B/C export parity but fails A/B because repaired W3 state is not representable. Either make the state wire-representable or constrain training to representable fields, then rerun the gate.
5. **Complete served A/B.** After the wire gate passes, collect the full window rail plus NLL and GPU-offloaded throughput. The current `AB64_RESULTS.jsonl` is baseline-only and partial.
6. **Research beyond W3 only after T1/T2 rail.** Strongest next hypotheses are a representation-aware d8 objective, joint codebook/scale optimization, and kernels supporting the chosen representation without a massive throughput penalty.

## Evidence rules

- Do not call a solver prediction measured.
- Do not compare d8 `KL_vs_fp8` directly with d4 `KL_vs_teacher`.
- Do not claim from a single window or a partial arm.
- Preserve negative arms and faded final panels.
- Every new row needs corpus pin, window count, manifest hash, package snapshot, and a sidecar checksum.
