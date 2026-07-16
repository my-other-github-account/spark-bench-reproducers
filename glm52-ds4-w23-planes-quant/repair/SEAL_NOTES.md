# Sealed Repair Replication

Status: **REPLICATED_ABOVE_FLOOR_HELDOUT_KLD_REPAIR_ZERO_LEAKAGE**

This seal records the campaign's first replicated above-floor pooled held-out KLD repair wins on the same 8 disjoint probes, with zero train/probe leakage.

- Baseline pooled KLD: `0.056753577664494514`
- Held-out probes: `[4, 84, 160, 236, 304, 373, 442, 511]`
- Manifest: `5a622d5139e73452b719d5d2cfeb2571`
- Binding pooled effect floor: `+2.6%`

## Mechanism 1: all-layer codebook repair
- arm4 (spark-1, lr=0.01, train_windows=16): step 48 `0.056753577664 -> 0.053548510303` = `+5.647340%`; best `+6.221544%` [SEALED ABOVE-FLOOR; natural_final].
- arm5 (spark-8, lr=0.003, train_windows=64): step 32 `0.056753577664 -> 0.053722218145` = `+5.341266%`; best `+5.341266%` [SEALED ABOVE-FLOOR; current_best_lower_bound_while_96step_run_continues].
- arm3 (spark-7, lr=0.003, train_windows=16): step 48 `0.056753577664 -> 0.055294484016` = `+2.570928%`; best `+3.016373%` [NOT SEALED / BELOW-FLOOR; natural_final].

Codebook replication gate: at least two independent arms selected above +2.6% on the same pooled 8-window panel.

## Mechanism 2: RMSNorm-gamma repair
- Source: step 24 `0.056753577664 -> 0.049061696744` = `+13.553121%`.
- Rotated-order replicate: step 16 `0.056753577664 -> 0.049096250907` = `+13.492236%`.

RMSNorm replication gate: source and fresh-tag rotated-order replicate both clear +2.6%, each after exact own step-0 parity.

## Floor handling
- arm4_nuclear `+0.31%`: SUB-FLOOR ZERO, not sealed.
- arm6 `-0.89%`: SUB-FLOOR ZERO, not sealed.
- arm3 final step 48 `+2.570928%`: FADED BELOW FLOOR, not sealed (best `+3.016373%`).

No single-window claim is made; every seal decision uses the pooled 8-window panel.
