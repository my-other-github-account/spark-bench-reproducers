# End-to-end repair campaign

The repair track optimizes the deployed quantized artifact against teacher logits. Every arm uses fixed held-out probes `[4,84,160,236,304,373,442,511]`, reports pooled KLD from exact per-window rows, and labels a gain claimable only when it exceeds the campaign's `2.6%` effect floor.

## Formal seal

`SEALED_REPAIR_REPLICATION.json` is authoritative:

- codebook arm4 final step 48: `+5.647340%` (`+6.221544%` best at step 40), replicated by arm5 current-best step 32 `+5.341266%`;
- RMSNorm-gamma source step 24: `+13.553121%`, replicated by rotated-order step 16 `+13.492236%`;
- arm3 best `+3.016373%`, but final `+2.570928%`: `FADED_BELOW_FLOOR_NOT_SEALED`;
- pilot1, pilot2, arm4_nuclear, and output-scale are sub-floor or negative at their binding panels.

Raw JSONL ledgers, status snapshots, final summaries, and launch configurations are checked in below. `PROBE_TABLES.md` and `PROBE_TABLES.json` are generated from those ledgers and contain every completed pooled panel plus all eight per-window values. The generated table, rather than provisional status-message rounding, is the source for individual-arm trajectories.

## Evidence boundary

- A training loss or single window is not a held-out claim.
- Best and final checkpoints are reported separately.
- Partial arms remain partial; no extrapolation is allowed.
- Mechanism deltas are not added arithmetically; combined artifacts require a new rail.
- In-memory repair is not a served result until export and checkpoint-to-wire gates pass.
