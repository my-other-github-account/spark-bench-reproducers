# End-to-end repair campaign

The repair track optimizes the deployed quantized artifact against teacher logits. Every arm uses fixed held-out probes `[4,84,160,236,304,373,442,511]`, reports pooled KLD from exact per-window rows, and labels a gain claimable only when it exceeds the campaign's `2.6%` effect floor.

[`TRAJECTORIES.md`](TRAJECTORIES.md) and `tooling/sweeps.py` also report the arithmetic mean of
per-window percentage deltas for LR-dose comparisons. That convention produces the familiar
arm4/arm5 `+6.4218%` / `+5.9809%` and RMSNorm `+11.1590%` / `+10.8783%` headlines; the formal
seal below remains pooled KLD. Raw arm6–10 snapshots live under `results/live-snapshots/`.

## Formal seal

`SEALED_REPAIR_REPLICATION.json` is authoritative:

- codebook arm4 final step 48: `+5.647340%` (`+6.221544%` best at step 40), replicated by arm5 current-best step 32 `+5.341266%`;
- RMSNorm-gamma source step 24: `+13.553121%`, replicated by rotated-order step 16 `+13.492236%`;
- arm3 best `+3.016373%`, but final `+2.570928%`: `FADED_BELOW_FLOOR_NOT_SEALED`;
- pilot1, pilot2, arm4_nuclear, and output-scale are sub-floor or negative at their binding panels.
- the separate 24-window external gate measured approximately `+5.1%` held-out;
- the exported-artifact rail then sealed all 512 windows: full KLD `0.092240` (`+6.781%`),
  but the claims-grade train-excluded 496-window row is KLD `0.094284` (`+5.176%`) and
  therefore does not cross the `0.0927` bar.

Raw JSONL ledgers, status snapshots, final summaries, and launch configurations are checked in below. `PROBE_TABLES.md` and `PROBE_TABLES.json` are generated from those ledgers and contain every completed pooled panel plus all eight per-window values. The generated table, rather than provisional status-message rounding, is the source for individual-arm trajectories.

## Evidence boundary

- A training loss or single window is not a held-out claim.
- Best and final checkpoints are reported separately.
- Partial arms remain partial; no extrapolation is allowed.
- Mechanism deltas are not added arithmetically; combined artifacts require a new rail.
- In-memory repair is not a served result until export and checkpoint-to-wire gates pass.

Use [`../RESUME.md`](../RESUME.md) for the exact next-run phases and copy-paste launch recipes.
