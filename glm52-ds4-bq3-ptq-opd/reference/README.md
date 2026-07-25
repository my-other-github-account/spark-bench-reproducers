# Public reference source

This directory is a scrubbed source-only export of the PTQ-OPD implementation used in the campaign.

## Files

- `ptq_opd.py` — bank validator, exact top-k+tail distribution checks, FP32 JSD/reverse-KL, static-gate logic, and parameter ranking.
- `train_contracts.py` — resume ladder, deterministic bank slicing, checkpoint/source hashing, and update-state validation.
- `train_ptq_opd.py` — full 43-layer trainer, static anchor, segmented own-rollout backward pass, durable checkpoints, and gate receipts.
- `adapter/` — BQ3 model construction inherited from the prior quantization reproducer. `adapter/vendor/` contains the source-only layer builder, pack reader, model rail, and plane readers needed by that adapter.
- `plans/static_anchor_control.json` — static calibration/anchor window schedule.
- `tests/` — CPU unit tests for bank, divergence, gate, update, and fsync contracts.

## External inputs

No model artifact is committed. `train_ptq_opd.py` requires:

- `TAILFIX_START_CKPT` pointing to BQ3 step0;
- BQ3 wire/model paths consumed by the adapter;
- the static teacher corpus;
- a PTQ-OPD teacher-score bank.

See `../REPRO.md`.

## Naming note

Some adapter module names retain historical internal `binrepair` / `tailfix` filenames because they define the BQ3 construction boundary. The public method implemented above that boundary is PTQ-OPD; community `IQ*` names are not used for BQ3.
