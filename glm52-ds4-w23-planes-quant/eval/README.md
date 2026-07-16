# Evaluation rail

The campaign uses a teacher-forced, per-window ledger rather than a single opaque aggregate.

## Canonical contract

- corpus: `windows_ds4_eval.json`;
- corpus MD5: `1701920b4ba96dea0b18fe9df0151876`;
- 512 windows, 1,024 scored positions each;
- teacher support: top 8,192 logits per position;
- divergence: renormalized `KL(reference || candidate)`;
- side metrics: top-1 agreement and Jensen-Shannon divergence;
- every row includes variant, manifest hash, corpus hash, completed-window count, and score-ledger hash.

## ECORPUS discipline

Two incidents established the rule:

1. a calibration/evaluation corpus mix made a promising row non-comparable;
2. an overwritten or mismatched window set let resume logic continue against a different corpus identity.

The fix is fail-closed: pin the corpus MD5 in every builder, teacher receipt, candidate launcher, chunk ledger, final row, and resume check. A chunk is reusable only when corpus, teacher, manifest, scorer version, and window range all match.

## Per-window ledger shape

See `LEDGER_SCHEMA.json`. The scorer writes append-only JSONL rows. Finalization rejects duplicates, gaps, mixed hashes, non-finite values, and incomplete 512-window coverage before emitting a sealed aggregate and checksum sidecar.
