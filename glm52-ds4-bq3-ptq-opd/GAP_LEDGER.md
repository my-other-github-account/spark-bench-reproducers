# Canonical Gap Ledger

Publication snapshot: 2026-07-31. This is the only canonical measurement-gap ledger for the release. A single HOLDOUT512 run closes its global value and six-class vector together; repeated TBD labels in comparison tables do not create additional gaps.

| Cell | Owner | Exact closure contract | Forbidden substitutes |
|---|---|---|---|
| IQ3 HOLDOUT512 | IQ3 HOLDOUT512 finalizer | Score the immutable IQ3 comparator on `HOLDOUT512_V1`: KL(teacher || candidate), support 8,192, cutoff 1,024, exactly 512 windows / 524,288 positions; publish global plus reasoning, chat, agentic, code, prose, and multilingual values with terminal receipt SHA-256. | FULL512, BALANCED64, prefix rows, or partial class coverage |
| IQ4 HOLDOUT512 | IQ4 HOLDOUT512 scorer | Score the immutable IQ4 comparator on the same `HOLDOUT512_V1` manifest and instrument; publish global plus the same six classes with terminal receipt SHA-256. | The sealed IQ4 DEV BALANCED64 row, FULL512, prefix rows, or partial class coverage |

Canonical open count: **2**.

## Naming-hygiene closure

The serving-stack naming hygiene is tracked separately from measurement gaps. Final fleet migration state, launcher updates, public-repository commit, and NOTICE hash are sealed in `HYGIENE_MOET_FOSSIL_COMPLETE.json`; this does not change the canonical measurement-gap count above.

## Non-canonical update-kernel performance gap

The P1436 grouped backward-VJP path is default-on and physically verified, but its engineering target remains open. On the same fresh 43-layer x 1,024-token `smash update` workload, grouped reduction collapsed 1,376 legacy reduction launches to 86 while improving backward wall from 35.233350853 s to 32.932033144 s (1.06988x), short of the requested 10–15 s range. Correctness passed the sealed result-SHA class at tolerance with bit-equal loss and winner indices. See `banana-smasher/benchmarks/P1436_GROUPED_VJP_AB.json`. This engineering note does not change the canonical HOLDOUT512 open count.
