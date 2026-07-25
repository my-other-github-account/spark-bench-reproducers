# Censoring, not a 16K attractor

## Discovery

HumanEval/116 and /132 repeatedly appeared as 4096-token `finish=length` rows with no visible answer. A capped observation cannot distinguish:

1. a true non-terminating reasoning attractor; from
2. a finite completion whose natural stop lies beyond the evaluation ceiling.

We kept frozen correctness at 4096 and introduced a separate diagnostic with a 16,384-token completion allowance. It is labeled **DIAGNOSTIC_UNCAPPED** everywhere and is not a replacement correctness claim.

## Full uncapped table

| checkpoint | role | task | reasoning | completion | finish | base | plus | ratio vs FP teacher |
|---|---|---:|---:|---:|---|---|---|---:|
| BQ3 step0 | real-dose baseline | 116 | 6,553 | 6,891 | stop | fail | fail | 4.42770x |
| BQ3 step0 | real-dose baseline | 132 | 5,753 | 5,936 | stop | pass | fail | 1.92989x |
| PTQ-OPD step4 | real dose | 116 | 3,745 | 4,032 | stop | pass | pass | 2.53041x |
| PTQ-OPD step4 | real dose | 132 | 12,688 | 12,878 | stop | pass | fail | 4.25629x |
| exploratory Track-C step8 | campaign-noncreditable | 116 | 5,980 | 6,256 | stop | pass | pass | 4.04054x |
| exploratory Track-C step8 | campaign-noncreditable | 132 | 3,350 | 3,563 | stop | pass | fail | 1.12378x |
| migrated step4 control | duplicate/control | 116 | 4,034 | 4,372 | stop | fail | fail | 2.72568x |
| migrated step4 control | duplicate/control | 132 | 6,856 | 7,176 | stop | pass | fail | 2.29990x |

All eight generations naturally stopped below 16K. A true >16K/non-terminating attractor is therefore refuted for these observations; the 4096 instrument was censoring finite completions.

Receipt identities:

- table JSON: `111e14c6adf6ac4e1327cf0c8a8473805c9030a5c26c174a0a41d355def1f4ba`
- human table: `d233fa480f10e343ee9c831722b4264fb44ebe8b5747a30471cb737a3da60dc5`
- 14-file manifest: `8e897741be5130e801523c5fa3fec56ec48a6930d286056078e46c3b44ac794b`
- FP-reasoning receipt: `b108d1088a682806e9d6149d0c4f8b0cdc667fd86772f7912572ff0ebb1d9705`

## The convergence curve is not monotone

The real-dose reasoning trajectories were:

```text
HumanEval/116: 6,553 -> 3,745 -> 5,980
HumanEval/132: 5,753 -> 12,688 -> 3,350
```

Censoring is confirmed, but generic monotone convergence and slope extrapolation are not. In particular, /132 becomes much longer at step4 before dropping below the 4096 ceiling at the exploratory step8 checkpoint.

The migrated step4 duplicate/control also disagreed sharply with canonical step4 under the same reported fingerprint. It is excluded from the dose curve and retained as a reproducibility alarm.

## Teacher reasoning counts

Provider-owned hidden-reasoning counts were recovered for four requested tasks:

| task | FP teacher reasoning tokens | finish |
|---:|---:|---|
| 99 | 472 | stop |
| 116 | 1,480 | stop |
| 132 | 2,981 | stop |
| 134 | 834 | stop |

The original persisted corpora did not retain equivalent hidden usage for /2, /57, or /93. Those values are reported as unavailable rather than reconstructed from visible text.

## Variance finding

Back-to-back temperature-zero generations on the same serve varied substantially. In the first transfer panel, designated replicates differed by:

| task | replicate 1 | replicate 2 | absolute delta |
|---:|---:|---:|---:|
| 116 | 4,492 | 3,664 | 18.43% |
| 132 | 5,476 | 3,979 | 27.34% |
| 99 | 1,072 | 737 | 31.25% |

Across broader diagnostic repetitions, same-serve changes were roughly ±18-31%. Therefore a single greedy length is not a stable estimator even at temperature zero.

## Pre-registered panel law

A directional decrease claim requires all of:

1. a fixed 12-prompt panel selected before generation;
2. negative median per-prompt reasoning change;
3. at least 8 of 12 prompts decreasing;
4. absolute median magnitude greater than the conservative variance floor;
5. the variance floor defined as the maximum absolute percent difference among designated same-serve replicate pairs;
6. exact serving identity and request shape;
7. a separate correctness/non-null report.

The first transfer update met conditions 2 and 3 but failed condition 4 (`17.81% < 31.25%`), so its result is explicitly inconclusive-directional.

## Interpretation boundary

Uncapped rows answer: “was the 4096 null a censored finite completion?”

They do not answer: “would the frozen product evaluation pass?” Frozen correctness remains 4096, and any answer appearing only after that ceiling remains a frozen-eval failure.
