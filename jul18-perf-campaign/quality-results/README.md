# July 18 quality rows

## Results

| Row | Score | Footprint | Provenance |
|---|---:|---:|---|
| R5 rerepair on the prerepaired three-tier pack, full-512 | **0.07506409375** | **193.063787137 GB** receipt-reported | canonical row SHA-256 `9baf4ad4c0d9030c20cccad3768c977ee56a62ef18f15974fbcf5b0a3b0620fb` |
| zero-train GPTQ overlay, gate64 | **0.09078628125** | 109,891,931,697-byte overlay receipt | gate-stage SHA-256 `2722a773a61c29e806942429b170c1fe263b22a618374c9edef5c5f6dbeb801b` |

Lower is better for this quality instrument. These rows must not be compared with throughput, MMLU, or differently normalized KLD instruments as if they were the same metric.

R5 is the measured new best within this repair lineage: it improves 18.1623% over its prerepaired full-512 base (0.091723068359375), 1.60167% over the straight ComboV2 repair row (0.076285939453125), and 19.0247% over the 0.0927 IQ4_XS bar. It scored top1 agreement 0.916774736328125 over 512 windows / 524,288 matched positions.

The GPTQ result is **not** a full-512 claim. It improves the exact corrected gate64 baseline 0.09323921875 by 2.6308% and was classified in the preregistered maintainer-decision band (>0.0885 and ≤0.092). The full-512 continuation remained separately gated.

## Methodology

- Gate64 and full-512 evaluations use paired windows and the same teacher/candidate scorer settings.
- The calibration study reports paired gate64/full-512 Spearman rank correlation **ρ=0.978**; campaign certainty planning put the equivalent unpaired budget at roughly **7,000 windows**.
- An independently measured unpaired-to-paired inflation factor is **1.0365**. It is a diagnostic for unpaired rows, not a correction to apply to already paired full-512 scores.
- Every public row needs a payload hash, scorer identity, window identity, and explicit paired/unpaired label.
- The shared evaluation corpus MD5 for these campaign rails is `1701920b4ba96dea0b18fe9df0151876`.

## Fixed-compose override coverage

A pack with `codebook_overrides` is valid only when every target row for every overridden tier is sourced from the same delta lineage—even rows whose assignment bytes happen to be numerically unchanged. The validator must emit `missing_target_rows=0`, bind target/base/delta manifests and codebook identity, and fail closed before gate64 otherwise. This rule excludes the earlier mixed-lineage 0.2458 corruption class.

See `../quality-predictor/` for the reusable calibration script and publication rules.

## Provenance gaps

The original sealed R5 artifact is represented here by its published digest and receipt summary; the large model payload is intentionally not vendored. The GPTQ row is similarly hash-bound. Reproduction requires the private/large model inputs named by the campaign, but no private path, worker name, or address is published.
