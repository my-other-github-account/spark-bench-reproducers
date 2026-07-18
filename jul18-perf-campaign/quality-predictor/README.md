# Gate64 → full-512 quality predictor

This package calibrates a cheap 64-window quality gate against the full 512-window rail without mixing paired and unpaired instruments.

## Campaign receipt

- paired gate64/full-512 Spearman rank correlation: **ρ=0.978**;
- campaign conclusion: an equivalent unpaired certainty budget would require approximately **7,000 windows**;
- median unpaired inflation factor: **1.0365**.

## Input format

CSV with numeric `gate64` and `full512` columns. Extra columns are ignored.

```csv
gate64,full512
0.0841,0.0837
0.0902,0.0898
```

## Fit and predict

```bash
python calibrate_gate64.py fit paired.csv --output fit.json
python calibrate_gate64.py predict fit.json 0.0860
```

The fit reports:

- OLS intercept and slope for `full512 ~ gate64`;
- Pearson correlation;
- Spearman rank correlation with average ranks for ties;
- median `full512/gate64` ratio;
- row count and min/max domain.

## Publication rules

1. Use rows with identical window pairing, teacher, scorer, token cutoff, and candidate bytes.
2. Never train on an unpaired score without labeling it separately.
3. Do not multiply a paired prediction by 1.0365. The factor is only a measured diagnostic for a known unpaired instrument.
4. Reject extrapolation outside the fitted gate64 range unless explicitly labeled.
5. The predictor ranks candidates; a promoted candidate still needs a measured full-512 row.

## Why pairing changes the sample budget

For matched candidate/baseline windows, the variance of a difference is
`var(A-B) = var(A) + var(B) - 2*cov(A,B)`. With similarly scaled arms and campaign correlation ρ=0.978, pairing removes most shared window difficulty. An unpaired design loses the covariance cancellation and therefore needs orders of magnitude more windows for the same decision certainty; the campaign planning calculation was approximately 7,000. Preserve the original paired window IDs rather than trying to buy back certainty with a much larger unpaired run.

The campaign-level ρ, approximately-7,000 unpaired certainty estimate, and inflation receipt are not reconstructed from synthetic data in this repository. The script is the rerunnable estimator; supply the original paired CSV to reproduce the campaign coefficients exactly.
