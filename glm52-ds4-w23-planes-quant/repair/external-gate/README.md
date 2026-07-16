# Disjoint 24-window external gate

The post-training external gate used 24 windows excluded from the codebook training pool, the eight binding probes, and the 64-window wide-data pool:

```text
9,32,53,74,96,117,138,159,181,202,224,245,
265,287,309,330,352,372,394,416,437,458,480,501
```

Campaign result: the exported arm4 artifact improved held-out KLD by approximately `+5.1%` on this panel. This is the evidence that the eight binding probes were not the entire effect.

Evidence boundary at this repository refresh:

- the window set and corpus identity are fixed;
- the result is a measured external-gate result, not a solver prediction;
- the independent baseline/arm4 remeasurement was active at the final refresh cut: baseline had
  18/24 durable rows and the arm4 half had not started, so no partial comparison is published;
- the independent run must write both per-window ledgers and a hash-bound aggregate before it replaces this campaign record;
- no served-quality claim follows from this gate until checkpoint → exported planes → offline harness → served logits all match.

To repeat it, use the export and sealed-streamer environment block in [`RESUME.md`](../../RESUME.md). Do not run this 24-window panel through the slow training harness unless debugging the harness itself.
