# Q2 95.75 GB repair-training progress

## Measured progress

At checkpoint step 45, the held-out panel moved from **0.0712644** before repair to **0.0545097**, a **23.5106% relative improvement**. Training was observed healthy through step 47.

This is a training-progress receipt only. The full-512 rail was pending, so the package does **not** claim a final quality gate or shipping verdict.

## Integrity contract

Before promoting a Q2 checkpoint:

1. verify the exact 95.75 GB pack, target manifest, base manifest, loader, and checkpoint digests;
2. fail closed if any codebook-override target row is absent, even if the row is numerically unchanged;
3. require the fixed override-coverage validator to pass over all layers and both projections;
4. bind the selected checkpoint to a measured gate64 row;
5. run the full paired-512 rail before claiming quality;
6. retain the prerepair panel, step ledger, and selected-checkpoint hash.

For any tier with a codebook override, the coverage receipt must additionally report `missing_target_rows=0`; unchanged target rows still come from the override-consistent delta. A seal that only proves the changed rows is insufficient.

The summary JSON captures only public-safe numerical progress and status.
