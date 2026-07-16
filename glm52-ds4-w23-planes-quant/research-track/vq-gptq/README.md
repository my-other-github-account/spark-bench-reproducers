# VQ-GPTQ error-feedback pilot

## Question

Can GPTQ-style error feedback improve a fixed d8 nearest-neighbour codebook assignment without changing the codebook or representation budget?

## Result

No raw-reconstruction improvement was possible on the tested units: fixed-codebook nearest-neighbour assignment is already the Euclidean optimum for each vector, so sequential error feedback lost on all 1,536 checked units.

A three-layer, six-window KLD probe was mixed rather than uniformly negative:

| layer | baseline `KL_vs_fp8` | pilot | relative change |
|---:|---:|---:|---:|
| 3 | 0.09844515 | 0.08111869 | +17.599% |
| 23 | 0.12765789 | 0.11446288 | +10.336% |
| 41 | 0.13328429 | 0.13226370 | +0.766% |
| pooled | 0.35938733 | 0.32784528 | +8.777% |

The pooled signal is exploratory only: three layers and six windows do not satisfy the campaign's full-artifact or held-out replication gates. The raw pilot directory was no longer present on its source host at publication time, so `PILOT_SUMMARY.json` is reconstructed from the completed Kanban run metadata and is marked accordingly rather than pretending a raw-file seal exists.

## Decision

Do not continue fixed-codebook error feedback as a production ladder lane. If revisited, change the objective or codebook itself and require a full 43-layer held-out rail.
