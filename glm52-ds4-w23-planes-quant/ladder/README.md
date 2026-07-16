# Quantization ladder and backpack solves

This directory contains the sealed anchor rows, mixed-bin rows, solver inputs, solver implementations, and prediction-versus-measurement calibration record for the DS4 expert-plane campaign.

## Measurement contract

All headline KLD values are teacher-forced `KL(reference || candidate)` over 512 windows / 524,288 positions, reference top-8192 support, positions `[0,1024)`, and corpus MD5 `1701920b4ba96dea0b18fe9df0151876`. The native released checkpoint is the teacher and therefore has KLD 0.

## Headline sealed rows

| family | wire rate / size | KLD | top1 | status |
|---|---:|---:|---:|---|
| VQ d=4, k=8192 uniform | 3.50 bpw / 128.8 GB total | 0.057692 | 0.929 | sealed |
| VQ d=4, k=4096 uniform, corrected | 3.25 bpw / 120.1 GB total | 0.067160 | 0.924427 | sealed |
| W3v2 GPTQ uniform | 3.25 bpw / 120.1 GB total | 0.072742 | 0.920 | sealed |
| IQ3 bin, k4096-menu intermediate | 101.95 GB total | 0.09894975 | 0.906254 | sealed; prediction error -0.0412% |
| IQ3 bin canonical | 2.927081 bpw / 101.95 GB total | 0.10052475 | 0.906021 | sealed |
| Q2 bin canonical | 2.734836 bpw / 95.75 GB total | 0.13135650 | 0.891554 | sealed |
| VQ d=8, k=256 uniform | 1.25 bpw | 1.757602 | 0.539780 | sealed cold-tail rung |
| VQ d=8, k=1024 uniform | 1.50 bpw | 1.030811 | 0.661455 | sealed cold-tail rung |

The raw rows and manifests are copied into `anchors/`, `bins/`, and `manifests/`. `solvers/` preserves each widening step rather than only the final solver so the menu evolution is reproducible.

## Prediction calibration

The k4096-menu IQ3 solve predicted 0.0989905456 and measured 0.09894975, an absolute residual of -0.0000407956 (-0.0412%). The three-rung d=4 menu predicted 0.0944570719 at the same 101.95 GB cap. Adding k1024 expanded the choice set and reduced the prediction monotonically to 0.0937067982. Q2 predictions similarly improved from 0.1274883712 to 0.1260708633. Intermediate predictions are retained even when a later menu superseded their rail.
