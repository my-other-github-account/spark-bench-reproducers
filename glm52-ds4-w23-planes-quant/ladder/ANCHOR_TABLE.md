# Sealed ladder and solve table

All rows are 512-window measurements unless explicitly marked `PRED`. Corpus pin: `1701920b4ba96dea0b18fe9df0151876`. The d4 and d8 sections use different reference distributions and must not be numerically compared.

## d4 uniform anchors (`KL_vs_teacher`)

| row | K | KLD | top-1 | whole GB | expert bpw | status |
|---|---:|---:|---:|---:|---:|---|
| K8192 | 8192 | 0.057692 | 0.929312 | 128.8 | 3.50 | MEASURED |
| K4096 corrected | 4096 | 0.067160 | 0.924427 | 120.1 | 3.25 | MEASURED |
| K2048 | 2048 | 0.098564 | 0.908140 | 111.4 | 3.00 | MEASURED |
| K1024 | 1024 | 0.147352 | 0.886419 | 102.78 | 2.75 | MEASURED |
| K512 | 512 | 0.235656 | 0.852175 | 94.12 | 2.50 | MEASURED |

The checked-in JSON rows and sidecar hashes in `anchors/` are authoritative when displayed precision differs.

## d8 sub-ternary anchors (`KL_vs_fp8`)

| K | KLD | top-1 | JS | whole GB | whole bpw | status |
|---:|---:|---:|---:|---:|---:|---|
| 256 | 1.757602 | 0.539780 | 0.280130 | 50.8356 | 1.397992 | MEASURED |
| 512 | 1.345675 | 0.603914 | 0.220681 | 55.1644 | 1.517038 | MEASURED |
| 1024 | 1.030811 | 0.661455 | 0.169766 | 59.4937 | 1.636093 | MEASURED |
| 2048 | 0.817378 | 0.704857 | 0.135224 | 63.8236 | 1.755167 | MEASURED |
| 4096 | 0.664968 | 0.739141 | 0.110607 | 68.1549 | 1.874280 | MEASURED |

## Measured two-bin rows

| budget | menu | KLD | top-1 | total GB | status |
|---|---|---:|---:|---:|---|
| Q2-BIN | K8192-era | 0.13135650 | 0.8916 | 95.75 | MEASURED |
| IQ3-BIN | K8192-era | 0.10052475 | 0.9060 | 101.95 | MEASURED |
| IQ3-BIN | K4096 menu | 0.09894975 | — | 101.95 | MEASURED; best measured campaign row |

## Full-menu predictions

| menu extension | IQ3-BIN PRED | Q2-BIN PRED |
|---|---:|---:|
| K2048 triple | 0.09445707 | 0.12748837 |
| K1024 quad | 0.09370680 | 0.12607086 |
| d8 penta | 0.09357928 | 0.12579005 |
| K512 hexa | 0.09358039 | 0.12579143 |

These are solver predictions, not rails. Manifests and exact allocation counts live in `manifests/` and `solve-results/`.
