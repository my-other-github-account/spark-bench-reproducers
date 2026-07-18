# Compression-loop acceleration

## Profiles

### Rail / full-menu materialization

A 64-window chunk spent roughly **465 s loading (47%)** and **527 s forwarding (53%)**, with total wall **16.7–16.9 min**. The sealed path fell back to 256 experts × two projections of scalar Python materialization, repeatedly uploading the same layer-shared codebooks.

These numbers come from 86 per-layer records in the exact production-loop log. That accounting closed within 2% of chunk wall and was preferred over a short sampling profile for the rail.

### Repair trainer

A 90-second sampling profile measured:

- 36.5% self time in autograd backward;
- 35.0% in repeated frozen codes/scales mmap row-gather plus H2D;
- 341–383 s per training step;
- 84–104 s probe panel every five steps (~5% amortized).

## Published optimizations

1. **Bulk `fill_layer`** (`src/r5_fullmenu_source.py`): group rows by source/tier/projection, upload each codebook once, batch row gathers, and vectorize the reduction-free dequant chain.
2. **Next-layer page-cache readahead:** a background thread sequentially warms the next layer while the current layer forwards. It changes no numerical operation.
3. **Frozen codes/scales GPU cache** (`src/fullmenu_surface_t1.py`): optional whole-layer LRU cache keyed by layer/projection/source/tier/rows. It skips repeated mmap+H2D while leaving the trainable codebook autograd path unchanged.

## Validation status

- CPU scalar-vs-bulk analog: **12/12 PASS**, including sub-batch invariance.
- Trainer cache unit suite: **10/10 PASS**, including disabled behavior, same-object hits, byte accounting, and whole-layer LRU eviction.
- Production status: **staged, not adopted**. An end-to-end eight-window GPU bit-identical rail gate and a fixed-seed one-step trainer loss-identity gate are still mandatory.

## Tests

```bash
python tests/r1_cpu_bitwise_test.py
python tests/t1_unit_test.py
```

The tests need PyTorch. The rail source additionally requires the campaign's model/pack harness and large inputs.

## Expected, not measured

The design estimated the single-host chunk could fall from ~1008 s toward 540–600 s, with a two-host window split providing further wall reduction. Those are projections, not published benchmark results.
