# R5 rail acceleration — design + bit-exactness argument (<source-task>)

## Baseline (sealed R5 rail, s3, <source-task>)
Per 64-window chunk (43 layers, mb=2), from full512_build.log (86 layer records):
  load = 465 s/chunk (47%)  -> scalar per-expert materialization
  fwd  = 527 s/chunk (53%)  -> prefill forward
  chunk wall 16.7-16.9 min; accounting closes within 2%.

## Fix R1: bulk fill_layer (port of <worker-B> fullmenu_delta_source.fill_layer)
The sealed harness t8192_ds4_build_v3.fill_plane_experts() already prefers a
`fill_layer` bulk hook when the source provides one (line ~373; instrument
sha 904d28d4... on both s3 and s6). The sealed R5 source lacks it, so the rail
falls back to 256 experts x 2 projections = 512 scalar closure calls per layer,
each doing its own H2D copies, its own codebook re-upload + fp16-wire cast, and
its own dequant kernel chain.

fill_layer batches rows per (source, tier, projection) group:
  - ONE codebook upload + fp16-wire cast per (tier, projection) per layer
    (R5 codebooks come from the sealed checkpoint; identical values to the
    per-expert path because the cast chain .to(DEV).to(fp16).float() is
    value-deterministic).
  - batched row gather from the mmap-backed pack, one .to(DEV) per batch.
  - vectorized dequant: codebook[codes.long()].reshape(B,R,-1) *
    exp2(scales.float()-127).repeat_interleave(32,-1)

## Bit-exactness argument (R1)
Every op in the dequant chain is ELEMENTWISE or GATHER — no reductions:
  gather (codebook[codes.long()]), reshape, exp2, sub, repeat_interleave,
  mul, dtype casts (fp16-wire -> float -> bf16).
Elementwise/gather results are independent of batch shape on both CPU and CUDA;
each output element is computed from the same scalar inputs with the same op
order as in the scalar path. Destination rows are disjoint (destination[ids] =
weights vs per-id assignment), so write order is irrelevant. Therefore the bulk
path is bitwise-identical BY CONSTRUCTION; the 8-window golden gate then
verifies this empirically end-to-end (md5 of q8192_win*.pt vs sealed
candidate/DONE.jsonl from <source-task>).

fp4 static tier: v3.deq_fp4_block32 already accepts batched rows (<worker-B>
precedent) and is likewise reduction-free.

## Fix R2: next-layer page-cache warmer
Scalar load time includes cold mmap page-in of ~3-4 GiB/layer of pack
components. A background thread sequentially reads layer L+1's component files
into the OS page cache while layer L is forwarding. Zero numerical effect (pure
readahead); torch.load mmap then hits warm pages.

## What is intentionally NOT changed
- mb stays 2 in validation arms (matches sealed golden batching pairs).
- attention stays eager; no torch.compile, no flash prefill (reduction-order
  risk). mb>2 is a separate bitwise EXPERIMENT arm, adopted only if the
  8-window md5 gate passes.
- Scoring (kld_score.py) untouched.

## Expected effect
load 465s -> ~40-90s (R1), largely hidden behind fwd (R2).
Chunk: ~1008s -> ~540-600s single-host (~1.7-1.9x).
mb=4/8 arm (if bitwise): fwd 527 -> ~300-400s => ~2.5-3.3x single-host.
Two-host window split (R4, default policy change): additional ~2x on full512
wall; R1+R2+R4 >= 3.6x with zero numerical risk.
