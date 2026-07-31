# VQ Warp GEMV kernel card

Status: real layer-042 correctness/perf passed; requires independent review and served RAW-AR attribution before promotion.

Target: DGX Spark GB10 (`sm_121`), BF16 activation/output, grouped d4/d8 8–12-bit row packs with K32 exponent scales.

Upstream surface: a standalone PyTorch custom op with `build.toml`; the CUDA source is runtime-independent and can be lifted into a Hugging Face Kernels Hub package. Public publication is intentionally not performed without owner approval.

Correctness: compare against the existing `VLLM_MOE_VQ_FAST=1` grouped Triton path using `microbench_vq_warp.py`; report full arrays and max/mean absolute error. The measured scalar-FP32 warp reduction is not bit-identical to tensor-core Triton reduction: VQ13/VQ2 both have max-abs 0.015625, mean-abs 2.82e-6/1.88e-6, cosine 1.0 on real layer-042 planes.

Performance: real layer-042 event medians are VQ13 1.693→0.235 ms (7.21x) and VQ2 0.825→0.126 ms (6.54x), N=30 alternating. Event-time microbench is Gate-2 signal; end-to-end raw-AR served tokens/s with on-path sentinel and A/B/A env rollback is the actual success criterion.
