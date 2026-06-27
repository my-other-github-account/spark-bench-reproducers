import py_compile, sys
# patch_skip_fp4_repack.py — GLM-5.2-NVFP4 dense campaign, RUN #26 deeper-source lever.
# Skips the TRAILING FP4 MoE repack call in ModelOptNvFp4FusedMoE.process_weights_after_loading
# (modelopt.py ~line 1607: self.moe_kernel.fused_experts.process_weights_after_loading(layer)).
# This is the documented ~2 GiB post-load NVFP4 finalize spike that NVRM-OOMs the 119.7 GiB thin
# nodes (thin nodes) on BOTH dense and sparse paths — it is the MoE finalize, not the attention indexer,
# which is why every profile/warmup/KV/util lever in Run #25 left it untouched and all wedged.
# The NVFP4 conversion (convert_to_nvfp4_moe_kernel_format) + make_nvfp4_moe_kernel ABOVE this line
# stay intact, so weights are still in kernel format; only the redundant deeper repack is skipped.
# Anchor matched byte-for-byte in this 0.23.1rc1.dev207 image (verified count==1 @ 1607). Idempotent.
P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/modelopt.py"
MARK = "SHORT_PROOF_FP4_FUSED_MOE_SKIP_REPACK"
s = open(P).read()
if MARK in s:
    print("[NOREPACK] already applied"); sys.exit(0)
lines = s.split("\n")
hits = [i for i, l in enumerate(lines)
        if l.strip() == "self.moe_kernel.fused_experts.process_weights_after_loading(layer)"]
if len(hits) != 1:
    print(f"[NOREPACK] anchor count={len(hits)} (expected 1) — ABORT"); sys.exit(2)
i = hits[0]
indent = lines[i][:len(lines[i]) - len(lines[i].lstrip())]
lines[i] = (f"{indent}# {MARK}: skip trailing FP4 MoE repack — OOMs finalize on GB10 unified mem (RUN #26)\n"
            f"{indent}pass  # was: self.moe_kernel.fused_experts.process_weights_after_loading(layer)")
open(P, "w").write("\n".join(lines))
py_compile.compile(P, doraise=True)
print(f"[NOREPACK] applied at line {i+1} + py_compile OK")
