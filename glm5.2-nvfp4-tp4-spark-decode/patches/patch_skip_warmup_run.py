#!/usr/bin/env python3
# patch_skip_warmup_run.py
# GB10 thin-node post-load WEDGE fix #2 — skip the compile_or_warm_up_model dummy-forward.
#
# ROOT CAUSE (py-spy run 2026-06-21, the DENSE+SKIP_PROFILE breakthrough run):
#   patch_skip_profile_run.py successfully removed the profile_run() killer — for the
#   FIRST time in 24 runs ALL FOUR nodes (incl. all three 119.7 GiB THIN nodes thin nodes)
#   passed determine_available_memory() and logged
#     "VLLM_SKIP_PROFILE_RUN=1: skipping profile_run ... reserved 1.86 GiB KV"
#   then "Skipping FlashInfer autotune". The wedge MOVED to a SECOND, separate dummy
#   forward: GpuWorker.compile_or_warm_up_model() runs
#     self.model_runner._dummy_run(num_tokens=max_num_reqs, cudagraph_runtime_mode=NONE)
#     + _dummy_sampler_run(...)
#   py-spy localized every worker (incl. the fat node at 127% CPU 20 min in) inside:
#     compile_or_warm_up_model (gpu_worker.py:730)
#       -> _dummy_run (gpu_model_runner.py:5948)
#         -> deepseek_v2 forward -> fused_moe runner -> meta_utils meta_tensor
#   i.e. the FIRST forward through the 78-layer 744B MoE under @support_torch_compile's
#   eager meta-tensor trace + per-expert kernel JIT. It is num_tokens=1 (tiny), so it is
#   NOT an activation-peak OOM like profile_run was — it is a pathological FIRST-FORWARD
#   trace/JIT that runs 20+ min and pins CPU, banner-starving the memory-tight thin nodes
#   so :8000 never binds.
#
#   Under --enforce-eager the warmup's stated purpose ("JIT compile triton kernels" /
#   "warm up sampler and preallocate logits buffer ... to avoid fragmentation") is
#   non-essential: there are no CUDA graphs to capture, and the sampler logits buffer
#   allocates lazily on the first real request. Skipping it lets the engine proceed
#   straight to binding :8000; the (same, one-time) lazy init is paid by the first real
#   decode instead of a 20-min pre-bind warmup that wedges the cluster.
#
# WHAT THIS PATCH DOES (surgical, idempotent, default-ON):
#   In compile_or_warm_up_model(), converts the warmup dispatch
#       if self.use_v2_model_runner:        # V2 path
#           warmup_kernels(...)
#       elif get_pp_group().is_last_rank:   # V1 path (our path) -> the slow dummy_run
#           ...
#   into a guarded form by prepending a new leading branch:
#       if VLLM_SKIP_WARMUP_RUN == "1":
#           <log skip>
#       elif self.use_v2_model_runner:
#           warmup_kernels(...)
#       elif get_pp_group().is_last_rank:
#           ...
#   so the existing branch bodies/indentation are untouched and the dummy-forward is
#   simply never reached when the env is set. Default is "1" (skip) because on this
#   GB10 fleet under --enforce-eager the warmup is always dead weight. Set
#   VLLM_SKIP_WARMUP_RUN=0 to restore stock behavior.
#
# Run INSIDE the engine container (launcher mounts read-only and execs it). Idempotent.
import sys, glob, os, py_compile

P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
src = open(P).read()

MARK = "VLLM_SKIP_WARMUP_RUN gate"
if MARK in src:
    print("[SKIPWARMUP] already patched")
    sys.exit(0)

lines = src.split("\n")
out = []
patched = 0
i = 0
while i < len(lines):
    line = lines[i]
    # Anchor: the warmup-dispatch `if self.use_v2_model_runner:` whose NEXT non-empty
    # line is the unique "# V2: Run full execute_model" comment. (The other
    # `if self.use_v2_model_runner:` sites are not followed by that comment.)
    nxt = lines[i + 1] if i + 1 < len(lines) else ""
    if (line.strip() == "if self.use_v2_model_runner:"
            and "# V2: Run full execute_model" in nxt):
        indent = line[:len(line) - len(line.lstrip())]
        body = indent + "    "
        out.append(f"{indent}# {MARK}: under --enforce-eager the compile_or_warm_up")
        out.append(f"{indent}# dummy-forward (first-forward trace/JIT through the 744B MoE)")
        out.append(f"{indent}# is dead weight that runs 20+ min and banner-wedges thin GB10")
        out.append(f"{indent}# nodes. Sampler logits buffer allocates lazily on 1st request.")
        out.append(f"{indent}import os as _os_warm")
        out.append(f'{indent}if _os_warm.environ.get("VLLM_SKIP_WARMUP_RUN", "1") == "1":')
        out.append(f"{body}import logging as _lg_warm")
        out.append(f"{body}_lg_warm.getLogger(__name__).warning(")
        out.append(f'{body}    "VLLM_SKIP_WARMUP_RUN=1: skipping compile_or_warm_up "')
        out.append(f'{body}    "dummy-forward (eager + pinned KV) to avoid GB10 thin-node "')
        out.append(f'{body}    "first-forward trace/JIT wedge; lazy-init on first request")')
        out.append(f"{indent}elif self.use_v2_model_runner:")
        patched += 1
        i += 1
        continue
    out.append(line)
    i += 1

if patched == 0:
    print("[SKIPWARMUP] FAIL: warmup `if self.use_v2_model_runner:` anchor not found (vLLM skew?)")
    sys.exit(2)

open(P, "w").write("\n".join(out))

for p in glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/__pycache__/*.pyc"):
    try:
        os.remove(p)
    except Exception:
        pass

py_compile.compile(P, doraise=True)
c = open(P).read()
ok = (MARK in c) and ('VLLM_SKIP_WARMUP_RUN", "1") == "1"' in c) and ("elif self.use_v2_model_runner:" in c)
print(f"[SKIPWARMUP] applied x{patched}")
print("[SKIPWARMUP] PY_COMPILE OK")
print(f"VERIFY skip_warmup={ok}")
