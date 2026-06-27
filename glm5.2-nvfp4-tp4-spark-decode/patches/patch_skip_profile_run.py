#!/usr/bin/env python3
# patch_skip_profile_run.py
# GB10 thin-node post-load OOM fix — skip the purposeless profile_run dummy-forward.
#
# ROOT CAUSE (py-spy run#19 + the dense-path RUN #24 isolation):
#   vLLM's GpuWorker.determine_available_memory() ALWAYS runs
#   self.model_runner.profile_run() — even when KV is pinned via
#   --kv-cache-memory-bytes (the "short" branch). profile_run() executes a dummy
#   forward of `max_num_tokens == max_num_batched_tokens` tokens through the full
#   744B MoE, then a _dummy_sampler_run with a tensor_model_parallel_all_gather.
#   That forward's transient ACTIVATION peak (sized for the profiling batch, NOT
#   for real 1-token decode) is the ~2 GiB that NVRM-OOMs the 119.7 GiB worker
#   nodes (thin nodes) while the 121.7 GiB fat node survives — the exact 2.0 GiB
#   MemTotal asymmetry. The dense path (index_topk=0, NO sparse indexer) wedged
#   the SAME thin nodes, proving the wedge is profile_run, not the indexer.
#
#   The short branch's own comment says profile_run is only "still need[ed] ... to
#   compile the model for max_num_batched_tokens". Under --enforce-eager there is
#   NOTHING to compile, so the forward is pure dead weight whose only effect is the
#   OOM. Real serving (1-token decode, max-num-seqs 1) needs far less memory than
#   the profiling forward demands, so skipping it lets the thin nodes serve.
#
# WHAT THIS PATCH DOES (surgical, opt-in, idempotent):
#   In determine_available_memory()'s `if kv_cache_memory_bytes :=` branch, gate the
#   profile_run() call behind env VLLM_SKIP_PROFILE_RUN=1. When set, skip the dummy
#   forward and proceed straight to `return kv_cache_memory_bytes` (the branch already
#   skips memory profiling by design). Comm buffers, if any, lazily allocate on the
#   first real forward (small; not the activation peak). Correctness is unaffected —
#   profile_run is a measurement/warmup step, it does not touch weights or outputs.
#
# REQUIRES the launcher to pass BOTH:
#   -e VLLM_SKIP_PROFILE_RUN=1   AND   --kv-cache-memory-bytes <N>  (routes to short branch)
#   plus --enforce-eager (so there is genuinely nothing to compile).
#
# Run INSIDE the engine container (launcher mounts read-only and execs it). Idempotent.
import sys, glob, os, py_compile

P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py"
src = open(P).read()

MARK = "VLLM_SKIP_PROFILE_RUN gate"
if MARK in src:
    print("[SKIPPROFILE] already patched")
    sys.exit(0)

lines = src.split("\n")
out = []
i = 0
patched = 0
seen_branch = False
while i < len(lines):
    line = lines[i]
    # Locate the pinned-KV short branch header.
    if "if kv_cache_memory_bytes :=" in line and "self.cache_config.kv_cache_memory_bytes" in line:
        seen_branch = True
        out.append(line)
        i += 1
        continue
    # First profile_run() call AFTER the short-branch header → gate it.
    if seen_branch and patched == 0 and line.strip() == "self.model_runner.profile_run()":
        indent = line[:len(line) - len(line.lstrip())]
        body = indent + "    "
        out.append(f"{indent}# {MARK}: under --enforce-eager there is nothing to compile,")
        out.append(f"{indent}# so the profiling dummy-forward (max_num_batched_tokens-sized,")
        out.append(f"{indent}# all_gather) is pure dead weight that NVRM-OOMs thin GB10 nodes.")
        out.append(f"{indent}import os as _os")
        out.append(f'{indent}if _os.environ.get("VLLM_SKIP_PROFILE_RUN", "0") == "1":')
        out.append(f"{body}import logging as _lg")
        out.append(f"{body}_lg.getLogger(__name__).warning(")
        out.append(f'{body}    "VLLM_SKIP_PROFILE_RUN=1: skipping profile_run dummy-forward "')
        out.append(f'{body}    "(eager + pinned KV) to avoid GB10 thin-node activation-peak OOM")')
        out.append(f"{indent}else:")
        out.append(f"{body}self.model_runner.profile_run()")
        patched += 1
        i += 1
        continue
    out.append(line)
    i += 1

if patched == 0:
    print("[SKIPPROFILE] FAIL: pinned-KV profile_run() call not found (vLLM version skew?)")
    sys.exit(2)

open(P, "w").write("\n".join(out))

# drop stale bytecode
for p in glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/__pycache__/*.pyc"):
    try:
        os.remove(p)
    except Exception:
        pass

py_compile.compile(P, doraise=True)
c = open(P).read()
ok = (MARK in c) and ('VLLM_SKIP_PROFILE_RUN", "0") == "1"' in c)
print(f"[SKIPPROFILE] applied x{patched}")
print("[SKIPPROFILE] PY_COMPILE OK")
print(f"VERIFY skip_profile={ok}")
