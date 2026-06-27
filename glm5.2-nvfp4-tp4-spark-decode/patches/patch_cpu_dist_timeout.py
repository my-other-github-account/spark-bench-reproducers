#!/usr/bin/env python3
# patch_cpu_dist_timeout.py
# GB10 TP=4 load-SKEW WEDGE fix — raise the gloo/CPU distributed barrier timeout.
#
# ROOT CAUSE (RUN #55 evidence, 2026-06-23 00:39:46Z — the genuinely-new finding that
# overturns the RUN #51-54 "b12x first-forward speed" narrative):
#   The b12x serve NEVER reached a forward. It died at EXECUTOR INIT. the fat node/rank3 (LOCAL
#   model, fastest) finished its weight load at 00:09:46, then entered the MQ-broadcaster
#   gloo barrier:
#       MultiprocExecutor._init_message_queues
#         -> get_inner_dp_world_group().create_mq_broadcaster
#           -> MessageQueue.create_from_process_group
#             -> in_the_same_node_as(pg) -> torch.distributed.barrier(group=pg)  [GLOO]
#   and died at EXACTLY 00:39:46 = start + 30:00:
#       RuntimeError: Application timeout caused pair closure
#   30:00 == 1800s == the gloo ProcessGroup DEFAULT operation timeout. The fast nodes
#   (the fat node (121.7 GiB) LOCAL, head rank0) abandoned the barrier while the THIN nodes (thin nodes, 119.7 GiB,
#   NFS-loading, banner-wedged but ping-ALIVE) were still finishing their much-slower load.
#   Head then reported only the generic downstream `Engine core initialization failed`.
#
#   vLLM exposes ParallelConfig.cpu_distributed_timeout_seconds, but it DEFAULTS TO None
#   (verified in-container: `ParallelConfig().cpu_distributed_timeout_seconds == None`) and
#   there is NO CLI flag to set it (`vllm serve --help` has no cpu-distributed-timeout). When
#   None, get_cpu_distributed_timeout_or_none() returns None -> gloo uses its 1800s default
#   -> the 30-min death. Raising VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS / TORCH_NCCL_HEARTBEAT_*
#   to 14400 (the longjit launcher already does) does NOT touch this gloo CPU-group timeout.
#
# WHAT THIS PATCH DOES (surgical, idempotent, default-ON):
#   Rewrites get_cpu_distributed_timeout_or_none() in vllm/distributed/utils.py so that when
#   the config value is None it returns a concrete timedelta (default 14400s = 4h, overridable
#   via env VLLM_CPU_DIST_TIMEOUT_SEC) instead of None. Every gloo/CPU process-group barrier
#   that reads this (including the MQ-broadcaster's in_the_same_node_as barrier) then tolerates
#   the LOCAL-vs-NFS load skew between the fat and thin nodes instead of pair-closing at 30 min.
#   Set VLLM_CPU_DIST_TIMEOUT_SEC=0 to restore stock None/default behavior.
#
# Run INSIDE the engine container (launcher mounts read-only and execs it). Idempotent.
import sys, glob, os, py_compile

P = "/usr/local/lib/python3.12/dist-packages/vllm/distributed/utils.py"
src = open(P).read()

MARK = "VLLM_CPU_DIST_TIMEOUT patch"
if MARK in src:
    print("[CPUDISTTIMEOUT] already patched")
    sys.exit(0)

OLD = (
    "    vllm_config = get_current_vllm_config_or_none()\n"
    "    if vllm_config is None:\n"
    "        return None\n"
    "    timeout_seconds = vllm_config.parallel_config.cpu_distributed_timeout_seconds\n"
    "    return timedelta(seconds=timeout_seconds) if timeout_seconds is not None else None"
)

NEW = (
    "    # " + MARK + ": on GB10 TP=4 the LOCAL (the fat node (121.7 GiB)) vs NFS (thin nodes) weight-load\n"
    "    # skew exceeds gloo's 1800s default barrier timeout, pair-closing the MQ-broadcaster\n"
    "    # in_the_same_node_as barrier at exactly start+30:00. cpu_distributed_timeout_seconds\n"
    "    # defaults to None (no CLI flag exists) -> gloo default. Force a long concrete timeout\n"
    "    # (env VLLM_CPU_DIST_TIMEOUT_SEC, default 14400s) so the slow thin-node load can finish.\n"
    "    import os as _os_cdt\n"
    "    _cdt_default = int(_os_cdt.environ.get(\"VLLM_CPU_DIST_TIMEOUT_SEC\", \"14400\"))\n"
    "    vllm_config = get_current_vllm_config_or_none()\n"
    "    timeout_seconds = None\n"
    "    if vllm_config is not None:\n"
    "        timeout_seconds = vllm_config.parallel_config.cpu_distributed_timeout_seconds\n"
    "    if timeout_seconds is None:\n"
    "        timeout_seconds = _cdt_default\n"
    "    if timeout_seconds == 0:\n"
    "        return None\n"
    "    return timedelta(seconds=timeout_seconds)"
)

if OLD not in src:
    print("[CPUDISTTIMEOUT] FAIL: get_cpu_distributed_timeout_or_none body anchor not found (vLLM skew?)")
    sys.exit(2)

src = src.replace(OLD, NEW, 1)
open(P, "w").write(src)

for p in glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/distributed/__pycache__/*.pyc"):
    try:
        os.remove(p)
    except Exception:
        pass

py_compile.compile(P, doraise=True)
c = open(P).read()
ok = (MARK in c) and ("VLLM_CPU_DIST_TIMEOUT_SEC" in c) and ("return timedelta(seconds=timeout_seconds)" in c)
print("[CPUDISTTIMEOUT] applied")
print("[CPUDISTTIMEOUT] PY_COMPILE OK")
print(f"VERIFY cpu_dist_timeout={ok}")
