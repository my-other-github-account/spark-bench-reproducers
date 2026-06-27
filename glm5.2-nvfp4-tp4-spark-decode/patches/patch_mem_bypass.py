#!/usr/bin/env python3
# patch_mem_bypass.py
# GB10 unified-memory startup free-memory check bypass.
#
# vLLM's request_memory() (vllm/v1/worker/utils.py) raises ValueError when the
# measured free memory at startup is less than gpu_memory_utilization * total.
# On GB10 the ~104.67 GiB resident NVFP4 weights mean free < 0.99*total at the
# moment of the check, so vanilla vLLM refuses to start even though the run is
# actually memory-safe (KV is pinned via --num-gpu-blocks-override 128 and the
# footprint trio caps activations). The proven dev156 image (glm5-repro:t2) bakes
# this bypass; a fresh from-source eugr build does NOT, so apply it here.
#
# Idempotent. Replaces the `raise ValueError(...)` inside `request_memory` with a
# logging.warning, leaving the rest of the function (return requested_memory) intact.
#
# Run INSIDE the engine container (the launcher mounts this read-only and execs it).
import sys, re, glob, os, py_compile
P = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/utils.py"
src = open(P).read()

MARK = "GB10 unified mem bypass"
if MARK in src:
    print("[MEMBYPASS] already patched")
    sys.exit(0)

lines = src.split("\n")
out = []
i = 0
patched = 0
while i < len(lines):
    line = lines[i]
    # find the guard line inside request_memory
    if line.strip() == "if init_snapshot.free_memory < requested_memory:" and \
       i + 1 < len(lines) and lines[i + 1].lstrip().startswith("raise ValueError("):
        indent = line[:len(line) - len(line.lstrip())]
        body_indent = indent + "    "
        out.append(line)
        # consume the raise ValueError( ... ) block by paren balance
        j = i + 1
        depth = 0
        started = False
        while j < len(lines):
            for ch in lines[j]:
                if ch == "(":
                    depth += 1; started = True
                elif ch == ")":
                    depth -= 1
            j += 1
            if started and depth <= 0:
                break
        # emit warning in place of the raise block
        out.append(f'{body_indent}import logging as _lg')
        out.append(f'{body_indent}_lg.getLogger(__name__).warning(')
        out.append(f'{body_indent}    "GB10 unified mem bypass: free=%s req=%s, continuing",')
        out.append(f'{body_indent}    init_snapshot.free_memory, requested_memory)')
        patched += 1
        i = j
        continue
    out.append(line)
    i += 1

if patched == 0:
    print("[MEMBYPASS] FAIL: guard/raise block not found (vLLM version skew?)")
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
ok = (MARK in c) and ("raise ValueError(\n            f\"Free memory on device" not in c)
print(f"[MEMBYPASS] applied x{patched}")
print("[MEMBYPASS] PY_COMPILE OK")
print(f"VERIFY mem_bypass={ok}")
sys.exit(0 if ok else 3)
