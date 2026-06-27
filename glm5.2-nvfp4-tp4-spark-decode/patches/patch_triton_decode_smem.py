#!/usr/bin/env python3
# patch_triton_decode_smem.py
# GB10 (sm_121) per-block shared-memory cap = 101376 bytes. The Triton grouped MLA
# decode-attention kernel (_decode_grouped_att_m_fwd -> _fwd_grouped_kernel_stage1) asks for
# 102400 bytes at num_stages=2 -> triton.runtime.errors.OutOfResources at the FIRST decode
# step (after the server has already bound :8000 and returned /health=200).
#
# vLLM already drops num_stages 2->1 to fit, but only behind a `BLOCK_DMODEL >= 1024` guard,
# and GLM-5.0's MLA decode has BLOCK_DMODEL < 1024 so the guard never fires. Relax it to all
# non-HIP. num_stages only controls software-pipelining depth, not numerics, so num_stages=1
# is numerically identical (and is exactly what vLLM does for the >=1024 case).
#
# Run INSIDE the engine container (the launcher mounts this read-only and execs it).
import os, glob, py_compile, sys
F = '/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_decode_attention.py'
s = open(F).read()
needle = '    elif not is_hip_ and BLOCK_DMODEL >= 1024:'
repl   = '    elif not is_hip_:  # GB10 sm_121 101376B SMEM cap: num_stages=1 for all MLA decode'

if 'GB10 sm_121 101376B SMEM cap' in s:
    print('[TSMEM] already patched')
elif needle in s:
    s = s.replace(needle, repl, 1)
    open(F, 'w').write(s)
    print('[TSMEM] guard relaxed -> non-HIP num_stages=1')
else:
    print('[TSMEM] FAIL: guard line not found (vLLM version skew?)')
    sys.exit(2)

# drop stale bytecode so the edit takes effect
for p in glob.glob('/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/__pycache__/*.pyc'):
    try:
        os.remove(p)
    except Exception:
        pass

py_compile.compile(F, doraise=True)
c = open(F).read()
ok = ('GB10 sm_121 101376B SMEM cap' in c) and (needle not in c)
print('[TSMEM] PY_COMPILE OK')
print('VERIFY num_stages_guard_relaxed=', ok)
sys.exit(0 if ok else 3)
