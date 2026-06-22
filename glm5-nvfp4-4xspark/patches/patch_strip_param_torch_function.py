#!/usr/bin/env python3
# patch_strip_param_torch_function.py
# RUN #33 lever (b) — kill the slow unquantized eager-MLA first-forward by stripping the
# per-op Python __torch_function__ dispatch off vLLM's weight Parameter subclasses.
#
# ROOT CAUSE (py-spy, RUN #32 head Worker_TP0, 5 samples; re-confirmed RUN #34):
#   The DENSE+SKIP_BOTH path BINDS :8000 cleanly (OOM/bind wall solved), but the FIRST real
#   forward CRAWLS at ~0.001 tok/s (>100 min for ~10 tokens). py-spy localized the hot path:
#     default_unquantized_gemm (layers/utils.py:98)  ==  F.linear(x, weight, bias)
#       <- MLA q/kv/o projections (mla_attention.py) x78 layers
#     + EVERY tensor op dispatches through BasevLLMParameter.__torch_function__
#       (parameter.py:126) = a Python-level per-op override on the read-only weights.
#   RUN #32 proved the MLA projections are UNQUANTIZED BF16 *by design* (config.json
#   quantization_config.ignore lists every self_attn*), so they CANNOT be moved to FP4 —
#   the only attackable overhead left is the __torch_function__ per-op Python dispatch that
#   wraps EVERY aten op on those BF16 weights inside the 78-layer attention.
#
# WHAT THE STOCK OVERRIDE DOES (parameter.py ~123-126):
#     @classmethod
#     def __torch_function__(cls, func, types, args=(), kwargs=None):
#         if kwargs is None: kwargs = {}
#         return super().__torch_function__(func, types, args, kwargs)
#   i.e. it adds NOTHING functional — it just re-dispatches to the default impl. It exists
#   only so the subclass participates in the torch-function protocol during weight LOADING
#   (the load_*_weight methods are plain Python calls, unaffected by this). During the
#   FORWARD it is pure per-op Python overhead on a read-only tensor.
#
# THE FIX (this patch — gated, reversible, idempotent):
#   Set  BasevLLMParameter.__torch_function__ = torch._C._disabled_torch_function_impl
#   This is the canonical PyTorch idiom for a Tensor subclass whose ops should behave like
#   plain tensors: aten ops execute directly via the C dispatcher (fast path), returning
#   plain tensors — NUMERICALLY IDENTICAL, just without the Python wrapper on every op.
#   All subclasses (_ColumnvLLMParameter, RowvLLMParameter, ModelWeightParameter, ...)
#   inherit it via MRO (none define their own __torch_function__ — verified single def).
#   Gated behind VLLM_STRIP_PARAM_TORCH_FUNCTION (default "1"=strip) and read ONCE at import
#   so there is ZERO per-call env overhead. Set =0 to restore stock dispatch.
#
# Run INSIDE the engine container (launcher mounts read-only and execs it). Idempotent.
import sys, glob, os, py_compile

P = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/parameter.py"
src = open(P).read()

MARK = "VLLM_STRIP_PARAM_TORCH_FUNCTION strip"
if MARK in src:
    print("[STRIPTF] already patched")
    sys.exit(0)

# Sanity: confirm the anchor class + the stock override exist (guard against vLLM skew).
if "class BasevLLMParameter(Parameter):" not in src:
    print("[STRIPTF] FAIL: BasevLLMParameter class not found (vLLM skew?)")
    sys.exit(2)
if "def __torch_function__(cls, func, types, args=(), kwargs=None):" not in src:
    print("[STRIPTF] FAIL: stock __torch_function__ signature not found (vLLM skew?)")
    sys.exit(2)

block = '''

# === {mark} (RUN #33 lever b) ===
# Strip the per-op Python __torch_function__ dispatch off the weight Parameter subclasses
# so aten ops on the (read-only) BF16/FP4 weights hit the fast C dispatcher path instead of
# re-entering Python on EVERY op during the eager-MLA first forward (py-spy parameter.py:126).
# Numerically identical (the stock override only re-dispatched to super); removes the
# per-op Python overhead that made the dense-MLA prefill crawl at ~0.001 tok/s.
import os as _os_striptf
import torch as _torch_striptf
if _os_striptf.environ.get("VLLM_STRIP_PARAM_TORCH_FUNCTION", "1") == "1":
    try:
        BasevLLMParameter.__torch_function__ = _torch_striptf._C._disabled_torch_function_impl
        import logging as _lg_striptf
        _lg_striptf.getLogger(__name__).warning(
            "VLLM_STRIP_PARAM_TORCH_FUNCTION=1: stripped BasevLLMParameter.__torch_function__ "
            "per-op dispatch (eager-MLA fast-path); plain-tensor aten dispatch on weights")
    except Exception as _e_striptf:
        import logging as _lg_striptf
        _lg_striptf.getLogger(__name__).warning(
            "VLLM_STRIP_PARAM_TORCH_FUNCTION strip FAILED, keeping stock dispatch: %s",
            _e_striptf)
'''.format(mark=MARK)

# Append at module scope, after the class is fully defined.
if not src.endswith("\n"):
    src += "\n"
src += block

open(P, "w").write(src)

for p in glob.glob("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/__pycache__/*.pyc"):
    try:
        os.remove(p)
    except Exception:
        pass

py_compile.compile(P, doraise=True)
c = open(P).read()
ok = (MARK in c) and ("_disabled_torch_function_impl" in c)
print("[STRIPTF] applied (module-level gated reassign appended)")
print("[STRIPTF] PY_COMPILE OK")
print(f"VERIFY strip_torch_function={ok}")
