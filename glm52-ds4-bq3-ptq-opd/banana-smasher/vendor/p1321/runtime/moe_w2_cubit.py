# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed experts on 2-bit tensor-sym planes (cubit moe_w2) for the
DeepSeek-V4 / GLM-5.x MoE family.

Opt-in via VLLM_MOE_W2=1. Replaces the stock routed-expert GEMM path:

  weights : checkpoint mxfp4 e2m1 codes -> {-4,-1,1,4} 2-bit planes built on
            GPU at load (QUANT_PROBE tensor-sym K=4: acceptance 2.73 vs 2.68
            baseline, 12/12 coherent; the sign-sym finding reproduces on
            GLM-5.2 — internal/glm52-sweep). Block-32 UE8M0 scale bytes
            verbatim. FP8 block-quant checkpoints (DS4-FP8, GLM-5.2-FP8) are
            re-quantized at load via build_layer_planes_fp8.
  compute : cubit `moe_w2_mm` SASS GEMM (M<=4 per pair, PRMT-LUT decode,
            QMMA.SF block-32 sfb, f32 act-scale fold) for BOTH w13 and w2.
  glue    : moe_align_block_size(block=4) pairs, fp8 group-128 activation
            quant, silu*up in torch, weighted scatter-add unpermute. All
            steps are tensor ops or driver launches on the current stream:
            CUDA-graph capturable, registered as one custom op.

VRAM: planes+scales ~1.73 GiB/layer (vs ~3.2 GiB raw fp4) -> 43 layers fit
a single 96 GB SM120 board together with the fp8 dense stack and KV.
The MTP drafter keeps the stock DeepGEMM-MXFP4 path: layer names containing
"mtp" are excluded, matching the QUANT_PROBE protocol (drafter unmodified).
"""

import ctypes
import functools
import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
    mxfp4_to_codes,
    pack_fragment_major,
    pack_scales,
)
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

_KERN = b"moe_w2_mm"
_DIR = os.getenv("VLLM_MOE_W2_CUBIT_DIR", "/cubit-share")
_BLOCK = 4                      # tokens per pair == kernel M limit
_NTHR = 256                     # NWARP=8 (K>=1024)


def _nwarp_for_k(k: int) -> int:
    """Split-K warp count baked into each cubin by gen_moe_w2.py (KSLICE=K/NWARP
    must be a multiple of 128). K>=1024 -> 8 warps; K=512 (the w2 GEMM under TP4)
    shards to 4. The launch block MUST match the cubin or the extra warps index
    past K (KSLICE*wid) and read garbage. Mirrors the generator's `_nwarp`."""
    nb = k // 128
    cap = 8 if k >= 1024 else 4
    for n in range(min(cap, nb), 0, -1):
        if nb % n == 0:
            return n
    return 1

_cu = None
_fns: dict = {}
_state = "uninit"
# PREFILL LEVER (default ON since the mc4afrag cubins ship): fragment-major
# activations so each lane's m16k32 QMMA A-fragment loads in ONE LDG.128 (vs 8
# strided 4-byte loads). Profile showed prefill moe_w2_mm is L1/load-issue bound
# (NOT weight-DRAM bound), so this cuts the dominant load class ~4x at identical
# occupancy -> measured 1.30x (K=4096) / 1.27x (K=2048) on the prefill GEMM.
# Numerics are bit-identical to mc4. Needs moe_w2_mm_mc4afrag_k{K}.cubin present
# (loader degrades to mc4 when missing). Opt out: VLLM_MOE_W2_AFRAG=0.
_AFRAG = os.getenv("VLLM_MOE_W2_AFRAG", "1") == "1"
_afrag_ok = False


def _to_fragment_major(a: torch.Tensor, pairs: int, K: int) -> torch.Tensor:
    """[pairs*16, K] fp8 row-major -> fragment-major per 16-token tile (matches the
    AFRAG kernel layout / tools.moe_w2_prefill_bench.pack_a_fragment_major):
    dims [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b].

    `a` MUST have EXACTLY pairs*16 rows (complete tiles). Callers pass the
    tile-aligned region ws['a1'][:pairs*16] -- NOT ws['a1'][:slots] (slots is the
    over-allocated, non-16-multiple sorted_ids size)."""
    assert a.shape[0] == pairs * 16, (a.shape, pairs)
    v = a.view(torch.uint8).view(pairs, 2, 8, K // 64, 4, 4, 4)
    v = v.permute(0, 3, 2, 5, 4, 1, 6).reshape(pairs * 16, K)
    return v.contiguous().view(a.dtype)

# layer_key -> dict(planes13, sc13, planes2, sc2, top_k, inter)
_LAYERS: dict[int, dict] = {}
_WS: dict = {}                  # shared workspaces, sized lazily


def enabled() -> bool:
    return os.getenv("VLLM_MOE_W2", "0") == "1"


@functools.cache
def _layer_cutoff() -> int:
    """Main-stack layer count: layers >= this are the MTP drafter. Taken from
    the model config when available (43 for DS4-Flash, 78 for GLM-5.2);
    VLLM_MOE_W2_NUM_LAYERS overrides."""
    v = os.getenv("VLLM_MOE_W2_NUM_LAYERS")
    if v is not None:
        return int(v)
    try:
        from vllm.config import get_current_vllm_config
        n = get_current_vllm_config().model_config.hf_config.num_hidden_layers
        if n:
            return int(n)
    except Exception:  # noqa: BLE001
        pass
    return 43


def is_w2_layer(layer_name: str) -> bool:
    """Main-model routed experts only. The MTP drafter (layer index >=
    num_hidden_layers, e.g. model.layers.43.* for the 43-layer main stack)
    keeps its original path: QUANT_PROBE's acceptance numbers were
    measured with the drafter unmodified."""
    if not enabled():
        return False
    name = layer_name or ""
    if "mtp" in name:
        return False
    import re
    m = re.search(r"\.layers\.(\d+)\.", name)
    if m is None:
        return False
    return int(m.group(1)) < _layer_cutoff()


def _driver():
    global _cu
    if _cu is None:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [
            ctypes.c_uint, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p]
        cu.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                    ctypes.c_char_p]
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.c_void_p, ctypes.c_char_p]
        _cu = cu
    return _cu


def _ck(r, what):
    if r:
        raise RuntimeError(f"moe_w2_cubit: CUDA error {r} in {what}")


def _ensure_ready() -> bool:
    global _state
    if _state == "ready":
        return True
    if _state == "unavailable":
        return False
    try:
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        cu = _driver()
        for tier, kern in (("w2", b"moe_w2_mm"), ("w4", b"moe_w4_mm"),
                           ("w2mc2", b"moe_w2_mm"), ("w2mc4", b"moe_w2_mm")):
            # GEMM contraction K: gate-up needs K=hidden (4096 DS4-Flash,
            # 6144 GLM-5.x); down needs K=I/TP (2048 @ TP1, 1024 @ TP2,
            # 512 @ TP4). Cubins are loaded opportunistically -- the plane
            # builders assert the shapes the model actually needs are present
            # (_assert_kernels fails loudly at weight load).
            for k in (6144, 4096, 2048, 1024, 512):
                if tier in ("w2mc2", "w2mc4"):
                    fname = f"moe_w2_mm_{tier[2:]}_k{k}.cubin"
                else:
                    fname = f"moe_{tier}_mm_k{k}.cubin"
                path = os.path.join(_DIR, fname)
                if not os.path.exists(path):
                    continue
                mod = ctypes.c_void_p()
                _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                    f"cuModuleLoad {path}")
                fn = ctypes.c_void_p()
                _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, kern),
                    "cuModuleGetFunction")
                _fns[(tier, k)] = fn
        global _afrag_ok
        if _AFRAG:
            try:
                for k in (6144, 4096, 2048, 1024, 512):
                    path = os.path.join(_DIR, f"moe_w2_mm_mc4afrag_k{k}.cubin")
                    if not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod, b"moe_w2_mm"),
                        "cuModuleGetFunction afrag")
                    _fns[("w2mc4afrag", k)] = fn
                _afrag_ok = True
                logger.info("moe_w2_cubit: AFRAG prefill cubins loaded")
            except Exception as e:  # noqa: BLE001
                logger.warning("moe_w2_cubit: AFRAG unavailable (%s); using mc4", e)
                _afrag_ok = False
        # W3 sibling kernels (SAPID P3b moe_w3_mm, 8-level LUT 3.25 bpw):
        # loaded opportunistically like the W2 tiers; engaged per-layer when
        # a prepacked planes meta.json carries codebook=="w3" (<source-task> R2).
        # Same desc/launch ABI as moe_w2_mm — only the plane stride differs
        # (N*K*3/8 vs N*K/4 bytes), which flows through the desc pointers.
        w3dir = os.getenv("VLLM_MOE_W3_CUBIT_DIR", _DIR)
        try:
            for tier, pat in (("w3", "moe_w3_mm_k{k}.cubin"),
                              ("w3mc4", "moe_w3_mm_mc4_k{k}.cubin"),
                              ("w3mc4afrag",
                               "moe_w3_mm_mc4afrag_k{k}.cubin")):
                for k in (6144, 4096, 2048, 1024, 512):
                    path = os.path.join(w3dir, pat.format(k=k))
                    if not os.path.exists(path):
                        continue
                    mod = ctypes.c_void_p()
                    _ck(cu.cuModuleLoad(ctypes.byref(mod), path.encode()),
                        f"cuModuleLoad {path}")
                    fn = ctypes.c_void_p()
                    _ck(cu.cuModuleGetFunction(ctypes.byref(fn), mod,
                                               b"moe_w3_mm"),
                        "cuModuleGetFunction w3")
                    _fns[(tier, k)] = fn
        except Exception as e:  # noqa: BLE001
            logger.warning("moe_w2_cubit: W3 cubins unavailable (%s)", e)
        _state = "ready"
        logger.info("moe_w2_cubit: cubins loaded: %s", sorted(_fns))
        return True
    except Exception as e:  # noqa: BLE001
        logger.error("moe_w2_cubit unavailable: %s", e)
        _state = "unavailable"
        return False


# --------------------------------------------------------------------------
# Load-time plane building
# --------------------------------------------------------------------------

def _require_kernels(K13: int, K2: int, need_w4: bool) -> None:
    """Fail loudly at weight load when the cubins this model's shapes need are
    missing from _DIR (they are loaded opportunistically in _ensure_ready)."""
    need = [("w2", K13), ("w2", K2), ("w2mc4", K13), ("w2mc4", K2)]
    if need_w4:
        need += [("w4", K13), ("w4", K2)]
    missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
    assert not missing, (
        f"moe_w2_cubit: missing cubins for K13={K13}/K2={K2}: {missing} "
        f"(dir {_DIR}; set VLLM_MOE_W2_CUBIT_DIR)")


def build_layer_planes(layer, layer_key: int) -> None:
    """Quantize one FusedMoE layer's experts to 2-bit planes (GPU, chunked).

    Reads the CPU-resident checkpoint params (w13_weight [E,2I,K/2] u8 etc.),
    builds fragment-major code planes + scale planes on the GPU, then
    replaces the originals with empty stubs.
    """
    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data          # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data    # [E, 2I, H/32] u8
    w2 = layer.w2_weight.data            # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data      # [E, H, I/32] u8
    E, N13, _ = w13.shape
    _, N2, _ = w2.shape
    K13, K2 = N2, N13 // 2               # H, I (4096/2048 on DS4-Flash TP1)
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())

    planes13 = _plane_alloc(E, N13 * K13 // 4, dev)
    sc13 = _plane_alloc(E, N13 * K13 // 32, dev)
    planes2 = _plane_alloc(E, N2 * K2 // 4, dev)
    sc2 = _plane_alloc(E, N2 * K2 // 32, dev)

    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        mxfp4_to_nibbles, pack_fp4_fragment_major)
    # Pass the PER-RANK FP4 plane sizes (N*K//2 bytes/expert) so the delta tier's
    # slots, host store, and pool indexing match the (TP-sharded) planes. On TP1
    # these equal the module constants -> the single-GPU path is unchanged.
    tier = moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=N13 * K13 // 2,
                                 w2_bytes=N2 * K2 // 2)
    fp13 = fp2 = None
    if tier is not None:
        fp13 = torch.empty(E, N13 * K13 // 2, dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, N2 * K2 // 2, dtype=torch.uint8, device=dev)

    chunk = 32
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = _h2d(w13[e0:e1], dev)
        sg = _h2d(s13[e0:e1], dev)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes13[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc13[e0 + i] = pack_scales(sg[i])
            if fp13 is not None:
                fp13[e0 + i] = pack_fp4_fragment_major(nib)
        wg = _h2d(w2[e0:e1], dev)
        sg = _h2d(s2[e0:e1], dev)
        for i in range(e1 - e0):
            nib = mxfp4_to_nibbles(wg[i])
            planes2[e0 + i] = pack_fragment_major(mxfp4_to_codes(wg[i]))
            sc2[e0 + i] = pack_scales(sg[i])
            if fp2 is not None:
                fp2[e0 + i] = pack_fp4_fragment_major(nib)

    if tier is not None:
        tier.add_layer_host_planes(layer_key, fp13, fp2)
        del fp13, fp2
        # (the background manager is started by get_tier when the tier is
        # created; the old "start on layer NUM_LAYERS-1" trigger never fired
        # under PP, where layer_keys are local per rank and never reach 42)

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


def build_layer_planes_fp8(layer, layer_key: int,
                           scale_suffix: str = "weight_scale_inv") -> None:
    """FP8 block-quant checkpoint variant of build_layer_planes (Fp8MoEMethod:
    DS4-Flash-FP8, GLM-5.2-FP8 — models without an FP4 release).

    Reads the CPU-staged fp8 params (w13_weight [E,2I,H] e4m3 +
    w13_weight_scale_inv [E,ceil(2I/128),ceil(H/128)] f32 etc.), re-quantizes
    each expert on GPU to the sweep-validated 2-bit pipeline (block-32 UE8M0 +
    e2m1 snap + tensor-sym {-4,-1,1,4}; internal/glm52-sweep/sweep.py), packs
    fragment-major planes, then replaces the originals with empty stubs. The
    e2m1 nibbles of the same requant feed the optional FP4 delta tier.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        fp8_block_to_codes_scales, pack_fp4_fragment_major)

    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data                       # [E, 2I, H] e4m3 (cpu)
    s13 = getattr(layer, f"w13_{scale_suffix}").data  # [E, 2I/128, H/128] f32
    w2 = layer.w2_weight.data                         # [E, H, I] e4m3
    s2 = getattr(layer, f"w2_{scale_suffix}").data    # [E, H/128, I/128] f32
    assert w13.dtype == torch.float8_e4m3fn, w13.dtype
    E, N13, K13 = w13.shape
    _, N2, K2 = w2.shape
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())

    planes13 = _plane_alloc(E, N13 * K13 // 4, dev)
    sc13 = _plane_alloc(E, N13 * K13 // 32, dev)
    planes2 = _plane_alloc(E, N2 * K2 // 4, dev)
    sc2 = _plane_alloc(E, N2 * K2 // 32, dev)

    tier = moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=N13 * K13 // 2,
                                 w2_bytes=N2 * K2 // 2)
    fp13 = fp2 = None
    if tier is not None:
        fp13 = torch.empty(E, N13 * K13 // 2, dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, N2 * K2 // 2, dtype=torch.uint8, device=dev)

    # fp8 experts are 4x the bytes of the mxfp4 path and the requant makes f32
    # temporaries -> smaller H2D chunks, per-expert quantize.
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = _h2d(w13[e0:e1], dev)
        sg = _h2d(s13[e0:e1], dev)
        for i in range(e1 - e0):
            codes, sbytes, nib = fp8_block_to_codes_scales(
                wg[i], sg[i], want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = pack_fp4_fragment_major(nib)
        wg = _h2d(w2[e0:e1], dev)
        sg = _h2d(s2[e0:e1], dev)
        for i in range(e1 - e0):
            codes, sbytes, nib = fp8_block_to_codes_scales(
                wg[i], sg[i], want_nibbles=fp2 is not None)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = pack_fp4_fragment_major(nib)

    if tier is not None:
        tier.add_layer_host_planes(layer_key, fp13, fp2)
        del fp13, fp2

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                   f"w2_{scale_suffix}"))


def build_layer_planes_nvfp4(layer, layer_key: int) -> None:
    """NVFP4 (modelopt) checkpoint variant of build_layer_planes
    (ModelOptNvFp4FusedMoE: nvidia/GLM-5.2-NVFP4 — e2m1 codes + e4m3
    block-16 scales + per-tensor scale_2).

    Reads the CPU-staged params (w13_weight [E,2I,H/2] u8 packed +
    w13_weight_scale [E,2I,H/16] e4m3 + w13_weight_scale_2 [E,2] f32 etc.),
    dequantizes each expert to f64 on GPU (exact) and re-quantizes to the
    sweep-validated sign-symmetric 2-bit pipeline; the e2m1 nibbles of the
    same requant feed the optional FP4 delta tier. The UE8M0 block-32 output
    scales absorb scale_2, so serving needs no extra per-tensor factor.
    """
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    from vllm.model_executor.layers.quantization.utils.moe_w2_planes import (
        nvfp4_to_codes_scales, pack_fp4_fragment_major)

    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    w13 = layer.w13_weight.data                 # [E, 2I, H/2] u8 (cpu)
    s13 = layer.w13_weight_scale.data           # [E, 2I, H/16] e4m3
    s13_2 = layer.w13_weight_scale_2.data       # [E, 2] f32 (w1, w3)
    w2 = layer.w2_weight.data                   # [E, H, I/2] u8
    s2 = layer.w2_weight_scale.data             # [E, H, I/16] e4m3
    s2_2 = layer.w2_weight_scale_2.data         # [E] f32
    assert w13.dtype == torch.uint8 and s13.dtype == torch.float8_e4m3fn, (
        w13.dtype, s13.dtype)
    E, N13, K13h = w13.shape
    K13 = K13h * 2
    _, N2, K2h = w2.shape
    K2 = K2h * 2
    group = K13 // s13.shape[2]                 # 16 for NVFP4
    _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())

    planes13 = _plane_alloc(E, N13 * K13 // 4, dev)
    sc13 = _plane_alloc(E, N13 * K13 // 32, dev)
    planes2 = _plane_alloc(E, N2 * K2 // 4, dev)
    sc2 = _plane_alloc(E, N2 * K2 // 32, dev)

    tier = moe_w2_delta.get_tier(n_experts=E, dev=dev,
                                 w13_bytes=N13 * K13 // 2,
                                 w2_bytes=N2 * K2 // 2)
    fp13 = fp2 = None
    if tier is not None:
        fp13 = torch.empty(E, N13 * K13 // 2, dtype=torch.uint8, device=dev)
        fp2 = torch.empty(E, N2 * K2 // 2, dtype=torch.uint8, device=dev)

    # f64 temporaries are 16x the packed nibbles -> small H2D chunks,
    # per-expert quantize (mirrors the fp8 loader).
    chunk = 8
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        wg = _h2d(w13[e0:e1], dev)
        sg = _h2d(s13[e0:e1], dev)
        s2g = _h2d(s13_2[e0:e1], dev)
        half = N13 // 2                          # rows [0:I]=w1, [I:2I]=w3
        for i in range(e1 - e0):
            s2_row = torch.cat((s2g[i, 0].expand(half), s2g[i, 1].expand(half)))
            codes, sbytes, nib = nvfp4_to_codes_scales(
                wg[i], sg[i], s2_row, group=group,
                want_nibbles=fp13 is not None)
            planes13[e0 + i] = pack_fragment_major(codes)
            sc13[e0 + i] = pack_scales(sbytes)
            if fp13 is not None:
                fp13[e0 + i] = pack_fp4_fragment_major(nib)
        wg = _h2d(w2[e0:e1], dev)
        sg = _h2d(s2[e0:e1], dev)
        s2g = _h2d(s2_2[e0:e1], dev)
        for i in range(e1 - e0):
            codes, sbytes, nib = nvfp4_to_codes_scales(
                wg[i], sg[i], s2g[i], group=group,
                want_nibbles=fp2 is not None)
            planes2[e0 + i] = pack_fragment_major(codes)
            sc2[e0 + i] = pack_scales(sbytes)
            if fp2 is not None:
                fp2[e0 + i] = pack_fp4_fragment_major(nib)

    if tier is not None:
        tier.add_layer_host_planes(layer_key, fp13, fp2)
        del fp13, fp2

    _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", "w13_weight_scale", "w2_weight",
                   "w2_weight_scale"))


_PIN_BOUNCE: dict = {}

# Plane storage: on unified-memory hosts, keep the 2-bit planes in PINNED
# HOST memory instead of the CUDA pool. Same LPDDR, same bandwidth, and the
# SASS kernels take raw (UVA-valid) pointers from the desc table — but it
# sidesteps the GB10 driver behavior where cudaMalloc'd content grows
# CPU-side mirror arenas during load (which OOM'd the box at ~26/43 layers).
# Override with VLLM_MOE_W2_HOST_PLANES=0/1.


def _host_planes() -> bool:
    v = os.getenv("VLLM_MOE_W2_HOST_PLANES")
    if v is not None:
        return v == "1"
    try:
        return bool(getattr(torch.cuda.get_device_properties(0),
                            "is_integrated", 0))
    except Exception:  # noqa: BLE001
        return False


def _plane_alloc(e: int, nbytes: int, dev) -> torch.Tensor:
    if _host_planes():
        # PLAIN PAGEABLE host memory — not pinned! cudaHostAlloc on GB10 is
        # shmem-backed with power-of-2 rounding (a 20 GiB ask costs ~33 GiB),
        # while the SASS kernels read pageable pointers coherently via ATS
        # at identical speed (op-validated: spark/moe_w2_check_hostb.py).
        return torch.empty(e, nbytes, dtype=torch.uint8)
    return torch.empty(e, nbytes, dtype=torch.uint8, device=dev)


def _h2d(t: torch.Tensor, dev) -> torch.Tensor:
    """H2D copy via a cached pinned bounce buffer.

    On unified-memory hosts (GB10), an async H2D from PAGEABLE memory is
    executed by the CPU writing straight into the UVM destination — those
    destination pages become CPU-resident on top of their device backing,
    and with allocator churn the double-resident pages accumulate to tens
    of GiB across a 40+-layer load. A pinned source makes the copy a DMA:
    the destination stays device-resident only. The bounce buffers are keyed
    by (shape, dtype) and reused across layers (~0.5 GiB total)."""
    if not t.is_cuda and not t.is_pinned():
        key = (tuple(t.shape), t.dtype)
        buf = _PIN_BOUNCE.get(key)
        if buf is None:
            buf = _PIN_BOUNCE[key] = torch.empty(
                t.shape, dtype=t.dtype, pin_memory=True)
        buf.copy_(t)
        t = buf
    # PERSISTENT device destination, reused for every chunk: on GB10 the
    # driver executes H2D with CPU writes, which leaves the destination's
    # UVM pages CPU-resident forever. Fresh caching-allocator blocks each
    # chunk accumulate ~2-3 GiB of such pages PER LAYER (the planes
    # fragment the pool, so transients keep landing on new extents);
    # reusing one buffer per shape bounds it at ~1 GiB total.
    key_d = ("dev", tuple(t.shape), t.dtype)
    dbuf = _PIN_BOUNCE.get(key_d)
    if dbuf is None:
        dbuf = _PIN_BOUNCE[key_d] = torch.empty(
            t.shape, dtype=t.dtype, device=dev)
    dbuf.copy_(t, non_blocking=True)
    return dbuf


def release_pin_bounce() -> None:
    _PIN_BOUNCE.clear()


# --------------------------------------------------------------------------
# Prepacked planes (spark/prepack_planes.py): load 2-bit planes straight from
# disk — no staging, no requant, no transient churn at serve time. The whole
# in-process conversion pipeline is the fallback when no prepack exists.
# --------------------------------------------------------------------------

def prepacked_path(layer_name: str) -> str | None:
    d = os.getenv("VLLM_MOE_W2_PREPACKED_DIR")
    if not d:
        return None
    import re as _re
    m = _re.search(r"\.layers\.(\d+)\.", layer_name or "")
    if m is None:
        return None
    p = os.path.join(os.path.expanduser(d), f"layer_{int(m.group(1)):03d}")
    return p if os.path.exists(p + ".meta.json") else None


def noop_weight_loader(param, loaded_weight, *args, **kwargs):
    """Substitute loader for prepacked layers: the checkpoint expert bytes
    are not needed (planes come from disk); report success so the model's
    load bookkeeping stays happy."""
    return True if kwargs.get("return_success") else None


def load_prepacked_layer_mixed(layer, key: int, prefix: str,
                               scale_suffix: str = "weight_scale") -> None:
    """R6 dynamic-experts (<source-task>): per-EXPERT mixed-tier planes.

    layer_LLL.meta.json carries mixed=true + tier_of/slot_of[E]; per-tier
    COMPACT plane files layer_LLL.{w2,w3,fp4}.{planes13,sc13,planes2,sc2}.npy
    hold only that tier's experts (rows in slot_of order). Tier kernels:
      w2  -> moe_w2_mm family (VLLM_MOE_W2_CUBIT_DIR)
      w3  -> moe_w3_mm family (VLLM_MOE_W3_CUBIT_DIR; e43-LUT build matching
             the dp_asym8-fit GPTQ planes, pool R27=0xb6bfc6cd R13=0x4d463c21)
      fp4 -> moe_w4_mm (ckpt e2m1 planes verbatim + ckpt UE8M0 scales)
    The manifest is fully resolved HERE at load; forward-time cost is the
    per-tier desc grouping only (other tiers' pairs early-EXIT on m=0).
    """
    import json as _json
    import numpy as _np
    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    meta = _json.load(open(prefix + ".meta.json"))
    E, N13, K13, N2, K2 = (meta[k] for k in ("E", "N13", "K13", "N2", "K2"))
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    assert not moe_w2_delta.enabled() and not moe_w2_delta.base_enabled(), \
        "mixed planes are incompatible with the delta/base tiers"
    need = [("w2", K13), ("w2", K2), ("w2mc4", K13), ("w2mc4", K2),
            ("w3", K13), ("w3", K2), ("w3mc4", K13), ("w3mc4", K2),
            ("w4", K13), ("w4", K2)]
    missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
    assert not missing, (
        f"moe_w2_cubit: mixed planes at {prefix} but missing cubins: "
        f"{missing} (set VLLM_MOE_W2_CUBIT_DIR / VLLM_MOE_W3_CUBIT_DIR)")
    # R7PP (<source-task>): per-PROJECTION manifests carry pp=true +
    # counts13/counts2 + tier_of13/slot_of13/tier_of2/slot_of2; an expert's
    # fused13 and down projections may live in DIFFERENT tiers, and each
    # tier's {planes13,sc13} / {planes2,sc2} files hold only that
    # projection's rows (independent slot spaces). Legacy per-expert mixed
    # metas (counts + tier_of/slot_of) load through the same path with
    # counts13 == counts2 == counts.
    pp = bool(meta.get("pp"))
    if pp:
        counts13, counts2 = meta["counts13"], meta["counts2"]
    else:
        counts13 = counts2 = meta["counts"]
    bytes13 = {"w2": N13 * K13 // 4, "w3": N13 * K13 * 3 // 8,
               "fp4": N13 * K13 // 2}
    bytes2 = {"w2": N2 * K2 // 4, "w3": N2 * K2 * 3 // 8,
              "fp4": N2 * K2 // 2}
    scb = {"13": N13 * K13 // 32, "2": N2 * K2 // 32}
    pb = {"13": bytes13, "2": bytes2}
    tiers = {}
    mmap_planes = os.getenv("VLLM_MOE_W2_PLANES_MMAP", "0") == "1"
    mmap_from = int(os.getenv("VLLM_MOE_W2_MMAP_FROM_LAYER", "0"))
    if mmap_planes and mmap_from:
        import re as _re2
        m2 = _re2.search(r"layer_(\d+)$", os.path.basename(prefix))
        if m2 is not None and int(m2.group(1)) < mmap_from:
            mmap_planes = False
    for t in ("w2", "w3", "fp4"):
        d = {}
        for which, cnts in (("13", counts13), ("2", counts2)):
            n = cnts.get(t, 0)
            if n == 0:
                # (tier, projection) empty in this layer: tier_of never
                # routes here; 1-row dummies keep the desc build's pointer
                # math trivially valid (m=0 pairs never dereference their
                # plane pointers).
                d["planes" + which] = torch.zeros(
                    1, pb[which][t], dtype=torch.uint8)
                d["sc" + which] = torch.zeros(
                    1, scb[which], dtype=torch.uint8)
                continue
            for tag in ("planes" + which, "sc" + which):
                arr = _np.load(f"{prefix}.{t}.{tag}.npy", mmap_mode="r")
                if mmap_planes:
                    d[tag] = torch.from_numpy(arr)
                    continue
                buf = _plane_alloc(arr.shape[0], arr.shape[1], dev)
                buf.copy_(torch.from_numpy(_np.ascontiguousarray(arr)))
                d[tag] = buf
            assert d["planes" + which].shape == (n, pb[which][t]), \
                (t, which, d["planes" + which].shape, n, pb[which][t])
            assert d["sc" + which].shape == (n, scb[which])
        tiers[t] = d
    st = dict(N13=N13, K13=K13, N2=N2, K2=K2, E=E, mixed=True,
              tiers=tiers)
    if pp:
        for nm in ("tier_of13", "slot_of13", "tier_of2", "slot_of2"):
            v = torch.tensor(meta[nm], dtype=torch.int32, device=dev)
            assert v.shape[0] == E, (nm, v.shape, E)
            st[nm] = v
        st["pp"] = True
    else:
        tier_of = torch.tensor(meta["tier_of"], dtype=torch.int32,
                               device=dev)
        slot_of = torch.tensor(meta["slot_of"], dtype=torch.int32,
                               device=dev)
        assert tier_of.shape[0] == E and slot_of.shape[0] == E
        st.update(tier_of=tier_of, slot_of=slot_of)
    _LAYERS[key] = st
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                 f"w2_{scale_suffix}"):
        old = getattr(layer, name, None)
        if old is not None and old.data.numel():
            old.data = torch.empty(0, dtype=old.data.dtype,
                                   device=old.data.device)
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    gib = sum(v.nbytes for d_ in tiers.values() for v in d_.values()) / 2**30
    logger.info("moe_w2: layer %d MIXED%s planes loaded (f13 w2=%d w3=%d "
                "fp4=%d | down w2=%d w3=%d fp4=%d, %.2f GiB) from %s",
                key, " PP" if pp else "", counts13.get("w2", 0),
                counts13.get("w3", 0), counts13.get("fp4", 0),
                counts2.get("w2", 0), counts2.get("w3", 0),
                counts2.get("fp4", 0), gib, os.path.basename(prefix))


def _load_vq_projection(prefix: str, which: str, experts: int, width: int, dev):
    """Load one mmap-backed IQ3 VQ projection plus CUDA-resident offset tables."""
    import numpy as _np
    from vllm.model_executor.layers.quantization.utils import moe_vq_triton

    names = ("codes", "scales", "codebooks", "code_offset", "scale_offset",
             "code_row_bytes", "dimension", "bits", "cb_offset")
    arrays = {
        name: _np.load(f"{prefix}.vq{which}.{name}.npy", mmap_mode="r")
        for name in names
    }
    meta = __import__("json").load(open(prefix + ".meta.json"))
    arrays["n_outputs"] = int(meta[f"N{which}"])
    moe_vq_triton.validate_projection_state(arrays, experts=experts, width=width)
    state = {
        "n_outputs": arrays["n_outputs"],
        "all_d4": bool((_np.asarray(arrays["dimension"]) == 4).all()),
    }
    for name in ("code_offset", "scale_offset", "code_row_bytes", "dimension",
                 "bits", "cb_offset"):
        state[name] = torch.tensor(_np.asarray(arrays[name]), device=dev)
    # Keep the large immutable blobs file-backed. GB10 kernels read these CPU
    # mmap pointers coherently through ATS; vq_gemm passes raw pointers to
    # Triton so its launcher never rejects a CPU tensor argument.
    for name in ("codes", "scales", "codebooks"):
        state[name] = torch.from_numpy(arrays[name])
    state["blob_ptrs"] = torch.tensor(
        [state[name].data_ptr() for name in ("codes", "scales", "codebooks")],
        dtype=torch.int64,
        device=dev,
    )
    return state


def load_prepacked_layer_vq(layer, key: int, prefix: str,
                            scale_suffix: str = "weight_scale") -> None:
    """Load exact compact d4/d8/VQA rows plus the existing FP4 tier.

    ``iq3-vq-wire-v1`` stores heterogeneous VQ codes at their exact 8--12 bit
    widths and executes them with ``moe_vq_triton``. No VQ row is rounded or
    requantized through W2/W3/FP4, which is the checkpoint-to-wire parity gate.
    """
    import json as _json
    import numpy as _np

    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    meta = _json.load(open(prefix + ".meta.json"))
    E, N13, K13, N2, K2 = (int(meta[k]) for k in
                            ("E", "N13", "K13", "N2", "K2"))
    kinds13 = torch.tensor(meta["kind13"], dtype=torch.int32, device=dev)
    kinds2 = torch.tensor(meta["kind2"], dtype=torch.int32, device=dev)
    slots13 = torch.tensor(meta["slot13"], dtype=torch.int32, device=dev)
    slots2 = torch.tensor(meta["slot2"], dtype=torch.int32, device=dev)
    assert kinds13.shape == kinds2.shape == slots13.shape == slots2.shape == (E,)
    fp4_13 = int((kinds13 == 2).sum().item())
    fp4_2 = int((kinds2 == 2).sum().item())
    need = []
    if fp4_13:
        need.append(("w4", K13))
    if fp4_2:
        need.append(("w4", K2))
    missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
    assert not missing, f"IQ3 VQ pack requires missing cubins: {missing}"

    def arr(name, *, dummy_bytes=1):
        path = f"{prefix}.{name}.npy"
        if not os.path.exists(path):
            return torch.zeros(1, dummy_bytes, dtype=torch.uint8)
        return torch.from_numpy(_np.load(path, mmap_mode="r"))

    fp4 = {
        "planes13": arr("fp4.planes13"), "sc13": arr("fp4.sc13"),
        "planes2": arr("fp4.planes2"), "sc2": arr("fp4.sc2"),
    }
    dummy = {name: torch.zeros(1, 1, dtype=torch.uint8)
             for name in ("planes13", "sc13", "planes2", "sc2")}
    _LAYERS[key] = dict(
        N13=N13, K13=K13, N2=N2, K2=K2, E=E,
        mixed=True, pp=True, vq_mixed=True,
        has_fp4_13=bool(fp4_13), has_fp4_2=bool(fp4_2),
        tiers={"w2": dummy, "w3": dummy, "fp4": fp4},
        tier_of13=kinds13, tier_of2=kinds2,
        slot_of13=slots13, slot_of2=slots2,
        vq13=_load_vq_projection(prefix, "13", E, K13, dev),
        vq2=_load_vq_projection(prefix, "2", E, K2, dev),
    )
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    for name in ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                 f"w2_{scale_suffix}"):
        old = getattr(layer, name, None)
        if old is not None and old.data.numel():
            old.data = torch.empty(0, dtype=old.data.dtype,
                                   device=old.data.device)
        layer.register_parameter(name, torch.nn.Parameter(stub, requires_grad=False))
    logger.info("moe_w2: layer %d IQ3-VQ loaded (vq13=%d fp4_13=%d "
                "vq2=%d fp4_2=%d) from %s", key, E - fp4_13, fp4_13,
                E - fp4_2, fp4_2, os.path.basename(prefix))


def load_prepacked_layer(layer, key: int, prefix: str,
                         scale_suffix: str = "weight_scale") -> None:
    import json as _json
    import numpy as _np
    assert _ensure_ready(), "moe_w2 cubins missing"
    dev = torch.device("cuda")
    meta = _json.load(open(prefix + ".meta.json"))
    if meta.get("format") == "iq3-vq-wire-v1":
        return load_prepacked_layer_vq(layer, key, prefix, scale_suffix)
    if meta.get("mixed"):
        return load_prepacked_layer_mixed(layer, key, prefix, scale_suffix)
    E, N13, K13, N2, K2 = (meta[k] for k in ("E", "N13", "K13", "N2", "K2"))
    is_w3 = meta.get("codebook") == "w3"
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if is_w3:
        # W3 planes (moe_w3_mm): decode + mc4 (+ afrag) tiers must be present
        # for this model's contraction shapes.
        need = [("w3", K13), ("w3", K2), ("w3mc4", K13), ("w3mc4", K2)]
        missing = [f"{t}_k{k}" for t, k in need if (t, k) not in _fns]
        assert not missing, (
            f"moe_w2_cubit: W3 planes at {prefix} but missing W3 cubins: "
            f"{missing} (set VLLM_MOE_W3_CUBIT_DIR)")
        assert not moe_w2_delta.enabled() and not moe_w2_delta.base_enabled(), \
            "W3 planes are incompatible with the delta/base tiers"
    else:
        _require_kernels(K13, K2, need_w4=moe_w2_delta.enabled())
    outs = []
    mmap_planes = os.getenv("VLLM_MOE_W2_PLANES_MMAP", "0") == "1"
    # Residency split (<source-task> R2): with the full plane set file-backed,
    # a 105G W3 set on a 121G box thrashes page cache on every expert sweep
    # (LRU evicts exactly what the next window needs). Layers BELOW this
    # threshold load into anonymous (pageable-host) memory via _plane_alloc;
    # layers >= threshold stay file-backed, confining major faults to a
    # small fixed file window that readahead handles well. Default (unset)
    # keeps the existing all-mmap behavior. Residency does not change bytes
    # or kernels — measurement-identical.
    mmap_from = int(os.getenv("VLLM_MOE_W2_MMAP_FROM_LAYER", "0"))
    if mmap_planes and mmap_from:
        import re as _re2
        m2 = _re2.search(r"layer_(\d+)$", os.path.basename(prefix))
        if m2 is not None and int(m2.group(1)) < mmap_from:
            mmap_planes = False
    for tag in ("planes13", "sc13", "planes2", "sc2"):
        arr = _np.load(f"{prefix}.{tag}.npy", mmap_mode="r")
        if mmap_planes:
            # keep the planes FILE-BACKED: the kernels read them via ATS
            # straight out of the page cache, and the kernel evicts cold
            # experts under memory pressure (free disk-backed expert cache;
            # GLM routes ~90% of tokens through ~20% of experts). Cuts each
            # rank's anonymous footprint by the full plane size.
            outs.append(torch.from_numpy(arr))
            continue
        t = _plane_alloc(arr.shape[0], arr.shape[1], dev)
        t.copy_(torch.from_numpy(_np.ascontiguousarray(arr)))
        outs.append(t)
    planes13, sc13, planes2, sc2 = outs
    _finish_layer(layer, key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E,
                  ("w13_weight", f"w13_{scale_suffix}", "w2_weight",
                   f"w2_{scale_suffix}"))
    if is_w3:
        _LAYERS[key]["w3"] = True
    logger.info("moe_w2: layer %d planes loaded PREPACKED%s from %s", key,
                " (W3 codebook)" if is_w3 else "",
                os.path.basename(prefix))


# --------------------------------------------------------------------------
# Eager per-layer build (unified-memory hosts: DGX Spark / GB10 etc.)
# --------------------------------------------------------------------------
# The hooks stage ALL layers' checkpoint experts in CPU RAM and convert to
# 2-bit planes only in process_weights_after_loading. On discrete-GPU hosts
# that staging is free; on unified-memory hosts CPU RAM *is* the GPU memory,
# so staging the full expert set (~137 GiB on DS4-Flash) OOMs the box before
# conversion starts. arm_eager_build() wraps the staged params'
# weight_loaders and builds a layer's planes the moment its last expert
# shard lands, freeing its staging immediately — peak staging cost becomes
# ONE layer instead of all of them.
#
# Default: on iff the device reports itself as integrated (unified memory);
# override with VLLM_MOE_W2_EAGER_BUILD=0/1.

_EAGER_LOCK = __import__("threading").Lock()
_EAGER_FALLBACK_WARNED = False
_EAGER_NEXT_KEY = 0

# Backpressure: vLLM loads checkpoint files with several reader threads, so
# staged-but-unconverted layers pile up (~3.2 GiB each on DS4) while the
# conversions serialize — unbounded, that runahead OOMs a unified-memory
# host. Gate the FIRST write of a new layer until the backlog drains. The
# gate never blocks mid-layer writes, so every started layer can always
# complete, build, and open the gate: no deadlock.
_EAGER_COND = __import__("threading").Condition(_EAGER_LOCK)
_EAGER_STARTED = 0
_EAGER_BUILT = 0
_EAGER_INFLIGHT = int(os.getenv("VLLM_MOE_W2_EAGER_INFLIGHT", "4"))


def eager_build_enabled() -> bool:
    v = os.getenv("VLLM_MOE_W2_EAGER_BUILD")
    if v is not None:
        return v == "1"
    try:
        props = torch.cuda.get_device_properties(0)
        return bool(getattr(props, "is_integrated", 0))
    except Exception:
        return False


def _eager_key() -> int:
    # unique per built layer; forward only needs consistency with _LAYERS
    return len(_LAYERS)


def arm_eager_build(layer, build_fn, scale_suffix: str = "weight_scale") -> None:
    """Wrap the four staged params' weight_loaders; when every expert shard
    of the layer has been written, build the planes and free the staging.

    Expected loader calls per param (per-expert shard loading, the DS4/GLM
    checkpoint layout): w13_* get 2 calls/expert (w1+w3), w2_* get 1. A
    whole-tensor load (expert_id None) completes its param outright. If a
    checkpoint loads in some other pattern the counts never complete and the
    layer just falls back to process_weights_after_loading (with a one-time
    warning: on unified memory that path can OOM)."""
    if not eager_build_enabled():
        return
    E = layer.w13_weight.data.shape[0]
    expected = {"w13_weight": 2 * E, f"w13_{scale_suffix}": 2 * E,
                "w2_weight": E, f"w2_{scale_suffix}": E}
    seen = {k: 0 for k in expected}
    done = {"built": False}

    def _complete() -> bool:
        return all(seen[k] >= expected[k] for k in expected)

    def _wrap(pname, inner):
        @functools.wraps(inner)
        def loader(param, loaded_weight, *args, **kwargs):
            global _EAGER_STARTED, _EAGER_BUILT, _EAGER_NEXT_KEY
            if not done.get("started"):
                with _EAGER_COND:
                    if not done.get("started"):
                        waited = 0.0
                        while (_EAGER_STARTED - _EAGER_BUILT
                               >= _EAGER_INFLIGHT and waited < 600):
                            _EAGER_COND.wait(5.0)
                            waited += 5.0
                        if waited >= 600:
                            logger.warning("moe_w2 eager: backpressure gate "
                                           "timed out — proceeding")
                        done["started"] = True
                        _EAGER_STARTED += 1
            ret = inner(param, loaded_weight, *args, **kwargs)
            # expert_id is the last positional (weight_name, shard_id,
            # expert_id) or a kwarg, depending on caller
            expert_id = kwargs.get("expert_id",
                                   args[-1] if args else None)
            with _EAGER_LOCK:
                if done["built"]:
                    logger.warning(
                        "moe_w2 eager: loader call AFTER build! layer=%s "
                        "pname=%s args=%r kwargs=%r", done.get("key"),
                        pname, args[:3], list(kwargs))
                    return ret
                seen[pname] += (expected[pname] - seen[pname]
                                if expert_id is None else 1)
                if not _complete():
                    return ret
                done["built"] = True
                key = done["key"] = _EAGER_NEXT_KEY
                _EAGER_NEXT_KEY += 1
            build_fn(layer, key)
            layer._moe_w2_key = key
            # Return freed transient blocks to the driver each layer: on
            # GB10, pool pages that were ever CPU-written stay host-resident
            # while cached in the allocator — releasing whole segments drops
            # both residencies.
            torch.cuda.empty_cache()
            # Drop the checkpoint's page cache: cudaMalloc on GB10 takes
            # system RAM but does NOT trigger page-cache reclaim, so the
            # mmap'd safetensors cache (up to the full 149 GiB checkpoint)
            # races weight allocation to the bottom of RAM and wins ("754 MiB
            # free" CUDA OOMs while tens of GiB sit in reclaimable cache).
            glob_pat = os.getenv("VLLM_MOE_W2_FADVISE_GLOB")
            if glob_pat:
                import glob as _glob
                for fpath in _glob.glob(os.path.expanduser(glob_pat)):
                    try:
                        fd = os.open(fpath, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0,
                                             os.POSIX_FADV_DONTNEED)
                        finally:
                            os.close(fd)
                    except OSError:
                        pass
            with _EAGER_COND:
                _EAGER_BUILT += 1
                _EAGER_COND.notify_all()
            logger.info("moe_w2: layer %d planes built EAGERLY "
                        "(staging freed at load, backlog %d)",
                        key, _EAGER_STARTED - _EAGER_BUILT)
            return ret
        return loader

    for pname in expected:
        p = getattr(layer, pname)
        if hasattr(p, "weight_loader"):
            p.weight_loader = _wrap(pname, p.weight_loader)


def eager_built(layer) -> bool:
    """True if arm_eager_build already converted this layer."""
    global _EAGER_FALLBACK_WARNED
    if getattr(layer, "_moe_w2_key", None) is not None:
        return True
    if eager_build_enabled() and not _EAGER_FALLBACK_WARNED:
        _EAGER_FALLBACK_WARNED = True
        logger.warning(
            "moe_w2: eager build armed but a layer completed loading "
            "without triggering it (unexpected shard pattern?) — falling "
            "back to build-at-end; on unified memory this may OOM")
    return False


def _finish_layer(layer, layer_key, dev, planes13, sc13, planes2, sc2,
                  N13, K13, N2, K2, E, param_names) -> None:
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    if moe_w2_delta.base_enabled():
        # BASE cache (inverted delta): the 2-bit planes go to PINNED HOST RAM
        # instead of staying GPU-resident; the GPU holds only the base tier's
        # slot pool. Slot layout per expert: [codes13 | sc13 | codes2 | sc2]
        # (the tier's "w13 section" = codes13+sc13, "w2 section" = codes2+sc2,
        # so add_layer_host_planes packs it verbatim).
        c13len, s13len = planes13.shape[1], sc13.shape[1]
        c2len, s2len = planes2.shape[1], sc2.shape[1]
        btier = moe_w2_delta.get_base_tier(
            _layer_cutoff() + 1, E, dev,
            w13_bytes=c13len + s13len, w2_bytes=c2len + s2len)
        btier.add_layer_host_planes(
            layer_key,
            torch.cat((planes13, sc13), dim=1),
            torch.cat((planes2, sc2), dim=1))
        _LAYERS[layer_key] = dict(
            N13=N13, K13=K13, N2=N2, K2=K2, E=E, base=True,
            off_s13=c13len, off_c2=c13len + s13len,
            off_s2=c13len + s13len + c2len,
        )
        del planes13, sc13, planes2, sc2
        stub = torch.empty(0, dtype=torch.uint8, device=dev)
        for name in param_names:
            layer.register_parameter(
                name, torch.nn.Parameter(stub, requires_grad=False))
        logger.info("moe_w2: layer %d planes HOST-staged (base cache, "
                    "%.2f GiB pinned)", layer_key,
                    E * btier.slot_bytes / 2**30)
        return

    _LAYERS[layer_key] = dict(
        planes13=planes13, sc13=sc13, planes2=planes2, sc2=sc2,
        N13=N13, K13=K13, N2=N2, K2=K2, E=E,
    )
    # Release checkpoint copies; keep CUDA stubs so device probes stay happy.
    # Free the OLD param objects' storage in place too: the model's
    # load_weights holds a params_dict snapshot that keeps them alive past
    # re-registration, which pins the full staged expert set until loading
    # ends — fatal on unified-memory hosts.
    stub = torch.empty(0, dtype=torch.uint8, device=dev)
    freed = 0
    for name in param_names:
        old = getattr(layer, name, None)
        if old is not None and old.data.numel():
            freed += old.data.numel() * old.data.element_size()
            old.data = torch.empty(0, dtype=old.data.dtype,
                                   device=old.data.device)
        layer.register_parameter(
            name, torch.nn.Parameter(stub, requires_grad=False))
    logger.debug("moe_w2: staging released %d MB", freed >> 20)
    logger.info("moe_w2: layer %d planes built (%.2f GiB)", layer_key,
                (planes13.nbytes + sc13.nbytes + planes2.nbytes + sc2.nbytes)
                / 2**30)


# --------------------------------------------------------------------------
# Forward
# --------------------------------------------------------------------------

def _workspaces(slots: int, tokens: int, dev, inter: int = 2048,
                hidden: int = 4096) -> dict:
    # `inter` = per-rank expert intermediate size I (2048 on 1 GPU; 1024 @ TP2,
    # 512 @ TP4 as the experts shard). The hidden H (4096 DS4, 6144 GLM-5.x) is
    # NOT sharded, so the A-side (a1), x-quant (xq) and w2 output (c2) buffers
    # stay H-wide; only the gate/up output (c13 = 2I), the intermediate
    # activation (act/a2 = I) and its group-128 scales (as2 = I/128) follow the
    # shard.
    if (_WS.get("slots", 0) < slots or _WS.get("tokens", 0) < tokens
            or _WS.get("inter") != inter or _WS.get("hidden") != hidden):
        slots = max(slots, _WS.get("slots", 0))
        tokens = max(tokens, _WS.get("tokens", 0))
        _WS.update(
            slots=slots,
            tokens=tokens,
            inter=inter,
            hidden=hidden,
            # token-side quant buffers; the LAST row is the permanent zero
            # pad row (gather source for filler slots) — quant only ever
            # writes rows [:T].
            xq=torch.zeros(tokens + 1, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            xs=torch.zeros(tokens + 1, hidden // 128, dtype=torch.float32,
                           device=dev),
            # Exact VQ tiers consume BF16 activations directly. The last xv row
            # is the permanent zero gather source for moe_align filler slots.
            xv=torch.zeros(tokens + 1, hidden, dtype=torch.bfloat16,
                           device=dev),
            a1vq=torch.zeros(slots + 4, hidden, dtype=torch.bfloat16,
                             device=dev),
            a1=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                           device=dev),
            as1=torch.zeros(slots + 4, hidden // 128, dtype=torch.float32,
                            device=dev),
            # zeros, not empty: pad-pair rows are never written by the kernel
            # (early EXIT) yet flow through silu/scatter math with weight 0;
            # uninitialized inf/nan would poison 0*x.
            c13=torch.zeros(slots + 4, 2 * inter, dtype=torch.bfloat16,
                            device=dev),
            act=torch.zeros(slots + 4, inter, dtype=torch.bfloat16, device=dev),
            a2=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                           device=dev),
            as2=torch.zeros(slots + 4, max(inter // 128, 1),
                            dtype=torch.float32, device=dev),
            c2=torch.zeros(slots + 4, hidden, dtype=torch.bfloat16,
                           device=dev),
            desc=torch.empty(4, slots // _BLOCK, 6, dtype=torch.int64,
                             device=dev),
            # R6 mixed layers: 6 tables (w2 w13/w2, w3 w13/w2, fp4 w13/w2).
            # fp4 tables carry up to 4 sub-entries per prefill pair (4 rows
            # each; moe_w4_mm is M<=4) -> at mblock=16 that is
            # (slots//16)*4 = slots//4 entries = the same cap.
            desc6=torch.empty(6, slots // _BLOCK, 6, dtype=torch.int64,
                              device=dev),
            no_slots=torch.full((256,), -1, dtype=torch.int32, device=dev),
        )
        if _afrag_ok:
            # AFRAG destination buffers: the triton repack streams row-major
            # a1/a2 into these (single pass, no copy-back); the desc tables
            # point the GEMM at them instead of a1/a2.
            _WS.update(
                a1f=torch.zeros(slots + 4, hidden, dtype=torch.float8_e4m3fn,
                                device=dev),
                a2f=torch.zeros(slots + 4, inter, dtype=torch.float8_e4m3fn,
                                device=dev),
            )
    return _WS


import triton
import triton.language as tl


@triton.jit
def _afrag_repack_kernel(src_ptr, dst_ptr, K: tl.constexpr):
    """Row-major fp8 [pairs*16, K] -> AFRAG fragment-major, single pass.

    One program = one (pair, j=k64) 16-row x 64-byte block = 256 u32 words;
    the permutation [pair, g2, g, j, quad, t, b] -> [pair, j, g, t, quad, g2, b]
    lands each program's words in one contiguous 1 KiB dst run. Bit-identical
    to _to_fragment_major (validated), ~3x faster than the torch permute+copy
    and needs no intermediate tensor."""
    p = tl.program_id(0)
    j = tl.program_id(1)
    w = tl.arange(0, 256)
    g2 = w & 1
    quad = (w >> 1) & 3
    t = (w >> 3) & 3
    g = (w >> 5) & 7
    src_off = (p * 16 + g2 * 8 + g) * (K // 4) + j * 16 + quad * 4 + t
    dst_off = p * 16 * (K // 4) + j * 256 + w
    tl.store(dst_ptr + dst_off, tl.load(src_ptr + src_off))


def _afrag_repack(src: torch.Tensor, dst: torch.Tensor, pairs: int, K: int):
    """Repack rows [:pairs*16] of `src` (fp8 row-major) into `dst` (AFRAG)."""
    src32 = src.view(torch.uint8).view(-1).view(torch.int32)
    dst32 = dst.view(torch.uint8).view(-1).view(torch.int32)
    _afrag_repack_kernel[(pairs, K // 64)](src32, dst32, K=K)


@triton.jit
def _desc_build_kernel(
    eids_ptr, npost_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    p13b, s13b, p2b, s2b, poolb,
    p13s, s13s, p2s, s2s,
    slot_bytes, w13_bytes,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """All four moe desc tables in one launch (24 columns per pair).

    d_ptr = [4, cap, 6] i64: 0 = w2-tier w13, 1 = w2-tier w2,
    2 = w4-tier w13, 3 = w4-tier w2. A pair is routed to exactly one tier
    via the m_rows field (the other tier's kernel sees m=0 -> early EXIT).
    slot_ptr = this layer's row of the delta slot table (-1 = base tier);
    poolb = delta pool base (w13 plane at slot start, w2 at +w13_bytes).
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    is4 = slot >= 0
    m2 = tl.where(live & ~is4, mblock, 0).to(tl.int64)
    m4 = tl.where(live & is4, mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    bs13 = s13b + e * s13s
    bs2 = s2b + e * s2s
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = p13b + e * p13s, bs13, a1, as1, c13, m2
        elif gi == 1:
            b, s, a, as_, c, m = p2b + e * p2s, bs2, a2, as2, c2, m2
        elif gi == 2:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes, bs13,
                                  a1, as1, c13, m4)
        else:
            b, s, a, as_, c, m = (poolb + slot_c * slot_bytes + w13_bytes,
                                  bs2, a2, as2, c2, m4)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_basecache(
    eids_ptr, npost_ptr, slot_ptr, miss_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    poolb, slot_bytes, off_s13, off_c2, off_s2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    BLOCK: tl.constexpr,
):
    """Base-cache variant of _desc_build_kernel: the 2-bit BASE planes live in
    a GPU pool (slot sections per expert: [codes13 | sc13 | codes2 | sc2]),
    not in resident per-layer planes. A live pair whose expert is NOT resident
    (slot < 0) gets m=0 (the GEMM early-EXITs; its c13/c2 rows stay zero, so
    the pair contributes nothing) and bumps `miss_ptr` — the runner fetches
    the missing experts and replays the step. Only the w2-tier tables d[0]
    (w13 GEMM) and d[1] (w2 GEMM) are written; the w4 tier is not used with
    the base cache."""
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    slot = tl.load(slot_ptr + e, mask=mask, other=-1).to(tl.int64)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    hit = slot >= 0
    m = tl.where(live & hit, mblock, 0).to(tl.int64)
    n_miss = tl.sum(tl.where(mask & live & ~hit, 1, 0))
    tl.atomic_add(miss_ptr, n_miss)
    base = p.to(tl.int64) * mblock
    slot_c = tl.maximum(slot, 0)
    sbase = poolb + slot_c * slot_bytes
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    for gi in tl.static_range(2):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c = sbase, sbase + off_s13, a1, as1, c13
        else:
            b, s, a, as_, c = sbase + off_c2, sbase + off_s2, a2, as2, c2
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)


@triton.jit
def _desc_build_kernel_mixed(
    eids_ptr, npost_ptr, tier_ptr, slot_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1r, a2r,
    w2p13, w2s13, w2p2, w2s2,
    w3p13, w3s13, w3p2, w3s2,
    f4p13, f4s13, f4p2, f4s2,
    w2p13s, w2s13s, w2p2s, w2s2s,
    w3p13s, w3s13s, w3p2s, w3s2s,
    f4p13s, f4s13s, f4p2s, f4s2s,
    n_w2, n_w3, n_f4,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    NSUB: tl.constexpr, BLOCK: tl.constexpr,
):
    """R6 mixed per-expert desc build (<source-task>): 6 tables, one launch.

    d_ptr = [6, cap, 6] i64: 0/1 = w2-tier w13/w2 GEMM, 2/3 = w3-tier,
    4/5 = fp4-tier (moe_w4_mm; M<=4, so NSUB=mblock//4 sub-entries per
    pair, 4 rows each, at table index p*NSUB+j). Each pair routes to
    exactly ONE tier via tier_ptr[e] (0=w2 1=w3 2=fp4); the other tiers'
    entries carry m=0 (kernel early-EXIT, pointers never dereferenced).
    slot_ptr[e] = the expert's row inside its OWN tier's compact plane
    files; cross-tier clamps keep pointer math in-bounds for m=0 rows.
    fp4 tables always read ROW-MAJOR activations (a1r/a2r): moe_w4_mm
    has no afrag variant.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    t = tl.load(tier_ptr + e, mask=mask, other=0).to(tl.int64)
    slot = tl.load(slot_ptr + e, mask=mask, other=0).to(tl.int64)
    slot = tl.maximum(slot, 0)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    m2 = tl.where(live & (t == 0), mblock, 0).to(tl.int64)
    m3 = tl.where(live & (t == 1), mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    s2c = tl.minimum(slot, n_w2 - 1)
    s3c = tl.minimum(slot, n_w3 - 1)
    s4c = tl.minimum(slot, n_f4 - 1)
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = (w2p13 + s2c * w2p13s,
                                  w2s13 + s2c * w2s13s, a1, as1, c13, m2)
        elif gi == 1:
            b, s, a, as_, c, m = (w2p2 + s2c * w2p2s,
                                  w2s2 + s2c * w2s2s, a2, as2, c2, m2)
        elif gi == 2:
            b, s, a, as_, c, m = (w3p13 + s3c * w3p13s,
                                  w3s13 + s3c * w3s13s, a1, as1, c13, m3)
        else:
            b, s, a, as_, c, m = (w3p2 + s3c * w3p2s,
                                  w3s2 + s3c * w3s2s, a2, as2, c2, m3)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)
    m4 = tl.where(live & (t == 2), 4, 0).to(tl.int64)
    for j in tl.static_range(NSUB):
        sb = base + j * 4
        a1j = a1r + sb * a1_rb
        as1j = as1b + sb * as1_rb
        c13j = c13b + sb * c13_rb
        a2j = a2r + sb * a2_rb
        as2j = as2b + sb * as2_rb
        c2j = c2b + sb * c2_rb
        for gi in tl.static_range(2):
            d = d_ptr + (4 + gi) * cap6 + (p * NSUB + j) * 6
            if gi == 0:
                b, s, a, as_, c = (f4p13 + s4c * f4p13s,
                                   f4s13 + s4c * f4s13s, a1j, as1j, c13j)
            else:
                b, s, a, as_, c = (f4p2 + s4c * f4p2s,
                                   f4s2 + s4c * f4s2s, a2j, as2j, c2j)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m4, mask=mask)


@triton.jit
def _desc_build_kernel_mixed_pp(
    eids_ptr, npost_ptr, t13_ptr, s13_ptr, t2_ptr, s2_ptr, d_ptr,
    a1b, as1b, c13b, a2b, as2b, c2b,
    a1r, a2r,
    w2p13, w2s13, w2p2, w2s2,
    w3p13, w3s13, w3p2, w3s2,
    f4p13, f4s13, f4p2, f4s2,
    w2p13s, w2s13s, w2p2s, w2s2s,
    w3p13s, w3s13s, w3p2s, w3s2s,
    f4p13s, f4s13s, f4p2s, f4s2s,
    n_w2_13, n_w3_13, n_f4_13, n_w2_2, n_w3_2, n_f4_2,
    a1_rb, as1_rb, c13_rb, a2_rb, as2_rb, c2_rb,
    n_experts, pairs, cap6, mblock,
    NSUB: tl.constexpr, BLOCK: tl.constexpr,
):
    """R7PP per-PROJECTION desc build (<source-task>): 6 tables, one launch.

    Same table layout as _desc_build_kernel_mixed (0/1 = w2-tier w13/w2
    GEMM, 2/3 = w3-tier, 4/5 = fp4-tier moe_w4_mm sub-entries), but the
    fused13 GEMM (tables 0/2/4) routes via t13_ptr[e]/s13_ptr[e] and the
    down GEMM (tables 1/3/5) via t2_ptr[e]/s2_ptr[e] INDEPENDENTLY — an
    expert split across tiers (Lever B) issues its two GEMMs in different
    tier families. Slot spaces are per (tier, projection): the compact
    plane files' planes13 and planes2 row counts differ, hence six clamp
    bounds. m=0 rows early-EXIT and never dereference their pointers.
    """
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = p < pairs
    e = tl.load(eids_ptr + p, mask=mask, other=0).to(tl.int64)
    e = tl.minimum(tl.maximum(e, 0), n_experts - 1)
    t13 = tl.load(t13_ptr + e, mask=mask, other=0).to(tl.int64)
    s13v = tl.load(s13_ptr + e, mask=mask, other=0).to(tl.int64)
    t2 = tl.load(t2_ptr + e, mask=mask, other=0).to(tl.int64)
    s2v = tl.load(s2_ptr + e, mask=mask, other=0).to(tl.int64)
    s13v = tl.maximum(s13v, 0)
    s2v = tl.maximum(s2v, 0)
    npost = tl.load(npost_ptr).to(tl.int64)
    live = p < npost // mblock
    mw2_13 = tl.where(live & (t13 == 0), mblock, 0).to(tl.int64)
    mw3_13 = tl.where(live & (t13 == 1), mblock, 0).to(tl.int64)
    mw2_2 = tl.where(live & (t2 == 0), mblock, 0).to(tl.int64)
    mw3_2 = tl.where(live & (t2 == 1), mblock, 0).to(tl.int64)
    base = p.to(tl.int64) * mblock
    a1 = a1b + base * a1_rb
    as1 = as1b + base * as1_rb
    c13 = c13b + base * c13_rb
    a2 = a2b + base * a2_rb
    as2 = as2b + base * as2_rb
    c2 = c2b + base * c2_rb
    sw2_13 = tl.minimum(s13v, n_w2_13 - 1)
    sw3_13 = tl.minimum(s13v, n_w3_13 - 1)
    sf4_13 = tl.minimum(s13v, n_f4_13 - 1)
    sw2_2 = tl.minimum(s2v, n_w2_2 - 1)
    sw3_2 = tl.minimum(s2v, n_w3_2 - 1)
    sf4_2 = tl.minimum(s2v, n_f4_2 - 1)
    for gi in tl.static_range(4):
        d = d_ptr + gi * cap6 + p * 6
        if gi == 0:
            b, s, a, as_, c, m = (w2p13 + sw2_13 * w2p13s,
                                  w2s13 + sw2_13 * w2s13s, a1, as1, c13,
                                  mw2_13)
        elif gi == 1:
            b, s, a, as_, c, m = (w2p2 + sw2_2 * w2p2s,
                                  w2s2 + sw2_2 * w2s2s, a2, as2, c2,
                                  mw2_2)
        elif gi == 2:
            b, s, a, as_, c, m = (w3p13 + sw3_13 * w3p13s,
                                  w3s13 + sw3_13 * w3s13s, a1, as1, c13,
                                  mw3_13)
        else:
            b, s, a, as_, c, m = (w3p2 + sw3_2 * w3p2s,
                                  w3s2 + sw3_2 * w3s2s, a2, as2, c2,
                                  mw3_2)
        tl.store(d + 0, a, mask=mask)
        tl.store(d + 1, as_, mask=mask)
        tl.store(d + 2, b, mask=mask)
        tl.store(d + 3, s, mask=mask)
        tl.store(d + 4, c, mask=mask)
        tl.store(d + 5, m, mask=mask)
    m4_13 = tl.where(live & (t13 == 2), 4, 0).to(tl.int64)
    m4_2 = tl.where(live & (t2 == 2), 4, 0).to(tl.int64)
    for j in tl.static_range(NSUB):
        sb = base + j * 4
        a1j = a1r + sb * a1_rb
        as1j = as1b + sb * as1_rb
        c13j = c13b + sb * c13_rb
        a2j = a2r + sb * a2_rb
        as2j = as2b + sb * as2_rb
        c2j = c2b + sb * c2_rb
        for gi in tl.static_range(2):
            d = d_ptr + (4 + gi) * cap6 + (p * NSUB + j) * 6
            if gi == 0:
                b, s, a, as_, c, m = (f4p13 + sf4_13 * f4p13s,
                                      f4s13 + sf4_13 * f4s13s,
                                      a1j, as1j, c13j, m4_13)
            else:
                b, s, a, as_, c, m = (f4p2 + sf4_2 * f4p2s,
                                      f4s2 + sf4_2 * f4s2s,
                                      a2j, as2j, c2j, m4_2)
            tl.store(d + 0, a, mask=mask)
            tl.store(d + 1, as_, mask=mask)
            tl.store(d + 2, b, mask=mask)
            tl.store(d + 3, s, mask=mask)
            tl.store(d + 4, c, mask=mask)
            tl.store(d + 5, m, mask=mask)


def _launch(tier: str, K: int, desc: torch.Tensor, n_rows: int, pairs: int,
            stream):
    fn = _fns[(tier, K)]
    args = [ctypes.c_uint64(desc.data_ptr()),
            ctypes.c_uint32(K),
            ctypes.c_uint32(K // 64),
            ctypes.c_uint32(n_rows * 2),
            ctypes.c_uint32(K // 128)]
    argv = (ctypes.c_void_p * len(args))(
        *[ctypes.cast(ctypes.byref(x), ctypes.c_void_p) for x in args])
    _ck(_driver().cuLaunchKernel(fn, n_rows // 16, pairs, 1,
                                 _nwarp_for_k(K) * 32, 1, 1, 0,
                                 stream, argv, None), "launch")


def _moe_w2_forward(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    if _decode_graph_wanted(x):
        return _moe_w2_forward_graphed(x, topk_weights, topk_ids, layer_key)
    from vllm.model_executor.layers.quantization.utils import prefill_timers
    with prefill_timers.span("moe_w2"):
        return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)


# --------------------------------------------------------------------------
# Per-layer decode CUDA graphs (<source-task> MOE-BOOKKEEPING-FUSION)
#
# py-spy on the live 6.59 tok/s serve showed ~100ms/token spent in the
# per-layer pre-GEMM bookkeeping micro-ops (moe_align_block_size, group
# quant, torch.where + 3x index_select, desc build) launched from python
# 43x per token.  At decode the whole _moe_w2_forward_timed body is
# static-shape and already graph-safe (the num_post.item() sync was removed
# earlier; tier code has is_current_stream_capturing() handling), so we
# capture ONE CUDA graph per (layer_key, T, top_k) and replay it.  This is
# deliberately per-layer — the earlier iter3 attempt graphed the whole step
# including dynamic router branching and regressed.
#
# Opt-in via VLLM_MOE_W2_DECODE_GRAPH=1 (default off = incumbent behavior,
# bit-for-bit).  Replay launches the exact same kernels with the same launch
# parameters on the same buffers, so outputs are bitwise-equal to eager.
#
# Safety notes:
# - _WS scratch buffers are captured by address.  _workspaces() only ever
#   REPLACES them when a larger shape arrives, so we keep a strong reference
#   to every tensor alive at capture time (ws_refs) — a later prefill regrow
#   then cannot free the captured buffers, and the graph keeps its own
#   consistent scratch (ordered on the same stream, so no cross-talk).
# - Weight/tier tables (st[...]) are long-lived module state whose CONTENTS
#   may mutate (delta fetches); the captured kernels read them by pointer at
#   replay, same as eager.
# - Each graph gets a private memory pool (intermediates at decode shapes
#   are ~100s of KiB; sharing pools would impose replay-order constraints).
# - Any capture failure permanently falls back to eager for that key.
_GRAPHS: dict = {}
_GRAPH_WARM: dict = {}
_GRAPH_BAD: set = set()
_GRAPH_REPLAY_LOGGED: set = set()
_VQ_DISPATCH_LOGGED: set = set()
_GRAPH_WARMUP_STEPS = 3


def _decode_graph_wanted(x: torch.Tensor) -> bool:
    if os.getenv("VLLM_MOE_W2_DECODE_GRAPH", "0") != "1":
        return False
    max_t = int(os.getenv("VLLM_MOE_W2_DECODE_GRAPH_MAX_T", "8"))
    return (x.shape[0] <= max_t
            and not torch.cuda.is_current_stream_capturing())


def decode_graph_key(layer_key, x, topk_ids) -> tuple[int, int, int]:
    """Key private graphs by layer and exact decode batch shape."""
    return (layer_key, x.shape[0], topk_ids.shape[1])


def _capture_layer_graph(x, topk_weights, topk_ids, layer_key):
    sx = x.detach().clone()
    stw = topk_weights.detach().clone()
    stids = topk_ids.detach().clone()
    # canonical capture recipe: warm up on a side stream first so lazy state
    # (triton JIT, cublas handles, sentinels) is materialized outside capture
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            _moe_w2_forward_timed(sx, stw, stids, layer_key)
    torch.cuda.current_stream().wait_stream(side)
    ws_refs = list(_WS.values())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _moe_w2_forward_timed(sx, stw, stids, layer_key)
    return {
        "graph": graph, "x": sx, "tw": stw, "tids": stids, "out": out,
        "ws_refs": ws_refs,
    }


def _moe_w2_forward_graphed(x, topk_weights, topk_ids, layer_key):
    key = decode_graph_key(layer_key, x, topk_ids)
    ent = _GRAPHS.get(key)
    if ent is None:
        if key in _GRAPH_BAD:
            return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)
        warm = _GRAPH_WARM.get(key, 0)
        if warm < _GRAPH_WARMUP_STEPS:
            _GRAPH_WARM[key] = warm + 1
            return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)
        try:
            ent = _capture_layer_graph(x, topk_weights, topk_ids, layer_key)
        except Exception:
            logger.exception(
                "moe_w2_cubit: decode-graph capture FAILED for key=%s — "
                "falling back to eager for this key", key)
            _GRAPH_BAD.add(key)
            return _moe_w2_forward_timed(x, topk_weights, topk_ids, layer_key)
        _GRAPHS[key] = ent
        if not _WS.get("_decode_graph_sentinel"):
            _WS["_decode_graph_sentinel"] = True
            logger.info(
                "moe_w2_cubit: DECODE-GRAPH ON-PATH sentinel — captured "
                "per-layer CUDA graph key=%s (T=%d top_k=%d)",
                key, x.shape[0], topk_ids.shape[1])
    ent["x"].copy_(x)
    ent["tw"].copy_(topk_weights)
    ent["tids"].copy_(topk_ids)
    ent["graph"].replay()
    if layer_key == 0 and key not in _GRAPH_REPLAY_LOGGED:
        _GRAPH_REPLAY_LOGGED.add(key)
        logger.info(
            "moe_w2_cubit: DECODE-GRAPH REPLAY PROBE key=%s T=%d top_k=%d",
            key, x.shape[0], topk_ids.shape[1])
    return ent["out"].clone()


def vq_valid_rows(tokens: int, mblock: int) -> int:
    """Return the live-row bound for each compact decode expert block."""
    return min(tokens, mblock) if mblock == 4 else mblock


def vq_pair_launch_bound(tokens: int, top_k: int, allocated_pairs: int) -> int:
    """Graph-safe upper bound for compacted MoE expert blocks.

    Every live expert block contains at least one routed assignment, so the
    compacted block count cannot exceed ``tokens * top_k``.  Bounding by that
    host-known shape removes the per-layer ``num_post.item()`` device sync while
    remaining exact for batch-1 decode (top-k experts are unique per token).
    """
    return min(allocated_pairs, tokens * top_k)


def _moe_w2_forward_timed(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )

    st = _LAYERS[layer_key]
    vq_mixed = bool(st.get("vq_mixed", False))
    T, H = x.shape
    top_k = topk_ids.shape[1]
    dev = x.device
    stream = ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream)

    # decode-sized calls use the proven 4-token kernel + delta tier;
    # prefill-sized calls use the MC4 kernel (16 tokens per pair-entry = full
    # QMMA-M, plane reads amortized 4x, ~1.5x over MC2) on the 2-bit base only.
    # 96 = the largest cudagraph capture size: anything above is necessarily a
    # prefill chunk; short tail chunks keep the delta-quality path.
    prefill = T > 96
    mblock = 16 if prefill else _BLOCK
    sorted_ids, expert_blocks, num_post = moe_align_block_size(
        topk_ids, mblock, st["E"])
    slots = sorted_ids.numel()
    pairs = slots // mblock
    # st["K2"] = per-rank expert intermediate I (w2 contraction), st["K13"] =
    # hidden H (w13 contraction) -> size the workspaces for the model's shapes
    # (and correctly under tensor parallelism).
    ws = _workspaces(slots, T, dev, inter=st["K2"], hidden=st["K13"])

    # ---- activation quant (group-128) into the padded buffer; the buffer's
    # last row is the permanent zero pad row for filler slots.
    xq = ws["xq"]
    pad_row = xq.shape[0] - 1
    _, xs = per_token_group_quant_fp8(x, 128, out_q=xq[:T])
    ws["xs"][:T] = xs
    valid = sorted_ids < T * top_k
    rows = torch.where(valid, sorted_ids // top_k,
                       torch.full_like(sorted_ids, pad_row))
    torch.index_select(xq.view(torch.uint8), 0, rows,
                       out=ws["a1"][:slots].view(torch.uint8))
    torch.index_select(ws["xs"], 0, rows, out=ws["as1"][:slots])
    if vq_mixed:
        ws["xv"][:T].copy_(x)
        ws["xv"][T].zero_()
        torch.index_select(ws["xv"], 0, rows, out=ws["a1vq"][:slots])

    # ---- desc tables in ONE triton launch
    from vllm.model_executor.layers.quantization.utils import moe_w2_delta
    base_mode = st.get("base", False)
    is_w3 = st.get("w3", False)
    mixed = st.get("mixed", False)
    # AFRAG (prefill): the GEMM reads fragment-major activations from the
    # dedicated a1f/a2f buffers (filled by the single-pass triton repack
    # below); point the desc 'a' fields there. w4 tables are decode-only,
    # so redirecting the shared base in prefill is safe.
    if mixed:
        # mixed layers launch BOTH afrag families; engage only when the w3
        # afrag cubins for this layer's shapes are loaded too (w2 afrag
        # presence is what _afrag_ok already asserts). The fp4 tables read
        # row-major a1/a2 regardless (moe_w4_mm has no afrag sibling).
        use_afrag = (not vq_mixed and prefill and _afrag_ok
                     and ("w3mc4afrag", st["K13"]) in _fns
                     and ("w3mc4afrag", st["K2"]) in _fns)
    elif is_w3:
        # W3 layers use the moe_w3_mm afrag siblings; engage only when the
        # w3 afrag cubins for this layer's shapes are loaded (_afrag_ok
        # gates the a1f/a2f workspaces' existence).
        use_afrag = (prefill and _afrag_ok
                     and ("w3mc4afrag", st["K13"]) in _fns
                     and ("w3mc4afrag", st["K2"]) in _fns)
    else:
        use_afrag = prefill and _afrag_ok
    a1_base = ws["a1f"] if use_afrag else ws["a1"]
    a2_base = ws["a2f"] if use_afrag else ws["a2"]
    d = ws["desc"]
    cap = d.shape[1]
    miss_rows = None
    if mixed:
        # R6 per-expert mixed tiers: desc + launches fully handled here;
        # the single-tier launch block below is skipped.
        tier = None
        d6 = ws["desc6"]
        cap6t = d6.shape[1]
        nsub = mblock // 4
        tw2, tw3, tf4 = (st["tiers"][t] for t in ("w2", "w3", "fp4"))
        is_pp = st.get("pp", False)
        if is_pp:
            _desc_build_kernel_mixed_pp[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post,
                st["tier_of13"], st["slot_of13"],
                st["tier_of2"], st["slot_of2"], d6,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(),
                ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                tw2["planes13"].data_ptr(), tw2["sc13"].data_ptr(),
                tw2["planes2"].data_ptr(), tw2["sc2"].data_ptr(),
                tw3["planes13"].data_ptr(), tw3["sc13"].data_ptr(),
                tw3["planes2"].data_ptr(), tw3["sc2"].data_ptr(),
                tf4["planes13"].data_ptr(), tf4["sc13"].data_ptr(),
                tf4["planes2"].data_ptr(), tf4["sc2"].data_ptr(),
                tw2["planes13"].shape[1], tw2["sc13"].shape[1],
                tw2["planes2"].shape[1], tw2["sc2"].shape[1],
                tw3["planes13"].shape[1], tw3["sc13"].shape[1],
                tw3["planes2"].shape[1], tw3["sc2"].shape[1],
                tf4["planes13"].shape[1], tf4["sc13"].shape[1],
                tf4["planes2"].shape[1], tf4["sc2"].shape[1],
                tw2["planes13"].shape[0], tw3["planes13"].shape[0],
                tf4["planes13"].shape[0],
                tw2["planes2"].shape[0], tw3["planes2"].shape[0],
                tf4["planes2"].shape[0],
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, cap6t * 6, mblock, NSUB=nsub, BLOCK=256)
        else:
            _desc_build_kernel_mixed[(triton.cdiv(pairs, 256),)](
                expert_blocks, num_post, st["tier_of"], st["slot_of"], d6,
                a1_base.data_ptr(), ws["as1"].data_ptr(),
                ws["c13"].data_ptr(),
                a2_base.data_ptr(), ws["as2"].data_ptr(),
                ws["c2"].data_ptr(),
                ws["a1"].data_ptr(), ws["a2"].data_ptr(),
                tw2["planes13"].data_ptr(), tw2["sc13"].data_ptr(),
                tw2["planes2"].data_ptr(), tw2["sc2"].data_ptr(),
                tw3["planes13"].data_ptr(), tw3["sc13"].data_ptr(),
                tw3["planes2"].data_ptr(), tw3["sc2"].data_ptr(),
                tf4["planes13"].data_ptr(), tf4["sc13"].data_ptr(),
                tf4["planes2"].data_ptr(), tf4["sc2"].data_ptr(),
                tw2["planes13"].shape[1], tw2["sc13"].shape[1],
                tw2["planes2"].shape[1], tw2["sc2"].shape[1],
                tw3["planes13"].shape[1], tw3["sc13"].shape[1],
                tw3["planes2"].shape[1], tw3["sc2"].shape[1],
                tf4["planes13"].shape[1], tf4["sc13"].shape[1],
                tf4["planes2"].shape[1], tf4["sc2"].shape[1],
                tw2["planes13"].shape[0], tw3["planes13"].shape[0],
                tf4["planes13"].shape[0],
                st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
                (st["K2"] // 128) * 4, 2 * st["K13"],
                st["E"], pairs, cap6t * 6, mblock, NSUB=nsub, BLOCK=256)
        if vq_mixed:
            from vllm.model_executor.layers.quantization.utils import moe_vq_triton
            # moe_align_block_size returns an over-allocated expert-block table.
            # In the opt-in fast path, a compacted block contains at least one
            # routed assignment, so host-known T*top_k is a graph-safe upper
            # bound.  Flag-off preserves the incumbent full descriptor grid.
            vq_pairs = (
                vq_pair_launch_bound(T, top_k, pairs)
                if os.getenv("VLLM_MOE_VQ_FAST", "0") == "1"
                else pairs
            )
            if mblock == 4 and not _WS.get("_vq_decode_geometry_sentinel"):
                _WS["_vq_decode_geometry_sentinel"] = True
                logger.info(
                    "moe_w2_cubit: IQ3 VQ decode geometry — T=%d top_k=%d "
                    "slots=%d pairs=%d live_pairs=%d",
                    T, top_k, slots, pairs, vq_pairs)
            if not _WS.get("_vq_sentinel"):
                _WS["_vq_sentinel"] = True
                logger.info(
                    "moe_w2_cubit: IQ3 exact VQ ON-PATH sentinel — layer_key=%d "
                    "mblock=%d K13=%d K2=%d",
                    layer_key, mblock, st["K13"], st["K2"])
            if (mblock == 4
                    and os.getenv("VLLM_MOE_VQ_CUDA_WARP", "0") == "1"
                    and not _WS.get("_vq_cuda_warp_sentinel")):
                _WS["_vq_cuda_warp_sentinel"] = True
                logger.info(
                    "moe_w2_cubit: IQ3 CUDA WARP-GEMV ON-PATH sentinel — "
                    "layer_key=%d pairs=%d K13=%d K2=%d",
                    layer_key, vq_pairs, st["K13"], st["K2"])
            # Every token can route to a given expert at most once, so an
            # expert block contains at most T valid assignments. moe_align
            # packs them before filler rows; min(T, mblock) is therefore the
            # exact global row bound for small-M warp dispatch.
            vq_valid_m = vq_valid_rows(T, mblock)
            if layer_key == 0:
                dispatch_label = moe_vq_triton.dispatch_probe_label(
                    st["vq13"], mblock, vq_valid_m)
                if dispatch_label not in _VQ_DISPATCH_LOGGED:
                    _VQ_DISPATCH_LOGGED.add(dispatch_label)
                    logger.info(
                        "moe_w2_cubit: VQ DISPATCH PROBE path=%s T=%d "
                        "mblock=%d valid_m=%d max_m=%s",
                        dispatch_label, T, mblock, vq_valid_m,
                        os.getenv("VLLM_MOE_VQ_CUDA_WARP_MAX_M", "1"))
            moe_vq_triton.vq_gemm(
                ws["a1vq"][:slots], ws["c13"][:slots], expert_blocks[:vq_pairs],
                num_post, st["tier_of13"], st["vq13"], n=st["N13"],
                k=st["K13"], mblock=mblock, valid_m=vq_valid_m)
            if st["has_fp4_13"]:
                _launch("w4", st["K13"], d6[4], st["N13"],
                        pairs * nsub, stream)
            act = ws["act"][:slots]
            torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
            _, qs2m = per_token_group_quant_fp8(
                act, 128, out_q=ws["a2"][:slots])
            ws["as2"][:slots] = qs2m
            moe_vq_triton.vq_gemm(
                act, ws["c2"][:slots], expert_blocks[:vq_pairs], num_post,
                st["tier_of2"], st["vq2"], n=st["N2"], k=st["K2"],
                mblock=mblock, valid_m=vq_valid_m)
            if st["has_fp4_2"]:
                _launch("w4", st["K2"], d6[5], st["N2"],
                        pairs * nsub, stream)
        else:
            wt2 = ("w2mc4afrag" if use_afrag else "w2mc4") if prefill else "w2"
            wt3 = ("w3mc4afrag" if use_afrag else "w3mc4") if prefill else "w3"
            if not _WS.get("_mixed_sentinel"):
                _WS["_mixed_sentinel"] = True
                logger.info(
                    "moe_w2_cubit: MIXED%s tier ON-PATH sentinel — layer_key=%d "
                    "w2tier=%s w3tier=%s fp4=moe_w4_mm nsub=%d K13=%d K2=%d",
                    " PP" if is_pp else "", layer_key, wt2, wt3, nsub,
                    st["K13"], st["K2"])
            if use_afrag:
                _afrag_repack(ws["a1"], ws["a1f"], pairs, st["K13"])
            _launch(wt2, st["K13"], d6[0], st["N13"], pairs, stream)
            _launch(wt3, st["K13"], d6[2], st["N13"], pairs, stream)
            _launch("w4", st["K13"], d6[4], st["N13"], pairs * nsub, stream)
            act = ws["act"][:slots]
            torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
            _, qs2m = per_token_group_quant_fp8(act, 128, out_q=ws["a2"][:slots])
            ws["as2"][:slots] = qs2m
            if use_afrag:
                _afrag_repack(ws["a2"], ws["a2f"], pairs, st["K2"])
            _launch(wt2, st["K2"], d6[1], st["N2"], pairs, stream)
            _launch(wt3, st["K2"], d6[3], st["N2"], pairs, stream)
            _launch("w4", st["K2"], d6[5], st["N2"], pairs * nsub, stream)
    elif base_mode:
        # BASE cache: 2-bit planes come from the base tier's GPU pool; a live
        # pair with a non-resident expert contributes zero and bumps the miss
        # counter (runner fetches + replays). Prefill fetches its whole layer
        # working set up-front (outside capture) — decode must stay
        # capturable, so misses are handled post-hoc.
        btier = moe_w2_delta._BASE_TIER
        if torch.cuda.is_current_stream_capturing():
            btier.notify_capture()
        elif prefill:
            btier.ensure_resident(layer_key, topk_ids.view(-1))
        moe_w2_delta.mark_seen(btier.seen[layer_key], topk_ids.view(-1).long())
        if layer_key == 0:
            # per-step counter reset, in-graph (layer 0 runs first each step)
            btier.miss_count.zero_()
        slot_row = btier.slot_table[layer_key]
        _desc_build_kernel_basecache[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, slot_row,
            btier.miss_count, d,
            a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
            a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
            btier.pool.data_ptr(), btier.slot_bytes,
            st["off_s13"], st["off_c2"], st["off_s2"],
            st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
            (st["K2"] // 128) * 4, 2 * st["K13"],
            st["E"], pairs, cap * 6, mblock, BLOCK=256)
        # Miss pairs get scatter weight 0: the GEMMs early-EXIT on m=0 and
        # never write their c13/c2 rows, but those workspace rows hold STALE
        # values from a previous forward — zeroing the WEIGHT (not the rows)
        # makes the miss contribution an exact 0 for free. Graph-safe (pure
        # tensor ops on captured buffers).
        e_pair = expert_blocks.to(torch.long).clamp_(0, st["E"] - 1)
        resident = (slot_row[e_pair] >= 0)
        miss_rows = resident.repeat_interleave(mblock)[:slots]
        tier = None                      # no FP4 delta with the base cache
    else:
        tier = moe_w2_delta._TIER       # peek only; created by the plane builder
        if tier is not None and not prefill:
            if torch.cuda.is_current_stream_capturing():
                tier.notify_capture()
            slot_row = tier.slot_table[layer_key]
            pool_ptr = tier.pool.data_ptr()
            moe_w2_delta.mark_seen(tier.seen[layer_key],
                                   topk_ids.view(-1).long())
        else:
            if tier is not None:
                moe_w2_delta.mark_seen(tier.seen[layer_key],
                                       topk_ids.view(-1).long())
            slot_row = ws["no_slots"]
            pool_ptr = ws["a1"].data_ptr()      # never dereferenced (m4=0)
        _desc_build_kernel[(triton.cdiv(pairs, 256),)](
            expert_blocks, num_post, slot_row, d,
            a1_base.data_ptr(), ws["as1"].data_ptr(), ws["c13"].data_ptr(),
            a2_base.data_ptr(), ws["as2"].data_ptr(), ws["c2"].data_ptr(),
            st["planes13"].data_ptr(), st["sc13"].data_ptr(),
            st["planes2"].data_ptr(), st["sc2"].data_ptr(), pool_ptr,
            st["planes13"].shape[1], st["sc13"].shape[1],
            st["planes2"].shape[1], st["sc2"].shape[1],
            (tier.slot_bytes if tier is not None else moe_w2_delta.SLOT_BYTES),
            (tier.w13_bytes if tier is not None else moe_w2_delta.W13_BYTES),
            # row strides (bytes). H-side: a1 fp8 [H], as1 f32 [H/128], c2 bf16
            # [H]. per-rank intermediate side: c13 bf16 [2I], a2 fp8 [I], as2
            # f32 [I/128]. K13 = H, K2 = I -> identical to the old literals on
            # DS4 TP1 (H=4096, I=2048); GLM-5.x gets H=6144, TP shards shrink I.
            st["K13"], (st["K13"] // 128) * 4, 4 * st["K2"], st["K2"],
            (st["K2"] // 128) * 4, 2 * st["K13"],
            st["E"], pairs, cap * 6, mblock, BLOCK=256)

    # ---- w13 GEMMs (both tiers) -> fused silu*up -> quant -> w2 GEMMs
    # AFRAG prefill: single-pass triton repack row-major a1/a2 -> fragment-major
    # a1f/a2f (desc built against a1f/a2f above) so the GEMM loads each m16k32
    # A-fragment in one LDG.128. Numerics bit-identical to mc4.
    # (mixed layers issued their 6-table launches inside the branch above.)
    if not mixed:
        if is_w3:
            w2tier = ("w3mc4afrag" if use_afrag else "w3mc4") if prefill else "w3"
            # one-time on-path sentinel (exact-site proof: the serve really
            # launches moe_w3_mm, not a silent W2 fallback)
            if not _WS.get("_w3_sentinel"):
                _WS["_w3_sentinel"] = True
                logger.info("moe_w2_cubit: W3 tier ON-PATH sentinel — layer_key=%d"
                            " tier=%s K13=%d K2=%d", layer_key, w2tier,
                            st["K13"], st["K2"])
        else:
            w2tier = ("w2mc4afrag" if use_afrag else "w2mc4") if prefill else "w2"
        # AFRAG repacks COMPLETE 16-row tiles. `slots` is moe_align's OVER-ALLOCATED
        # row count (sorted_ids.numel() = topk*T + E*15), NOT a multiple of 16; the
        # desc/kernel only ever touch the first `pairs*16` rows (num_post <= pairs*16),
        # so repack exactly that tile-aligned region. Rows [pairs*16:slots] are unused
        # filler (never read). Capacity is fine: pairs*16 <= slots <= a1.shape[0]-4.
        if use_afrag:
            _afrag_repack(ws["a1"], ws["a1f"], pairs, st["K13"])
        _launch(w2tier, st["K13"], d[0], st["N13"], pairs, stream)
        if tier is not None and not prefill:
            _launch("w4", st["K13"], d[2], st["N13"], pairs, stream)
        act = ws["act"][:slots]
        torch.ops._C.silu_and_mul(act, ws["c13"][:slots])
        _, qs2 = per_token_group_quant_fp8(act, 128, out_q=ws["a2"][:slots])
        ws["as2"][:slots] = qs2
        if use_afrag:
            _afrag_repack(ws["a2"], ws["a2f"], pairs, st["K2"])
        _launch(w2tier, st["K2"], d[1], st["N2"], pairs, stream)
        if tier is not None and not prefill:
            _launch("w4", st["K2"], d[3], st["N2"], pairs, stream)

    # ---- weighted unpermute (pad slots masked out), DETERMINISTIC.
    # The old `out.index_add_(0, rows, c2*w)` scattered with atomics, so the
    # f32 accumulation ORDER varied run-to-run: identical inputs wobbled by
    # up to ~1.6e-2 abs on prefill, and single-token probes produced a small
    # set of bit-distinct logit variants — the root cause of the "greedy
    # decode is not reproducible" investigation (PP_DETERMINISM.md; it was
    # never PP-specific). Deterministic scheme: every VALID slot owns a
    # unique (token, j) coordinate (valid sorted_ids are a permutation of
    # token*top_k + j), so index_copy_ into [T*top_k (+1 dump row), H] has no
    # write collisions except filler slots, which all target the discarded
    # dump row. The final sum(dim=1) reduces top_k in a fixed order.
    # Static shapes + no host branches -> cudagraph-capture-safe.
    w = topk_weights.reshape(-1)[sorted_ids.clamp(max=T * top_k - 1)]
    w = torch.where(valid, w, torch.zeros_like(w)).to(torch.float32)
    if miss_rows is not None:
        # base cache: rows of non-resident pairs hold stale workspace values
        # (their GEMMs early-EXITed) — zero their scatter weight so a miss
        # contributes exactly nothing (the replay recomputes them properly).
        w = w * miss_rows.to(torch.float32)
    dump = T * top_k                       # collision row for filler slots
    dst = torch.where(valid, sorted_ids,
                      torch.full_like(sorted_ids, dump)).long()
    gath = torch.zeros(dump + 1, H, dtype=torch.float32, device=dev)
    gath.index_copy_(0, dst, ws["c2"][:slots].float() * w.unsqueeze(1))
    return gath[:dump].view(T, top_k, H).sum(dim=1).to(x.dtype)


def _moe_w2_forward_fake(
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    layer_key: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    "moe_w2_forward",
    _moe_w2_forward,
    fake_impl=_moe_w2_forward_fake,
)


def moe_w2_forward(x, topk_weights, topk_ids, layer_key):
    return torch.ops.vllm.moe_w2_forward(x, topk_weights, topk_ids, layer_key)


@functools.cache
def ready() -> bool:
    return enabled() and _ensure_ready()
