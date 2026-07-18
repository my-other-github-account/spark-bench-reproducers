// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 @banana_bae and contributors
//
// Decode-only learned-VQ GEMV for canonical row-major packed d4/d8 planes.
//
// Geometry deliberately differs from the retired K=32/barrier prototype:
//   * one warp owns one output row;
//   * 16 warps/CTA amortize one full-K activation staging pass;
//   * each lane decodes eight consecutive packed indices per iteration;
//   * a two-BF16 pad per eight-code activation chunk removes shared-bank aliasing;
//   * FP16 d4/d8 codewords are gathered through the read-only/L1 path;
//   * K is reduced once, warp-locally, with no split-K workspace or atomics.
//
// The canonical weights are reconstructed as BF16 before scalar FMA, matching
// the incumbent operand precision. The reduction order is intentionally not the
// tensor-core order used by the Triton oracle, so integration uses a calibrated
// BF16 tolerance rather than claiming bit identity.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>

namespace {

constexpr int kWarpsPerBlock = 16;
constexpr int kThreads = kWarpsPerBlock * 32;
constexpr int kCodesPerLaneItem = 8;
constexpr int kPadBf16PerItem = 2;
constexpr int kRowStride = 4;
constexpr int kComputeM = 1;
constexpr unsigned kFullWarp = 0xffffffffu;

__device__ __forceinline__ uint32_t load_cg_u8(const uint8_t* ptr) {
  uint32_t value;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(value) : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint32_t load_cg_u32(const uint32_t* ptr) {
  uint32_t value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(ptr));
  return value;
}

__device__ __forceinline__ uint32_t load_packed_index(
    const uint8_t* row, int64_t row_bytes, int group, int index_bits) {
  const int bit = group * index_bits;
  const int byte = bit >> 3;
  const int shift = bit & 7;
  uint32_t word = load_cg_u8(row + byte);
  if (byte + 1 < row_bytes) {
    word |= load_cg_u8(row + byte + 1) << 8;
  }
  if (byte + 2 < row_bytes) {
    word |= load_cg_u8(row + byte + 2) << 16;
  }
  return (word >> shift) & ((1u << index_bits) - 1u);
}

__device__ __forceinline__ uint32_t extract_code(
    uint32_t w0, uint32_t w1, uint32_t w2, int code_in_item, int index_bits) {
  const int bit = code_in_item * index_bits;
  const uint32_t mask = (1u << index_bits) - 1u;
  if (bit < 32) {
    const uint64_t pair = static_cast<uint64_t>(w0) |
        (static_cast<uint64_t>(w1) << 32);
    return static_cast<uint32_t>((pair >> bit) & mask);
  }
  const uint64_t pair = static_cast<uint64_t>(w1) |
      (static_cast<uint64_t>(w2) << 32);
  return static_cast<uint32_t>((pair >> (bit - 32)) & mask);
}

__device__ __forceinline__ void load_eight_codes(
    const uint8_t* ptr,
    int bytes_available,
    int index_bits,
    uint32_t (&codes)[kCodesPerLaneItem]) {
  // Fast path: four aligned cg.u32 loads instead of 8..12 scalar byte loads.
  // The last lane item falls back to exact bounded bytes, so no mmap overread.
  uint32_t w0 = 0;
  uint32_t w1 = 0;
  uint32_t w2 = 0;
  if (bytes_available >= 16) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
    const int shift = static_cast<int>((address & 3u) * 8u);
    const auto* base = reinterpret_cast<const uint32_t*>(address & ~uintptr_t{3});
    const uint32_t a = load_cg_u32(base + 0);
    const uint32_t b = load_cg_u32(base + 1);
    const uint32_t c = load_cg_u32(base + 2);
    if (shift == 0) {
      w0 = a;
      w1 = b;
      w2 = c;
    } else {
      const uint32_t d = load_cg_u32(base + 3);
      w0 = (a >> shift) | (b << (32 - shift));
      w1 = (b >> shift) | (c << (32 - shift));
      w2 = (c >> shift) | (d << (32 - shift));
    }
  } else {
#pragma unroll
    for (int byte = 0; byte < 12; ++byte) {
      if (byte < index_bits) {
        const uint32_t value = load_cg_u8(ptr + byte);
        if (byte < 4) {
          w0 |= value << (byte * 8);
        } else if (byte < 8) {
          w1 |= value << ((byte - 4) * 8);
        } else {
          w2 |= value << ((byte - 8) * 8);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kCodesPerLaneItem; ++i) {
    codes[i] = extract_code(w0, w1, w2, i, index_bits);
  }
}

union Half4 {
  uint2 words;
  half values[4];
};

union Half8 {
  uint4 words;
  half values[8];
};

__device__ __forceinline__ float rounded_bf16_weight(half value, float scale) {
  const float reconstructed = __half2float(value) * scale;
  return __bfloat162float(__float2bfloat16_rn(reconstructed));
}

template <int D>
__device__ __forceinline__ void accumulate_item(
    const half* __restrict__ codebook,
    const uint32_t (&codes)[kCodesPerLaneItem],
    const __nv_bfloat16* __restrict__ x_smem,
    int item,
    float scale0,
    float scale1,
    float& acc) {
  constexpr int kItemElems = kCodesPerLaneItem * D;
  const int x_base = item * (kItemElems + kPadBf16PerItem);
#pragma unroll
  for (int code_i = 0; code_i < kCodesPerLaneItem; ++code_i) {
    const float scale = (D == 8 && code_i >= 4) ? scale1 : scale0;
    if constexpr (D == 4) {
      // The immutable per-expert table is only 2..64 KiB. __ldg is deliberate:
      // real GB10 layer-042 A/B measured the otherwise-identical ld.global.cg
      // variant at 2.546 ms versus 0.235 ms here. Packed codes and scales keep
      // their cache-global loads so the table, not the streaming row pack, owns
      // read-only/L1 capacity.
      const Half4 packed = {
          __ldg(reinterpret_cast<const uint2*>(codebook + codes[code_i] * D))};
#pragma unroll
      for (int component = 0; component < D; ++component) {
        const float weight = rounded_bf16_weight(packed.values[component], scale);
        const int x_index = x_base + code_i * D + component;
        const float x = __bfloat162float(x_smem[x_index]);
        acc = fmaf(x, weight, acc);
      }
    } else {
      const Half8 packed = {
          __ldg(reinterpret_cast<const uint4*>(codebook + codes[code_i] * D))};
#pragma unroll
      for (int component = 0; component < D; ++component) {
        const float weight = rounded_bf16_weight(packed.values[component], scale);
        const int x_index = x_base + code_i * D + component;
        const float x = __bfloat162float(x_smem[x_index]);
        acc = fmaf(x, weight, acc);
      }
    }
  }
}

__global__ void vq_warp_gemv_kernel(
    const __nv_bfloat16* __restrict__ a,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ expert_blocks,
    const int32_t* __restrict__ num_post,
    const int32_t* __restrict__ kind,
    const uint8_t* __restrict__ codes,
    const uint8_t* __restrict__ scales,
    const half* __restrict__ codebooks,
    const int64_t* __restrict__ code_offset,
    const int64_t* __restrict__ scale_offset,
    const int32_t* __restrict__ code_row_bytes,
    const uint8_t* __restrict__ dimension,
    const uint8_t* __restrict__ bits,
    const int64_t* __restrict__ cb_offset,
    int n,
    int k) {
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int pair = blockIdx.y;
  const int row_in_pair = blockIdx.z;
  const int64_t activation_row =
      static_cast<int64_t>(pair) * kRowStride + row_in_pair;

  const int live_pairs = num_post[0] / kRowStride;
  if (pair >= live_pairs) return;
  const int expert = expert_blocks[pair];
  if (kind[expert] != 0) return;

  const int d = dimension[expert];
  const int index_bits = bits[expert];
  if ((d != 4 && d != 8) || index_bits < 8 || index_bits > 12) return;

  const int item_elems = kCodesPerLaneItem * d;
  const int items = k / item_elems;
  extern __shared__ __align__(16) __nv_bfloat16 x_smem[];

  // Full-K stage with two BF16 pad elements per eight-code lane item. The
  // resulting d4/d8 item strides avoid repeated 32-bank alignment.
  for (int kk = tid; kk < k; kk += blockDim.x) {
    const int item = kk / item_elems;
    const int within = kk - item * item_elems;
    x_smem[item * (item_elems + kPadBf16PerItem) + within] =
        a[activation_row * k + kk];
  }
  __syncthreads();

  const int out_row = blockIdx.x * kWarpsPerBlock + warp;
  if (out_row >= n) return;

  const int64_t row_bytes = static_cast<int64_t>(code_row_bytes[expert]);
  const uint8_t* code_row = codes + code_offset[expert] +
      static_cast<int64_t>(out_row) * row_bytes;
  const uint8_t* scale_row = scales + scale_offset[expert] +
      static_cast<int64_t>(out_row) * (k / 32);
  const half* codebook = codebooks + cb_offset[expert];

  float acc = 0.0f;
  // Give each lane a contiguous K interval before the warp tree reduction.
  // This preserves the fast one-warp-per-output geometry while moving the
  // FP32 association toward the contiguous-K accumulation used by dense BF16
  // GEMM, rather than summing 1K/2K-strided fragments in each lane.
  const int items_per_lane = (items + 31) / 32;
  const int item_begin = lane * items_per_lane;
  const int item_limit = item_begin + items_per_lane;
  const int item_end = item_limit < items ? item_limit : items;
  for (int item = item_begin; item < item_end; ++item) {
    uint32_t item_codes[kCodesPerLaneItem];
    const int item_byte = item * index_bits;
    load_eight_codes(code_row + item_byte,
                     static_cast<int>(row_bytes) - item_byte,
                     index_bits, item_codes);
    if (d == 4) {
      const float scale = ldexpf(
          1.0f, static_cast<int>(load_cg_u8(scale_row + item)) - 127);
      accumulate_item<4>(codebook, item_codes, x_smem, item,
                         scale, scale, acc);
    } else {
      const float scale0 = ldexpf(
          1.0f, static_cast<int>(load_cg_u8(scale_row + item * 2)) - 127);
      const float scale1 = ldexpf(
          1.0f, static_cast<int>(load_cg_u8(scale_row + item * 2 + 1)) - 127);
      accumulate_item<8>(codebook, item_codes, x_smem, item,
                         scale0, scale1, acc);
    }
  }

#pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1) {
    acc += __shfl_down_sync(kFullWarp, acc, delta);
  }
  if (lane == 0) {
    out[activation_row * n + out_row] = __float2bfloat16_rn(acc);
  }
}

at::Tensor vq_gemm(
    const at::Tensor& a,
    at::Tensor out,
    const at::Tensor& expert_blocks,
    const at::Tensor& num_post,
    const at::Tensor& kind,
    const at::Tensor& codes,
    const at::Tensor& scales,
    const at::Tensor& codebooks,
    const at::Tensor& code_offset,
    const at::Tensor& scale_offset,
    const at::Tensor& code_row_bytes,
    const at::Tensor& dimension,
    const at::Tensor& bits,
    const at::Tensor& cb_offset,
    int64_t n64,
    int64_t k64,
    int64_t mblock64,
    int64_t valid_m64) {
  TORCH_CHECK(a.is_cuda() && out.is_cuda(), "a/out must be CUDA tensors");
  TORCH_CHECK(a.scalar_type() == at::kBFloat16 &&
              out.scalar_type() == at::kBFloat16,
              "a/out must be BF16");
  TORCH_CHECK(a.is_contiguous() && out.is_contiguous(),
              "a/out must be contiguous");
  TORCH_CHECK(mblock64 == kRowStride,
              "vq_warp_gemv row stride must be 4");
  TORCH_CHECK(valid_m64 >= 1 && valid_m64 <= kRowStride,
              "valid_m must be in [1, 4]");
  TORCH_CHECK(n64 > 0 && n64 <= INT_MAX && k64 > 0 && k64 <= INT_MAX,
              "n/k out of range");
  TORCH_CHECK(k64 % 64 == 0,
              "k must be a multiple of 64 for eight-code d4/d8 lane items");
  TORCH_CHECK(a.size(1) == k64 && out.size(1) == n64,
              "a/out shape does not match n/k");
  TORCH_CHECK(a.size(0) >= expert_blocks.numel() * kRowStride &&
              out.size(0) >= expert_blocks.numel() * kRowStride,
              "a/out rows do not cover compact pair grid");

  TORCH_CHECK(expert_blocks.is_cuda() && num_post.is_cuda() && kind.is_cuda(),
              "routing metadata must be CUDA-resident");
  TORCH_CHECK(expert_blocks.scalar_type() == at::kInt &&
              num_post.scalar_type() == at::kInt && kind.scalar_type() == at::kInt,
              "expert_blocks/num_post/kind must be int32");
  TORCH_CHECK(code_offset.is_cuda() && scale_offset.is_cuda() &&
              code_row_bytes.is_cuda() && dimension.is_cuda() &&
              bits.is_cuda() && cb_offset.is_cuda(),
              "projection metadata must be CUDA-resident");
  TORCH_CHECK(code_offset.scalar_type() == at::kLong &&
              scale_offset.scalar_type() == at::kLong &&
              cb_offset.scalar_type() == at::kLong,
              "offset tensors must be int64");
  TORCH_CHECK(code_row_bytes.scalar_type() == at::kInt,
              "code_row_bytes must be int32");
  TORCH_CHECK(dimension.scalar_type() == at::kByte &&
              bits.scalar_type() == at::kByte,
              "canonical dimension/bits metadata must be uint8");
  TORCH_CHECK(codes.scalar_type() == at::kByte && codes.is_contiguous(),
              "codes must be contiguous uint8 CPU-UVA or CUDA storage");
  TORCH_CHECK(scales.scalar_type() == at::kByte && scales.is_contiguous(),
              "real wire scales must be contiguous uint8 exponents");
  TORCH_CHECK(codebooks.scalar_type() == at::kHalf && codebooks.is_contiguous(),
              "codebooks must be contiguous FP16 CPU-UVA or CUDA storage");

  const c10::cuda::CUDAGuard guard(a.device());
  const int n = static_cast<int>(n64);
  const int k = static_cast<int>(k64);
  const int valid_m = static_cast<int>(valid_m64);
  const int max_items = k / (kCodesPerLaneItem * 4);
  const int shared_bytes =
      (k + max_items * kPadBf16PerItem) * sizeof(__nv_bfloat16);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      vq_warp_gemv_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes));

  const auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();
  const dim3 grid((n + kWarpsPerBlock - 1) / kWarpsPerBlock,
                  expert_blocks.numel(), valid_m);
  vq_warp_gemv_kernel<<<grid, kThreads, shared_bytes, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
      expert_blocks.data_ptr<int32_t>(), num_post.data_ptr<int32_t>(),
      kind.data_ptr<int32_t>(), codes.data_ptr<uint8_t>(),
      scales.data_ptr<uint8_t>(),
      reinterpret_cast<const half*>(codebooks.data_ptr<at::Half>()),
      code_offset.data_ptr<int64_t>(), scale_offset.data_ptr<int64_t>(),
      code_row_bytes.data_ptr<int32_t>(), dimension.data_ptr<uint8_t>(),
      bits.data_ptr<uint8_t>(), cb_offset.data_ptr<int64_t>(), n, k);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

}  // namespace

TORCH_LIBRARY(vq_warp_gemv, m) {
  m.def("gemm(Tensor a, Tensor(a!) out, Tensor expert_blocks, Tensor num_post, "
        "Tensor kind, Tensor codes, Tensor scales, Tensor codebooks, "
        "Tensor code_offset, Tensor scale_offset, Tensor code_row_bytes, "
        "Tensor dimension, Tensor bits, Tensor cb_offset, int n, int k, "
        "int mblock, int valid_m) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(vq_warp_gemv, CUDA, m) {
  m.impl("gemm", &vq_gemm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
