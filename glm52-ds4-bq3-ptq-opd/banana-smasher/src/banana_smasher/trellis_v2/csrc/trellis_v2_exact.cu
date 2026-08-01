#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <map>
#include <mutex>
#include <tuple>
#include <vector>

namespace {
constexpr int STEPS = 128;
constexpr int PAIRS = 64;
constexpr int PREFIXES = 4096;
constexpr int STATES = 65536;
constexpr int BATCH = 256;
// One paired-step CTA owns eight independent sequences. The transition LUT
// values are lane-uniform and are therefore fetched once and reused across all
// eight rows before the packed two-step backpointer is written.
constexpr int B_TILE = 8;
constexpr int ROWS_PER_LANE = B_TILE / 2;
constexpr int WARPS = 16;
constexpr int THREADS = WARPS * 32;
constexpr int AXIS = 16;
constexpr int WAVES = AXIS / WARPS;
constexpr int COMPACT_PREFIXES = 128;
constexpr int COMPACT_VALUES = B_TILE * COMPACT_PREFIXES;
constexpr int FULL_VALUES = B_TILE * 256;
constexpr size_t SHARED_BYTES =
    2 * COMPACT_VALUES * sizeof(float) + FULL_VALUES * sizeof(uint8_t);

__device__ __forceinline__ float emission(
    float x0, float x1, float l0, float l1) {
  const float d0 = __fsub_rn(l0, x0);
  const float d1 = __fsub_rn(l1, x1);
  // Match the canonical Triton PTX exactly: mul(d1,d1), then
  // fma.rn(d0,d0,d1_squared).  The fusion point is winner-visible on
  // near-tied FIRST64 rows and is therefore part of the exact contract.
  return __fmaf_rn(d0, d0, __fmul_rn(d1, d1));
}

__device__ __forceinline__ float candidate(
    float prev, float x0, float x1, float l0, float l1) {
  const float d0 = __fsub_rn(l0, x0);
  const float d1 = __fsub_rn(l1, x1);
  // Canonical Triton emits two ordered fma.rn operations:
  //   acc0 = fma(d0,d0,prev); candidate = fma(d1,d1,acc0).
  return __fmaf_rn(d1, d1, __fmaf_rn(d0, d0, prev));
}

// SM100+ executes these float2 intrinsics component-wise with the same scalar
// round-to-nearest operations.  Pairing independent sequence rows halves the
// steady-state FP32 transition instruction stream without coupling winners.
__device__ __forceinline__ float2 candidate2(
    float2 previous, float2 x0, float2 x1, float l0, float l1) {
  const float2 d0 = __fadd2_rn(
      make_float2(l0, l0), make_float2(-x0.x, -x0.y));
  const float2 d1 = __fadd2_rn(
      make_float2(l1, l1), make_float2(-x1.x, -x1.y));
  const float2 acc = __ffma2_rn(d0, d0, previous);
  return __ffma2_rn(d1, d1, acc);
}

// nvcc lowers the source-level two-result `if (value < best)` to a divergent
// BRA/BSSY/BSYNC region for every row and q branch.  The PTX predicate/select
// sequence is exactly the same ordered strict comparison, including false on
// equality and unordered NaN, but has no control-flow reconvergence cost.
__device__ __forceinline__ void update_best_strict(
    float value, uint32_t q, float& best, uint32_t& best_q) {
  asm volatile(
      "{ .reg .pred better;\n\t"
      "setp.lt.f32 better, %2, %0;\n\t"
      "selp.f32 %0, %2, %0, better;\n\t"
      "selp.u32 %1, %3, %1, better;\n\t"
      "}"
      : "+f"(best), "+r"(best_q)
      : "f"(value), "r"(q));
}

// Score a 4-sequence x 256-prefix slice.  Each half-warp owns two exact
// sequence rows without changing q order, arithmetic, or winners.
template <bool NO_PREVIOUS>
__device__ __forceinline__ void score_step(
    const half* __restrict__ x,
    const float* __restrict__ lut_aos,
    const float* __restrict__ previous,
    float* __restrict__ output,
    uint8_t* __restrict__ q_output,
    int seq_base,
    int group_b) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_group = lane >> 4;
  const int suffix = lane & 15;

  float x0s[ROWS_PER_LANE];
  float x1s[ROWS_PER_LANE];
#pragma unroll
  for (int k = 0; k < ROWS_PER_LANE; ++k) {
    const int row = row_group + 2 * k;
    const int seq = seq_base + row;
    x0s[k] = __half2float(x[seq]);
    x1s[k] = __half2float(x[BATCH + seq]);
  }

#pragma unroll
  for (int wave = 0; wave < WAVES; ++wave) {
    const int tile = wave * WARPS + warp;
    const int prefix = (tile * 16 + group_b) * 16 + suffix;
    const int parity = group_b & 1;
    uint32_t staged_q[8];
#pragma unroll
    for (int branch = 0; branch < 8; ++branch) {
      staged_q[branch] = static_cast<uint32_t>(branch * 2 + parity);
    }
    float best[ROWS_PER_LANE];
    uint32_t best_q[ROWS_PER_LANE];
#pragma unroll
    for (int k = 0; k < ROWS_PER_LANE; ++k) {
      best[k] = INFINITY;
      best_q[k] = 0;
    }
#pragma unroll
    for (int branch = 0; branch < 8; ++branch) {
      const uint32_t q = staged_q[branch];
      const float2 lut_pair = reinterpret_cast<const float2*>(lut_aos)[
          q * PREFIXES + prefix];
      const float l0 = lut_pair.x;
      const float l1 = lut_pair.y;
      if constexpr (NO_PREVIOUS) {
#pragma unroll
        for (int k = 0; k < ROWS_PER_LANE; ++k) {
          const float value = emission(x0s[k], x1s[k], l0, l1);
          update_best_strict(value, q, best[k], best_q[k]);
        }
      } else {
#pragma unroll
        for (int k = 0; k < ROWS_PER_LANE; k += 2) {
          const int row0 = row_group + 2 * k;
          const int row1 = row_group + 2 * (k + 1);
          const float2 values = candidate2(
              make_float2(
                  previous[row0 * COMPACT_PREFIXES + (q >> 1) * 16 + tile],
                  previous[row1 * COMPACT_PREFIXES + (q >> 1) * 16 + tile]),
              make_float2(x0s[k], x0s[k + 1]),
              make_float2(x1s[k], x1s[k + 1]),
              l0, l1);
          update_best_strict(values.x, q, best[k], best_q[k]);
          update_best_strict(values.y, q, best[k + 1], best_q[k + 1]);
        }
      }
    }
#pragma unroll
    for (int k = 0; k < ROWS_PER_LANE; ++k) {
      const int row = row_group + 2 * k;
      const int full_local = row * 256 + tile * 16 + suffix;
      if (((tile + suffix) & 1) != 0) {
        const int compact =
            row * COMPACT_PREFIXES + (tile >> 1) * 16 + suffix;
        output[compact] = best[k];
      }
      q_output[full_local] = static_cast<uint8_t>(best_q[k]);
    }
  }
  __syncthreads();
}

// The odd step is terminal for the pair: write its exact costs and the packed
// even/odd winner nibbles directly to graph-owned global sinks.  This removes
// the second shared winner plane and the generic copy-out pass without changing
// q visitation order, arithmetic, or storage order.
__device__ __forceinline__ void score_odd_to_sink(
    const half* __restrict__ x,
    const float* __restrict__ lut_aos,
    const float* __restrict__ previous,
    const uint8_t* __restrict__ q_even,
    float* __restrict__ cost_sink,
    uint8_t* __restrict__ packed_sink,
    int seq_base,
    int group_b) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_group = lane >> 4;
  const int suffix = lane & 15;

  float x0s[ROWS_PER_LANE];
  float x1s[ROWS_PER_LANE];
#pragma unroll
  for (int k = 0; k < ROWS_PER_LANE; ++k) {
    const int row = row_group + 2 * k;
    const int seq = seq_base + row;
    x0s[k] = __half2float(x[2 * BATCH + seq]);
    x1s[k] = __half2float(x[3 * BATCH + seq]);
  }

#pragma unroll
  for (int wave = 0; wave < WAVES; ++wave) {
    const int tile = wave * WARPS + warp;
    const int prefix = (group_b * 16 + tile) * 16 + suffix;
    const int parity = (tile + 1) & 1;
    uint32_t staged_q[8];
#pragma unroll
    for (int branch = 0; branch < 8; ++branch) {
      staged_q[branch] = static_cast<uint32_t>(branch * 2 + parity);
    }
    float best[ROWS_PER_LANE];
    uint32_t best_q[ROWS_PER_LANE];
#pragma unroll
    for (int k = 0; k < ROWS_PER_LANE; ++k) {
      best[k] = INFINITY;
      best_q[k] = 0;
    }
#pragma unroll
    for (int branch = 0; branch < 8; ++branch) {
      const uint32_t q = staged_q[branch];
      const float2 lut_pair = reinterpret_cast<const float2*>(lut_aos)[
          q * PREFIXES + prefix];
#pragma unroll
      for (int k = 0; k < ROWS_PER_LANE; k += 2) {
        const int row0 = row_group + 2 * k;
        const int row1 = row_group + 2 * (k + 1);
        const float2 values = candidate2(
            make_float2(
                previous[row0 * COMPACT_PREFIXES + (q >> 1) * 16 + tile],
                previous[row1 * COMPACT_PREFIXES + (q >> 1) * 16 + tile]),
            make_float2(x0s[k], x0s[k + 1]),
            make_float2(x1s[k], x1s[k + 1]),
            lut_pair.x, lut_pair.y);
        update_best_strict(values.x, q, best[k], best_q[k]);
        update_best_strict(values.y, q, best[k + 1], best_q[k + 1]);
      }
    }
#pragma unroll
    for (int k = 0; k < ROWS_PER_LANE; ++k) {
      const int row = row_group + 2 * k;
      const int seq = seq_base + row;
      const int local = row * 256 + tile * 16 + suffix;
      const int64_t sink = static_cast<int64_t>(seq) * PREFIXES + prefix;
      cost_sink[sink] = best[k];
      packed_sink[sink] = static_cast<uint8_t>(
          (best_q[k] << 4) | (q_even[local] & 15));
    }
  }
}

template <bool FIRST_PAIR, bool HAS_OVERLAP>
__global__ void _persistent_k2_viterbi(
    const half* __restrict__ x,
    const float* __restrict__ lut_aos,
    const float* __restrict__ cost_in,
    float* __restrict__ cost_out,
    const int32_t* __restrict__ overlap,
    uint8_t* __restrict__ packed_backpointer) {
  extern __shared__ unsigned char smem[];
  float* local0 = reinterpret_cast<float*>(smem);
  float* local1 = local0 + COMPACT_VALUES;
  uint8_t* q_even = reinterpret_cast<uint8_t*>(local1 + COMPACT_VALUES);

  const int seq_base = blockIdx.x * B_TILE;
  const int group_b = blockIdx.y;

  if constexpr (!FIRST_PAIR) {
    for (int i = threadIdx.x; i < COMPACT_VALUES; i += THREADS) {
      const int row = i >> 7;
      const int rem = i & 127;
      const int q = 2 * (rem >> 4) + (group_b & 1);
      const int axis_a = rem & 15;
      const int seq = seq_base + row;
      local0[i] = cost_in[seq * PREFIXES + q * 256 + axis_a * 16 + group_b];
    }
    __syncthreads();
  }

  if constexpr (FIRST_PAIR && HAS_OVERLAP) {
    for (int i = threadIdx.x; i < FULL_VALUES; i += THREADS) {
      const int row = i >> 8;
      const int rem = i & 255;
      const int axis_a = rem >> 4;
      const int suffix = rem & 15;
      const int seq = seq_base + row;
      float value = INFINITY;
      uint8_t q_value = 0;
      const int ov = overlap[seq];
      const int ov_q = ov >> 8;
      const int ov_residue = ov & 255;
      const int ov_a = ov_residue >> 4;
      const int ov_b = ov_residue & 15;
      if (group_b == ov_b && axis_a == ov_a) {
        const int prefix = axis_a * 256 + group_b * 16 + suffix;
        const int state = ov_q * PREFIXES + prefix;
        const float x0 = __half2float(x[seq]);
        const float x1 = __half2float(x[BATCH + seq]);
        const float2 lut_pair =
            reinterpret_cast<const float2*>(lut_aos)[state];
        value = emission(x0, x1, lut_pair.x, lut_pair.y);
        q_value = static_cast<uint8_t>(ov_q);
      }
      if (((axis_a + suffix) & 1) != 0) {
        const int compact =
            row * COMPACT_PREFIXES + (axis_a >> 1) * 16 + suffix;
        local1[compact] = value;
      }
      q_even[i] = q_value;
    }
    __syncthreads();
  } else {
    score_step<FIRST_PAIR>(
        x, lut_aos, local0, local1, q_even,
        seq_base, group_b);
  }

  score_odd_to_sink(
      x, lut_aos, local1, q_even, cost_out, packed_backpointer,
      seq_base, group_b);
}

template <bool HAS_OVERLAP>
__global__ void backtrack_kernel(
    const float* __restrict__ final_cost,
    const uint8_t* __restrict__ packed_backpointer,
    const int32_t* __restrict__ overlap,
    int32_t* __restrict__ states) {
  const int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const int seq = global_thread >> 5;
  const int lane = threadIdx.x & 31;

  int prefix = 0;
  if constexpr (HAS_OVERLAP) {
    prefix = overlap[seq];
  } else {
    float best = INFINITY;
    int best_prefix = 0;
    for (int j = lane; j < PREFIXES; j += 32) {
      const float value = final_cost[seq * PREFIXES + j];
      if (value < best) {
        best = value;
        best_prefix = j;
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other = __shfl_down_sync(0xffffffffu, best, offset);
      const int other_prefix =
          __shfl_down_sync(0xffffffffu, best_prefix, offset);
      if (other < best || (other == best && other_prefix < best_prefix)) {
        best = other;
        best_prefix = other_prefix;
      }
    }
    prefix = __shfl_sync(0xffffffffu, best_prefix, 0);
  }
  if (lane != 0) return;
  for (int step = STEPS - 1; step >= 0; --step) {
    const int pair = step >> 1;
    uint8_t packed;
    if (step & 1) {
      packed = packed_backpointer[
          (static_cast<int64_t>(pair) * BATCH + seq) * PREFIXES + prefix] >> 4;
    } else {
      const int axis_a = prefix >> 8;
      const int axis_b = (prefix >> 4) & 15;
      const int suffix = prefix & 15;
      const int storage_prefix = axis_b * 256 + axis_a * 16 + suffix;
      packed = packed_backpointer[
          (static_cast<int64_t>(pair) * BATCH + seq) * PREFIXES + storage_prefix] & 15;
    }
    const int state = static_cast<int>(packed) * PREFIXES + prefix;
    states[step * BATCH + seq] = state;
    prefix = state >> 4;
  }
}

struct GraphState {
  torch::Tensor x;
  torch::Tensor lut;
  torch::Tensor overlap;
  torch::Tensor states;
  torch::Tensor packed_bp;
  torch::Tensor cost0;
  torch::Tensor cost1;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};

using GraphKey = std::tuple<int, int, bool, uintptr_t>;
std::mutex graph_cache_mutex;
std::map<GraphKey, GraphState*> graph_cache;

void build_exact_graph(
    GraphState* state,
    int B,
    bool has_overlap) {
  C10_CUDA_CHECK(cudaGraphCreate(&state->graph, 0));
  const dim3 grid(BATCH / B_TILE, AXIS);
  float* in_ptr = state->cost0.data_ptr<float>();
  float* out_ptr = state->cost1.data_ptr<float>();
  cudaGraphNode_t previous = nullptr;
  for (int pair = 0; pair < PAIRS; ++pair) {
    const half* x_ptr =
        reinterpret_cast<const half*>(state->x.data_ptr<c10::Half>()) +
        static_cast<int64_t>(pair) * 4 * BATCH;
    const float* lut_ptr = state->lut.data_ptr<float>();
    const float* cost_in_ptr = in_ptr;
    float* cost_out_ptr = out_ptr;
    const int32_t* overlap_ptr =
        has_overlap ? state->overlap.data_ptr<int32_t>() : nullptr;
    uint8_t* packed_ptr = state->packed_bp.data_ptr<uint8_t>() +
        static_cast<int64_t>(pair) * BATCH * PREFIXES;
    void* arguments[] = {
        &x_ptr, &lut_ptr, &cost_in_ptr, &cost_out_ptr, &overlap_ptr,
        &packed_ptr};
    cudaKernelNodeParams parameters{};
    if (pair == 0) {
      parameters.func = has_overlap
          ? reinterpret_cast<void*>(_persistent_k2_viterbi<true, true>)
          : reinterpret_cast<void*>(_persistent_k2_viterbi<true, false>);
    } else {
      parameters.func = reinterpret_cast<void*>(
          _persistent_k2_viterbi<false, false>);
    }
    parameters.gridDim = grid;
    parameters.blockDim = dim3(THREADS);
    parameters.sharedMemBytes = SHARED_BYTES;
    parameters.kernelParams = arguments;
    cudaGraphNode_t node = nullptr;
    C10_CUDA_CHECK(cudaGraphAddKernelNode(
        &node, state->graph, previous == nullptr ? nullptr : &previous,
        previous == nullptr ? 0 : 1, &parameters));
    previous = node;
    float* temporary = in_ptr;
    in_ptr = out_ptr;
    out_ptr = temporary;
  }
  const float* final_cost_ptr = in_ptr;
  const uint8_t* packed_ptr = state->packed_bp.data_ptr<uint8_t>();
  const int32_t* overlap_ptr =
      has_overlap ? state->overlap.data_ptr<int32_t>() : nullptr;
  int32_t* states_ptr = state->states.data_ptr<int32_t>();
  void* arguments[] = {
      &final_cost_ptr, &packed_ptr, &overlap_ptr, &states_ptr};
  cudaKernelNodeParams parameters{};
  parameters.func = has_overlap
      ? reinterpret_cast<void*>(backtrack_kernel<true>)
      : reinterpret_cast<void*>(backtrack_kernel<false>);
  parameters.gridDim = dim3(BATCH / 8);
  parameters.blockDim = dim3(256);
  parameters.sharedMemBytes = 0;
  parameters.kernelParams = arguments;
  cudaGraphNode_t backtrack = nullptr;
  C10_CUDA_CHECK(cudaGraphAddKernelNode(
      &backtrack, state->graph, &previous, 1, &parameters));
}

GraphState* graph_state_for(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const torch::Tensor& overlap_tensor,
    bool has_overlap,
    int B,
    cudaStream_t stream) {
  const int device = x.get_device();
  const GraphKey key(device, B, has_overlap, reinterpret_cast<uintptr_t>(stream));
  std::lock_guard<std::mutex> lock(graph_cache_mutex);
  const auto existing = graph_cache.find(key);
  if (existing != graph_cache.end()) return existing->second;

  // Intentionally process-lifetime storage.  Destroying graph handles or CUDA
  // tensors from extension globals after the CUDA runtime begins shutdown is
  // unsafe; public smash producers are short-lived processes.
  auto* state = new GraphState();
  state->x = torch::empty_like(x);
  state->lut = torch::empty_like(lut_aos);
  if (has_overlap) state->overlap = torch::empty_like(overlap_tensor);
  state->states = torch::empty({STEPS, B}, x.options().dtype(torch::kInt32));
  state->packed_bp =
      torch::empty({PAIRS, B, PREFIXES}, x.options().dtype(torch::kUInt8));
  state->cost0 = torch::empty({B, PREFIXES}, x.options().dtype(torch::kFloat32));
  state->cost1 = torch::empty_like(state->cost0);

  build_exact_graph(state, B, has_overlap);
  TORCH_CHECK(state->graph != nullptr, "QTIP2 CUDA graph capture returned null graph");
  C10_CUDA_CHECK(
      cudaGraphInstantiate(&state->exec, state->graph, nullptr, nullptr, 0));
  TORCH_CHECK(state->exec != nullptr, "QTIP2 CUDA graph instantiate returned null exec");
  graph_cache.emplace(key, state);
  return state;
}

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be CUDA")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
}  // namespace

std::vector<torch::Tensor> trellis_v2_exact_cuda(
    const torch::Tensor& x,
    const torch::Tensor& lut_aos,
    const c10::optional<torch::Tensor>& overlap) {
  CHECK_CUDA(x); CHECK_CONTIG(x);
  CHECK_CUDA(lut_aos); CHECK_CONTIG(lut_aos);
  TORCH_CHECK(
      x.scalar_type() == torch::kFloat16 && x.dim() == 2 && x.size(0) == 256,
      "public smash solve must produce contiguous CUDA float16 x [256,B] before trellis-v2-exact");
  TORCH_CHECK(
      lut_aos.scalar_type() == torch::kFloat32 &&
          lut_aos.dim() == 2 && lut_aos.size(0) == STATES && lut_aos.size(1) == 2,
      "public smash solve must produce contiguous CUDA float32 LUT [65536,2] before trellis-v2-exact");
  const int B = static_cast<int>(x.size(1));
  TORCH_CHECK(
      B == BATCH,
      "exact public QTIP2 fast path requires B=256; got ", B);
  const bool has_overlap = overlap.has_value();
  torch::Tensor overlap_tensor;
  if (has_overlap) {
    overlap_tensor = overlap.value();
    CHECK_CUDA(overlap_tensor); CHECK_CONTIG(overlap_tensor);
    TORCH_CHECK(
        overlap_tensor.scalar_type() == torch::kInt32 && overlap_tensor.numel() == B,
        "overlap must be contiguous CUDA int32 [B]");
  }
  const c10::cuda::CUDAGuard guard(x.device());
  auto states = torch::empty({STEPS, B}, x.options().dtype(torch::kInt32));
  const auto stream = at::cuda::getCurrentCUDAStream(x.get_device()).stream();
  GraphState* state = graph_state_for(
      x, lut_aos, overlap_tensor, has_overlap, B, stream);
  C10_CUDA_CHECK(cudaMemcpyAsync(
      state->x.data_ptr(), x.data_ptr(), x.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      state->lut.data_ptr(), lut_aos.data_ptr(), lut_aos.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  if (has_overlap) {
    C10_CUDA_CHECK(cudaMemcpyAsync(
        state->overlap.data_ptr(), overlap_tensor.data_ptr(),
        overlap_tensor.nbytes(), cudaMemcpyDeviceToDevice, stream));
  }
  C10_CUDA_CHECK(cudaGraphLaunch(state->exec, stream));
  C10_CUDA_CHECK(cudaMemcpyAsync(
      states.data_ptr(), state->states.data_ptr(), states.nbytes(),
      cudaMemcpyDeviceToDevice, stream));
  return {states};
}
