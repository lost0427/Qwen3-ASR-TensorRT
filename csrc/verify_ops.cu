// Row-batched elementwise kernels for the verification pass.
//
// The verification block runs M rows through the same layer, so the fused
// single-row kernels in decode_fusions.cu need M-row forms. Fusing here matters
// more than on the decode path: the graph replays 28 layers, and every extra
// elementwise launch costs a few microseconds even when the tensors are tiny.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ void block_sum_broadcast(float& value, float* scratch) {
  const int tid = threadIdx.x;
  value = warp_sum(value);
  if ((tid & 31) == 0) scratch[tid >> 5] = value;
  __syncthreads();
  const int warps = blockDim.x / 32;
  if (tid < 32) {
    float total = (tid < warps) ? scratch[tid] : 0.f;
    total = warp_sum(total);
    if (tid == 0) scratch[0] = total;
  }
  __syncthreads();
  value = scratch[0];
  __syncthreads();
}

// One block per (row, head): RMS-normalise the head vector, scale by the head
// norm weight and apply rotary embedding in place.
__global__ void qk_norm_rope_rows_kernel(__half* __restrict__ vectors,
                                         const __half* __restrict__ weight,
                                         const __half* __restrict__ cos,
                                         const __half* __restrict__ sin,
                                         int row_stride, int offset, int heads, int head_dim,
                                         float eps) {
  __shared__ float scratch[32];
  __shared__ float quad[128];
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int half = head_dim >> 1;

  __half* vector = vectors + static_cast<long long>(row) * row_stride + offset +
                   static_cast<long long>(head) * head_dim;
  const __half* row_cos = cos + static_cast<long long>(row) * head_dim;
  const __half* row_sin = sin + static_cast<long long>(row) * head_dim;

  const float value = __half2float(vector[tid]);
  quad[tid] = value;
  float squares = value * value;
  block_sum_broadcast(squares, scratch);
  const float norm = rsqrtf(squares / head_dim + eps);
  quad[tid] = value * norm * __half2float(weight[tid]);
  __syncthreads();
  const float rotated = tid < half ? -quad[tid + half] : quad[tid - half];
  vector[tid] = __float2half(quad[tid] * __half2float(row_cos[tid]) +
                             rotated * __half2float(row_sin[tid]));
}

// out = silu(gate) * up for every row of a fused (rows, 2 * width) projection.
__global__ void silu_mul_rows_kernel(const __half* __restrict__ gate_up,
                                     __half* __restrict__ out, int width) {
  const int row = blockIdx.y;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width) return;
  const __half* src = gate_up + static_cast<long long>(row) * 2 * width;
  const float gate = __half2float(src[index]);
  const float up = __half2float(src[width + index]);
  out[static_cast<long long>(row) * width + index] =
      __float2half(gate / (1.f + __expf(-gate)) * up);
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

// Causal GQA attention for `rows` queries: query r sees cache positions
// [0, base + r]. One block per (head, row), online softmax over 64-key tiles so
// the length comes from the device position instead of a host-side shape.
__global__ void verify_attention_kernel(const __half* __restrict__ q, int q_stride,
                                        const __half* __restrict__ k_cache,
                                        const __half* __restrict__ v_cache,
                                        const int64_t* __restrict__ position,
                                        __half* __restrict__ out, int head_dim, int capacity,
                                        int groups) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int depth = tid % head_dim;
  const int length = static_cast<int>(position[0]) + row + 1;

  extern __shared__ __half shared[];
  __half* sq = shared;
  __half* sk = sq + head_dim;
  __half* sv = sk + 64 * head_dim;
  float* sp = reinterpret_cast<float*>(sv + 64 * head_dim);
  float* scratch = sp + 64;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const __half* qh = q + static_cast<long long>(row) * q_stride +
                     static_cast<long long>(head) * head_dim;
  const __half* kh = k_cache + static_cast<long long>(head / groups) * capacity * head_dim;
  const __half* vh = v_cache + static_cast<long long>(head / groups) * capacity * head_dim;

  sq[depth] = qh[depth];
  __syncthreads();

  float acc = 0.f;
  float running_max = -INFINITY;
  float running_sum = 0.f;
  const int warps = blockDim.x / 32;

  for (int start = 0; start < length; start += 64) {
    const int count = min(64, length - start);
    const int values = count * head_dim;
    for (int base = 0; base < values; base += blockDim.x * 8) {
      const int index = base + tid * 8;
      if (index + 8 <= values) {
        const long long offset = static_cast<long long>(start) * head_dim + index;
        *reinterpret_cast<uint4*>(&sk[index]) = __ldcs(reinterpret_cast<const uint4*>(&kh[offset]));
        *reinterpret_cast<uint4*>(&sv[index]) = __ldcs(reinterpret_cast<const uint4*>(&vh[offset]));
      }
    }
    __syncthreads();

    if (tid < count) {
      const __half2* keys = reinterpret_cast<const __half2*>(sk + tid * head_dim);
      const __half2* query = reinterpret_cast<const __half2*>(sq);
      float dot = 0.f;
      for (int index = 0; index < head_dim / 2; ++index) {
        const float2 kv = __half22float2(keys[index]);
        const float2 qv = __half22float2(query[index]);
        dot += kv.x * qv.x + kv.y * qv.y;
      }
      sp[tid] = dot * scale;
    }
    __syncthreads();

    float best = (tid < count) ? sp[tid] : -INFINITY;
    best = warp_max(best);
    if ((tid & 31) == 0) scratch[tid >> 5] = best;
    __syncthreads();
    float tile_max = -INFINITY;
    if (tid < 32) {
      tile_max = (tid < warps) ? scratch[tid] : -INFINITY;
      tile_max = warp_max(tile_max);
      if (tid == 0) scratch[0] = tile_max;
    }
    __syncthreads();
    tile_max = scratch[0];

    const float next_max = fmaxf(running_max, tile_max);
    const float correction = running_max == -INFINITY ? 0.f : __expf(running_max - next_max);
    float weight = 0.f;
    if (tid < count) {
      weight = __expf(sp[tid] - next_max);
      sp[tid] = weight;
    }
    __syncthreads();
    float tile_sum = weight;
    block_sum_broadcast(tile_sum, scratch);

    acc *= correction;
    for (int t = 0; t < count; ++t) {
      acc += sp[t] * __half2float(sv[t * head_dim + depth]);
    }
    running_sum = running_sum * correction + tile_sum;
    running_max = next_max;
    __syncthreads();
  }

  out[(static_cast<long long>(row) * gridDim.x + head) * head_dim + depth] =
      __float2half(acc / running_sum);
}

}  // namespace

void qk_norm_rope_rows_launch(void* vectors, int row_stride, int offset, const void* weight,
                              const void* cos, const void* sin, int rows, int heads, int head_dim,
                              float eps, cudaStream_t stream) {
  dim3 grid(heads, rows);
  qk_norm_rope_rows_kernel<<<grid, head_dim, 0, stream>>>(
      reinterpret_cast<__half*>(vectors), reinterpret_cast<const __half*>(weight),
      reinterpret_cast<const __half*>(cos), reinterpret_cast<const __half*>(sin),
      row_stride, offset, heads, head_dim, eps);
}

void silu_mul_rows_launch(const void* gate_up, void* out, int rows, int width,
                          cudaStream_t stream) {
  const int threads = 256;
  dim3 grid((width + threads - 1) / threads, rows);
  silu_mul_rows_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(gate_up), reinterpret_cast<__half*>(out), width);
}

void verify_attention_launch(const void* q, int q_stride, const void* k_cache, const void* v_cache,
                             const void* position, void* out, int rows, int heads, int head_dim,
                             int capacity, int groups, cudaStream_t stream) {
  const int threads = 128;
  const size_t shared = (head_dim + 2 * 64 * head_dim) * sizeof(__half) +
                        64 * sizeof(float) + 32 * sizeof(float);
  dim3 grid(heads, rows);
  verify_attention_kernel<<<grid, threads, shared, stream>>>(
      reinterpret_cast<const __half*>(q), q_stride, reinterpret_cast<const __half*>(k_cache),
      reinterpret_cast<const __half*>(v_cache), reinterpret_cast<const int64_t*>(position),
      reinterpret_cast<__half*>(out), head_dim, capacity, groups);
}
