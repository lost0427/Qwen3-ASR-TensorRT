// Single-query decode attention with a device-side sequence length.
//
// One block per attention head. The key/value cache is scanned in tiles and the
// softmax is maintained online, so the kernel never needs the host to know how
// long the context is: the length comes straight from the position buffer. That
// removes the attention bucketing (and the per-bucket CUDA graphs) from the
// decoder entirely.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kTile = 64;

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_reduce_sum(float value, float* scratch, int tid, int warps) {
  value = warp_reduce_sum(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();
  if (tid < 32) {
    float total = (tid < warps) ? scratch[tid] : 0.f;
    total = warp_reduce_sum(total);
    if (tid == 0) {
      scratch[31] = total;
    }
  }
  __syncthreads();
  return scratch[31];
}

__device__ __forceinline__ float block_reduce_max(float value, float* scratch, int tid, int warps) {
  value = warp_reduce_max(value);
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();
  if (tid < 32) {
    float best = (tid < warps) ? scratch[tid] : -INFINITY;
    best = warp_reduce_max(best);
    if (tid == 0) {
      scratch[31] = best;
    }
  }
  __syncthreads();
  return scratch[31];
}

__global__ void decode_attention_kernel(const __half* __restrict__ q,
                                        const __half* __restrict__ k_cache,
                                        const __half* __restrict__ v_cache,
                                        const int64_t* __restrict__ position,
                                        __half* __restrict__ out,
                                        int head_dim, int max_cache, int groups) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int depth = tid % head_dim;          // this thread owns one head dimension

  extern __shared__ __half shared[];
  __half* sq = shared;                                  // head_dim
  __half* sk = sq + head_dim;                           // kTile * head_dim
  __half* sv = sk + kTile * head_dim;                   // kTile * head_dim
  float* sp = reinterpret_cast<float*>(sv + kTile * head_dim);   // kTile
  float* scratch = sp + kTile;                                   // 32

  const int64_t cached = position[0];
  const int length = static_cast<int>(cached) + 1;      // cached positions + this token
  const float scale = rsqrtf(static_cast<float>(head_dim));

  const __half* qh = q + static_cast<long>(head) * head_dim;
  const __half* kh = k_cache + static_cast<long>(head / groups) * max_cache * head_dim;
  const __half* vh = v_cache + static_cast<long>(head / groups) * max_cache * head_dim;

  sq[depth] = qh[depth];
  __syncthreads();

  float acc = 0.f;
  float running_max = -INFINITY;
  float running_sum = 0.f;
  const int warps = blockDim.x / 32;

  for (int start = 0; start < length; start += kTile) {
    const int count = min(kTile, length - start);
    const int values = count * head_dim;
    for (int base = 0; base < values; base += blockDim.x * 8) {
      const int index = base + tid * 8;
      if (index + 8 <= values) {
        const long offset = static_cast<long>(start) * head_dim + index;
        *reinterpret_cast<uint4*>(&sk[index]) = __ldcs(reinterpret_cast<const uint4*>(&kh[offset]));
        *reinterpret_cast<uint4*>(&sv[index]) = __ldcs(reinterpret_cast<const uint4*>(&vh[offset]));
      }
    }
    __syncthreads();

    // one thread per key computes the dot product with the query
    if (tid < count) {
      const __half2* row = reinterpret_cast<const __half2*>(sk + tid * head_dim);
      const __half2* query = reinterpret_cast<const __half2*>(sq);
      float dot = 0.f;
      for (int index = 0; index < head_dim / 2; ++index) {
        const float2 k = __half22float2(row[index]);
        const float2 qv = __half22float2(query[index]);
        dot += k.x * qv.x + k.y * qv.y;
      }
      sp[tid] = dot * scale;
    }
    __syncthreads();

    const float tile_max = block_reduce_max(tid < count ? sp[tid] : -INFINITY, scratch, tid, warps);
    const float next_max = fmaxf(running_max, tile_max);
    const float correction = running_max == -INFINITY ? 0.f : __expf(running_max - next_max);
    float weight = 0.f;
    if (tid < count) {
      weight = __expf(sp[tid] - next_max);
      sp[tid] = weight;
    }
    __syncthreads();
    const float tile_sum = block_reduce_sum(weight, scratch, tid, warps);
    __syncthreads();

    acc *= correction;
    for (int t = 0; t < count; ++t) {
      acc += sp[t] * __half2float(sv[t * head_dim + depth]);
    }

    running_sum = running_sum * correction + tile_sum;
    running_max = next_max;
    __syncthreads();
  }

  out[static_cast<long>(head) * head_dim + depth] = __float2half(acc / running_sum);
}

}  // namespace

void decode_attention_launch(const void* q, const void* k_cache, const void* v_cache,
                             const void* position, void* out,
                             int heads, int head_dim, int max_cache, int groups,
                             cudaStream_t stream) {
  const int threads = 128;
  const size_t shared = (head_dim + 2 * kTile * head_dim) * sizeof(__half)
                        + kTile * sizeof(float) + 32 * sizeof(float);
  decode_attention_kernel<<<heads, threads, shared, stream>>>(
      reinterpret_cast<const __half*>(q),
      reinterpret_cast<const __half*>(k_cache),
      reinterpret_cast<const __half*>(v_cache),
      reinterpret_cast<const int64_t*>(position),
      reinterpret_cast<__half*>(out),
      head_dim, max_cache, groups);
}
