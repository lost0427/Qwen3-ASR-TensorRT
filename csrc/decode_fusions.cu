// Small fused kernels for the single-token decode step.
//
// The decode step spends most of its time in fixed-cost kernel launches for
// elementwise work, so q/k normalisation + RoPE and the gated MLP activation are
// folded into one kernel each.

#include <cuda_fp16.h>
#include <cstdint>

namespace {

__device__ __forceinline__ float block_sum(float value, float* scratch, int tid, int size) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset);
  }
  if ((tid & 31) == 0) {
    scratch[tid >> 5] = value;
  }
  __syncthreads();
  value = (tid < size / 32) ? scratch[tid] : 0.f;
  if (tid < 32) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
  }
  return value;
}

// One block per attention head: RMS-normalise, scale by the head norm weight and
// apply rotary embedding, in place.
__global__ void qk_norm_rope_kernel(__half* __restrict__ vectors,
                                    const __half* __restrict__ weight,
                                    const __half* __restrict__ cos,
                                    const __half* __restrict__ sin,
                                    int head_dim, float eps) {
  __shared__ float scratch[32];
  __shared__ float values[128];
  __shared__ float norm;

  const int tid = threadIdx.x;
  __half* vector = vectors + blockIdx.x * head_dim;
  const int index = tid;

  float value = __half2float(vector[index]);
  values[index] = value;
  float squares = block_sum(value * value, scratch, tid, head_dim);
  if (tid == 0) {
    norm = rsqrtf(squares / head_dim + eps);
  }
  __syncthreads();

  const float scaled = values[index] * norm * __half2float(weight[index]);
  __syncthreads();
  values[index] = scaled;
  __syncthreads();

  const int half = head_dim >> 1;
  const float rotated = index < half ? -values[index + half] : values[index - half];
  vector[index] = __float2half(scaled * __half2float(cos[index]) + rotated * __half2float(sin[index]));
}

// out = silu(gate) * up, where gate/up are the two halves of a fused projection.
__global__ void silu_mul_kernel(const __half* __restrict__ gate_up,
                                __half* __restrict__ out,
                                int width) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= width) {
    return;
  }
  const float gate = __half2float(gate_up[index]);
  const float up = __half2float(gate_up[width + index]);
  const float silu = gate / (1.f + __expf(-gate));
  out[index] = __float2half(silu * up);
}

// Appends one token to the grouped-query KV cache: the GQA expansion happens on
// the fly, replacing a repeat_interleave plus two index_copy_ launches.
__global__ void cache_write_kernel(__half* __restrict__ k_cache,
                                   __half* __restrict__ v_cache,
                                   const __half* __restrict__ k,
                                   const __half* __restrict__ v,
                                   const int64_t* __restrict__ position,
                                   int total, int head_dim, int max_cache) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int head = index / head_dim;
  const int depth = index % head_dim;
  const long destination = static_cast<long>(head) * max_cache * head_dim
                           + static_cast<long>(position[0]) * head_dim + depth;
  k_cache[destination] = k[index];
  v_cache[destination] = v[index];
}

}  // namespace

void qk_norm_rope_launch(void* vectors, const void* weight, const void* cos, const void* sin,
                         int heads, int head_dim, float eps, cudaStream_t stream) {
  qk_norm_rope_kernel<<<heads, head_dim, 0, stream>>>(
      reinterpret_cast<__half*>(vectors),
      reinterpret_cast<const __half*>(weight),
      reinterpret_cast<const __half*>(cos),
      reinterpret_cast<const __half*>(sin),
      head_dim, eps);
}

void silu_mul_launch(const void* gate_up, void* out, int width, cudaStream_t stream) {
  const int threads = 256;
  silu_mul_kernel<<<(width + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<const __half*>(gate_up), reinterpret_cast<__half*>(out), width);
}

void cache_write_launch(void* k_cache, void* v_cache, const void* k, const void* v,
                        const void* position, int total, int head_dim,
                        int max_cache, cudaStream_t stream) {
  const int threads = 256;
  cache_write_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      reinterpret_cast<__half*>(k_cache), reinterpret_cast<__half*>(v_cache),
      reinterpret_cast<const __half*>(k), reinterpret_cast<const __half*>(v),
      reinterpret_cast<const int64_t*>(position), total, head_dim, max_cache);
}
