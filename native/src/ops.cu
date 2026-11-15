// Missing pieces the Python runtime gets from PyTorch: RMSNorm, embedding
// gather, audio scatter and argmax. The GEMV, RoPE, cache and attention kernels
// come from ../../csrc.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

__global__ void rmsnorm_kernel(const __half* __restrict__ x, const __half* __restrict__ weight,
                               __half* __restrict__ out, int n, float eps) {
  __shared__ float sums[32];
  const int tid = threadIdx.x;
  float sum = 0.0f;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = __half2float(x[i]);
    sum += v * v;
  }
  sum /= static_cast<float>(n);
  // warp reduce then block reduce
  for (int offset = 16; offset > 0; offset >>= 1) sum += __shfl_down_sync(0xffffffff, sum, offset);
  if ((tid & 31) == 0) sums[tid >> 5] = sum;
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
    const int warps = blockDim.x / 32;
    for (int i = 0; i < warps; ++i) total += sums[i];
    sums[0] = rsqrtf(total + eps);
  }
  __syncthreads();
  const float scale = sums[0];
  for (int i = tid; i < n; i += blockDim.x) {
    out[i] = __float2half(__half2float(x[i]) * scale * __half2float(weight[i]));
  }
}

__global__ void embed_kernel(__half* __restrict__ out, const __half* __restrict__ table,
                             const int* __restrict__ ids, int count, int hidden) {
  const int token = blockIdx.x;
  if (token >= count) return;
  const int id = ids[token];
  const __half* row = table + static_cast<long long>(id) * hidden;
  __half* dst = out + static_cast<long long>(token) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = row[i];
}

// Replaces the placeholder positions with encoder embeddings (bfloat16 -> fp16).
__global__ void scatter_audio_kernel(__half* __restrict__ out, const __nv_bfloat16* __restrict__ audio,
                                     int first, int audio_count, int hidden) {
  const int index = blockIdx.x;
  if (index >= audio_count) return;
  const __nv_bfloat16* row = audio + static_cast<long long>(index) * hidden;
  __half* dst = out + static_cast<long long>(first + index) * hidden;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = __float2half(__bfloat162float(row[i]));
}

__global__ void argmax_kernel(const __half* __restrict__ logits, int n, int* __restrict__ out) {
  __shared__ float values[1024];
  __shared__ int indices[1024];
  float best = -1e30f;
  int best_index = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const float value = __half2float(logits[i]);
    if (value > best) {
      best = value;
      best_index = i;
    }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = best_index;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (values[threadIdx.x + stride] > values[threadIdx.x]) {
        values[threadIdx.x] = values[threadIdx.x + stride];
        indices[threadIdx.x] = indices[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *out = indices[0];
}

}  // namespace

void rmsnorm_launch(const void* x, const void* weight, void* out, int n, float eps, cudaStream_t stream) {
  rmsnorm_kernel<<<1, 256, 0, stream>>>(static_cast<const __half*>(x), static_cast<const __half*>(weight),
                                        static_cast<__half*>(out), n, eps);
}

void embed_launch(void* out, const void* table, const void* ids, int count, int hidden, cudaStream_t stream) {
  embed_kernel<<<count, 256, 0, stream>>>(static_cast<__half*>(out), static_cast<const __half*>(table),
                                          static_cast<const int*>(ids), count, hidden);
}

void scatter_audio_launch(void* out, const void* audio, int first, int audio_count, int hidden,
                          cudaStream_t stream) {
  scatter_audio_kernel<<<audio_count, 256, 0, stream>>>(static_cast<__half*>(out),
                                                        static_cast<const __nv_bfloat16*>(audio), first,
                                                        audio_count, hidden);
}

void argmax_launch(const void* logits, int n, void* out, cudaStream_t stream) {
  argmax_kernel<<<1, 1024, 0, stream>>>(static_cast<const __half*>(logits), n, static_cast<int*>(out));
}
