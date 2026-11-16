// Log-mel front end on the GPU: windowed DFT, power spectrum, mel accumulation,
// log10 clamp and the /4 normalisation, in two passes over the frames.

#include <cuda_runtime.h>

#include <cmath>

namespace {

__device__ float atomic_max_float(float* address, float value) {
  // Works for negatives too: the unsigned order is reversed below zero.
  if (value >= 0.0f) {
    return __int_as_float(atomicMax(reinterpret_cast<int*>(address), __float_as_int(value)));
  }
  return __uint_as_float(atomicMin(reinterpret_cast<unsigned int*>(address), __float_as_uint(value)));
}

__device__ float reflect_sample(const float* pcm, int length, int index) {
  if (index < 0) index = -index;
  if (index >= length) index = 2 * (length - 1) - index;
  if (index < 0) index = 0;
  if (index >= length) index = length - 1;
  return pcm[index];
}

__global__ void mel_frames_kernel(const float* __restrict__ pcm, int length,
                                  const float2* __restrict__ twiddle,
                                  const float* __restrict__ window,
                                  const float* __restrict__ filters, int n_mel, int n_freq, int n_fft,
                                  int hop, int pad, int frames, float* __restrict__ out,
                                  float* __restrict__ max_log) {
  extern __shared__ float shared[];
  float* frame = shared;              // n_fft
  float* power = frame + n_fft;       // n_freq
  float* bins = power + n_freq;       // n_mel

  const int index = blockIdx.x;
  const int tid = threadIdx.x;

  const int start = index * hop;
  for (int n = tid; n < n_fft; n += blockDim.x) {
    frame[n] = reflect_sample(pcm, length, start - pad + n) * window[n];
  }
  __syncthreads();

  for (int k = tid; k < n_freq; k += blockDim.x) {
    const float2* row = twiddle + static_cast<long long>(k) * n_fft;
    float re = 0.0f;
    float im = 0.0f;
    for (int n = 0; n < n_fft; ++n) {
      const float value = frame[n];
      const float2 factor = row[n];
      re += value * factor.x;
      im -= value * factor.y;
    }
    power[k] = re * re + im * im;
  }
  __syncthreads();

  for (int m = tid; m < n_mel; m += blockDim.x) {
    const float* filter = filters + static_cast<long long>(m) * n_freq;
    float sum = 0.0f;
    for (int k = 0; k < n_freq; ++k) sum += filter[k] * power[k];
    if (sum < 1e-10f) sum = 1e-10f;
    const float value = log10f(sum);
    bins[m] = value;
    out[static_cast<long long>(m) * frames + index] = value;
    atomic_max_float(max_log, value);
  }
}

__global__ void mel_finish_kernel(float* __restrict__ out, long long count, const float* __restrict__ max_log) {
  const float floor_value = max_log[0] - 8.0f;
  for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    float value = out[i];
    if (value < floor_value) value = floor_value;
    out[i] = (value + 4.0f) / 4.0f;
  }
}

__global__ void fill_initial_max(float* value) { value[0] = -INFINITY; }

}  // namespace

void mel_forward_launch(const void* pcm, int length, const void* twiddle, const void* window,
                        const void* filters, int n_mel, int n_freq, int n_fft, int hop, int frames,
                        void* out, void* max_log, cudaStream_t stream) {
  fill_initial_max<<<1, 1, 0, stream>>>(static_cast<float*>(max_log));
  const int pad = n_fft / 2;
  const size_t shared_bytes = static_cast<size_t>(n_fft + n_freq + n_mel) * sizeof(float);
  mel_frames_kernel<<<frames, 128, shared_bytes, stream>>>(
      static_cast<const float*>(pcm), length, static_cast<const float2*>(twiddle),
      static_cast<const float*>(window), static_cast<const float*>(filters), n_mel, n_freq, n_fft, hop,
      pad, frames, static_cast<float*>(out), static_cast<float*>(max_log));
  const long long count = static_cast<long long>(n_mel) * frames;
  mel_finish_kernel<<<256, 256, 0, stream>>>(static_cast<float*>(out), count,
                                             static_cast<const float*>(max_log));
}
