#include "qasr.h"

#include <cuda_runtime.h>

#include <cmath>
#include <utility>
#include <vector>

#include "qasr.h"

void mel_forward_launch(const void* pcm, int length, const void* twiddle, const void* window,
                        const void* filters, int n_mel, int n_freq, int n_fft, int hop, int frames,
                        void* out, void* max_log, cudaStream_t stream);

namespace qasr {

namespace {

void* upload(const std::vector<float>& values) {
  if (values.empty()) return nullptr;
  void* device = nullptr;
  cuda_check(cudaMalloc(&device, values.size() * sizeof(float)), "cudaMalloc mel table");
  cuda_check(cudaMemcpy(device, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice),
             "copy mel table");
  return device;
}

}  // namespace

MelDeviceTables::~MelDeviceTables() {
  for (void* pointer : {cos, sin, window, filters}) {
    if (pointer) cudaFree(pointer);
  }
}

MelDeviceTables& MelDeviceTables::operator=(MelDeviceTables&& other) noexcept {
  if (this != &other) {
    std::swap(cos, other.cos);
    std::swap(sin, other.sin);
    std::swap(window, other.window);
    std::swap(filters, other.filters);
  }
  return *this;
}

int mel_frame_count(int length, int hop, int multiple, int& valid_frames) {
  if (length < 8000) length = 8000;
  valid_frames = length / hop;
  int frames = valid_frames;
  if (multiple > 1 && frames % multiple) frames += multiple - frames % multiple;
  return frames;
}

MelDeviceTables upload_mel_tables(const MelAssets& assets) {
  const int n_fft = assets.n_fft;
  const int n_freq = assets.n_freq;
  std::vector<float> twiddle(static_cast<size_t>(n_freq) * n_fft * 2);
  // Rotate a unit vector instead of calling sin/cos 80k times.
  const double two_pi = 6.283185307179586476925286766559;
  for (int k = 0; k < n_freq; ++k) {
    const double step = two_pi * k / n_fft;
    const double step_cos = std::cos(step);
    const double step_sin = std::sin(step);
    double current_cos = 1.0;
    double current_sin = 0.0;
    for (int n = 0; n < n_fft; ++n) {
      const size_t at = (static_cast<size_t>(k) * n_fft + n) * 2;
      twiddle[at] = static_cast<float>(current_cos);
      twiddle[at + 1] = static_cast<float>(current_sin);
      const double next_cos = current_cos * step_cos - current_sin * step_sin;
      current_sin = current_sin * step_cos + current_cos * step_sin;
      current_cos = next_cos;
    }
  }
  MelDeviceTables tables;
  tables.cos = upload(twiddle);
  tables.window = upload(assets.window);
  tables.filters = upload(assets.filters);
  return tables;
}

void mel_forward_gpu(const float* pcm, int length, const MelDeviceTables& tables, const MelAssets& assets,
                     float* out, int frames) {
  static float* max_log = nullptr;
  if (max_log == nullptr) {
    cuda_check(cudaMalloc(&max_log, sizeof(float)), "cudaMalloc mel max");
  }
  mel_forward_launch(pcm, length, tables.cos, tables.window, tables.filters, assets.n_mel, assets.n_freq,
                     assets.n_fft, assets.hop, frames, out, max_log, nullptr);
}

}  // namespace qasr
