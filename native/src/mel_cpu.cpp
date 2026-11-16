// CPU log-mel front end using pocketfft, the library torch's CPU STFT builds on.

#include <cmath>
#include <complex>
#include <vector>

#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"
#include "qasr.h"

namespace qasr {

namespace {

float reflect_sample(const std::vector<float>& pcm, int length, int index) {
  if (index < 0) index = -index;
  if (index >= length) index = 2 * (length - 1) - index;
  if (index < 0) index = 0;
  if (index >= length) index = length - 1;
  return pcm[static_cast<size_t>(index)];
}

}  // namespace

// Writes [n_mel][frames] log-mel features, matching WhisperFeatureExtractor.
void mel_forward_cpu(const float* samples, int length, const MelAssets& assets, float* out, int frames) {
  const int n_fft = assets.n_fft;
  const int n_freq = assets.n_freq;
  const int n_mel = assets.n_mel;
  const int hop = assets.hop;
  const int pad = n_fft / 2;

  std::vector<float> pcm(samples, samples + length);
  std::vector<float> frame(static_cast<size_t>(n_fft));
  std::vector<std::complex<float>> spectrum(static_cast<size_t>(n_freq));
  std::vector<float> power(static_cast<size_t>(n_freq));
  const pocketfft::shape_t shape_in{static_cast<size_t>(n_fft)};
  const pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(sizeof(float))};
  const pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
  const pocketfft::shape_t axes{0};

  float max_log = -INFINITY;
  for (int f = 0; f < frames; ++f) {
    const int start = f * hop;
    for (int n = 0; n < n_fft; ++n) {
      frame[static_cast<size_t>(n)] = reflect_sample(pcm, length, start - pad + n) * assets.window[n];
    }
    pocketfft::r2c(shape_in, stride_in, stride_out, axes, true, frame.data(), spectrum.data(), 1.0f);
    for (int k = 0; k < n_freq; ++k) {
      const auto value = spectrum[static_cast<size_t>(k)];
      power[static_cast<size_t>(k)] = value.real() * value.real() + value.imag() * value.imag();
    }
    for (int m = 0; m < n_mel; ++m) {
      const float* filter = &assets.filters[static_cast<size_t>(m) * n_freq];
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      int k = 0;
      for (; k + 4 <= n_freq; k += 4) {
        sum0 += filter[k] * power[static_cast<size_t>(k)];
        sum1 += filter[k + 1] * power[static_cast<size_t>(k + 1)];
        sum2 += filter[k + 2] * power[static_cast<size_t>(k + 2)];
        sum3 += filter[k + 3] * power[static_cast<size_t>(k + 3)];
      }
      for (; k < n_freq; ++k) sum0 += filter[k] * power[static_cast<size_t>(k)];
      float sum = (sum0 + sum1) + (sum2 + sum3);
      if (sum < 1e-10f) sum = 1e-10f;
      const float value = std::log10(sum);
      out[static_cast<size_t>(m) * frames + f] = value;
      if (value > max_log) max_log = value;
    }
  }

  const float floor_value = max_log - 8.0f;
  const size_t count = static_cast<size_t>(n_mel) * frames;
  for (size_t i = 0; i < count; ++i) {
    float value = out[i];
    if (value < floor_value) value = floor_value;
    out[i] = (value + 4.0f) / 4.0f;
  }
}

}  // namespace qasr
