// CPU log-mel front end using pocketfft, the library torch's CPU STFT builds on.

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

#define POCKETFFT_CACHE_SIZE 16
#include "pocketfft_hdronly.h"
#include "qasr.h"

namespace qasr {

namespace {

}  // namespace

// Writes [n_mel][frames] log-mel features, matching WhisperFeatureExtractor.
void mel_forward_cpu(const float* padded, int valid_frames, int frames, const MelAssets& assets,
                     float* out) {
  const int n_fft = assets.n_fft;
  const int n_freq = assets.n_freq;
  const int n_mel = assets.n_mel;
  const int hop = assets.hop;

  std::vector<float> frame(static_cast<size_t>(n_fft));
  std::vector<std::complex<float>> spectrum(static_cast<size_t>(n_freq));
  std::vector<float> power(static_cast<size_t>(n_freq));
  const pocketfft::shape_t shape_in{static_cast<size_t>(n_fft)};
  const pocketfft::stride_t stride_in{static_cast<ptrdiff_t>(sizeof(float))};
  const pocketfft::stride_t stride_out{static_cast<ptrdiff_t>(sizeof(std::complex<float>))};
  const pocketfft::shape_t axes{0};

  // Frames are produced contiguously in [valid_frames][n_mel] order and then
  // transposed, because writing out[m * frames + f] per frame touches one line
  // per mel bin and costs more than the FFT itself on long inputs.
  std::vector<float> rows(static_cast<size_t>(valid_frames) * n_mel);
  // The mel filters are triangular, so each row has one contiguous nonzero span
  // and the dense 80 x 201 product wastes most of its multiply-adds.
  std::vector<int> span_begin(static_cast<size_t>(n_mel));
  std::vector<int> span_end(static_cast<size_t>(n_mel));
  for (int m = 0; m < n_mel; ++m) {
    const float* filter = &assets.filters[static_cast<size_t>(m) * n_freq];
    int begin = 0;
    while (begin < n_freq && filter[begin] == 0.0f) ++begin;
    int end = n_freq;
    while (end > begin && filter[end - 1] == 0.0f) --end;
    span_begin[static_cast<size_t>(m)] = begin;
    span_end[static_cast<size_t>(m)] = end;
  }
  float max_log = -INFINITY;
  float* row = rows.data();
  for (int f = 0; f < valid_frames; ++f, row += n_mel) {
    const int start = f * hop;
    for (int n = 0; n < n_fft; ++n) {
      frame[static_cast<size_t>(n)] = padded[start + n] * assets.window[n];
    }
    pocketfft::r2c(shape_in, stride_in, stride_out, axes, true, frame.data(), spectrum.data(), 1.0f);
    for (int k = 0; k < n_freq; ++k) {
      const auto value = spectrum[static_cast<size_t>(k)];
      power[static_cast<size_t>(k)] = value.real() * value.real() + value.imag() * value.imag();
    }
    for (int m = 0; m < n_mel; ++m) {
      const float* filter = &assets.filters[static_cast<size_t>(m) * n_freq];
      float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
      const int begin = span_begin[static_cast<size_t>(m)];
      const int end = span_end[static_cast<size_t>(m)];
      int k = begin;
      for (; k + 4 <= end; k += 4) {
        sum0 += filter[k] * power[static_cast<size_t>(k)];
        sum1 += filter[k + 1] * power[static_cast<size_t>(k + 1)];
        sum2 += filter[k + 2] * power[static_cast<size_t>(k + 2)];
        sum3 += filter[k + 3] * power[static_cast<size_t>(k + 3)];
      }
      for (; k < end; ++k) sum0 += filter[k] * power[static_cast<size_t>(k)];
      float sum = (sum0 + sum1) + (sum2 + sum3);
      if (sum < 1e-10f) sum = 1e-10f;
      const float value = std::log10(sum);
      row[m] = value;
      if (value > max_log) max_log = value;
    }
  }

  // Blocked transpose that fuses the floor and the (x + 4) / 4 scaling.
  const float floor_value = max_log - 8.0f;
  constexpr int kBlock = 32;
  for (int f0 = 0; f0 < valid_frames; f0 += kBlock) {
    const int f1 = std::min(f0 + kBlock, valid_frames);
    for (int m0 = 0; m0 < n_mel; m0 += kBlock) {
      const int m1 = std::min(m0 + kBlock, n_mel);
      for (int m = m0; m < m1; ++m) {
        float* dst = out + static_cast<size_t>(m) * frames + f0;
        const float* src = rows.data() + static_cast<size_t>(f0) * n_mel + m;
        for (int f = f0; f < f1; ++f) {
          float value = src[static_cast<size_t>(f - f0) * n_mel];
          if (value < floor_value) value = floor_value;
          dst[f - f0] = (value + 4.0f) / 4.0f;
        }
      }
    }
  }
}

}  // namespace qasr
