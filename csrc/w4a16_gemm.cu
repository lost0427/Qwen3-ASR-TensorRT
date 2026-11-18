// Batched W4A16 GEMM: y[m, n] = sum_k x[m, k] * dequant(qweight[n, k], scale[n, k / group])
//
// Speculative verification feeds K candidate tokens through the same weights, so
// M = K is small (2..8) and the weights dominate the traffic. One warp owns
// kRowGroup output rows at a time and keeps M accumulators per row: the weight
// bytes are read once per row, and the M activation rows are loaded once per
// kRowGroup rows instead of once per row.
//
// Layout matches w4a16_kernel.cu: two INT4 values per byte along the contraction
// axis, low nibble holds the even column, one FP16 scale per `group` values.

#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 8;
constexpr int kBlock = kWarps * 32;
constexpr int kNibblesPerLane = 32;
constexpr int kValuesPerWarpStep = 32 * kNibblesPerLane;  // 1024
// Output rows per warp iteration. Every extra row reuses the activation
// registers already loaded for the block, which is what the single-row kernel
// spent most of its time re-reading.
constexpr int kRowGroup = 4;

template <int kGroup, int kMaxRows>
__global__ void w4a16_gemm_kernel(const __half* __restrict__ x,
                                  const uint8_t* __restrict__ qweight,
                                  const __half* __restrict__ scales,
                                  const __half* __restrict__ residual,
                                  __half* __restrict__ y,
                                  int rows, int k, int m_rows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int groups = k / kGroup;
  const int packed_row = k >> 1;
  const int steps = k / kValuesPerWarpStep;
  constexpr int kLanesPerGroup = kGroup / kNibblesPerLane;
  constexpr int kGroupsPerStep = kValuesPerWarpStep / kGroup;
  const int lane_group = lane / kLanesPerGroup;

  const int stride = gridDim.x * kWarps * kRowGroup;
  for (int group_base = blockIdx.x * kWarps * kRowGroup + warp * kRowGroup; group_base < rows;
       group_base += stride) {
    const int count = min(kRowGroup, rows - group_base);
    float acc[kRowGroup][kMaxRows];
#pragma unroll
    for (int r = 0; r < kRowGroup; ++r) {
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) acc[r][m] = 0.f;
    }

    for (int step = 0; step < steps; ++step) {
      const int base = step * kValuesPerWarpStep + lane * kNibblesPerLane;
      uint4 packed[kRowGroup];
      float scale[kRowGroup];
#pragma unroll
      for (int r = 0; r < kRowGroup; ++r) {
        if (r < count) {
          const long row_index = static_cast<long>(group_base + r);
          packed[r] = __ldcs(reinterpret_cast<const uint4*>(
              qweight + row_index * packed_row + (base >> 1)));
          scale[r] = __half2float(
              scales[row_index * groups + step * kGroupsPerStep + lane_group]);
        }
      }

      float lo[kRowGroup][kMaxRows];
      float hi[kRowGroup][kMaxRows];
#pragma unroll
      for (int r = 0; r < kRowGroup; ++r) {
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) {
          lo[r][m] = 0.f;
          hi[r][m] = 0.f;
        }
      }
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        // One uint4 of every input row covers this word's four pairs.
        uint4 chunk[kMaxRows];
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) {
          if (m < m_rows) {
            chunk[m] = __ldg(reinterpret_cast<const uint4*>(
                &x[static_cast<long>(m) * k + base + (word << 3)]));
          }
        }
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
#pragma unroll
          for (int r = 0; r < kRowGroup; ++r) {
            if (r < count) {
              const uint32_t word_value = reinterpret_cast<const uint32_t*>(&packed[r])[word];
              const uint32_t byte = (word_value >> (8 * pair)) & 0xFFu;
              const int lo_value = ((byte & 0xF) ^ 8) - 8;
              const int hi_value = (((byte >> 4) & 0xF) ^ 8) - 8;
#pragma unroll
              for (int m = 0; m < kMaxRows; ++m) {
                if (m < m_rows) {
                  const uint32_t packed_pair =
                      reinterpret_cast<const uint32_t*>(&chunk[m])[pair];
                  const __half2 xv = *reinterpret_cast<const __half2*>(&packed_pair);
                  const float2 pair_value = __half22float2(xv);
                  lo[r][m] += lo_value * pair_value.x;
                  hi[r][m] += hi_value * pair_value.y;
                }
              }
            }
          }
        }
      }
#pragma unroll
      for (int r = 0; r < kRowGroup; ++r) {
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) acc[r][m] += (lo[r][m] + hi[r][m]) * scale[r];
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int r = 0; r < kRowGroup; ++r) {
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) acc[r][m] += __shfl_xor_sync(0xffffffffu, acc[r][m], offset);
      }
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kRowGroup; ++r) {
        if (r >= count) break;
        const long row_index = static_cast<long>(group_base + r);
#pragma unroll
        for (int m = 0; m < kMaxRows; ++m) {
          if (m < m_rows) {
            const float total =
                residual == nullptr
                    ? acc[r][m]
                    : acc[r][m] + __half2float(residual[static_cast<long>(m) * rows + row_index]);
            y[static_cast<long>(m) * rows + row_index] = __float2half(total);
          }
        }
      }
    }
  }
}

template <int kGroup, int kMaxRows>
void launch_one(const void* x, const void* qweight, const void* scales, const void* residual,
                void* y, int rows, int k, int m_rows, cudaStream_t stream) {
  const int per_block = kWarps * kRowGroup;
  const int needed = (rows + per_block - 1) / per_block;
  const int grid = needed < 65535 ? needed : 65535;
  w4a16_gemm_kernel<kGroup, kMaxRows><<<grid, kBlock, 0, stream>>>(
      reinterpret_cast<const __half*>(x), reinterpret_cast<const uint8_t*>(qweight),
      reinterpret_cast<const __half*>(scales), reinterpret_cast<const __half*>(residual),
      reinterpret_cast<__half*>(y), rows, k, m_rows);
}

template <int kGroup>
void launch_rows(const void* x, const void* qweight, const void* scales, const void* residual,
                 void* y, int rows, int k, int m_rows, cudaStream_t stream) {
  switch (m_rows) {
    case 1: launch_one<kGroup, 1>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 2: launch_one<kGroup, 2>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 3: launch_one<kGroup, 3>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 4: launch_one<kGroup, 4>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 5: launch_one<kGroup, 5>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 6: launch_one<kGroup, 6>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    case 7: launch_one<kGroup, 7>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
    default: launch_one<kGroup, 8>(x, qweight, scales, residual, y, rows, k, m_rows, stream); break;
  }
}

}  // namespace

void w4a16_gemm_launch(const void* x, const void* qweight, const void* scales, const void* residual,
                       void* y, int rows, int k, int m_rows, int group, cudaStream_t stream) {
  if (group == 32) {
    launch_rows<32>(x, qweight, scales, residual, y, rows, k, m_rows, stream);
  } else {
    launch_rows<64>(x, qweight, scales, residual, y, rows, k, m_rows, stream);
  }
}
