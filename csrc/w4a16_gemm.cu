// Batched W4A16 GEMM: y[m, n] = sum_k x[m, k] * dequant(qweight[n, k], scale[n, k / group])
//
// Speculative verification feeds K candidate tokens through the same weights, so
// M = K is small (2..8) and the weights still dominate the traffic. Each warp
// owns one output row and keeps M accumulators, so every weight byte is read
// once per row instead of once per (row, token) pair as a GEMV loop would.
//
// The M activations are staged in registers: one lane covers 32 values of each
// row per step, loaded as two 16-byte vectors. Staging them in shared memory
// measured slower than the L1 hits this produces.
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

  for (int n = blockIdx.x * kWarps + warp; n < rows; n += gridDim.x * kWarps) {
    const uint8_t* row = qweight + static_cast<long>(n) * packed_row;
    const __half* row_scales = scales + static_cast<long>(n) * groups;
    float acc[kMaxRows];
#pragma unroll
    for (int m = 0; m < kMaxRows; ++m) acc[m] = 0.f;
    for (int step = 0; step < steps; ++step) {
      const int base = step * kValuesPerWarpStep + lane * kNibblesPerLane;
      const uint4 packed = __ldcs(reinterpret_cast<const uint4*>(row + (base >> 1)));
      const uint32_t* words = reinterpret_cast<const uint32_t*>(&packed);
      const float scale = __half2float(row_scales[step * kGroupsPerStep + lane_group]);
      // One lane owns 32 halves of every input row: four 16-byte vectors cover them.
      uint4 chunk[kMaxRows][4];
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) {
        if (m < m_rows) {
          const uint4* source = reinterpret_cast<const uint4*>(
              &x[static_cast<long>(m) * k + base]);
#pragma unroll
          for (int part = 0; part < 4; ++part) chunk[m][part] = __ldg(source + part);
        }
      }
      float lo_acc[kMaxRows];
      float hi_acc[kMaxRows];
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) {
        lo_acc[m] = 0.f;
        hi_acc[m] = 0.f;
      }
#pragma unroll
      for (int word = 0; word < 4; ++word) {
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
          const uint32_t byte = (words[word] >> (8 * pair)) & 0xFFu;
          const int lo = ((byte & 0xF) ^ 8) - 8;
          const int hi = (((byte >> 4) & 0xF) ^ 8) - 8;
          const int slot = word * 4 + pair;
#pragma unroll
          for (int m = 0; m < kMaxRows; ++m) {
            if (m < m_rows) {
              const uint4* vectors = reinterpret_cast<const uint4*>(chunk[m]);
              const uint32_t packed_pair = reinterpret_cast<const uint32_t*>(&vectors[slot >> 2])[slot & 3];
              const __half2 xv = *reinterpret_cast<const __half2*>(&packed_pair);
              const float2 pair_value = __half22float2(xv);
              lo_acc[m] += lo * pair_value.x;
              hi_acc[m] += hi * pair_value.y;
            }
          }
        }
      }
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) acc[m] += (lo_acc[m] + hi_acc[m]) * scale;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], offset);
    }
    if (lane == 0) {
#pragma unroll
      for (int m = 0; m < kMaxRows; ++m) {
        if (m < m_rows) {
          const float total = residual == nullptr
                                  ? acc[m]
                                  : acc[m] + __half2float(residual[static_cast<long>(m) * rows + n]);
          y[static_cast<long>(m) * rows + n] = __float2half(total);
        }
      }
    }
  }
}

template <int kGroup, int kMaxRows>
void launch_one(const void* x, const void* qweight, const void* scales, const void* residual,
                void* y, int rows, int k, int m_rows, cudaStream_t stream) {
  const int needed = (rows + kWarps - 1) / kWarps;
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
