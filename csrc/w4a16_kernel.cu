// W4A16 GEMV: y[n] = sum_k x[k] * dequant(qweight[n, k], scale[n, k / 64])
//
// Decode always has M = 1, so this is a bandwidth-bound GEMV. Weights are packed
// two INT4 values per byte along the contraction axis (low nibble = even k) and
// scales are one FP16 value per 64-element group along K.
//
// Each lane owns 8 consecutive nibbles, i.e. 8 lanes cover one 64-element group,
// so a lane's 8 products share a single scale. Summing the scaled partials at the
// end means one warp reduction per row instead of one per group.
//
// Kept free of torch headers so nvcc does not blow up on template instantiation.

#include <cuda_fp16.h>
#include <cstdint>

namespace {

constexpr int kWarps = 8;
constexpr int kBlock = kWarps * 32;
constexpr int kNibblesPerLane = 8;
constexpr int kValuesPerWarpStep = 32 * kNibblesPerLane;  // 256

__global__ void w4a16_gemv_kernel(const __half* __restrict__ x,
                                  const uint8_t* __restrict__ qweight,
                                  const __half* __restrict__ scales,
                                  const __half* __restrict__ residual,
                                  __half* __restrict__ y,
                                  int rows, int k) {
  extern __shared__ __half shared_x[];
  for (int index = threadIdx.x; index < k; index += kBlock) {
    shared_x[index] = x[index];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int groups = k >> 6;
  const int packed_row = k >> 1;
  const int steps = k / kValuesPerWarpStep;
  const int lane_group = lane >> 3;   // 4 groups live in one 256-value step
  const int group_scale = lane & 7;   // 8 lanes share each group

  for (int n = blockIdx.x * kWarps + warp; n < rows; n += gridDim.x * kWarps) {
    const uint8_t* row = qweight + static_cast<long>(n) * packed_row;
    const __half* row_scales = scales + static_cast<long>(n) * groups;
    float acc = 0.f;
    for (int step = 0; step < steps; ++step) {
      const int base = step * kValuesPerWarpStep + lane * kNibblesPerLane;
      const uint32_t packed =
          *reinterpret_cast<const uint32_t*>(row + (step * kValuesPerWarpStep + lane * kNibblesPerLane) / 2);
      const uint4 xchunk = *reinterpret_cast<const uint4*>(&shared_x[base]);
      const __half2* x2 = reinterpret_cast<const __half2*>(&xchunk);
      float lo_acc = 0.f;
      float hi_acc = 0.f;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const uint32_t byte = (packed >> (8 * pair)) & 0xFFu;
        const int lo = ((byte & 0xF) ^ 8) - 8;
        const int hi = (((byte >> 4) & 0xF) ^ 8) - 8;
        const float2 xv = __half22float2(x2[pair]);
        lo_acc += lo * xv.x;
        hi_acc += hi * xv.y;
      }
      const int scale_index = step * 4 + lane_group;
      acc += (lo_acc + hi_acc) * __half2float(row_scales[scale_index]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
      const float total = residual == nullptr ? acc : acc + __half2float(residual[n]);
      y[n] = __float2half(total);
    }
    (void)group_scale;
  }
}

}  // namespace

void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales,
                       const void* residual, void* y, int rows, int k, cudaStream_t stream) {
  const int grid = (rows + kWarps - 1) / kWarps < 65535 ? (rows + kWarps - 1) / kWarps : 65535;
  w4a16_gemv_kernel<<<grid, kBlock, k * sizeof(__half), stream>>>(
      reinterpret_cast<const __half*>(x),
      reinterpret_cast<const uint8_t*>(qweight),
      reinterpret_cast<const __half*>(scales),
      reinterpret_cast<const __half*>(residual),
      reinterpret_cast<__half*>(y),
      rows, k);
}

void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales, void* y,
                       int rows, int k, cudaStream_t stream) {
  w4a16_gemv_launch(x, qweight, scales, nullptr, y, rows, k, stream);
}
