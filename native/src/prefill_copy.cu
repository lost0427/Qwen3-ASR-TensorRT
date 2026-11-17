// Copies the prefill KV cache into the per-layer decode caches in one launch.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kRowTile = 64;

// Source layout is (seq, layers, kv_heads, head_dim); each decode cache is
// (kv_heads, capacity, head_dim). A block copies a tile of rows for one
// (layer, head, K-or-V) slice, and blocks are ordered so that every tile is
// read from a contiguous region of the source: a whole prefill needs a single
// kernel instead of 2 * layers * kv_heads 2-D memcopies.
__global__ void prefill_cache_copy_kernel(const __half* __restrict__ src_k,
                                          const __half* __restrict__ src_v,
                                          __half* const* __restrict__ dst_table, int layers,
                                          int kv_heads, int head_dim, int capacity, int seq) {
  const int head = blockIdx.x;
  const int layer = blockIdx.y >> 1;
  const int which = blockIdx.y & 1;
  const int row_begin = blockIdx.z * kRowTile;
  const int row_end = min(row_begin + kRowTile, seq);
  if (row_begin >= row_end) return;

  const __half* src = (which == 0 ? src_k : src_v) +
                      (static_cast<long long>(layer) * kv_heads + head) * head_dim;
  __half* dst = dst_table[which * layers + layer] +
                static_cast<long long>(head) * capacity * head_dim;
  const long long src_stride = static_cast<long long>(layers) * kv_heads * head_dim;
  const int tid = threadIdx.x;
  const int rows = row_end - row_begin;

  if ((head_dim % 8) == 0 && head_dim / 8 <= static_cast<int>(blockDim.x)) {
    // head_dim * 2 bytes per row is a multiple of 16, so the rows are uint4
    // aligned. Threads are laid out per row so no index division is needed.
    const int chunks = head_dim / 8;
    const int lane = tid % chunks;
    const int row_step = blockDim.x / chunks;
    for (int row = tid / chunks; row < rows; row += row_step) {
      *reinterpret_cast<uint4*>(dst + static_cast<long long>(row + row_begin) * head_dim +
                                static_cast<long long>(lane) * 8) =
          *reinterpret_cast<const uint4*>(src + static_cast<long long>(row + row_begin) * src_stride +
                                          static_cast<long long>(lane) * 8);
    }
  } else {
    const long long total = static_cast<long long>(rows) * head_dim;
    for (long long index = tid; index < total; index += blockDim.x) {
      const long long row = index / head_dim;
      const int column = static_cast<int>(index - row * head_dim);
      dst[(row + row_begin) * head_dim + column] = src[(row + row_begin) * src_stride + column];
    }
  }
}

}  // namespace

void prefill_cache_copy_launch(const void* src_k, const void* src_v, void* dst_table, int layers,
                               int kv_heads, int head_dim, int capacity, int seq,
                               cudaStream_t stream) {
  const int tiles = (seq + kRowTile - 1) / kRowTile;
  const dim3 grid(kv_heads, layers * 2, tiles);
  prefill_cache_copy_kernel<<<grid, 256, 0, stream>>>(
      static_cast<const __half*>(src_k), static_cast<const __half*>(src_v),
      static_cast<__half* const*>(dst_table), layers, kv_heads, head_dim, capacity, seq);
}
