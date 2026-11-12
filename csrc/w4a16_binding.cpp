// Host-side glue for the W4A16 GEMV: packing, scales, and the torch binding.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales, void* y,
                       int rows, int k, cudaStream_t stream);
void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales,
                       const void* residual, void* y, int rows, int k, cudaStream_t stream);
void qk_norm_rope_launch(void* vectors, const void* weight, const void* cos, const void* sin,
                         int heads, int head_dim, float eps, cudaStream_t stream);
void silu_mul_launch(const void* gate_up, void* out, int width, cudaStream_t stream);
void cache_write_launch(void* k_cache, void* v_cache, const void* k, const void* v,
                        const void* position, int total, int head_dim,
                        int max_cache, cudaStream_t stream);
void decode_attention_launch(const void* q, const void* k_cache, const void* v_cache,
                             const void* position, void* out,
                             int heads, int head_dim, int max_cache, int groups,
                             cudaStream_t stream);

namespace {

torch::Tensor w4a16_gemv(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales) {
  TORCH_CHECK(x.is_cuda() && qweight.is_cuda() && scales.is_cuda(), "tensors must be on CUDA");
  TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
  TORCH_CHECK(qweight.scalar_type() == torch::kByte, "qweight must be uint8");
  TORCH_CHECK(scales.scalar_type() == torch::kHalf, "scales must be fp16");
  TORCH_CHECK(x.dim() == 2 && x.size(0) == 1, "expects a single-token activation [1, K]");
  const int k = static_cast<int>(x.size(1));
  const int rows = static_cast<int>(qweight.size(0));
  TORCH_CHECK(qweight.size(1) == k / 2, "packed weight width must be K / 2");
  TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");

  auto y = torch::empty({1, rows}, x.options());
  w4a16_gemv_launch(x.data_ptr(), qweight.data_ptr(), scales.data_ptr(), y.data_ptr(),
                    rows, k, c10::cuda::getCurrentCUDAStream().stream());
  return y;
}

torch::Tensor w4a16_gemv_residual(torch::Tensor x, torch::Tensor qweight, torch::Tensor scales,
                                 torch::Tensor residual) {
  const int k = static_cast<int>(x.size(1));
  const int rows = static_cast<int>(qweight.size(0));
  auto y = torch::empty({1, rows}, x.options());
  w4a16_gemv_launch(x.contiguous().data_ptr(), qweight.data_ptr(), scales.data_ptr(),
                    residual.contiguous().data_ptr(), y.data_ptr(), rows, k,
                    c10::cuda::getCurrentCUDAStream().stream());
  return y;
}

void qk_norm_rope(torch::Tensor q, torch::Tensor k, torch::Tensor q_weight,
                  torch::Tensor k_weight, torch::Tensor cos, torch::Tensor sin, double eps) {
  const int head_dim = static_cast<int>(q_weight.numel());
  const int heads = static_cast<int>(q.numel()) / head_dim;
  const int kv_heads = static_cast<int>(k.numel()) / head_dim;
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  qk_norm_rope_launch(q.data_ptr(), q_weight.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                      heads, head_dim, static_cast<float>(eps), stream);
  qk_norm_rope_launch(k.data_ptr(), k_weight.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                      kv_heads, head_dim, static_cast<float>(eps), stream);
}

torch::Tensor silu_mul(torch::Tensor gate_up, int64_t width) {
  auto options = gate_up.options();
  auto out = torch::empty({1, 1, width}, options);
  silu_mul_launch(gate_up.data_ptr(), out.data_ptr(), static_cast<int>(width),
                  c10::cuda::getCurrentCUDAStream().stream());
  return out;
}

void cache_write(torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor k,
                 torch::Tensor v, torch::Tensor position) {
  const int heads = static_cast<int>(k_cache.size(1));
  const int head_dim = static_cast<int>(k_cache.size(3));
  const int max_cache = static_cast<int>(k_cache.size(2));
  cache_write_launch(k_cache.data_ptr(), v_cache.data_ptr(), k.data_ptr(), v.data_ptr(),
                     position.data_ptr(), heads * head_dim, head_dim, max_cache,
                     c10::cuda::getCurrentCUDAStream().stream());
}

torch::Tensor decode_attention(torch::Tensor q, torch::Tensor k_cache, torch::Tensor v_cache,
                               torch::Tensor position, torch::Tensor out, int64_t groups) {
  const int heads = static_cast<int>(k_cache.size(1)) * static_cast<int>(groups);
  const int head_dim = static_cast<int>(k_cache.size(3));
  const int max_cache = static_cast<int>(k_cache.size(2));
  decode_attention_launch(q.data_ptr(), k_cache.data_ptr(), v_cache.data_ptr(),
                          position.data_ptr(), out.data_ptr(), heads, head_dim, max_cache,
                          static_cast<int>(groups), c10::cuda::getCurrentCUDAStream().stream());
  return out;
}

torch::Tensor pack_w4a16(torch::Tensor weight) {
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == torch::kHalf, "weight must be fp16 CUDA");
  TORCH_CHECK(weight.dim() == 2, "weight must be [rows, K]");
  const auto rows = weight.size(0);
  const auto k = weight.size(1);
  TORCH_CHECK(k % 64 == 0, "K must be a multiple of 64");

  auto values = weight.to(torch::kFloat).view({rows, k / 64, 64});
  auto scales = values.abs().amax(-1) / 7.0;
  scales = torch::where(scales > 0, scales, torch::ones_like(scales)).to(torch::kHalf);
  auto quantized = torch::round(values / scales.to(torch::kFloat).unsqueeze(-1))
                       .clamp(-7.0, 7.0)
                       .to(torch::kInt8)
                       .view({rows, k});
  auto nibbles = (quantized & 0xF).to(torch::kInt32);
  auto packed = (nibbles.slice(1, 0, k, 2) | (nibbles.slice(1, 1, k, 2) * 16)).to(torch::kByte);
  return packed.contiguous();
}

torch::Tensor scale_w4a16(torch::Tensor weight) {
  const auto rows = weight.size(0);
  const auto k = weight.size(1);
  auto values = weight.to(torch::kFloat).view({rows, k / 64, 64});
  auto scales = values.abs().amax(-1) / 7.0;
  return torch::where(scales > 0, scales, torch::ones_like(scales)).to(torch::kHalf).contiguous();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemv", &w4a16_gemv, "W4A16 GEMV (decode step)");
  m.def("gemv_residual", &w4a16_gemv_residual, "W4A16 GEMV with residual add");
  m.def("qk_norm_rope", &qk_norm_rope, "fused q/k RMSNorm + RoPE (in place)");
  m.def("silu_mul", &silu_mul, "fused silu(gate) * up");
  m.def("attention", &decode_attention, "single-query decode attention with device length");
  m.def("cache_write", &cache_write, "append one token to the GQA KV cache");
  m.def("pack", &pack_w4a16, "pack fp16 weights to INT4");
  m.def("scale", &scale_w4a16, "per-group INT4 scales");
}

}  // namespace
