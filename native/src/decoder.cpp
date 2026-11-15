#include "decoder.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

#include "qasr.h"

// Kernels shared with the Python runtime (csrc/).
void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales, void* y, int rows, int k,
                       int group, cudaStream_t stream);
void w4a16_gemv_launch(const void* x, const void* qweight, const void* scales, const void* residual,
                       void* y, int rows, int k, int group, cudaStream_t stream);
void qk_norm_rope_launch(void* vectors, const void* weight, const void* cos, const void* sin, int heads,
                         int head_dim, float eps, cudaStream_t stream);
void silu_mul_launch(const void* gate_up, void* out, int width, cudaStream_t stream);
void cache_write_launch(void* k_cache, void* v_cache, const void* k, const void* v, const void* position,
                        int total, int head_dim, int max_cache, cudaStream_t stream);
void decode_attention_launch(const void* q, const void* k_cache, const void* v_cache, const void* position,
                             void* out, int heads, int head_dim, int max_cache, int groups,
                             cudaStream_t stream);

// Kernels this runtime adds (ops.cu).
void rmsnorm_launch(const void* x, const void* weight, void* out, int n, float eps, cudaStream_t stream);
void embed_launch(void* out, const void* table, const void* ids, int count, int hidden, cudaStream_t stream);
void scatter_audio_launch(void* out, const void* audio, int first, int audio_count, int hidden,
                          cudaStream_t stream);
void argmax_launch(const void* logits, int n, void* out, cudaStream_t stream);

namespace qasr {

namespace {

int find_int(const std::string& text, const char* key, int fallback = -1) {
  const std::string pattern = std::string("\"") + key + "\"";
  size_t position = text.find(pattern);
  while (position != std::string::npos) {
    const size_t colon = text.find(':', position + pattern.size());
    if (colon == std::string::npos) break;
    size_t index = colon + 1;
    while (index < text.size() && (text[index] == ' ' || text[index] == '\n')) ++index;
    if (index < text.size() && (std::isdigit(static_cast<unsigned char>(text[index])) || text[index] == '-')) {
      return std::atoi(text.c_str() + index);
    }
    position = text.find(pattern, position + 1);
  }
  return fallback;
}

double find_double(const std::string& text, const char* key, double fallback = -1.0) {
  const std::string pattern = std::string("\"") + key + "\"";
  const size_t position = text.find(pattern);
  if (position == std::string::npos) return fallback;
  const size_t colon = text.find(':', position);
  if (colon == std::string::npos) return fallback;
  return std::atof(text.c_str() + colon + 1);
}

struct TensorView {
  const uint8_t* data = nullptr;
  std::vector<int64_t> shape;
  bool fp16 = false;
};

std::map<std::string, TensorView> load_safetensors(const std::string& path, std::vector<uint8_t>& blob) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  blob.assign(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
  if (blob.size() < 8) throw std::runtime_error("bad safetensors file");
  uint64_t header_size = 0;
  std::memcpy(&header_size, blob.data(), 8);
  const std::string header(reinterpret_cast<const char*>(blob.data() + 8), header_size);
  const uint8_t* base = blob.data() + 8 + header_size;

  std::map<std::string, TensorView> tensors;
  size_t position = 0;
  while ((position = header.find("\"dtype\"", position)) != std::string::npos) {
    // walk back to the key that owns this object
    size_t key_start = header.rfind('"', header.rfind('"', position) - 1);
    const size_t name_end = header.rfind('"', position - 1);
    while (key_start > 0 && header[key_start - 1] != '"') --key_start;
    const std::string name = header.substr(key_start, name_end - key_start);
    const size_t dtype_pos = header.find(':', position);
    const size_t dtype_begin = header.find('"', dtype_pos) + 1;
    const size_t dtype_end = header.find('"', dtype_begin);
    const std::string dtype = header.substr(dtype_begin, dtype_end - dtype_begin);

    TensorView view;
    view.fp16 = dtype == "F16";
    const size_t shape_pos = header.find("\"shape\"", dtype_end);
    const size_t open = header.find('[', shape_pos);
    const size_t close = header.find(']', open);
    std::string shape_text = header.substr(open + 1, close - open - 1);
    size_t cursor = 0;
    while (cursor < shape_text.size()) {
      while (cursor < shape_text.size() &&
             !std::isdigit(static_cast<unsigned char>(shape_text[cursor]))) ++cursor;
      if (cursor >= shape_text.size()) break;
      view.shape.push_back(std::atoll(shape_text.c_str() + cursor));
      while (cursor < shape_text.size() &&
             std::isdigit(static_cast<unsigned char>(shape_text[cursor]))) ++cursor;
    }
    const size_t offset_pos = header.find("\"data_offsets\"", close);
    const size_t offset_open = header.find('[', offset_pos);
    const size_t offset_close = header.find(']', offset_open);
    const std::string offsets = header.substr(offset_open + 1, offset_close - offset_open - 1);
    const size_t comma = offsets.find(',');
    const long long begin = std::atoll(offsets.c_str());
    const long long end = std::atoll(offsets.c_str() + comma + 1);
    view.data = base + begin;
    (void)end;
    tensors[name] = view;
    position = offset_close;
  }
  return tensors;
}

void* upload(const void* source, size_t bytes) {
  void* device = nullptr;
  cuda_check(cudaMalloc(&device, bytes), "cudaMalloc");
  cuda_check(cudaMemcpy(device, source, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
  return device;
}

size_t tensor_bytes(const TensorView& view) {
  size_t count = 1;
  for (int64_t dim : view.shape) count *= static_cast<size_t>(dim);
  return count * (view.fp16 ? 2 : 1);
}

}  // namespace

ModelConfig read_model_config(const std::string& directory) {
  std::ifstream stream(directory + "/config.json");
  if (!stream) throw std::runtime_error("cannot open config.json in " + directory);
  std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  // The audio tower repeats several of these keys, so read the text section only.
  const size_t text_config = text.find("\"text_config\"");
  if (text_config != std::string::npos) text = text.substr(text_config);
  ModelConfig config;
  config.hidden = find_int(text, "hidden_size");
  config.layers = find_int(text, "num_hidden_layers");
  config.heads = find_int(text, "num_attention_heads");
  config.kv_heads = find_int(text, "num_key_value_heads");
  config.head_dim = find_int(text, "head_dim");
  config.intermediate = find_int(text, "intermediate_size");
  config.vocab = find_int(text, "vocab_size");
  config.eps = static_cast<float>(find_double(text, "rms_norm_eps", 1e-6));
  config.rope_theta = static_cast<float>(find_double(text, "rope_theta", 1000000.0));
  if (config.hidden <= 0 || config.layers <= 0) throw std::runtime_error("incomplete config.json");
  return config;
}

Decoder::Decoder(const std::string& directory, int max_cache) {
  config_ = read_model_config(directory);
  if (std::getenv("QASR_DEBUG")) {
    std::fprintf(stderr,
                 "[cfg] hidden %d layers %d heads %d kv %d hd %d inter %d vocab %d eps %.3e theta %.0f\n",
                 config_.hidden, config_.layers, config_.heads, config_.kv_heads, config_.head_dim,
                 config_.intermediate, config_.vocab, config_.eps, config_.rope_theta);
  }
  capacity_ = max_cache;
  load_weights(directory);
  allocate_cache(capacity_);
}

Decoder::~Decoder() {
  auto free_ptr = [](void*& pointer) {
    if (pointer) cudaFree(pointer);
    pointer = nullptr;
  };
  free_ptr(embed_);
  free_ptr(lm_head_.packed);
  free_ptr(lm_head_.scales);
  free_ptr(final_norm_.weight);
  for (auto& list : {&qkv_, &o_proj_, &gu_, &down_}) {
    for (auto& linear : *list) {
      free_ptr(linear.packed);
      free_ptr(linear.scales);
    }
  }
  for (auto& list : {&in_norm_, &post_norm_, &q_norm_, &k_norm_}) {
    for (auto& norm : *list) free_ptr(norm.weight);
  }
  for (auto& pointer : cache_k_) free_ptr(pointer);
  for (auto& pointer : cache_v_) free_ptr(pointer);
  free_ptr(cos_table_);
  free_ptr(sin_table_);
  for (void** pointer : {&hidden_, &normed_, &qkv_buf_, &attended_, &gate_up_, &gated_, &logits_,
                         &token_dev_, &pos_dev_, &ids_dev_, &prompt_dev_, &prefill_head_, &prefill_k_,
                         &prefill_v_}) {
    free_ptr(*pointer);
  }
}

void Decoder::load_weights(const std::string& directory) {
  std::vector<uint8_t> blob;
  auto tensors = load_safetensors(directory + "/model.safetensors", blob);
  const int hidden = config_.hidden;
  const int q_dim = config_.heads * config_.head_dim;
  const int kv_dim = config_.kv_heads * config_.head_dim;

  auto linear = [&](const std::string& name, int rows, int k) {
    Linear result;
    const auto& packed = tensors.at(name + ".packed");
    const auto& scales = tensors.at(name + ".scales");
    result.packed = upload(packed.data, tensor_bytes(packed));
    result.scales = upload(scales.data, tensor_bytes(scales));
    result.rows = rows;
    result.k = k;
    return result;
  };
  auto norm = [&](const std::string& name) {
    Norm result;
    const auto& weight = tensors.at(name);
    result.weight = upload(weight.data, tensor_bytes(weight));
    return result;
  };

  embed_ = upload(tensors.at("embed_tokens.weight").data, tensor_bytes(tensors.at("embed_tokens.weight")));
  final_norm_ = norm("norm.weight");
  lm_head_ = linear("lm_head", config_.vocab, hidden);
  for (int i = 0; i < config_.layers; ++i) {
    const std::string prefix = "layers." + std::to_string(i);
    qkv_.push_back(linear(prefix + ".qkv", q_dim + 2 * kv_dim, hidden));
    o_proj_.push_back(linear(prefix + ".o", hidden, q_dim));
    gu_.push_back(linear(prefix + ".gu", 2 * config_.intermediate, hidden));
    down_.push_back(linear(prefix + ".down", hidden, config_.intermediate));
    in_norm_.push_back(norm(prefix + ".input_layernorm.weight"));
    post_norm_.push_back(norm(prefix + ".post_attention_layernorm.weight"));
    q_norm_.push_back(norm(prefix + ".q_norm.weight"));
    k_norm_.push_back(norm(prefix + ".k_norm.weight"));
  }
}

void Decoder::allocate_cache(int capacity) {
  const int kv_dim = config_.kv_heads * config_.head_dim;
  for (int i = 0; i < config_.layers; ++i) {
    void* k = nullptr;
    void* v = nullptr;
    cuda_check(cudaMalloc(&k, static_cast<size_t>(capacity) * kv_dim * 2), "cudaMalloc kv");
    cuda_check(cudaMalloc(&v, static_cast<size_t>(capacity) * kv_dim * 2), "cudaMalloc kv");
    cuda_check(cudaMemset(k, 0, static_cast<size_t>(capacity) * kv_dim * 2), "cudaMemset kv");
    cuda_check(cudaMemset(v, 0, static_cast<size_t>(capacity) * kv_dim * 2), "cudaMemset kv");
    cache_k_.push_back(k);
    cache_v_.push_back(v);
  }

  const int head_dim = config_.head_dim;
  std::vector<float> inv_freq(head_dim / 2);
  for (int i = 0; i < head_dim / 2; ++i) {
    inv_freq[i] = 1.0f / std::pow(config_.rope_theta, static_cast<float>(2 * i) / head_dim);
  }
  std::vector<__half> cos_table(static_cast<size_t>(capacity) * head_dim);
  std::vector<__half> sin_table(static_cast<size_t>(capacity) * head_dim);
  for (int pos = 0; pos < capacity; ++pos) {
    for (int i = 0; i < head_dim / 2; ++i) {
      const float angle = static_cast<float>(pos) * inv_freq[i];
      cos_table[static_cast<size_t>(pos) * head_dim + i] = __float2half(std::cos(angle));
      cos_table[static_cast<size_t>(pos) * head_dim + head_dim / 2 + i] = __float2half(std::cos(angle));
      sin_table[static_cast<size_t>(pos) * head_dim + i] = __float2half(std::sin(angle));
      sin_table[static_cast<size_t>(pos) * head_dim + head_dim / 2 + i] = __float2half(std::sin(angle));
    }
  }
  cos_table_ = upload(cos_table.data(), cos_table.size() * 2);
  sin_table_ = upload(sin_table.data(), sin_table.size() * 2);

  cuda_check(cudaMalloc(&hidden_, config_.hidden * 2), "cudaMalloc hidden");
  cuda_check(cudaMalloc(&normed_, config_.hidden * 2), "cudaMalloc normed");
  cuda_check(cudaMalloc(&qkv_buf_, (config_.heads + 2 * config_.kv_heads) * head_dim * 2), "cudaMalloc qkv");
  cuda_check(cudaMalloc(&attended_, config_.heads * head_dim * 2), "cudaMalloc attended");
  cuda_check(cudaMalloc(&gate_up_, 2 * config_.intermediate * 2), "cudaMalloc gate_up");
  cuda_check(cudaMalloc(&gated_, config_.intermediate * 2), "cudaMalloc gated");
  cuda_check(cudaMalloc(&logits_, config_.vocab * 2), "cudaMalloc logits");
  cuda_check(cudaMalloc(&token_dev_, sizeof(int)), "cudaMalloc token");
  cuda_check(cudaMalloc(&pos_dev_, sizeof(int64_t)), "cudaMalloc pos");
}

int Decoder::prefill(Engine& engine, const std::vector<int32_t>& ids, const void* audio, int audio_count,
                     int first_audio) {
  const int seq = static_cast<int>(ids.size());
  const int hidden = config_.hidden;
  prefill_seq_ = seq;
  if (!prompt_dev_) {
    cuda_check(cudaMalloc(&prompt_dev_, static_cast<size_t>(seq) * hidden * 2), "cudaMalloc prompt");
  }
  if (!ids_dev_) cuda_check(cudaMalloc(&ids_dev_, static_cast<size_t>(seq) * sizeof(int)), "cudaMalloc ids");
  cuda_check(cudaMemcpy(ids_dev_, ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice), "copy ids");
  embed_launch(prompt_dev_, embed_, ids_dev_, seq, hidden, nullptr);
  if (audio_count > 0) {
    scatter_audio_launch(prompt_dev_, audio, first_audio, audio_count, hidden, nullptr);
  }

  const int kv_dim = config_.kv_heads * config_.head_dim;
  const size_t cache_bytes = static_cast<size_t>(seq) * config_.layers * kv_dim * 2;
  if (!prefill_k_) cuda_check(cudaMalloc(&prefill_k_, cache_bytes), "cudaMalloc prefill cache");
  if (!prefill_v_) cuda_check(cudaMalloc(&prefill_v_, cache_bytes), "cudaMalloc prefill cache");
  const bool emits_logits = [&] {
    for (int i = 0; i < engine.num_io(); ++i)
      if (std::strcmp(engine.name(i), "logits") == 0) return true;
    return false;
  }();
  if (!prefill_head_) {
    const int width = emits_logits ? config_.vocab : hidden;
    cuda_check(cudaMalloc(&prefill_head_, static_cast<size_t>(width) * 2), "cudaMalloc head");
  }

  engine.set_shape("inputs_embeds", nvinfer1::Dims{3, {1, seq, hidden}});
  engine.set_shape("position_ids", nvinfer1::Dims{2, {1, seq}});
  // The prefill graph takes int64 position ids, like torch.arange.
  void* positions = nullptr;
  std::vector<int64_t> position_ids(seq);
  for (int i = 0; i < seq; ++i) position_ids[i] = i;
  cuda_check(cudaMalloc(&positions, seq * sizeof(int64_t)), "cudaMalloc positions");
  cuda_check(cudaMemcpy(positions, position_ids.data(), seq * sizeof(int64_t), cudaMemcpyHostToDevice),
             "copy positions");

  engine.bind("inputs_embeds", prompt_dev_);
  engine.bind("position_ids", positions);
  engine.bind(emits_logits ? "logits" : "hidden", prefill_head_);
  engine.bind("cache_k", prefill_k_);
  engine.bind("cache_v", prefill_v_);
  engine.run();
  cuda_check(cudaDeviceSynchronize(), "prefill sync");

  for (int layer = 0; layer < config_.layers; ++layer) {
    cuda_check(
        cudaMemcpy2D(cache_k_[layer], kv_dim * 2,
                     static_cast<const uint8_t*>(prefill_k_) + static_cast<size_t>(layer) * kv_dim * 2,
                     static_cast<size_t>(config_.layers) * kv_dim * 2, kv_dim * 2, seq,
                     cudaMemcpyDeviceToDevice),
        "copy cache");
    cuda_check(
        cudaMemcpy2D(cache_v_[layer], kv_dim * 2,
                     static_cast<const uint8_t*>(prefill_v_) + static_cast<size_t>(layer) * kv_dim * 2,
                     static_cast<size_t>(config_.layers) * kv_dim * 2, kv_dim * 2, seq,
                     cudaMemcpyDeviceToDevice),
        "copy cache");
  }

  if (const char* dump = std::getenv("QASR_DUMP")) {
    std::vector<__half> host(8);
    std::ofstream cache_out(std::string(dump) + "/cache0.txt");
    for (int t = 0; t < 2; ++t) {
      cuda_check(cudaMemcpy(host.data(), static_cast<const __half*>(cache_k_[0]) +
                                             static_cast<size_t>(t) * kv_dim,
                            host.size() * 2, cudaMemcpyDeviceToHost),
                 "dbg cache");
      for (int i = 0; i < 8; ++i) cache_out << __half2float(host[i]) << " ";
      cache_out << "\n";
    }
  }

  if (!emits_logits) {
    rmsnorm_launch(prefill_head_, final_norm_.weight, normed_, hidden, config_.eps, nullptr);
    w4a16_gemv_launch(normed_, lm_head_.packed, lm_head_.scales, logits_, config_.vocab, hidden, 32, nullptr);
  } else {
    cuda_check(cudaMemcpy(logits_, prefill_head_, static_cast<size_t>(config_.vocab) * 2, cudaMemcpyDeviceToDevice),
               "copy logits");
  }
  int token = 0;
  argmax_launch(logits_, config_.vocab, token_dev_, nullptr);
  cuda_check(cudaMemcpy(&token, token_dev_, sizeof(int), cudaMemcpyDeviceToHost), "copy token");
  position_ = seq;
  return token;
}

int Decoder::step() {
  const int hidden = config_.hidden;
  const int head_dim = config_.head_dim;
  const int q_dim = config_.heads * head_dim;
  const int kv_dim = config_.kv_heads * head_dim;
  const int groups = config_.heads / config_.kv_heads;

  // The cache and attention kernels read the position as int64.
  const int64_t position64 = position_;
  cuda_check(cudaMemcpy(pos_dev_, &position64, sizeof(int64_t), cudaMemcpyHostToDevice), "copy position");
  const void* cos_row = static_cast<const __half*>(cos_table_) + static_cast<size_t>(position_) * head_dim;
  const void* sin_row = static_cast<const __half*>(sin_table_) + static_cast<size_t>(position_) * head_dim;

  __half* hidden_buffer = static_cast<__half*>(hidden_);
  embed_launch(hidden_buffer, embed_, token_dev_, 1, hidden, nullptr);
  if (std::getenv("QASR_DEBUG")) {
    std::vector<__half> host(static_cast<size_t>(hidden));
    cuda_check(cudaMemcpy(host.data(), hidden_buffer, host.size() * 2, cudaMemcpyDeviceToHost), "dbg hidden");
    std::fprintf(stderr, "[dbg] embed0 %.4f embed1 %.4f\n", __half2float(host[0]), __half2float(host[1]));
  }
  for (int layer = 0; layer < config_.layers; ++layer) {
    rmsnorm_launch(hidden_buffer, in_norm_[layer].weight, normed_, hidden, config_.eps, nullptr);
    w4a16_gemv_launch(normed_, qkv_[layer].packed, qkv_[layer].scales, qkv_buf_,
                      qkv_[layer].rows, hidden, 32, nullptr);
    auto* qkv = static_cast<__half*>(qkv_buf_);
    __half* q = qkv;
    __half* k = qkv + q_dim;
    __half* v = qkv + q_dim + kv_dim;
    qk_norm_rope_launch(q, q_norm_[layer].weight, cos_row, sin_row, config_.heads, head_dim, config_.eps, nullptr);
    qk_norm_rope_launch(k, k_norm_[layer].weight, cos_row, sin_row, config_.kv_heads, head_dim, config_.eps,
                        nullptr);
    cache_write_launch(cache_k_[layer], cache_v_[layer], k, v, pos_dev_, kv_dim, head_dim, capacity_, nullptr);
    decode_attention_launch(q, cache_k_[layer], cache_v_[layer], pos_dev_, attended_, config_.heads, head_dim,
                            capacity_, groups, nullptr);
    // Residual overload: (x, packed, scales, residual, y, rows, k, group, stream).
    w4a16_gemv_launch(attended_, o_proj_[layer].packed, o_proj_[layer].scales, hidden_buffer,
                      hidden_buffer, hidden, q_dim, 32, nullptr);
    if (std::getenv("QASR_DEBUG") && layer == 0) {
      std::vector<__half> host(static_cast<size_t>(hidden));
      cuda_check(cudaMemcpy(host.data(), hidden_buffer, host.size() * 2, cudaMemcpyDeviceToHost), "dbg attn");
      std::fprintf(stderr, "[dbg] after-attn %.4f %.4f\n", __half2float(host[0]), __half2float(host[1]));
    }
    rmsnorm_launch(hidden_buffer, post_norm_[layer].weight, normed_, hidden, config_.eps, nullptr);
    w4a16_gemv_launch(normed_, gu_[layer].packed, gu_[layer].scales, gate_up_, gu_[layer].rows, hidden, 32,
                      nullptr);
    silu_mul_launch(gate_up_, gated_, config_.intermediate, nullptr);
    w4a16_gemv_launch(gated_, down_[layer].packed, down_[layer].scales, hidden_buffer, hidden_buffer,
                      hidden, config_.intermediate, 32, nullptr);
  }
  rmsnorm_launch(hidden_buffer, final_norm_.weight, normed_, hidden, config_.eps, nullptr);
  w4a16_gemv_launch(normed_, lm_head_.packed, lm_head_.scales, logits_, config_.vocab, hidden, 32, nullptr);
  argmax_launch(logits_, config_.vocab, token_dev_, nullptr);
  int token = 0;
  cuda_check(cudaMemcpy(&token, token_dev_, sizeof(int), cudaMemcpyDeviceToHost), "copy token");
  if (std::getenv("QASR_DEBUG")) {
    std::vector<__half> host(static_cast<size_t>(config_.vocab));
    cuda_check(cudaMemcpy(host.data(), logits_, host.size() * 2, cudaMemcpyDeviceToHost), "copy logits");
    float best = -1e30f, second = -1e30f;
    for (const __half value : host) {
      const float v = __half2float(value);
      if (v > best) {
        second = best;
        best = v;
      } else if (v > second) {
        second = v;
      }
    }
    std::fprintf(stderr, "[step] pos %d token %d logits max %.4f second %.4f\n", position_, token, best,
                 second);
  }
  ++position_;
  return token;
}

}  // namespace qasr
