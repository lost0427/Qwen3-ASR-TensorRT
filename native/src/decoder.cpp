#include "decoder.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
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
void commit_token_launch(const void* token, void* produced, void* generated, void* position,
                         cudaStream_t stream);
void gather_row_launch(const void* table, const void* position, void* out, int head_dim,
                       cudaStream_t stream);

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
  const std::vector<char> raw = read_file(path);
  blob.assign(raw.begin(), raw.end());
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

namespace {

void debug_dump(const char* label, const void* device, int count) {
  std::vector<__half> host(static_cast<size_t>(count));
  cudaMemcpy(host.data(), device, host.size() * 2, cudaMemcpyDeviceToHost);
  std::fprintf(stderr, "[dbg] %s:", label);
  for (int i = 0; i < count; ++i) std::fprintf(stderr, " %.6f", __half2float(host[i]));
  std::fprintf(stderr, "\n");
}

}  // namespace

ModelConfig read_model_config(const std::string& directory) {
  const std::vector<char> raw = read_file(directory + "/config.json");
  std::string text(raw.begin(), raw.end());
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
                         &prefill_v_, &produced_dev_, &generated_dev_, &cos_row_dev_, &sin_row_dev_}) {
    free_ptr(*pointer);
  }
  if (stream_) cudaStreamDestroy(static_cast<cudaStream_t>(stream_));
  if (graph_exec_) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph_exec_));
  if (graph_) cudaGraphDestroy(static_cast<cudaGraph_t>(graph_));
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
  cuda_check(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream_)), "cudaStreamCreate");
  cuda_check(cudaMalloc(&cos_row_dev_, static_cast<size_t>(head_dim) * 2), "cudaMalloc cos row");
  cuda_check(cudaMalloc(&sin_row_dev_, static_cast<size_t>(head_dim) * 2), "cudaMalloc sin row");
  cuda_check(cudaMalloc(&produced_dev_, static_cast<size_t>(capacity) * sizeof(int)), "cudaMalloc produced");
  cuda_check(cudaMalloc(&generated_dev_, sizeof(int)), "cudaMalloc generated");
}

int Decoder::prefill(Engine& engine, const std::vector<int32_t>& ids, const void* audio, int audio_count,
                     int first_audio) {
  const int seq = static_cast<int>(ids.size());
  const int hidden = config_.hidden;
  const int kv_dim = config_.kv_heads * config_.head_dim;
  if (seq > prefill_seq_) {
    // Buffers sized for an earlier, shorter prompt have to grow.
    if (prompt_dev_) cudaFree(prompt_dev_);
    if (ids_dev_) cudaFree(ids_dev_);
    if (prefill_k_) cudaFree(prefill_k_);
    if (prefill_v_) cudaFree(prefill_v_);
    prompt_dev_ = nullptr;
    ids_dev_ = nullptr;
    prefill_k_ = nullptr;
    prefill_v_ = nullptr;
    if (positions_dev_) cudaFree(positions_dev_);
    positions_dev_ = nullptr;
    cuda_check(cudaMalloc(&prompt_dev_, static_cast<size_t>(seq) * hidden * 2), "cudaMalloc prompt");
    cuda_check(cudaMalloc(&ids_dev_, static_cast<size_t>(seq) * sizeof(int)), "cudaMalloc ids");
    cuda_check(cudaMalloc(&positions_dev_, static_cast<size_t>(seq) * sizeof(int64_t)),
               "cudaMalloc positions");
    const size_t cache_bytes = static_cast<size_t>(seq) * config_.layers * kv_dim * 2;
    cuda_check(cudaMalloc(&prefill_k_, cache_bytes), "cudaMalloc prefill cache");
    cuda_check(cudaMalloc(&prefill_v_, cache_bytes), "cudaMalloc prefill cache");
    prefill_seq_ = seq;
  }
  cuda_check(cudaMemcpy(ids_dev_, ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice), "copy ids");
  embed_launch(prompt_dev_, embed_, ids_dev_, seq, hidden, nullptr);
  if (audio_count > 0) {
    scatter_audio_launch(prompt_dev_, audio, first_audio, audio_count, hidden, nullptr);
  }

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
  std::vector<int64_t> position_ids(seq);
  for (int i = 0; i < seq; ++i) position_ids[i] = i;
  cuda_check(cudaMemcpy(positions_dev_, position_ids.data(), seq * sizeof(int64_t),
                        cudaMemcpyHostToDevice), "copy positions");

  engine.bind("inputs_embeds", prompt_dev_);
  engine.bind("position_ids", positions_dev_);
  engine.bind(emits_logits ? "logits" : "hidden", prefill_head_);
  engine.bind("cache_k", prefill_k_);
  engine.bind("cache_v", prefill_v_);
  engine.run();
  cuda_check(cudaDeviceSynchronize(), "prefill sync");

  // Source layout is (seq, layers, kv_heads, head_dim); the runtime buffer is
  // (kv_heads, capacity, head_dim), so each head needs its own row copy.
  const int head_dim = config_.head_dim;
  const int kv_heads = config_.kv_heads;
  const size_t source_pitch = static_cast<size_t>(config_.layers) * kv_dim * 2;
  const size_t dest_pitch = static_cast<size_t>(head_dim) * 2;
  for (int layer = 0; layer < config_.layers; ++layer) {
    for (int head = 0; head < kv_heads; ++head) {
      const size_t source_offset = (static_cast<size_t>(layer) * kv_heads + head) * head_dim * 2;
      const size_t dest_offset = static_cast<size_t>(head) * capacity_ * head_dim * 2;
      cuda_check(cudaMemcpy2D(static_cast<uint8_t*>(cache_k_[layer]) + dest_offset, dest_pitch,
                              static_cast<const uint8_t*>(prefill_k_) + source_offset, source_pitch,
                              head_dim * 2, seq, cudaMemcpyDeviceToDevice),
                 "copy cache k");
      cuda_check(cudaMemcpy2D(static_cast<uint8_t*>(cache_v_[layer]) + dest_offset, dest_pitch,
                              static_cast<const uint8_t*>(prefill_v_) + source_offset, source_pitch,
                              head_dim * 2, seq, cudaMemcpyDeviceToDevice),
                 "copy cache v");
    }
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
  const int64_t position64 = position_;
  cuda_check(cudaMemcpy(pos_dev_, &position64, sizeof(int64_t), cudaMemcpyHostToDevice), "copy position");
  const int produced_first = token;
  const int generated_one = 1;
  cuda_check(cudaMemcpy(produced_dev_, &produced_first, sizeof(int), cudaMemcpyHostToDevice), "copy token");
  cuda_check(cudaMemcpy(generated_dev_, &generated_one, sizeof(int), cudaMemcpyHostToDevice), "copy count");
  return token;
}

void Decoder::enqueue_step() {
  const int hidden = config_.hidden;
  const int head_dim = config_.head_dim;
  const int q_dim = config_.heads * head_dim;
  const int kv_dim = config_.kv_heads * head_dim;
  const int groups = config_.heads / config_.kv_heads;

  // The cache and attention kernels read the position as int64.
  const int64_t position64 = position_;
  // The position lives on the device so the whole step can be graph-captured;
  // the rotary row for the current position is gathered into a fixed buffer.
  gather_row_launch(cos_table_, pos_dev_, cos_row_dev_, head_dim, static_cast<cudaStream_t>(stream_));
  gather_row_launch(sin_table_, pos_dev_, sin_row_dev_, head_dim, static_cast<cudaStream_t>(stream_));
  const void* cos_row = cos_row_dev_;
  const void* sin_row = sin_row_dev_;

  __half* hidden_buffer = static_cast<__half*>(hidden_);
  embed_launch(hidden_buffer, embed_, token_dev_, 1, hidden, static_cast<cudaStream_t>(stream_));
  if (std::getenv("QASR_DEBUG")) {
    std::vector<__half> host(static_cast<size_t>(hidden));
    cuda_check(cudaMemcpy(host.data(), hidden_buffer, host.size() * 2, cudaMemcpyDeviceToHost), "dbg hidden");
    std::fprintf(stderr, "[dbg] embed0 %.4f embed1 %.4f\n", __half2float(host[0]), __half2float(host[1]));
  }
  for (int layer = 0; layer < config_.layers; ++layer) {
    const bool trace = std::getenv("QASR_DEBUG") && layer == 0;
    rmsnorm_launch(hidden_buffer, in_norm_[layer].weight, normed_, hidden, config_.eps, static_cast<cudaStream_t>(stream_));
    if (trace) debug_dump("normed", normed_, 2);
    w4a16_gemv_launch(normed_, qkv_[layer].packed, qkv_[layer].scales, qkv_buf_,
                      qkv_[layer].rows, hidden, 32, static_cast<cudaStream_t>(stream_));
    if (trace) debug_dump("qkv", qkv_buf_, 2);
    auto* qkv = static_cast<__half*>(qkv_buf_);
    __half* q = qkv;
    __half* k = qkv + q_dim;
    __half* v = qkv + q_dim + kv_dim;
    qk_norm_rope_launch(q, q_norm_[layer].weight, cos_row, sin_row, config_.heads, head_dim, config_.eps, static_cast<cudaStream_t>(stream_));
    qk_norm_rope_launch(k, k_norm_[layer].weight, cos_row, sin_row, config_.kv_heads, head_dim, config_.eps,
                        static_cast<cudaStream_t>(stream_));
    if (trace) debug_dump("q_rope", q, 2);
    cache_write_launch(cache_k_[layer], cache_v_[layer], k, v, pos_dev_, kv_dim, head_dim, capacity_, static_cast<cudaStream_t>(stream_));
    decode_attention_launch(q, cache_k_[layer], cache_v_[layer], pos_dev_, attended_, config_.heads, head_dim,
                            capacity_, groups, static_cast<cudaStream_t>(stream_));
    if (trace) {
      debug_dump("attended", attended_, 2);
      debug_dump("cache_row", static_cast<const __half*>(cache_k_[layer]) +
                                  static_cast<size_t>(position_) * head_dim, 4);
      debug_dump("kv_head0", k, 4);
      debug_dump("cache_row30", static_cast<const __half*>(cache_k_[layer]) + static_cast<size_t>(30) * head_dim, 4);
    }
    // Residual overload: (x, packed, scales, residual, y, rows, k, group, stream).
    w4a16_gemv_launch(attended_, o_proj_[layer].packed, o_proj_[layer].scales, hidden_buffer,
                      hidden_buffer, hidden, q_dim, 32, static_cast<cudaStream_t>(stream_));
    if (std::getenv("QASR_DEBUG") && layer == 0) {
      std::vector<__half> host(static_cast<size_t>(hidden));
      cuda_check(cudaMemcpy(host.data(), hidden_buffer, host.size() * 2, cudaMemcpyDeviceToHost), "dbg attn");
      std::fprintf(stderr, "[dbg] after-attn %.4f %.4f\n", __half2float(host[0]), __half2float(host[1]));
    }
    rmsnorm_launch(hidden_buffer, post_norm_[layer].weight, normed_, hidden, config_.eps, static_cast<cudaStream_t>(stream_));
    w4a16_gemv_launch(normed_, gu_[layer].packed, gu_[layer].scales, gate_up_, gu_[layer].rows, hidden, 32,
                      static_cast<cudaStream_t>(stream_));
    silu_mul_launch(gate_up_, gated_, config_.intermediate, static_cast<cudaStream_t>(stream_));
    w4a16_gemv_launch(gated_, down_[layer].packed, down_[layer].scales, hidden_buffer, hidden_buffer,
                      hidden, config_.intermediate, 32, static_cast<cudaStream_t>(stream_));
  }
  rmsnorm_launch(hidden_buffer, final_norm_.weight, normed_, hidden, config_.eps, static_cast<cudaStream_t>(stream_));
  w4a16_gemv_launch(normed_, lm_head_.packed, lm_head_.scales, logits_, config_.vocab, hidden, 32, static_cast<cudaStream_t>(stream_));
  argmax_launch(logits_, config_.vocab, token_dev_, static_cast<cudaStream_t>(stream_));
  commit_token_launch(token_dev_, produced_dev_, generated_dev_, pos_dev_, static_cast<cudaStream_t>(stream_));
}

void Decoder::capture() {
  if (graph_exec_) return;
  // The warm-up and the capture both execute the step, so save the decoder state
  // and roll it back afterwards. Cache rows written at the scratch positions are
  // overwritten by the real decode.
  int64_t saved_position = 0;
  int saved_generated = 0;
  int saved_first = 0;
  int saved_token = 0;
  cuda_check(cudaMemcpy(&saved_position, pos_dev_, sizeof(int64_t), cudaMemcpyDeviceToHost), "save position");
  cuda_check(cudaMemcpy(&saved_generated, generated_dev_, sizeof(int), cudaMemcpyDeviceToHost),
             "save count");
  cuda_check(cudaMemcpy(&saved_first, produced_dev_, sizeof(int), cudaMemcpyDeviceToHost), "save token");
  cuda_check(cudaMemcpy(&saved_token, token_dev_, sizeof(int), cudaMemcpyDeviceToHost), "save input token");

  cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  enqueue_step();
  cuda_check(cudaStreamSynchronize(stream), "graph warm-up");

  cudaGraph_t graph = nullptr;
  cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
  enqueue_step();
  cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
  cudaGraphExec_t exec = nullptr;
  cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate graph");
  graph_ = graph;
  graph_exec_ = exec;

  cuda_check(cudaMemcpy(pos_dev_, &saved_position, sizeof(int64_t), cudaMemcpyHostToDevice), "restore position");
  cuda_check(cudaMemcpy(generated_dev_, &saved_generated, sizeof(int), cudaMemcpyHostToDevice),
             "restore count");
  cuda_check(cudaMemcpy(produced_dev_, &saved_first, sizeof(int), cudaMemcpyHostToDevice), "restore token");
  // The captured step overwrites the input token, so put it back as well.
  cuda_check(cudaMemcpy(token_dev_, &saved_token, sizeof(int), cudaMemcpyHostToDevice),
             "restore input token");
}

std::vector<int32_t> Decoder::decode(int max_new_tokens, int chunk) {
  capture();
  const int64_t position_start = position_;
  int generated = 1;
  cuda_check(cudaMemcpy(&generated, generated_dev_, sizeof(int), cudaMemcpyDeviceToHost), "read count");
  std::vector<int32_t> tokens(static_cast<size_t>(generated));
  cuda_check(cudaMemcpy(tokens.data(), produced_dev_, tokens.size() * sizeof(int), cudaMemcpyDeviceToHost),
             "read token");

  const int64_t limit = static_cast<int64_t>(capacity_) - 1;
  while (generated < max_new_tokens && (position_start + generated - 1) < limit &&
         tokens.back() != 151643 && tokens.back() != 151645) {
    int steps = std::min(chunk, max_new_tokens - generated);
    const int64_t room = limit - (position_start + generated - 1);
    if (steps > room) steps = static_cast<int>(room);
    if (steps <= 0) break;
    cudaGraphExec_t exec = static_cast<cudaGraphExec_t>(graph_exec_);
    cudaStream_t stream = static_cast<cudaStream_t>(stream_);
    for (int i = 0; i < steps; ++i) cuda_check(cudaGraphLaunch(exec, stream), "graph launch");
    cuda_check(cudaStreamSynchronize(stream), "graph sync");
    cuda_check(cudaMemcpy(&generated, generated_dev_, sizeof(int), cudaMemcpyDeviceToHost), "read count");
    tokens.resize(static_cast<size_t>(generated));
    cuda_check(cudaMemcpy(tokens.data(), produced_dev_, tokens.size() * sizeof(int), cudaMemcpyDeviceToHost),
               "read tokens");
    // The end token can land anywhere inside the chunk, not just at its end.
    const auto stop = std::find_if(tokens.begin(), tokens.end(),
                                   [](int32_t value) { return value == 151643 || value == 151645; });
    if (stop != tokens.end()) {
      tokens.erase(stop, tokens.end());
      break;
    }
  }
  position_ = position_start + generated - 1;
  return tokens;
}

void Decoder::bench(int iterations) {
  capture();
  cudaStream_t stream = static_cast<cudaStream_t>(stream_);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cuda_check(cudaEventCreate(&start), "event");
  cuda_check(cudaEventCreate(&stop), "event");
  auto run = [&](const char* label, auto&& body) {
    cuda_check(cudaStreamSynchronize(stream), "bench sync");
    cuda_check(cudaEventRecord(start, stream), "record");
    for (int i = 0; i < iterations; ++i) body();
    cuda_check(cudaEventRecord(stop, stream), "record");
    cuda_check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    cuda_check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    std::fprintf(stderr, "[bench] %-22s %.3f us/iter\n", label, ms * 1000.0f / iterations);
  };

  run("argmax", [&] { argmax_launch(logits_, config_.vocab, token_dev_, stream); });
  run("gather row x2", [&] {
    gather_row_launch(cos_table_, pos_dev_, cos_row_dev_, config_.head_dim, stream);
    gather_row_launch(sin_table_, pos_dev_, sin_row_dev_, config_.head_dim, stream);
  });
  run("commit token", [&] {
    commit_token_launch(token_dev_, produced_dev_, generated_dev_, pos_dev_, stream);
  });
  run("graph replay (step)", [&] { cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph_exec_), stream); });

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

}  // namespace qasr
