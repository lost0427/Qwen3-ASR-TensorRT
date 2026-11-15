#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "qasr.h"

namespace qasr {

struct ModelConfig {
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, intermediate = 0, vocab = 0;
  float eps = 0.0f, rope_theta = 0.0f;
};

ModelConfig read_model_config(const std::string& directory);

class Decoder {
 public:
  Decoder(const std::string& directory, int max_cache);
  ~Decoder();

  const ModelConfig& config() const { return config_; }
  int capacity() const { return capacity_; }
  int position() const { return position_; }

  // Builds the prompt embeddings on the host and runs the prefill engine.
  // Returns the first generated token.
  int prefill(Engine& engine, const std::vector<int32_t>& ids, const void* audio, int audio_count,
              int first_audio);
  // One decode step; returns the token the step produced.
  int step();

 private:
  void allocate_cache(int capacity);
  void load_weights(const std::string& directory);

  ModelConfig config_;
  int capacity_ = 0;
  int position_ = 0;
  int max_new_ = 0;

  struct Linear {
    void* packed = nullptr;
    void* scales = nullptr;
    int rows = 0;
    int k = 0;
  };
  struct Norm {
    void* weight = nullptr;
  };

  void* embed_ = nullptr;
  Linear lm_head_;
  Norm final_norm_;
  std::vector<Linear> qkv_, o_proj_, gu_, down_;
  std::vector<Norm> in_norm_, post_norm_, q_norm_, k_norm_;

  std::vector<void*> cache_k_, cache_v_;
  void* cos_table_ = nullptr;
  void* sin_table_ = nullptr;

  // scratch (device)
  void* hidden_ = nullptr;
  void* normed_ = nullptr;
  void* qkv_buf_ = nullptr;
  void* attended_ = nullptr;
  void* gate_up_ = nullptr;
  void* gated_ = nullptr;
  void* logits_ = nullptr;
  void* token_dev_ = nullptr;
  void* pos_dev_ = nullptr;
  void* ids_dev_ = nullptr;
  void* prompt_dev_ = nullptr;
  void* prefill_head_ = nullptr;
  void* prefill_k_ = nullptr;
  void* prefill_v_ = nullptr;
  int prefill_seq_ = 0;
};

}  // namespace qasr
