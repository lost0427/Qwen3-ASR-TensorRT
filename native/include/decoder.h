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
  // `reserve_tokens` sizes the KV cache so the decode loop cannot run out of
  // slots; the cache grows in place if the prompt plus that reserve does not fit.
  int prefill(Engine& engine, const std::vector<int32_t>& ids, const void* audio, int audio_count,
              int first_audio, int reserve_tokens);
  // Runs the decode loop on the device: the step is captured into a CUDA graph
  // and replayed `chunk` times before the host reads the tokens back.
  std::vector<int32_t> decode(int max_new_tokens, int chunk = 8);

  // ---- speculative decoding ------------------------------------------------

  // Resets the cache length, the next input token and the produced counter so a
  // draft block can start from a prefix the target accepted.
  void reset(int position, int32_t token);
  // Replays the captured step `count` times and returns the tokens it produced.
  std::vector<int32_t> draft(int count);
  // Runs the target over `candidates` in one batched pass, appends the accepted
  // tokens plus the target's correction or bonus token, and returns how many
  // candidates matched.
  int verify(const std::vector<int32_t>& candidates);
  // Returns everything the produced buffer holds so far.
  std::vector<int32_t> produced();

  // Times the pieces of a decode step so the gap against the Python runtime can
  // be attributed instead of guessed.
  void bench(int iterations);

 private:
  void allocate_cache(int capacity);
  void allocate_kv(int capacity);
  void build_rope_tables(int begin, int capacity);
  void allocate_scratch();
  void free_kv();
  void grow_cache(int needed);
  void enqueue_step();
  void capture();
  void load_weights(const std::string& directory);
  void allocate_verify(int rows);

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
  void* cos_row_dev_ = nullptr;
  void* sin_row_dev_ = nullptr;
  void* produced_dev_ = nullptr;
  void* generated_dev_ = nullptr;
  void* host_scratch_ = nullptr;
  void* stream_ = nullptr;
  void* graph_ = nullptr;
  void* graph_exec_ = nullptr;
  void* ids_dev_ = nullptr;
  void* positions_dev_ = nullptr;
  void* prompt_dev_ = nullptr;
  void* prefill_head_ = nullptr;
  void* prefill_k_ = nullptr;
  void* prefill_v_ = nullptr;
  void* prefill_ptrs_ = nullptr;
  int prefill_seq_ = 0;

  // Batched verification scratch, sized for at most 8 rows (7 draft tokens).
  int verify_rows_ = 0;
  void* v_ids_ = nullptr;
  void* v_positions_ = nullptr;
  void* v_hidden_ = nullptr;
  void* v_normed_ = nullptr;
  void* v_qkv_ = nullptr;
  void* v_attended_ = nullptr;
  void* v_gate_up_ = nullptr;
  void* v_gated_ = nullptr;
  void* v_logits_ = nullptr;
  void* v_predicted_ = nullptr;
  void* v_cos_ = nullptr;
  void* v_sin_ = nullptr;
};

}  // namespace qasr
