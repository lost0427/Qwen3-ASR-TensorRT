#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "qasr.h"

namespace qasr {

// Byte-level BPE vocabulary of the forced aligner, plus the Unicode class table
// the text splitter needs.
struct AlignerTokenizer {
  std::vector<std::string> tokens;
  std::vector<int> merge_left;
  std::vector<int> merge_right;
  std::vector<int> merge_result;
  std::unordered_map<uint64_t, int> merge_rank;
  std::array<int, 256> byte_ids{};
  std::vector<std::array<uint32_t, 3>> ranges;  // (first, last, class), sorted
  int audio_start = 0;
  int audio_pad = 0;
  int audio_end = 0;
  int timestamp = 0;
  int asr_text = 0;
};

AlignerTokenizer load_aligner_tokenizer(const std::string& path);

struct AlignerItem {
  std::string text;
  float start = 0.0f;
  float end = 0.0f;
};

struct AlignerTimings {
  double mel_ms = 0.0;
  double encoder_ms = 0.0;
  double decoder_ms = 0.0;
  double total_ms = 0.0;
  int frames = 0;
  int tokens = 0;
  int words = 0;
};

// Forced alignment: TensorRT audio encoder + single-pass TensorRT text decoder.
// Unlike the ASR decoder there is no KV cache and no decode loop: one forward
// over the prompt emits a timestamp bin per <timestamp> slot.
class Aligner {
 public:
  Aligner(const std::string& assets, const std::string& encoder_engine,
          const std::string& decoder_engine, const std::string& mel_device);
  ~Aligner();

  std::vector<AlignerItem> align(const std::vector<float>& pcm, const std::string& text,
                                 const std::string& language, AlignerTimings* timings = nullptr);

 private:
  struct Item {
    std::string text;
    int start = 0;
    int end = 0;
  };

  void ensure(int sequence, int timestamps, int frames, size_t pcm_samples);
  int class_of(int code) const;
  std::vector<std::string> split_words(const std::string& text, const std::string& language) const;
  std::vector<int> encode_word(const std::string& word) const;
  void merge_chunk(const std::vector<int>& codes, size_t begin, size_t end,
                   std::vector<int>& out) const;
  size_t letter_chunk_end(const std::vector<int>& codes, size_t begin) const;
  std::vector<Item> parse(const std::vector<std::string>& words, const std::vector<int>& bins) const;

  std::string mel_device_;
  bool warmed_ = false;
  AlignerTokenizer tokenizer_;
  MelAssets mel_assets_;
  MelDeviceTables tables_;
  bool tables_ready_ = false;
  Engine* encoder_ = nullptr;
  Engine* decoder_ = nullptr;

  void* embed_table_ = nullptr;
  void* stream_ = nullptr;
  void* features_ = nullptr;
  size_t features_bytes_ = 0;
  void* mask_ = nullptr;
  size_t mask_bytes_ = 0;
  void* pcm_device_ = nullptr;
  size_t pcm_bytes_ = 0;
  void* audio_ = nullptr;
  size_t audio_bytes_ = 0;
  void* valid_ = nullptr;
  void* ids_ = nullptr;
  size_t ids_bytes_ = 0;
  void* hidden_ = nullptr;
  size_t hidden_bytes_ = 0;
  void* positions_ = nullptr;
  size_t positions_bytes_ = 0;
  void* logits_ = nullptr;
  size_t logits_bytes_ = 0;
  void* bins_ = nullptr;
  size_t bins_bytes_ = 0;
};

}  // namespace qasr
