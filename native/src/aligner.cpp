// Forced alignment: byte-level BPE prompt construction, TensorRT audio encoder
// and a single-pass TensorRT decoder that emits one timestamp bin per slot.

#include "aligner.h"

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cuda_runtime.h>

// Kernels this runtime adds (ops.cu).
void embed_launch(void* out, const void* table, const void* ids, int count, int hidden,
                  cudaStream_t stream);
void scatter_audio_half_launch(void* out, const void* audio, int first, int audio_count, int hidden,
                               cudaStream_t stream);
void argmax_rows_launch(const void* logits, int rows, int columns, void* out, cudaStream_t stream);

namespace qasr {

namespace {

constexpr int kHidden = 1024;
constexpr int kTimestampBins = 5000;
constexpr float kTimestampStepMs = 80.0f;

// Unicode classes as written by native/tools/gen_aligner_assets.py.
enum CharClass : int {
  kDropped = 0,
  kUpper = 1,
  kLower = 2,
  kUncased = 3,
  kDigit = 4,
};

int decode_utf8(const std::string& text, size_t& index) {
  const auto byte = static_cast<unsigned char>(text[index]);
  if (byte < 0x80) {
    ++index;
    return byte;
  }
  int extra = 0;
  int code = 0;
  if ((byte & 0xE0) == 0xC0) {
    extra = 1;
    code = byte & 0x1F;
  } else if ((byte & 0xF0) == 0xE0) {
    extra = 2;
    code = byte & 0x0F;
  } else if ((byte & 0xF8) == 0xF0) {
    extra = 3;
    code = byte & 0x07;
  } else {
    ++index;
    return byte;
  }
  if (index + static_cast<size_t>(extra) >= text.size()) {
    index = text.size();
    return code;
  }
  for (int i = 1; i <= extra; ++i) {
    code = (code << 6) | (static_cast<unsigned char>(text[index + i]) & 0x3F);
  }
  index += static_cast<size_t>(extra) + 1;
  return code;
}

void append_utf8(std::string& text, int code) {
  if (code < 0x80) {
    text.push_back(static_cast<char>(code));
  } else if (code < 0x800) {
    text.push_back(static_cast<char>(0xC0 | (code >> 6)));
    text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else if (code < 0x10000) {
    text.push_back(static_cast<char>(0xE0 | (code >> 12)));
    text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  } else {
    text.push_back(static_cast<char>(0xF0 | (code >> 18)));
    text.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
    text.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
    text.push_back(static_cast<char>(0x80 | (code & 0x3F)));
  }
}

bool is_cjk(int code) {
  return (code >= 0x4E00 && code <= 0x9FFF) || (code >= 0x3400 && code <= 0x4DBF) ||
         (code >= 0x20000 && code <= 0x2A6DF) || (code >= 0x2A700 && code <= 0x2B73F) ||
         (code >= 0x2B740 && code <= 0x2B81F) || (code >= 0x2B820 && code <= 0x2CEAF) ||
         (code >= 0xF900 && code <= 0xFAFF);
}

bool is_space(int code) {
  return code == ' ' || (code >= 0x09 && code <= 0x0D) || (code >= 0x1C && code <= 0x1F) ||
         code == 0x85 || code == 0xA0 || code == 0x1680 || (code >= 0x2000 && code <= 0x200A) ||
         code == 0x2028 || code == 0x2029 || code == 0x202F || code == 0x205F || code == 0x3000;
}

std::vector<int> codepoints(const std::string& text) {
  std::vector<int> out;
  size_t index = 0;
  while (index < text.size()) out.push_back(decode_utf8(text, index));
  return out;
}

void reserve(void** pointer, size_t* capacity, size_t bytes) {
  if (bytes <= *capacity) return;
  if (*pointer) cudaFree(*pointer);
  cuda_check(cudaMalloc(pointer, bytes), "cudaMalloc aligner buffer");
  *capacity = bytes;
}

}  // namespace

AlignerTokenizer load_aligner_tokenizer(const std::string& path) {
  const std::vector<char> blob = read_file(path);
  size_t offset = 0;
  auto read_u32 = [&]() {
    uint32_t value = 0;
    std::memcpy(&value, blob.data() + offset, 4);
    offset += 4;
    return value;
  };
  const uint32_t magic = read_u32();
  const uint32_t version = read_u32();
  if (magic != 0x4B544151 || version != 1) throw std::runtime_error("bad aligner tokenizer asset");
  const uint32_t vocab_size = read_u32();
  const uint32_t merge_count = read_u32();
  const uint32_t range_count = read_u32();
  AlignerTokenizer tokenizer;
  tokenizer.audio_start = static_cast<int>(read_u32());
  tokenizer.audio_pad = static_cast<int>(read_u32());
  tokenizer.audio_end = static_cast<int>(read_u32());
  tokenizer.timestamp = static_cast<int>(read_u32());
  tokenizer.asr_text = static_cast<int>(read_u32());
  for (int i = 0; i < 256; ++i) tokenizer.byte_ids[i] = static_cast<int>(read_u32());
  tokenizer.tokens.resize(vocab_size);
  for (uint32_t i = 0; i < vocab_size; ++i) {
    const uint32_t length = read_u32();
    tokenizer.tokens[i].assign(blob.data() + offset, length);
    offset += length;
  }
  tokenizer.merge_left.resize(merge_count);
  tokenizer.merge_right.resize(merge_count);
  tokenizer.merge_result.resize(merge_count);
  tokenizer.merge_rank.reserve(static_cast<size_t>(merge_count) * 2);
  for (uint32_t i = 0; i < merge_count; ++i) {
    const int left = static_cast<int>(read_u32());
    const int right = static_cast<int>(read_u32());
    tokenizer.merge_left[i] = left;
    tokenizer.merge_right[i] = right;
    tokenizer.merge_result[i] = static_cast<int>(read_u32());
    tokenizer.merge_rank[(static_cast<uint64_t>(left) << 32) | static_cast<uint32_t>(right)] =
        static_cast<int>(i);
  }
  tokenizer.ranges.resize(range_count);
  for (uint32_t i = 0; i < range_count; ++i) {
    tokenizer.ranges[i] = {read_u32(), read_u32(), read_u32()};
  }
  return tokenizer;
}

Aligner::Aligner(const std::string& assets, const std::string& encoder_engine,
                 const std::string& decoder_engine, const std::string& mel_device)
    : mel_device_(mel_device) {
  mel_assets_ = load_mel_assets(assets + "/mel.bin");
  tokenizer_ = load_aligner_tokenizer(assets + "/aligner_tokenizer.bin");
  if (mel_device_ != "cpu") {
    tables_ = upload_mel_tables(mel_assets_);
    tables_ready_ = true;
  }
  encoder_ = new Engine(encoder_engine);
  decoder_ = new Engine(decoder_engine);

  const std::vector<char> blob = read_file(assets + "/aligner_embed.bin");
  if (blob.size() < 12) throw std::runtime_error("bad aligner embedding asset");
  uint32_t magic = 0;
  uint32_t rows = 0;
  uint32_t columns = 0;
  std::memcpy(&magic, blob.data(), 4);
  std::memcpy(&rows, blob.data() + 4, 4);
  std::memcpy(&columns, blob.data() + 8, 4);
  if (magic != 0x424D4551 || columns != kHidden) throw std::runtime_error("bad aligner embedding asset");
  const size_t bytes = static_cast<size_t>(rows) * columns * 2;
  cuda_check(cudaMalloc(&embed_table_, bytes), "cudaMalloc aligner embedding");
  cuda_check(cudaMemcpy(embed_table_, blob.data() + 12, bytes, cudaMemcpyHostToDevice),
             "copy aligner embedding");
  cuda_check(cudaMalloc(&valid_, 8), "cudaMalloc valid");

  // Warm both engines so the first request does not pay TRT tactic and kernel
  // setup; one second of silence is enough to run every kernel once.
  align(std::vector<float>(16000, 0.0f), "warm up", "English");
  warmed_ = true;
}

Aligner::~Aligner() {
  for (void** pointer : {&embed_table_, &features_, &mask_, &pcm_device_, &audio_, &valid_, &ids_,
                         &hidden_, &positions_, &logits_, &bins_}) {
    if (*pointer) cudaFree(*pointer);
    *pointer = nullptr;
  }
  delete encoder_;
  delete decoder_;
}

void Aligner::ensure(int sequence, int timestamps, int frames, size_t pcm_samples) {
  reserve(&features_, &features_bytes_, static_cast<size_t>(frames) * mel_assets_.n_mel * 4);
  reserve(&mask_, &mask_bytes_, static_cast<size_t>(frames) * 4);
  reserve(&pcm_device_, &pcm_bytes_, pcm_samples * 4);
  const int audio_rows = frames / 100 * 13 + 16;
  reserve(&audio_, &audio_bytes_, static_cast<size_t>(audio_rows) * kHidden * 2);
  reserve(&ids_, &ids_bytes_, static_cast<size_t>(sequence) * 4);
  reserve(&hidden_, &hidden_bytes_, static_cast<size_t>(sequence) * kHidden * 2);
  reserve(&positions_, &positions_bytes_, static_cast<size_t>(timestamps) * 8);
  reserve(&logits_, &logits_bytes_, static_cast<size_t>(timestamps) * kTimestampBins * 2);
  reserve(&bins_, &bins_bytes_, static_cast<size_t>(timestamps) * 4);
}

int Aligner::class_of(int code) const {
  size_t low = 0;
  size_t high = tokenizer_.ranges.size();
  while (low < high) {
    const size_t middle = (low + high) / 2;
    if (static_cast<uint32_t>(code) < tokenizer_.ranges[middle][0]) {
      high = middle;
    } else if (static_cast<uint32_t>(code) > tokenizer_.ranges[middle][1]) {
      low = middle + 1;
    } else {
      return static_cast<int>(tokenizer_.ranges[middle][2]);
    }
  }
  return kDropped;
}

std::vector<std::string> Aligner::split_words(const std::string& text, const std::string& language) const {
  std::vector<std::string> words;
  std::string lower = language;
  for (char& ch : lower) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  const bool cjk_language = lower == "japanese" || lower == "korean";

  const std::vector<int> codes = codepoints(text);
  size_t index = 0;
  while (index < codes.size()) {
    while (index < codes.size() && is_space(codes[index])) ++index;
    std::string segment;
    while (index < codes.size() && !is_space(codes[index])) {
      const int code = codes[index++];
      if (class_of(code) != kDropped || code == '\'') append_utf8(segment, code);
    }
    if (segment.empty()) continue;
    if (cjk_language) {
      for (const int code : codepoints(segment)) {
        std::string word;
        append_utf8(word, code);
        words.push_back(std::move(word));
      }
      continue;
    }
    std::string buffer;
    for (const int code : codepoints(segment)) {
      if (is_cjk(code)) {
        if (!buffer.empty()) {
          words.push_back(buffer);
          buffer.clear();
        }
        std::string word;
        append_utf8(word, code);
        words.push_back(std::move(word));
      } else {
        append_utf8(buffer, code);
      }
    }
    if (!buffer.empty()) words.push_back(buffer);
  }
  return words;
}

void Aligner::merge_chunk(const std::vector<int>& codes, size_t begin, size_t end,
                          std::vector<int>& out) const {
  std::string chunk;
  for (size_t i = begin; i < end; ++i) append_utf8(chunk, codes[i]);
  std::vector<int> symbols;
  symbols.reserve(chunk.size());
  for (const unsigned char byte : chunk) symbols.push_back(tokenizer_.byte_ids[byte]);
  while (symbols.size() > 1) {
    int best_rank = -1;
    size_t best_at = 0;
    for (size_t i = 0; i + 1 < symbols.size(); ++i) {
      const uint64_t key = (static_cast<uint64_t>(symbols[i]) << 32) | static_cast<uint32_t>(symbols[i + 1]);
      const auto found = tokenizer_.merge_rank.find(key);
      if (found != tokenizer_.merge_rank.end() && (best_rank < 0 || found->second < best_rank)) {
        best_rank = found->second;
        best_at = i;
      }
    }
    if (best_rank < 0) break;
    symbols[best_at] = tokenizer_.merge_result[best_rank];
    symbols.erase(symbols.begin() + static_cast<long long>(best_at) + 1);
  }
  out.insert(out.end(), symbols.begin(), symbols.end());
}

size_t Aligner::letter_chunk_end(const std::vector<int>& codes, size_t begin) const {
  size_t upper_end = begin;
  while (upper_end < codes.size()) {
    const int cls = class_of(codes[upper_end]);
    if (cls == kUpper || cls == kUncased) {
      ++upper_end;
    } else {
      break;
    }
  }
  size_t lower_end = upper_end;
  while (lower_end < codes.size()) {
    const int cls = class_of(codes[lower_end]);
    if (cls == kLower || cls == kUncased) {
      ++lower_end;
    } else {
      break;
    }
  }
  if (lower_end > upper_end) return lower_end;
  if (upper_end > begin) return upper_end;
  return begin + 1;
}

std::vector<int> Aligner::encode_word(const std::string& word) const {
  const std::vector<int> codes = codepoints(word);
  std::vector<int> result;
  size_t index = 0;
  while (index < codes.size()) {
    const int cls = class_of(codes[index]);
    if (cls == kDigit) {
      merge_chunk(codes, index, index + 1, result);
      ++index;
      continue;
    }
    if (cls == kDropped) {
      // A single apostrophe before letters joins the following letter chunk,
      // exactly like the reference pre-tokenizer.
      if (codes[index] == '\'' && index + 1 < codes.size()) {
        const int next = class_of(codes[index + 1]);
        if (next == kUpper || next == kLower || next == kUncased) {
          const size_t end = letter_chunk_end(codes, index + 1);
          merge_chunk(codes, index, end, result);
          index = end;
          continue;
        }
      }
      size_t end = index;
      while (end < codes.size() && class_of(codes[end]) == kDropped) ++end;
      merge_chunk(codes, index, end, result);
      index = end;
      continue;
    }
    const size_t end = letter_chunk_end(codes, index);
    merge_chunk(codes, index, end, result);
    index = end;
  }
  return result;
}

std::vector<Aligner::Item> Aligner::parse(const std::vector<std::string>& words,
                                          const std::vector<int>& bins) const {
  // Longest non-decreasing subsequence, then interpolate the outliers, matching
  // the Python reference so both produce the same timestamps.
  const size_t count = bins.size();
  std::vector<int> length(count, 1);
  std::vector<int> parent(count, -1);
  for (size_t i = 1; i < count; ++i) {
    for (size_t j = 0; j < i; ++j) {
      if (bins[j] <= bins[i] && length[j] + 1 > length[i]) {
        length[i] = length[j] + 1;
        parent[i] = static_cast<int>(j);
      }
    }
  }
  std::vector<bool> normal(count, false);
  if (count > 0) {
    size_t best = 0;
    for (size_t i = 1; i < count; ++i) {
      if (length[i] > length[best]) best = i;
    }
    int cursor = static_cast<int>(best);
    while (cursor >= 0) {
      normal[static_cast<size_t>(cursor)] = true;
      cursor = parent[static_cast<size_t>(cursor)];
    }
  }
  std::vector<int> fixed = bins;
  size_t i = 0;
  while (i < count) {
    if (normal[i]) {
      ++i;
      continue;
    }
    size_t j = i;
    while (j < count && !normal[j]) ++j;
    int left = 0;
    bool has_left = false;
    for (size_t k = i; k-- > 0;) {
      if (normal[k]) {
        left = fixed[k];
        has_left = true;
        break;
      }
    }
    int right = 0;
    bool has_right = false;
    for (size_t k = j; k < count; ++k) {
      if (normal[k]) {
        right = fixed[k];
        has_right = true;
        break;
      }
    }
    const size_t gap = j - i;
    if (gap <= 2) {
      for (size_t k = i; k < j; ++k) {
        if (!has_left) {
          fixed[k] = right;
        } else if (!has_right) {
          fixed[k] = left;
        } else {
          fixed[k] = (k - i + 1) <= (j - k) ? left : right;
        }
      }
    } else if (has_left && has_right) {
      const double step = static_cast<double>(right - left) / static_cast<double>(gap + 1);
      for (size_t k = i; k < j; ++k) {
        fixed[k] = static_cast<int>(left + step * static_cast<double>(k - i + 1));
      }
    } else if (has_left) {
      for (size_t k = i; k < j; ++k) fixed[k] = left;
    } else if (has_right) {
      for (size_t k = i; k < j; ++k) fixed[k] = right;
    }
    i = j;
  }

  std::vector<Item> items;
  items.reserve(words.size());
  for (size_t index = 0; index < words.size(); ++index) {
    Item item;
    item.text = words[index];
    const size_t first = index * 2;
    item.start = first < fixed.size() ? fixed[first] : 0;
    item.end = first + 1 < fixed.size() ? fixed[first + 1] : item.start;
    items.push_back(std::move(item));
  }
  return items;
}

std::vector<AlignerItem> Aligner::align(const std::vector<float>& pcm, const std::string& text,
                                        const std::string& language, AlignerTimings* timings) {
  const auto started = std::chrono::steady_clock::now();
  auto elapsed_ms = [&](std::chrono::steady_clock::time_point point) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - point).count();
  };

  std::vector<float> samples = pcm;
  if (samples.size() < 8000) samples.resize(8000, 0.0f);

  int valid_frames = 0;
  const int frames = mel_frame_count(static_cast<int>(samples.size()), mel_assets_.hop, 100, valid_frames);
  const int encoder_max = encoder_->profile_max("input_features", 2);
  if (encoder_max > 0 && frames > encoder_max) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "audio of %d frames (%.1f s) exceeds this encoder's %d-frame profile (%.1f s); "
                  "rebuild it with `qwen3-asr build-aligner --preset long`",
                  frames, frames / 100.0, encoder_max, encoder_max / 100.0);
    throw std::runtime_error(message);
  }

  const std::vector<std::string> words = split_words(text, language);
  if (words.empty()) throw std::runtime_error("aligner: the text has no words");

  int audio_tokens = valid_frames / 100 * 13;
  int remainder = valid_frames % 100;
  for (int i = 0; i < 3; ++i) remainder = (remainder + 1) / 2;
  audio_tokens += remainder;

  std::vector<int32_t> ids;
  ids.reserve(static_cast<size_t>(audio_tokens) + words.size() * 16 + 8);
  ids.push_back(tokenizer_.audio_start);
  ids.insert(ids.end(), static_cast<size_t>(audio_tokens), tokenizer_.audio_pad);
  ids.push_back(tokenizer_.audio_end);
  std::vector<int32_t> positions;
  positions.reserve(words.size() * 2);
  for (const std::string& word : words) {
    const std::vector<int> encoded = encode_word(word);
    ids.insert(ids.end(), encoded.begin(), encoded.end());
    positions.push_back(static_cast<int32_t>(ids.size()));
    ids.push_back(tokenizer_.timestamp);
    positions.push_back(static_cast<int32_t>(ids.size()));
    ids.push_back(tokenizer_.timestamp);
  }

  const int sequence = static_cast<int>(ids.size());
  const int timestamps = static_cast<int>(positions.size());
  if (warmed_ && std::getenv("QASR_DUMP_IDS")) {
    std::fprintf(stderr, "[ids]");
    for (const int32_t id : ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n[ts]");
    for (const int32_t position : positions) std::fprintf(stderr, " %d", position);
    std::fprintf(stderr, "\n");
  }
  const int decoder_seq_max = decoder_->profile_max("inputs_embeds", 1);
  if (decoder_seq_max > 0 && sequence > decoder_seq_max) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "aligner prompt of %d tokens exceeds this decoder's %d-token profile; "
                  "rebuild it with `qwen3-asr build-aligner --preset long`",
                  sequence, decoder_seq_max);
    throw std::runtime_error(message);
  }
  const int decoder_ts_max = decoder_->profile_max("ts_positions", 1);
  if (decoder_ts_max > 0 && timestamps > decoder_ts_max) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "%d timestamp slots (two per word) exceed this decoder's %d-slot profile; "
                  "rebuild it with `qwen3-asr build-aligner --preset long`",
                  timestamps, decoder_ts_max);
    throw std::runtime_error(message);
  }

  const auto padded = reflect_pad(samples, mel_assets_.n_fft / 2);
  ensure(sequence, timestamps, frames, padded.size());

  const auto mel_started = std::chrono::steady_clock::now();
  const bool gpu_mel = mel_device_ == "cuda" || (mel_device_ == "auto" && frames >= 200);
  if (gpu_mel) {
    cuda_check(cudaMemcpy(pcm_device_, padded.data(), padded.size() * 4, cudaMemcpyHostToDevice), "copy pcm");
    mel_forward_gpu(static_cast<const float*>(pcm_device_), tables_, mel_assets_,
                    static_cast<float*>(features_), valid_frames, frames);
    cuda_check(cudaDeviceSynchronize(), "mel sync");
  } else {
    std::vector<float> host(static_cast<size_t>(frames) * mel_assets_.n_mel, 0.0f);
    mel_forward_cpu(padded.data(), valid_frames, frames, mel_assets_, host.data());
    cuda_check(cudaMemcpy(features_, host.data(), static_cast<size_t>(frames) * mel_assets_.n_mel * 4,
                          cudaMemcpyHostToDevice),
               "copy mel");
  }
  const double mel_ms = elapsed_ms(mel_started);

  std::vector<int32_t> mask(frames, 0);
  for (int i = 0; i < valid_frames; ++i) mask[static_cast<size_t>(i)] = 1;
  cuda_check(cudaMemcpy(mask_, mask.data(), static_cast<size_t>(frames) * 4, cudaMemcpyHostToDevice),
             "copy mask");
  cuda_check(cudaMemcpy(ids_, ids.data(), static_cast<size_t>(sequence) * 4, cudaMemcpyHostToDevice),
             "copy ids");
  std::vector<int64_t> wide(static_cast<size_t>(timestamps));
  for (int i = 0; i < timestamps; ++i) wide[static_cast<size_t>(i)] = positions[static_cast<size_t>(i)];
  cuda_check(cudaMemcpy(positions_, wide.data(), static_cast<size_t>(timestamps) * 8,
                        cudaMemcpyHostToDevice),
             "copy ts positions");

  const auto encoder_started = std::chrono::steady_clock::now();
  encoder_->set_shape("input_features", nvinfer1::Dims{3, {1, mel_assets_.n_mel, frames}});
  encoder_->set_shape("input_features_mask", nvinfer1::Dims{2, {1, frames}});
  encoder_->bind("input_features", features_);
  encoder_->bind("input_features_mask", mask_);
  encoder_->bind("audio_embeddings", audio_);
  encoder_->bind("valid_tokens", valid_);
  encoder_->run();
  if (std::getenv("QASR_DEBUG")) {
    cuda_check(cudaDeviceSynchronize(), "encoder sync");
    int64_t encoded = 0;
    cuda_check(cudaMemcpy(&encoded, valid_, 8, cudaMemcpyDeviceToHost), "valid tokens");
    if (encoded != audio_tokens) {
      throw std::runtime_error("aligner: encoder returned " + std::to_string(encoded) + " tokens for " +
                               std::to_string(audio_tokens) + " placeholders");
    }
  }
  const double encoder_ms = elapsed_ms(encoder_started);

  const auto decoder_started = std::chrono::steady_clock::now();
  embed_launch(hidden_, embed_table_, ids_, sequence, kHidden, nullptr);
  scatter_audio_half_launch(hidden_, audio_, 1, audio_tokens, kHidden, nullptr);

  decoder_->set_shape("inputs_embeds", nvinfer1::Dims{3, {1, sequence, kHidden}});
  decoder_->set_shape("ts_positions", nvinfer1::Dims{2, {1, timestamps}});
  decoder_->bind("inputs_embeds", hidden_);
  decoder_->bind("ts_positions", positions_);
  decoder_->bind("logits", logits_);
  decoder_->run();
  cuda_check(cudaDeviceSynchronize(), "aligner decoder sync");
  argmax_rows_launch(logits_, timestamps, kTimestampBins, bins_, nullptr);
  std::vector<int32_t> host_bins(static_cast<size_t>(timestamps));
  cuda_check(cudaMemcpy(host_bins.data(), bins_, static_cast<size_t>(timestamps) * 4,
                        cudaMemcpyDeviceToHost),
             "copy bins");
  const double decoder_ms = elapsed_ms(decoder_started);

  const std::vector<int> values(host_bins.begin(), host_bins.end());
  const std::vector<Item> parsed = parse(words, values);
  std::vector<AlignerItem> items;
  items.reserve(parsed.size());
  for (const Item& item : parsed) {
    AlignerItem out;
    out.text = item.text;
    out.start = static_cast<float>(item.start) * kTimestampStepMs / 1000.0f;
    out.end = static_cast<float>(item.end) * kTimestampStepMs / 1000.0f;
    items.push_back(std::move(out));
  }
  if (timings) {
    timings->mel_ms = mel_ms;
    timings->encoder_ms = encoder_ms;
    timings->decoder_ms = decoder_ms;
    timings->total_ms = elapsed_ms(started);
    timings->frames = frames;
    timings->tokens = sequence;
    timings->words = static_cast<int>(words.size());
  }
  return items;
}

}  // namespace qasr
