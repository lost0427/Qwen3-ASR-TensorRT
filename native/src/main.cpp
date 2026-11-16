// Command line entry point: loads the engines once and transcribes one or more
// files, printing one JSON object per file.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "decoder.h"
#include "qasr.h"

namespace {

using qasr::cuda_check;

struct Options {
  std::string model;
  std::string encoder;
  std::string decoder;
  std::string assets = "assets";
  std::string language = "English";
  std::string mel_device = "cuda";
  int max_new_tokens = 256;
  int max_cache = 2048;
  std::vector<std::string> audio;
};

bool timing_enabled() { return std::getenv("QASR_TIMING") != nullptr; }

struct Timer {
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  }
};

std::string escape(const std::string& text) {
  std::string out;
  for (char ch : text) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (ch == '\n') {
      out += "\\n";
    } else {
      out += ch;
    }
  }
  return out;
}

class Transcriber {
 public:
  Transcriber(const Options& options) : options_(options) {
    mel_assets_ = qasr::load_mel_assets(options.assets + "/mel.bin");
    tokenizer_ = qasr::load_tokenizer_assets(options.assets + "/tokenizer.bin");
    if (options.mel_device == "cuda") {
      tables_ = qasr::upload_mel_tables(mel_assets_);
      tables_ready_ = true;
    }
    decoder_ = new qasr::Decoder(options.model, options.max_cache);
    encoder_ = new qasr::Engine(options.encoder);
    prefill_ = new qasr::Engine(options.decoder);
    qasr::cuda_check(cudaMalloc(&mask_, 0), "mask");
  }

  ~Transcriber() {
    for (void* pointer : {features_, mask_, embeddings_, valid_, pcm_device_, output_}) {
      if (pointer) cudaFree(pointer);
    }
    delete decoder_;
    delete encoder_;
    delete prefill_;
  }

  void transcribe(const std::string& path);

 private:
  void ensure(size_t features_bytes, size_t mask_bytes, size_t pcm_bytes, size_t embeddings_bytes);

  Options options_;
  qasr::MelAssets mel_assets_;
  qasr::TokenizerAssets tokenizer_;
  qasr::MelDeviceTables tables_;
  bool tables_ready_ = false;
  qasr::Decoder* decoder_ = nullptr;
  qasr::Engine* encoder_ = nullptr;
  qasr::Engine* prefill_ = nullptr;
  void* features_ = nullptr;
  size_t features_bytes_ = 0;
  void* mask_ = nullptr;
  size_t mask_bytes_ = 0;
  void* pcm_device_ = nullptr;
  size_t pcm_bytes_ = 0;
  void* embeddings_ = nullptr;
  size_t embeddings_bytes_ = 0;
  void* valid_ = nullptr;
  void* output_ = nullptr;
  size_t output_bytes_ = 0;
};

void Transcriber::ensure(size_t features_bytes, size_t mask_bytes, size_t pcm_bytes, size_t embeddings_bytes) {
  auto grow = [](void*& pointer, size_t& current, size_t wanted) {
    if (wanted <= current) return;
    if (pointer) cudaFree(pointer);
    cuda_check(cudaMalloc(&pointer, wanted), "cudaMalloc buffer");
    current = wanted;
  };
  grow(features_, features_bytes_, features_bytes);
  grow(mask_, mask_bytes_, mask_bytes);
  grow(pcm_device_, pcm_bytes_, pcm_bytes);
  grow(embeddings_, embeddings_bytes_, embeddings_bytes);
}

void Transcriber::transcribe(const std::string& path) {
  const auto started = std::chrono::steady_clock::now();
  int sample_rate = 0;
  auto pcm = qasr::load_audio(path, sample_rate);
  if (sample_rate != 16000) throw std::runtime_error("only 16 kHz audio is supported");
  if (pcm.size() < 8000) pcm.resize(8000, 0.0f);
  const double load_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - started).count();

  int valid_frames = 0;
  const int frames = qasr::mel_frame_count(static_cast<int>(pcm.size()), mel_assets_.hop, 100, valid_frames);
  const size_t mel_bytes = static_cast<size_t>(frames) * mel_assets_.n_mel * 4;
  ensure(mel_bytes, static_cast<size_t>(frames) * 4, pcm.size() * 4, 0);

  Timer mel_timer;
  cuda_check(cudaMemset(features_, 0, mel_bytes), "zero features");
  if (options_.mel_device == "cuda") {
    cuda_check(cudaMemcpy(pcm_device_, pcm.data(), pcm.size() * 4, cudaMemcpyHostToDevice), "copy pcm");
    qasr::mel_forward_gpu(static_cast<const float*>(pcm_device_), static_cast<int>(pcm.size()), tables_,
                          mel_assets_, static_cast<float*>(features_), frames);
    cuda_check(cudaDeviceSynchronize(), "mel sync");
  } else {
    std::vector<float> host(static_cast<size_t>(frames) * mel_assets_.n_mel, 0.0f);
    qasr::mel_forward_cpu(pcm.data(), static_cast<int>(pcm.size()), mel_assets_, host.data(), frames);
    cuda_check(cudaMemcpy(features_, host.data(), mel_bytes, cudaMemcpyHostToDevice), "copy mel");
  }
  const double mel_ms = mel_timer.ms();

  Timer encoder_timer;
  encoder_->set_shape("input_features", nvinfer1::Dims{3, {1, mel_assets_.n_mel, frames}});
  encoder_->set_shape("input_features_mask", nvinfer1::Dims{2, {1, frames}});
  std::vector<int32_t> mask_host(frames, 1);
  cuda_check(cudaMemcpy(mask_, mask_host.data(), static_cast<size_t>(frames) * 4, cudaMemcpyHostToDevice),
             "copy mask");
  const nvinfer1::Dims out_dims = encoder_->shape_after("audio_embeddings");
  const int hidden = static_cast<int>(out_dims.d[out_dims.nbDims - 1]);
  const int rows = static_cast<int>(out_dims.d[out_dims.nbDims - 2]);
  ensure(features_bytes_, mask_bytes_, pcm_bytes_, static_cast<size_t>(rows) * hidden * 2);
  if (!valid_) cuda_check(cudaMalloc(&valid_, 8), "valid");
  encoder_->bind("input_features", features_);
  encoder_->bind("input_features_mask", mask_);
  encoder_->bind("audio_embeddings", embeddings_);
  encoder_->bind("valid_tokens", valid_);
  encoder_->run();
  cuda_check(cudaDeviceSynchronize(), "encoder sync");
  const double encoder_ms = encoder_timer.ms();

  int64_t encoded_tokens = 0;
  cuda_check(cudaMemcpy(&encoded_tokens, valid_, 8, cudaMemcpyDeviceToHost), "valid tokens");
  int audio_tokens = valid_frames / 100 * 13;
  int remainder = valid_frames % 100;
  for (int i = 0; i < 3; ++i) remainder = (remainder + 1) / 2;
  audio_tokens += remainder;
  if (audio_tokens <= 0 || audio_tokens > encoded_tokens) throw std::runtime_error("bad audio token count");

  const auto suffix = tokenizer_.suffix.find(options_.language);
  if (suffix == tokenizer_.suffix.end()) throw std::runtime_error("unknown language: " + options_.language);
  std::vector<int32_t> ids = tokenizer_.prefix;
  const int first_audio = static_cast<int>(ids.size());
  ids.insert(ids.end(), static_cast<size_t>(audio_tokens), tokenizer_.audio_token);
  ids.insert(ids.end(), suffix->second.begin(), suffix->second.end());

  Timer prefill_timer;
  int token = decoder_->prefill(*prefill_, ids, embeddings_, static_cast<int>(audio_tokens), first_audio);
  const double prefill_ms = prefill_timer.ms();

  Timer decode_timer;
  (void)token;
  std::vector<int32_t> produced = decoder_->decode(options_.max_new_tokens);
  const double decode_ms = decode_timer.ms();
  const int eos_a = 151643;
  const int eos_b = 151645;
  if (!produced.empty() && (produced.back() == eos_a || produced.back() == eos_b)) produced.pop_back();

  const std::string text = qasr::decode_tokens(tokenizer_, produced);
  std::printf("{\"text\": \"%s\"}\n", escape(text).c_str());
  std::fflush(stdout);

  if (timing_enabled()) {
    const double total_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr,
                 "[time] %s load %.1f mel %.2f encoder %.2f prefill %.2f decode %.2f total %.1f | chunks %d "
                 "frames %d prompt %zu tokens %zu\n",
                 path.c_str(), load_ms, mel_ms, encoder_ms, prefill_ms, decode_ms, total_ms, audio_tokens,
                 frames, ids.size(), produced.size());
  }
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + arg);
      return argv[++i];
    };
    if (arg == "--model") {
      options.model = next();
    } else if (arg == "--encoder") {
      options.encoder = next();
    } else if (arg == "--decoder") {
      options.decoder = next();
    } else if (arg == "--assets") {
      options.assets = next();
    } else if (arg == "--language") {
      options.language = next();
    } else if (arg == "--mel-device") {
      options.mel_device = next();
    } else if (arg == "--max-new-tokens") {
      options.max_new_tokens = std::atoi(next().c_str());
    } else if (arg == "--max-cache") {
      options.max_cache = std::atoi(next().c_str());
    } else if (!arg.empty() && arg[0] != '-') {
      options.audio.push_back(arg);
    } else {
      std::fprintf(stderr, "unknown option %s\n", arg.c_str());
      return 2;
    }
  }
  if (options.model.empty() || options.encoder.empty() || options.decoder.empty() || options.audio.empty()) {
    std::fprintf(stderr,
                 "usage: qwen3-asr-native --model DIR --encoder ENGINE --decoder ENGINE AUDIO... "
                 "[--assets DIR] [--language NAME] [--mel-device cuda|cpu] [--max-cache N]\n");
    return 2;
  }

  try {
    Transcriber transcriber(options);
    for (const auto& path : options.audio) transcriber.transcribe(path);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
