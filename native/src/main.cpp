// Command line entry point: transcribes files, or serves the same HTTP endpoint
// as the Python runtime.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "decoder.h"
#include "httplib.h"
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
  int max_cache = 4096;
  bool serve = false;
  std::string host = "127.0.0.1";
  int port = 8000;
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
  explicit Transcriber(const Options& options) : options_(options) {
    mel_assets_ = qasr::load_mel_assets(options.assets + "/mel.bin");
    tokenizer_ = qasr::load_tokenizer_assets(options.assets + "/tokenizer.bin");
    if (options.mel_device == "cuda") {
      tables_ = qasr::upload_mel_tables(mel_assets_);
      tables_ready_ = true;
    }
    decoder_ = new qasr::Decoder(options.model, options.max_cache);
    encoder_ = new qasr::Engine(options.encoder);
    prefill_ = new qasr::Engine(options.decoder);
  }

  ~Transcriber() {
    for (void* pointer : {features_, mask_, embeddings_, valid_, pcm_device_}) {
      if (pointer) cudaFree(pointer);
    }
    delete decoder_;
    delete encoder_;
    delete prefill_;
  }

  void set_language(const std::string& language) { options_.language = language; }

  std::string process(const std::vector<float>& input, const char* label);

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

std::string Transcriber::process(const std::vector<float>& input, const char* label) {
  const auto started = std::chrono::steady_clock::now();
  std::vector<float> pcm = input;
  if (pcm.size() < 8000) pcm.resize(8000, 0.0f);

  int valid_frames = 0;
  const int frames = qasr::mel_frame_count(static_cast<int>(pcm.size()), mel_assets_.hop, 100, valid_frames);
  const int encoder_max = encoder_->profile_max("input_features", 2);
  if (encoder_max > 0 && frames > encoder_max) {
    char message[512];
    std::snprintf(message, sizeof(message),
                  "audio of %d frames (%.1f s) exceeds this encoder's %d-frame profile (%.1f s); "
                  "rebuild it with `qwen3-asr build --max-frames %d`",
                  frames, frames / 100.0, encoder_max, encoder_max / 100.0, frames);
    throw std::runtime_error(message);
  }
  const size_t mel_bytes = static_cast<size_t>(frames) * mel_assets_.n_mel * 4;

  Timer mel_timer;
  const auto padded = qasr::reflect_pad(pcm, mel_assets_.n_fft / 2);
  ensure(mel_bytes, static_cast<size_t>(frames) * 4, padded.size() * 4, 0);
  if (options_.mel_device == "cuda") {
    cuda_check(cudaMemcpy(pcm_device_, padded.data(), padded.size() * 4, cudaMemcpyHostToDevice), "copy pcm");
    qasr::mel_forward_gpu(static_cast<const float*>(pcm_device_), tables_, mel_assets_,
                          static_cast<float*>(features_), valid_frames, frames);
    cuda_check(cudaDeviceSynchronize(), "mel sync");
  } else {
    std::vector<float> host(static_cast<size_t>(frames) * mel_assets_.n_mel, 0.0f);
    qasr::mel_forward_cpu(padded.data(), valid_frames, frames, mel_assets_, host.data());
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
  (void)decoder_->prefill(*prefill_, ids, embeddings_, static_cast<int>(audio_tokens), first_audio,
                          options_.max_new_tokens);
  const double prefill_ms = prefill_timer.ms();

  if (std::getenv("QASR_BENCH")) decoder_->bench(std::atoi(std::getenv("QASR_BENCH")));

  Timer decode_timer;
  const std::vector<int32_t> produced = decoder_->decode(options_.max_new_tokens);
  const double decode_ms = decode_timer.ms();

  if (timing_enabled()) {
    const double total_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started).count();
    std::fprintf(stderr,
                 "[time] %s mel %.2f encoder %.2f prefill %.2f decode %.2f total %.1f | frames %d prompt "
                 "%zu tokens %zu\n",
                 label, mel_ms, encoder_ms, prefill_ms, decode_ms, total_ms, frames, ids.size(),
                 produced.size());
  }
  return qasr::decode_tokens(tokenizer_, produced);
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
    } else if (arg == "--serve") {
      options.serve = true;
    } else if (arg == "--host") {
      options.host = next();
    } else if (arg == "--port") {
      options.port = std::atoi(next().c_str());
    } else if (!arg.empty() && arg[0] != '-') {
      options.audio.push_back(arg);
    } else {
      std::fprintf(stderr, "unknown option %s\n", arg.c_str());
      return 2;
    }
  }
  if (options.model.empty() || options.encoder.empty() || options.decoder.empty()) {
    std::fprintf(stderr,
                 "usage: qwen3-asr-native --model DIR --encoder ENGINE --decoder ENGINE "
                 "(AUDIO... | --serve [--host H] [--port P]) [--assets DIR] [--language NAME] "
                 "[--mel-device cuda|cpu] [--max-cache N]\n");
    return 2;
  }
  if (!options.serve && options.audio.empty()) {
    std::fprintf(stderr, "error: no audio files and not serving\n");
    return 2;
  }

  try {
    Transcriber transcriber(options);
    if (options.serve) {
      httplib::Server server;
      server.set_payload_max_length(128ull * 1024 * 1024);
      std::mutex lock;
      server.Post("/v1/transcribe", [&](const httplib::Request& request, httplib::Response& response) {
        std::string language = options.language;
        if (request.has_param("language")) language = request.get_param_value("language");
        try {
          std::lock_guard<std::mutex> guard(lock);
          transcriber.set_language(language);
          int sample_rate = 0;
          const auto pcm = qasr::load_audio_memory(request.body.data(), request.body.size(), sample_rate);
          if (sample_rate != 16000) throw std::runtime_error("only 16 kHz audio is supported");
          const std::string text = transcriber.process(pcm, "http");
          response.set_content("{\"text\": \"" + escape(text) + "\"}", "application/json");
        } catch (const std::exception& error) {
          response.status = 400;
          response.set_content("{\"error\": \"" + escape(error.what()) + "\"}", "application/json");
        }
      });
      std::fprintf(stderr, "listening on http://%s:%d/v1/transcribe\n", options.host.c_str(), options.port);
      if (!server.listen(options.host.c_str(), options.port)) {
        std::fprintf(stderr, "error: cannot bind %s:%d\n", options.host.c_str(), options.port);
        return 1;
      }
      return 0;
    }

    for (const auto& path : options.audio) {
      int sample_rate = 0;
      const auto pcm = qasr::load_audio(path, sample_rate);
      if (sample_rate != 16000) throw std::runtime_error("only 16 kHz audio is supported");
      const std::string text = transcriber.process(pcm, path.c_str());
      std::printf("{\"text\": \"%s\"}\n", escape(text).c_str());
      std::fflush(stdout);
    }
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
