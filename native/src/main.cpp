// Command line entry point: transcribes files, or serves the same HTTP endpoint
// as the Python runtime.

#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "aligner.h"
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
  std::string mel_device = "auto";
  int max_new_tokens = 256;
  int max_cache = 512;
  bool serve = false;
  bool align = false;
  std::string align_encoder;
  std::string align_decoder;
  std::string text;
  std::string text_file;
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
    if (options.mel_device != "cpu") {
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
  // pocketfft wins below ~200 frames (the GPU kernel is launch bound there).
  const bool gpu_mel =
      options_.mel_device == "cuda" || (options_.mel_device == "auto" && frames >= 200);
  if (gpu_mel) {
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
  // The reference runtime feeds the encoder an all-ones mask, so padding frames
  // stay visible; keep that behaviour to stay transcript-identical.
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

namespace {

std::string to_utf8(const wchar_t* text) {
  if (!text) return {};
  const int bytes = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  if (bytes <= 1) return {};
  std::string out(static_cast<size_t>(bytes - 1), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text, -1, out.data(), bytes, nullptr, nullptr);
  return out;
}

}  // namespace

int run(const std::vector<std::string>& args) {
  Options options;
  for (size_t i = 1; i < args.size(); ++i) {
    const std::string arg = args[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= args.size()) throw std::runtime_error("missing value for " + arg);
      return args[++i];
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
    } else if (arg == "--align") {
      options.align = true;
    } else if (arg == "--align-encoder") {
      options.align_encoder = next();
    } else if (arg == "--align-decoder") {
      options.align_decoder = next();
    } else if (arg == "--text") {
      options.text = next();
    } else if (arg == "--text-file") {
      options.text_file = next();
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
  if ((!options.align && options.model.empty()) || options.encoder.empty() || options.decoder.empty()) {
    std::fprintf(stderr,
                 "usage: qwen3-asr-native --model DIR --encoder ENGINE --decoder ENGINE "
                 "(AUDIO... | --serve [--host H] [--port P]) [--assets DIR] [--language NAME] "
                 "[--mel-device cuda|cpu] [--max-cache N]\n"
                 "       qwen3-asr-native --align --encoder ALIGNER_ENCODER --decoder ALIGNER_DECODER "
                 "(--text TEXT | --text-file FILE) AUDIO [--assets DIR] [--language NAME]\n");
    return 2;
  }
  if (!options.serve && options.audio.empty()) {
    std::fprintf(stderr, "error: no audio files and not serving\n");
    return 2;
  }

  try {
    if (options.align) {
      if (options.text.empty() && options.text_file.empty()) {
        std::fprintf(stderr, "error: --align requires --text or --text-file\n");
        return 2;
      }
      std::string text = options.text;
      if (text.empty()) {
        const std::vector<char> blob = qasr::read_file(options.text_file);
        text.assign(blob.begin(), blob.end());
      }
      while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.pop_back();
      size_t first = 0;
      while (first < text.size() && std::isspace(static_cast<unsigned char>(text[first]))) ++first;
      text.erase(0, first);

      qasr::Aligner aligner(options.assets, options.encoder, options.decoder, options.mel_device);
      for (const auto& path : options.audio) {
        int sample_rate = 0;
        const auto pcm = qasr::load_audio(path, sample_rate);
        if (sample_rate != 16000) throw std::runtime_error("only 16 kHz audio is supported");
        qasr::AlignerTimings timings;
        const auto items = aligner.align(pcm, text, options.language, &timings);

        std::string json = "{\"items\": [";
        for (size_t i = 0; i < items.size(); ++i) {
          char buffer[512];
          std::snprintf(buffer, sizeof(buffer),
                        "%s{\"text\": \"%s\", \"start_time\": %.3f, \"end_time\": %.3f}",
                        i ? ", " : "", escape(items[i].text).c_str(), items[i].start, items[i].end);
          json += buffer;
        }
        char tail[512];
        std::snprintf(tail, sizeof(tail),
                      "], \"timings\": {\"melMs\": %.2f, \"encoderMs\": %.2f, \"decoderMs\": %.2f, "
                      "\"totalMs\": %.2f, \"frames\": %d, \"tokens\": %d, \"words\": %d}}",
                      timings.mel_ms, timings.encoder_ms, timings.decoder_ms, timings.total_ms,
                      timings.frames, timings.tokens, timings.words);
        json += tail;
        std::printf("%s\n", json.c_str());
        std::fflush(stdout);
        if (timing_enabled()) {
          std::fprintf(stderr, "[time] %s mel %.2f encoder %.2f decoder %.2f total %.1f | frames %d tokens %d words %d\n",
                       path.c_str(), timings.mel_ms, timings.encoder_ms, timings.decoder_ms,
                       timings.total_ms, timings.frames, timings.tokens, timings.words);
        }
      }
      return 0;
    }

    Transcriber transcriber(options);
    if (options.serve) {
      httplib::Server server;
      server.set_payload_max_length(128ull * 1024 * 1024);
      std::mutex lock;
      std::unique_ptr<qasr::Aligner> aligner;
      if (!options.align_encoder.empty() && !options.align_decoder.empty()) {
        aligner.reset(new qasr::Aligner(options.assets, options.align_encoder, options.align_decoder,
                                        options.mel_device));
      }
      auto escape_json = [](const std::vector<qasr::AlignerItem>& items) {
        std::string json = "[";
        for (size_t i = 0; i < items.size(); ++i) {
          char buffer[512];
          std::snprintf(buffer, sizeof(buffer),
                        "%s{\"text\": \"%s\", \"start_time\": %.3f, \"end_time\": %.3f}",
                        i ? ", " : "", escape(items[i].text).c_str(), items[i].start, items[i].end);
          json += buffer;
        }
        return json + "]";
      };
      if (aligner) {
        server.Post("/v1/align", [&](const httplib::Request& request, httplib::Response& response) {
          std::string language = options.language;
          if (request.has_param("language")) language = request.get_param_value("language");
          const std::string text = request.has_param("text") ? request.get_param_value("text") : "";
          if (text.empty()) {
            response.status = 400;
            response.set_content("{\"error\": \"the text query parameter is required\"}", "application/json");
            return;
          }
          try {
            std::lock_guard<std::mutex> guard(lock);
            int sample_rate = 0;
            const auto pcm = qasr::load_audio_memory(request.body.data(), request.body.size(), sample_rate);
            if (sample_rate != 16000) throw std::runtime_error("only 16 kHz audio is supported");
            const auto items = aligner->align(pcm, text, language);
            response.set_content("{\"items\": " + escape_json(items) + "}", "application/json");
          } catch (const std::exception& error) {
            response.status = 400;
            response.set_content("{\"error\": \"" + escape(error.what()) + "\"}", "application/json");
          }
        });
      }
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

#ifdef _WIN32
int wmain(int argc, wchar_t** wide) {
  std::vector<std::string> args;
  args.reserve(static_cast<size_t>(argc));
  for (int i = 0; i < argc; ++i) args.push_back(to_utf8(wide[i]));
  return run(args);
}
#else
int main(int argc, char** argv) {
  std::vector<std::string> args(argv, argv + argc);
  return run(args);
}
#endif
