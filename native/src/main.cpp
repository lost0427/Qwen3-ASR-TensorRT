#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "decoder.h"
#include "qasr.h"

namespace {

struct Options {
  std::string model;
  std::string encoder;
  std::string decoder;
  std::string assets = "assets";
  std::string language = "English";
  std::string audio;
  int max_new_tokens = 256;
  int max_cache = 2048;
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
    } else if (arg == "--max-new-tokens") {
      options.max_new_tokens = std::atoi(next().c_str());
    } else if (arg == "--max-cache") {
      options.max_cache = std::atoi(next().c_str());
    } else if (arg.size() && arg[0] != '-') {
      options.audio = arg;
    } else {
      std::fprintf(stderr, "unknown option %s\n", arg.c_str());
      return 2;
    }
  }
  if (options.model.empty() || options.encoder.empty() || options.decoder.empty() || options.audio.empty()) {
    std::fprintf(stderr,
                 "usage: qwen3-asr-native --model DIR --encoder ENGINE --decoder ENGINE "
                 "[--assets DIR] [--language NAME] [--max-cache N] AUDIO\n");
    return 2;
  }

  try {
    const auto mel_assets = qasr::load_mel_assets(options.assets + "/mel.bin");
    const auto tokenizer = qasr::load_tokenizer_assets(options.assets + "/tokenizer.bin");

    int sample_rate = 0;
    const auto pcm = qasr::load_audio(options.audio, sample_rate);
    if (sample_rate != 16000) {
      throw std::runtime_error("only 16 kHz audio is supported by the native runtime");
    }

    int frames = 0;
    int valid_frames = 0;
    const auto mel = qasr::log_mel(pcm, mel_assets, 100, frames, valid_frames);

    qasr::Engine encoder(options.encoder);
    void* features = nullptr;
    void* mask = nullptr;
    void* embeddings = nullptr;
    void* valid = nullptr;
    qasr::cuda_check(cudaMalloc(&features, static_cast<size_t>(frames) * mel_assets.n_mel * 4), "features");
    qasr::cuda_check(cudaMalloc(&mask, static_cast<size_t>(frames) * 4), "mask");
    qasr::cuda_check(cudaMemcpy(features, mel.data(), static_cast<size_t>(frames) * mel_assets.n_mel * 4,
                                cudaMemcpyHostToDevice), "copy mel");
    std::vector<int32_t> mask_host(frames, 1);
    qasr::cuda_check(cudaMemcpy(mask, mask_host.data(), mask_host.size() * 4, cudaMemcpyHostToDevice), "copy mask");

    encoder.set_shape("input_features", nvinfer1::Dims{3, {1, mel_assets.n_mel, frames}});
    encoder.set_shape("input_features_mask", nvinfer1::Dims{2, {1, frames}});
    const nvinfer1::Dims out_dims = encoder.shape_after("audio_embeddings");
    const int hidden = static_cast<int>(out_dims.d[out_dims.nbDims - 1]);
    const int rows = static_cast<int>(out_dims.d[out_dims.nbDims - 2]);
    qasr::cuda_check(cudaMalloc(&embeddings, static_cast<size_t>(rows) * hidden * 2), "embeddings");
    qasr::cuda_check(cudaMalloc(&valid, 8), "valid");
    encoder.bind("input_features", features);
    encoder.bind("input_features_mask", mask);
    encoder.bind("audio_embeddings", embeddings);
    encoder.bind("valid_tokens", valid);
    encoder.run();
    qasr::cuda_check(cudaDeviceSynchronize(), "encoder sync");

    int64_t encoded_tokens = 0;
    qasr::cuda_check(cudaMemcpy(&encoded_tokens, valid, 8, cudaMemcpyDeviceToHost), "copy valid tokens");
    // The processor turns the real frame count into placeholder tokens; the
    // encoder output is wider because it also saw the padded tail.
    int audio_tokens = valid_frames / 100 * 13;
    int remainder = valid_frames % 100;
    for (int i = 0; i < 3; ++i) remainder = (remainder + 1) / 2;
    audio_tokens += remainder;
    if (audio_tokens <= 0) throw std::runtime_error("encoder produced no audio tokens");
    if (audio_tokens > encoded_tokens) {
      throw std::runtime_error("placeholder count exceeds the encoder output");
    }

    if (const char* dump = std::getenv("QASR_DUMP")) {
      const std::string dir = dump;
      std::ofstream mel_out(dir + "/mel.f32", std::ios::binary);
      mel_out.write(reinterpret_cast<const char*>(mel.data()),
                    static_cast<std::streamsize>(mel.size() * sizeof(float)));
      std::ofstream audio_out(dir + "/audio.f32", std::ios::binary);
      std::vector<uint16_t> audio_host(static_cast<size_t>(rows) * hidden);
      qasr::cuda_check(cudaMemcpy(audio_host.data(), embeddings, audio_host.size() * 2,
                                  cudaMemcpyDeviceToHost), "copy audio");
      std::vector<float> audio_float(audio_host.size());
      for (size_t i = 0; i < audio_host.size(); ++i) {
        const uint32_t bits = static_cast<uint32_t>(audio_host[i]) << 16;  // bf16 -> f32
        std::memcpy(&audio_float[i], &bits, 4);
      }
      audio_out.write(reinterpret_cast<const char*>(audio_float.data()),
                      static_cast<std::streamsize>(audio_float.size() * sizeof(float)));
      std::ofstream meta(dir + "/meta.txt");
      meta << "frames " << frames << " valid " << valid_frames << " audio_tokens " << audio_tokens
           << " encoded " << encoded_tokens << " rows " << rows << " hidden " << hidden << "\n";
    }

    const auto suffix = tokenizer.suffix.find(options.language);
    if (suffix == tokenizer.suffix.end()) {
      throw std::runtime_error("unknown language: " + options.language);
    }
    std::vector<int32_t> ids = tokenizer.prefix;
    const int first_audio = static_cast<int>(ids.size());
    ids.insert(ids.end(), static_cast<size_t>(audio_tokens), tokenizer.audio_token);
    ids.insert(ids.end(), suffix->second.begin(), suffix->second.end());

    if (const char* dump = std::getenv("QASR_DUMP")) {
      std::ofstream ids_out(std::string(dump) + "/ids.txt");
      for (int32_t id : ids) ids_out << id << "\n";
    }

    qasr::Decoder decoder(options.model, options.max_cache);
    qasr::Engine prefill(options.decoder);
    int token = decoder.prefill(prefill, ids, embeddings, static_cast<int>(audio_tokens), first_audio);

    if (const char* dump = std::getenv("QASR_DUMP")) {
      std::ofstream first(std::string(dump) + "/first.txt");
      first << token << " prompt " << ids.size() << "\n";
    }

    std::vector<int32_t> produced{token};
    const int eos_a = 151643;
    const int eos_b = 151645;
    while (static_cast<int>(produced.size()) < options.max_new_tokens && token != eos_a && token != eos_b &&
           decoder.position() < decoder.capacity()) {
      token = decoder.step();
      produced.push_back(token);
    }
    if (!produced.empty() && (produced.back() == eos_a || produced.back() == eos_b)) produced.pop_back();

    if (const char* dump = std::getenv("QASR_DUMP")) {
      std::ofstream produced_out(std::string(dump) + "/produced.txt");
      for (int32_t id : produced) produced_out << id << "\n";
    }
    const std::string text = qasr::decode_tokens(tokenizer, produced);
    std::printf("{\"text\": \"%s\"}\n", escape(text).c_str());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
