#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <NvInfer.h>

namespace qasr {

struct Logger : nvinfer1::ILogger {
  void log(Severity severity, const char* message) noexcept override;
};

void cuda_check(int code, const char* what);

class Engine {
 public:
  explicit Engine(const std::string& path);
  ~Engine();

  int num_io() const { return engine_ ? engine_->getNbIOTensors() : 0; }
  const char* name(int index) const;
  bool is_input(int index) const;
  nvinfer1::Dims dims(int index) const;
  nvinfer1::DataType dtype(int index) const;
  void set_shape(const char* name, nvinfer1::Dims value);
  // Shape resolved by the context after the inputs were set (for dynamic outputs).
  nvinfer1::Dims shape_after(const char* name) const;
  void bind(const char* name, void* pointer);
  void run();

 private:
  Logger logger_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
};

struct MelAssets {
  int n_mel = 0, n_freq = 0, n_fft = 0, hop = 0;
  std::vector<float> filters;  // [n_mel][n_freq]
  std::vector<float> window;   // [n_fft]
};

struct TokenizerAssets {
  std::vector<std::vector<uint8_t>> tokens;
  std::vector<int32_t> special;
  std::vector<int32_t> prefix;
  int32_t audio_token = 0;
  std::map<std::string, std::vector<int32_t>> suffix;
};

MelAssets load_mel_assets(const std::string& path);
TokenizerAssets load_tokenizer_assets(const std::string& path);
std::string decode_tokens(const TokenizerAssets& assets, const std::vector<int32_t>& ids);

// Reads PCM through libsndfile (loaded at run time) and returns mono float32.
std::vector<float> load_audio(const std::string& path, int& sample_rate);
std::vector<float> load_audio_memory(const void* data, size_t size, int& sample_rate);

// Log-mel features as [n_mel][frames], frames padded to a multiple of `multiple`.
// The CPU path uses pocketfft, the library torch's CPU STFT is built on.
void mel_forward_cpu(const float* pcm, int length, const MelAssets& assets, float* out, int frames);

// The same features on the GPU: windowed DFT, mel accumulation and log in two passes.
struct MelDeviceTables {
  void* cos = nullptr;
  void* sin = nullptr;
  void* window = nullptr;
  void* filters = nullptr;
  MelDeviceTables() = default;
  MelDeviceTables(const MelDeviceTables&) = delete;
  MelDeviceTables& operator=(const MelDeviceTables&) = delete;
  MelDeviceTables(MelDeviceTables&& other) noexcept { *this = std::move(other); }
  MelDeviceTables& operator=(MelDeviceTables&& other) noexcept;
  ~MelDeviceTables();
};

MelDeviceTables upload_mel_tables(const MelAssets& assets);
void mel_forward_gpu(const float* pcm, int length, const MelDeviceTables& tables, const MelAssets& assets,
                     float* out, int frames);

int mel_frame_count(int length, int hop, int multiple, int& valid_frames);

}  // namespace qasr
