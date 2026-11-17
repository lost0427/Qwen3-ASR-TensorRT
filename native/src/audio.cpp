#include "qasr.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#endif

namespace qasr {

namespace {

#ifdef _WIN32
struct SndFileInfo {
  int64_t frames;
  int samplerate;
  int channels;
  int format;
  int sections;
  int seekable;
};

struct MemoryFile {
  const uint8_t* data = nullptr;
  int64_t size = 0;
  int64_t offset = 0;
};

struct SfVirtualIo {
  int64_t (*get_filelen)(void*);
  int64_t (*seek)(int64_t, int, void*);
  int64_t (*read)(void*, int64_t, void*);
  int64_t (*write)(const void*, int64_t, void*);
  int64_t (*tell)(void*);
};

int64_t memory_length(void* user) { return static_cast<MemoryFile*>(user)->size; }

int64_t memory_seek(int64_t offset, int whence, void* user) {
  auto* file = static_cast<MemoryFile*>(user);
  int64_t target = offset;
  if (whence == 1) target += file->offset;
  if (whence == 2) target += file->size;
  if (target < 0 || target > file->size) return -1;
  file->offset = target;
  return target;
}

int64_t memory_read(void* out, int64_t count, void* user) {
  auto* file = static_cast<MemoryFile*>(user);
  const int64_t available = file->size - file->offset;
  if (count > available) count = available;
  std::memcpy(out, file->data + file->offset, static_cast<size_t>(count));
  file->offset += count;
  return count;
}

int64_t memory_write(const void*, int64_t, void*) { return 0; }

int64_t memory_tell(void* user) { return static_cast<MemoryFile*>(user)->offset; }

void* open_sndfile_library() {
  const char* names[] = {"libsndfile_x64.dll", "libsndfile.dll"};
  for (const char* name : names) {
    if (HMODULE handle = LoadLibraryA(name)) return handle;
  }
  return nullptr;
}
#endif

}  // namespace

std::vector<float> load_audio(const std::string& path, int& sample_rate) {
#ifdef _WIN32
  void* handle = open_sndfile_library();
  if (!handle) throw std::runtime_error("libsndfile not found on PATH (set it next to the executable)");
  using OpenFn = void* (*)(const char*, int, SndFileInfo*);
  using ReadFn = int64_t (*)(void*, float*, int64_t);
  using CloseFn = int (*)(void*);
  auto open = reinterpret_cast<OpenFn>(GetProcAddress(static_cast<HMODULE>(handle), "sf_open"));
  auto read = reinterpret_cast<ReadFn>(GetProcAddress(static_cast<HMODULE>(handle), "sf_readf_float"));
  auto close = reinterpret_cast<CloseFn>(GetProcAddress(static_cast<HMODULE>(handle), "sf_close"));
  if (!open || !read || !close) throw std::runtime_error("libsndfile is missing symbols");

  SndFileInfo info{};
  const int read_mode = 0x10;  // SFM_READ
  void* file = open(path.c_str(), read_mode, &info);
  if (!file) throw std::runtime_error("cannot open audio: " + path);

  std::vector<float> interleaved(static_cast<size_t>(info.frames) * info.channels);
  const int64_t got = read(file, interleaved.data(), info.frames);
  close(file);
  sample_rate = info.samplerate;
  if (got <= 0) throw std::runtime_error("empty audio: " + path);

  std::vector<float> mono(static_cast<size_t>(got));
  for (int64_t i = 0; i < got; ++i) {
    float sum = 0.0f;
    for (int c = 0; c < info.channels; ++c) sum += interleaved[i * info.channels + c];
    mono[i] = sum / static_cast<float>(info.channels);
  }
  return mono;
#else
  (void)path;
  (void)sample_rate;
  throw std::runtime_error("native audio loading is implemented for Windows only");
#endif
}

std::vector<float> reflect_pad(const std::vector<float>& pcm, int pad) {
  const int length = static_cast<int>(pcm.size());
  std::vector<float> padded(static_cast<size_t>(length) + 2 * pad);
  for (int i = 0; i < pad; ++i) padded[static_cast<size_t>(i)] = pcm[static_cast<size_t>(pad - i)];
  std::memcpy(padded.data() + pad, pcm.data(), static_cast<size_t>(length) * sizeof(float));
  for (int i = 0; i < pad; ++i) {
    padded[static_cast<size_t>(pad + length + i)] = pcm[static_cast<size_t>(length - 2 - i)];
  }
  return padded;
}

std::vector<char> read_file(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open " + path);
  const std::streamsize size = stream.tellg();
  if (size <= 0) throw std::runtime_error("empty file: " + path);
  std::vector<char> blob(static_cast<size_t>(size));
  stream.seekg(0, std::ios::beg);
  if (!stream.read(blob.data(), size)) throw std::runtime_error("short read: " + path);
  return blob;
}

std::vector<float> load_audio_memory(const void* data, size_t size, int& sample_rate) {
#ifdef _WIN32
  void* handle = open_sndfile_library();
  if (!handle) throw std::runtime_error("libsndfile not found on PATH");
  using OpenVirtualFn = void* (*)(SfVirtualIo*, int, SndFileInfo*, void*);
  using ReadFn = int64_t (*)(void*, float*, int64_t);
  using CloseFn = int (*)(void*);
  auto open = reinterpret_cast<OpenVirtualFn>(
      GetProcAddress(static_cast<HMODULE>(handle), "sf_open_virtual"));
  auto read = reinterpret_cast<ReadFn>(GetProcAddress(static_cast<HMODULE>(handle), "sf_readf_float"));
  auto close = reinterpret_cast<CloseFn>(GetProcAddress(static_cast<HMODULE>(handle), "sf_close"));
  if (!open || !read || !close) throw std::runtime_error("libsndfile is missing symbols");

  MemoryFile file{static_cast<const uint8_t*>(data), static_cast<int64_t>(size), 0};
  SfVirtualIo io{memory_length, memory_seek, memory_read, memory_write, memory_tell};
  SndFileInfo info{};
  const int read_mode = 0x10;  // SFM_READ
  void* stream = open(&io, read_mode, &info, &file);
  if (!stream) throw std::runtime_error("cannot decode audio from memory");

  std::vector<float> interleaved(static_cast<size_t>(info.frames) * info.channels);
  const int64_t got = read(stream, interleaved.data(), info.frames);
  close(stream);
  sample_rate = info.samplerate;
  if (got <= 0) throw std::runtime_error("empty audio");

  std::vector<float> mono(static_cast<size_t>(got));
  for (int64_t i = 0; i < got; ++i) {
    float sum = 0.0f;
    for (int c = 0; c < info.channels; ++c) sum += interleaved[i * info.channels + c];
    mono[i] = sum / static_cast<float>(info.channels);
  }
  return mono;
#else
  (void)data;
  (void)size;
  (void)sample_rate;
  throw std::runtime_error("memory audio loading is implemented for Windows only");
#endif
}

std::vector<float> log_mel(const std::vector<float>& input, const MelAssets& assets,
                           int multiple, int& frames, int& valid_frames) {
  constexpr int kMinLength = 8000;
  std::vector<float> pcm = input;
  if (static_cast<int>(pcm.size()) < kMinLength) pcm.resize(kMinLength, 0.0f);

  const int pad = assets.n_fft / 2;
  const int length = static_cast<int>(pcm.size());
  auto sample = [&](int index) -> double {
    if (index < 0) index = -index;                        // reflect
    if (index >= length) index = 2 * (length - 1) - index;  // reflect
    if (index < 0) index = 0;
    if (index >= length) index = length - 1;
    return static_cast<double>(pcm[index]);
  };

  valid_frames = length / assets.hop;
  frames = valid_frames;
  if (multiple > 1 && frames % multiple) frames += multiple - frames % multiple;

  std::vector<float> mel(static_cast<size_t>(assets.n_mel) * frames, 0.0f);
  std::vector<double> power(assets.n_freq);
  std::vector<double> cos_table(static_cast<size_t>(assets.n_freq) * assets.n_fft);
  std::vector<double> sin_table(static_cast<size_t>(assets.n_freq) * assets.n_fft);
  const double two_pi = 6.283185307179586476925286766559;
  for (int k = 0; k < assets.n_freq; ++k) {
    for (int n = 0; n < assets.n_fft; ++n) {
      const double angle = two_pi * k * n / assets.n_fft;
      cos_table[static_cast<size_t>(k) * assets.n_fft + n] = std::cos(angle);
      sin_table[static_cast<size_t>(k) * assets.n_fft + n] = std::sin(angle);
    }
  }

  std::vector<double> frame(assets.n_fft);
  double max_log = -1e30;
  for (int f = 0; f < valid_frames; ++f) {
    const int start = f * assets.hop;
    for (int n = 0; n < assets.n_fft; ++n) {
      frame[n] = sample(start - pad + n) * static_cast<double>(assets.window[n]);
    }
    for (int k = 0; k < assets.n_freq; ++k) {
      double re = 0.0, im = 0.0;
      const double* cos_row = &cos_table[static_cast<size_t>(k) * assets.n_fft];
      const double* sin_row = &sin_table[static_cast<size_t>(k) * assets.n_fft];
      for (int n = 0; n < assets.n_fft; ++n) {
        re += frame[n] * cos_row[n];
        im -= frame[n] * sin_row[n];
      }
      power[k] = re * re + im * im;
    }
    for (int m = 0; m < assets.n_mel; ++m) {
      const float* filter = &assets.filters[static_cast<size_t>(m) * assets.n_freq];
      double sum = 0.0;
      for (int k = 0; k < assets.n_freq; ++k) sum += static_cast<double>(filter[k]) * power[k];
      if (sum < 1e-10) sum = 1e-10;
      const double value = std::log10(sum);
      mel[static_cast<size_t>(m) * frames + f] = static_cast<float>(value);
      if (value > max_log) max_log = value;
    }
  }

  const double floor_value = max_log - 8.0;
  for (size_t index = 0; index < mel.size(); ++index) {
    double value = mel[index];
    if (value < floor_value) value = floor_value;
    mel[index] = static_cast<float>((value + 4.0) / 4.0);
  }
  return mel;
}

}  // namespace qasr
