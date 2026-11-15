#include "qasr.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace qasr {

namespace {

template <typename T>
T read_value(std::ifstream& stream) {
  T value{};
  stream.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!stream) throw std::runtime_error("truncated asset file");
  return value;
}

}  // namespace

MelAssets load_mel_assets(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  MelAssets assets;
  assets.n_mel = read_value<uint32_t>(stream);
  assets.n_freq = read_value<uint32_t>(stream);
  assets.n_fft = read_value<uint32_t>(stream);
  assets.hop = read_value<uint32_t>(stream);
  assets.filters.resize(static_cast<size_t>(assets.n_mel) * assets.n_freq);
  stream.read(reinterpret_cast<char*>(assets.filters.data()), assets.filters.size() * sizeof(float));
  assets.window.resize(assets.n_fft);
  stream.read(reinterpret_cast<char*>(assets.window.data()), assets.window.size() * sizeof(float));
  if (!stream) throw std::runtime_error("truncated " + path);
  return assets;
}

TokenizerAssets load_tokenizer_assets(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open " + path);
  TokenizerAssets assets;
  const uint32_t count = read_value<uint32_t>(stream);
  assets.tokens.resize(count);
  std::vector<char> buffer;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t length = read_value<uint32_t>(stream);
    buffer.resize(length);
    if (length) stream.read(buffer.data(), length);
    assets.tokens[i].assign(buffer.begin(), buffer.begin() + length);
  }
  const uint32_t special_count = read_value<uint32_t>(stream);
  assets.special.resize(special_count);
  for (uint32_t i = 0; i < special_count; ++i) assets.special[i] = static_cast<int32_t>(read_value<uint32_t>(stream));
  const uint32_t prefix_count = read_value<uint32_t>(stream);
  assets.prefix.resize(prefix_count);
  for (uint32_t i = 0; i < prefix_count; ++i) assets.prefix[i] = static_cast<int32_t>(read_value<uint32_t>(stream));
  assets.audio_token = static_cast<int32_t>(read_value<uint32_t>(stream));
  const uint32_t languages = read_value<uint32_t>(stream);
  for (uint32_t i = 0; i < languages; ++i) {
    const uint32_t name_length = read_value<uint32_t>(stream);
    std::string name(name_length, '\0');
    if (name_length) stream.read(name.data(), name_length);
    const uint32_t id_count = read_value<uint32_t>(stream);
    std::vector<int32_t> ids(id_count);
    for (uint32_t j = 0; j < id_count; ++j) ids[j] = static_cast<int32_t>(read_value<uint32_t>(stream));
    assets.suffix[name] = std::move(ids);
  }
  if (!stream) throw std::runtime_error("truncated " + path);
  return assets;
}

std::string decode_tokens(const TokenizerAssets& assets, const std::vector<int32_t>& ids) {
  std::vector<uint8_t> bytes;
  for (const int32_t id : ids) {
    if (id < 0 || id >= static_cast<int32_t>(assets.tokens.size())) continue;
    bool skip = false;
    for (const int32_t special : assets.special) {
      if (special == id) {
        skip = true;
        break;
      }
    }
    if (skip) continue;
    bytes.insert(bytes.end(), assets.tokens[id].begin(), assets.tokens[id].end());
  }
  return std::string(bytes.begin(), bytes.end());
}

}  // namespace qasr
