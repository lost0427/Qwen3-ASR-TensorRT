---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-0.6B-hf
library_name: exllamav3
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- exll3
- exllamav3
- quantized
- 3bpw
- 4bpw
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 0.6B EXL3 Decoder

## English

EXL3 quantized text-decoder weights derived from [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf). Use with a compatible [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder). The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) integrates both parts for transcription. These weights use the ExLlamaV3 format and are loaded by the project runtime.

### Variants

| Variant | Use |
| --- | --- |
| `decoder-exl3-3bpw` | Lower memory use |
| `decoder-exl3-4bpw` | Higher-precision alternative |

Each variant includes EXL3 `model.safetensors`, tokenizer files, model and generation configuration, and quantization configuration.

### Requirements

- NVIDIA GPU and CUDA
- ExLlamaV3 1.6.6 with compatible PyTorch and CUDA versions
- Reference environment: PyTorch 2.10.0, CUDA 12.8, Python 3.10–3.13
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which includes the encoder/decoder integration

Windows is tested and supported. Linux should work with compatible CUDA, TensorRT, PyTorch, and ExLlamaV3 packages.

### Usage

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-3bpw `
  --language Chinese
```

### License and citation

These weights are derived from `Qwen/Qwen3-ASR-0.6B-hf` and are published under Apache-2.0. Retain the upstream attribution and license. ExLlamaV3 and runtime dependencies have their own licenses. Please cite Qwen3-ASR, ExLlamaV3, and the runtime project.

## 中文

本仓库提供基于 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 转换的 EXL3 量化文本解码器权重。使用时需配套兼容的 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)集成了 encoder 和 decoder，可用于完整转写。本权重采用 ExLlamaV3 格式，由项目运行时加载。

### 量化版本

| 版本 | 用途 |
| --- | --- |
| `decoder-exl3-3bpw` | 显存占用较低 |
| `decoder-exl3-4bpw` | 精度较高的替代版本 |

每个版本包含 EXL3 `model.safetensors`、tokenizer、模型与生成配置以及量化配置。

### 运行要求

- NVIDIA GPU 与 CUDA
- ExLlamaV3 1.6.6，以及相互兼容的 PyTorch 和 CUDA 版本
- 参考环境：PyTorch 2.10.0、CUDA 12.8、Python 3.10–3.13
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，其中已集成 encoder 和 decoder

项目已在 Windows 上测试并可用。Linux 环境在 CUDA、TensorRT、PyTorch 和 ExLlamaV3 软件包版本兼容时也可运行。

### 使用

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-3bpw `
  --language Chinese
```

### 许可证与引用

这些权重基于 `Qwen/Qwen3-ASR-0.6B-hf`，以 Apache-2.0 发布。请保留上游归属说明和许可证。ExLlamaV3 与运行时依赖各自遵循其许可证。请引用 Qwen3-ASR、ExLlamaV3 及运行时项目。
