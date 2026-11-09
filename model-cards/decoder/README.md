---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B-hf
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
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 1.7B EXL3 Decoder

## English

EXL3 quantized text-decoder weights derived from [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf). Use with a compatible [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder). The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) integrates the encoder and decoder for transcription. These weights use the ExLlamaV3 format and are loaded by the project runtime.

### Variants

| Variant | Approximate size | Use |
| --- | ---: | --- |
| `decoder-exl3-3bpw` | 1.30 GiB | Legacy 3 bpw export |
| `decoder-exl3-4bpw` | 1.46 GiB | Legacy 4 bpw export |
| [`decoder-exl3-3bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) | 1.30 GiB | Separately calibrated 3 bpw export |
| [`decoder-exl3-4bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) | 1.46 GiB | Separately calibrated 4 bpw export |

Each variant includes EXL3 `model.safetensors`, tokenizer files, model and generation configuration, and quantization configuration. EXL3 uses quantized weight tensors alongside floating-point scale, normalization, and auxiliary tensors; the variants are identified by their EXL3 bitrate.

### Calibration

The existing 3 bpw and 4 bpw files are legacy exports. Their quantization metadata records 250 rows of 2,048 tokens but does not identify the calibration corpus.

The `decoder-exl3-3bpw-train-clean-100` and `decoder-exl3-4bpw-train-clean-100` directories contain the calibrated exports. Their calibration tensors contain 200 packed rows of 2,048 tokens, built from 1,603 paired audio requests and transcripts in LibriSpeech `train-clean-100`. No `test-clean` samples were used. Calibration changes quantization statistics; it is not fine-tuning.

On all 2,620 LibriSpeech `test-clean` utterances, the calibrated 4 bpw variant measured 1.9484% WER and 196.26 ms median latency, compared with 1.9842% and 191.05 ms for the legacy 4 bpw file. The calibrated 3 bpw variant measured 2.1329% WER and 194.25 ms median, compared with 2.0237% and 189.19 ms for the legacy 3 bpw file; calibration did not improve the 3 bpw tradeoff. Each variant was run once; timings are not repeated-run averages.

### Requirements

- NVIDIA GPU and CUDA
- ExLlamaV3 1.6.6 with compatible PyTorch and CUDA versions
- Reference environment: PyTorch 2.10.0, CUDA 12.8, Python 3.10–3.13
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which includes the encoder/decoder integration

Windows is tested and supported. Linux should work with compatible CUDA, TensorRT, PyTorch, and ExLlamaV3 packages.

### Usage

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100 `
  --language Chinese
```

Start the HTTP API with the same backend:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf serve `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100
```

### License and citation

These weights are derived from `Qwen/Qwen3-ASR-1.7B-hf` and are published under Apache-2.0. Retain the upstream attribution and license. ExLlamaV3 and runtime dependencies have their own licenses. Please cite Qwen3-ASR, ExLlamaV3, and the runtime project.

## 中文

本仓库提供基于 [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 转换的 EXL3 量化文本解码器权重。使用时需配套兼容的 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)已集成 encoder 和 decoder，可用于完整转写。本权重采用 ExLlamaV3 格式，由项目运行时加载。

### 量化版本

| 版本 | 大小约 | 用途 |
| --- | ---: | --- |
| `decoder-exl3-3bpw` | 1.30 GiB | 旧版 3 bpw 权重 |
| `decoder-exl3-4bpw` | 1.46 GiB | 旧版 4 bpw 权重 |
| [`decoder-exl3-3bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) | 1.30 GiB | 单独校准的新版 3 bpw 权重 |
| [`decoder-exl3-4bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) | 1.46 GiB | 单独校准的新版 4 bpw 权重 |

每个版本包含 EXL3 `model.safetensors`、tokenizer、模型与生成配置、量化配置。EXL3 权重由量化张量以及浮点缩放、归一化和辅助张量组成，版本以 EXL3 bpw 标识。

### 校准数据

现有的 3 bpw 和 4 bpw 权重是旧版导出。量化配置记录了 250 行、每行 2,048 个 token，但未注明校准语料来源。

`decoder-exl3-3bpw-train-clean-100` 和 `decoder-exl3-4bpw-train-clean-100` 目录保存新版权重。校准张量由 LibriSpeech `train-clean-100` 的 1,603 组音频请求和转写文本打包为 200 行，每行 2,048 个 token。校准未使用 `test-clean` 样本。校准会影响量化统计，不会微调模型。

在全部 2,620 条 LibriSpeech `test-clean` 音频上，新版 4 bpw 的 WER 为 1.9484%，中位延迟为 196.26 ms；旧版 4 bpw 分别为 1.9842% 和 191.05 ms。新版 3 bpw 的 WER 为 2.1329%，中位延迟为 194.25 ms；旧版 3 bpw 分别为 2.0237% 和 189.19 ms，校准没有改善 3 bpw 的取舍。每个版本只运行了一次，延迟不是多轮平均值。

### 运行要求

- NVIDIA GPU 与 CUDA
- ExLlamaV3 1.6.6，以及相互兼容的 PyTorch 和 CUDA 版本
- 参考环境：PyTorch 2.10.0、CUDA 12.8、Python 3.10–3.13
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，其中已集成 encoder 和 decoder

项目已在 Windows 上测试并可用。Linux 环境在 CUDA、TensorRT、PyTorch 和 ExLlamaV3 软件包版本兼容时也可运行。

### 使用

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100 `
  --language Chinese
```

使用相同 backend 启动 HTTP API：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf serve `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100
```

### 许可证与引用

这些权重基于 `Qwen/Qwen3-ASR-1.7B-hf`，以 Apache-2.0 发布。请保留上游归属说明和许可证。ExLlamaV3 与运行时依赖各自遵循其许可证。请引用 Qwen3-ASR、ExLlamaV3 及运行时项目。
