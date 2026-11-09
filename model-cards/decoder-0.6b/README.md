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
| `decoder-exl3-3bpw` | Legacy 3 bpw export, ExLlamaV3 default calibration |
| `decoder-exl3-4bpw` | Legacy 4 bpw export, ExLlamaV3 default calibration |
| [`decoder-exl3-3bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) | Separately calibrated 3 bpw export |
| [`decoder-exl3-4bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) | Separately calibrated 4 bpw export |

Each variant includes EXL3 `model.safetensors`, tokenizer files, model and generation configuration, and quantization configuration.

### Calibration

The existing `decoder-exl3-3bpw` and `decoder-exl3-4bpw` files are legacy exports. They were quantized with ExLlamaV3's default calibration data: 250 rows of 2,048 tokens.

The separately published `decoder-exl3-3bpw-train-clean-100` and `decoder-exl3-4bpw-train-clean-100` directories contain the calibrated exports. Their calibration tensors contain 200 packed rows of 2,048 tokens, built from 1,603 paired audio requests and transcripts in LibriSpeech `train-clean-100`. No `test-clean` samples were used. Calibration changes quantization statistics; it is not fine-tuning.

On all 2,620 LibriSpeech `test-clean` utterances, the calibrated 3 bpw variant measured 2.5715% WER and 114.37 ms median latency, compared with 2.9029% and 115.17 ms for the legacy 3 bpw file. The calibrated 4 bpw variant measured 2.3776% WER and 115.05 ms median, compared with 2.4228% and 113.99 ms for the legacy 4 bpw file. These are two-pass 2026-11-10 rerun medians; timings are not repeated-run averages.

The `train-clean-100` variants are tuned for LibriSpeech-style English read speech. For general-purpose audio and Chinese ASR, the legacy decoders remain the default recommendation.

### Comparison with GGUF Q4_K

![Corpus WER and median latency](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/resolve/main/qwen3-0.6b-calibration-vs-gguf.png)

![Median latency by stage](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/resolve/main/qwen3-0.6b-stage-latency.png)

All configurations use all 2,620 test-clean utterances. The EXL3 decoders are about 2.6× faster than GGUF Q4_K. The official `qwen-asr` Transformers backend reaches 2.3212% WER at 787.12 ms, about 6.8× slower than the best EXL3 variant. The chart labels give WER, median latency, and the speed ratio.

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
  --decoder-dir .\decoder-exl3-4bpw `
  --language Chinese
```

### License and citation

These weights are derived from `Qwen/Qwen3-ASR-0.6B-hf` and are published under Apache-2.0. Retain the upstream attribution and license. ExLlamaV3 and runtime dependencies have their own licenses. Please cite Qwen3-ASR, ExLlamaV3, and the runtime project.

## 中文

本仓库提供基于 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 转换的 EXL3 量化文本解码器权重。使用时需配套兼容的 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)集成了 encoder 和 decoder，可用于完整转写。本权重采用 ExLlamaV3 格式，由项目运行时加载。

### 量化版本

| 版本 | 用途 |
| --- | --- |
| `decoder-exl3-3bpw` | 旧版 3 bpw 权重，ExLlamaV3 默认校准 |
| `decoder-exl3-4bpw` | 旧版 4 bpw 权重，ExLlamaV3 默认校准 |
| [`decoder-exl3-3bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) | 单独校准的新版 3 bpw 权重 |
| [`decoder-exl3-4bpw-train-clean-100`](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) | 单独校准的新版 4 bpw 权重 |

每个版本包含 EXL3 `model.safetensors`、tokenizer、模型与生成配置以及量化配置。

### 校准数据

现有的 `decoder-exl3-3bpw` 和 `decoder-exl3-4bpw` 是旧版权重，使用 ExLlamaV3 默认校准数据量化：250 行、每行 2,048 个 token。

独立目录 `decoder-exl3-3bpw-train-clean-100` 和 `decoder-exl3-4bpw-train-clean-100` 中保存新版权重。校准张量由 LibriSpeech `train-clean-100` 的 1,603 组音频请求和转写文本打包为 200 行，每行 2,048 个 token。校准未使用 `test-clean` 样本。校准会影响量化统计，不会微调模型。

在全部 2,620 条 LibriSpeech `test-clean` 音频上，新版 3 bpw 的 WER 为 2.5715%，中位延迟为 114.37 ms；旧版 3 bpw 分别为 2.9029% 和 115.17 ms。新版 4 bpw 的 WER 为 2.3776%，中位延迟为 115.05 ms；旧版 4 bpw 分别为 2.4228% 和 113.99 ms。以上为 2026-11-10 两轮复测的中位数，延迟不是多轮平均值。

`train-clean-100` 版本针对 LibriSpeech 风格的英语朗读调优。通用音频和中文 ASR 仍默认推荐 legacy 版本。

### 与 GGUF Q4_K 对比

![速度与 WER](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/resolve/main/qwen3-0.6b-calibration-vs-gguf.png)

![分阶段延迟](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/resolve/main/qwen3-0.6b-stage-latency.png)

全部配置均使用 2,620 条 test-clean。EXL3 约比 GGUF Q4_K 快 2.6 倍。官方 `qwen-asr` Transformers 后端的 WER 为 2.3212%，中位延迟为 787.12 ms，比最佳 EXL3 版本慢约 6.8 倍。图中标注了 WER、中位延迟和速度倍数。

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
  --decoder-dir .\decoder-exl3-4bpw `
  --language Chinese
```

### 许可证与引用

这些权重基于 `Qwen/Qwen3-ASR-0.6B-hf`，以 Apache-2.0 发布。请保留上游归属说明和许可证。ExLlamaV3 与运行时依赖各自遵循其许可证。请引用 Qwen3-ASR、ExLlamaV3 及运行时项目。
