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
- 4bpw
- train-clean-100
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100"><img src="https://img.shields.io/badge/Hugging_Face-new%204%20bpw-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 0.6B EXL3 4 bpw Decoder (train-clean-100)

## English

This is a separately calibrated 4 bpw decoder variant derived from [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf). It is published in a new directory in the existing [0.6B decoder repository](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder). The legacy `decoder-exl3-3bpw` and `decoder-exl3-4bpw` files remain unchanged. The legacy decoders use ExLlamaV3 default calibration data.

### Calibration

The EXL3 calibration tensor was generated from paired audio requests and transcripts in LibriSpeech `train-clean-100`: 1,603 pairs packed into 200 rows of 2,048 tokens. No `test-clean` samples were used. Calibration affects quantization statistics; it is not fine-tuning.

### Benchmark

On all 2,620 LibriSpeech `test-clean` utterances, this variant measured 2.3776% corpus WER, 115.05 ms median latency, and 253.18 ms P90 latency. The legacy 4 bpw variant measured 2.4228% WER, 113.99 ms median, and 249.98 ms P90 in the 2026-11-10 two-pass rerun. Each pass ran once; timings are not repeated-run averages.

The `train-clean-100` variants are tuned for LibriSpeech-style English read speech. For general-purpose audio and Chinese ASR, the legacy decoders remain the default recommendation.

### Requirements and usage

- NVIDIA GPU and CUDA
- ExLlamaV3 with compatible PyTorch and CUDA versions
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which integrates the TensorRT encoder and EXL3 decoder

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100 `
  --language Chinese
```

These weights are derived from `Qwen/Qwen3-ASR-0.6B-hf` and are published under Apache-2.0. Retain the upstream attribution and license. ExLlamaV3 and runtime dependencies have their own licenses.

## 中文

这是基于 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 重新校准的 4 bpw 文本解码器权重，单独保存在原有的 [0.6B decoder 仓库](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)的新目录中。原来的 `decoder-exl3-3bpw` 和 `decoder-exl3-4bpw` 文件保持不变。legacy 版本使用 ExLlamaV3 默认校准数据。

### 校准数据

EXL3 校准张量来自 LibriSpeech `train-clean-100` 的音频请求与转写文本，共 1,603 组，打包为 200 行、每行 2,048 个 token。未使用 `test-clean` 样本。校准会影响量化统计，不会微调模型。

### 评测结果

在全部 2,620 条 LibriSpeech `test-clean` 音频上，本版本的 corpus WER 为 2.3776%，中位延迟为 115.05 ms，P90 为 253.18 ms。2026-11-10 两轮复测中，旧版 4 bpw 的 WER 为 2.4228%，中位延迟为 113.99 ms，P90 为 249.98 ms。每轮各运行一次；延迟不是多轮平均值。

`train-clean-100` 版本针对 LibriSpeech 风格的英语朗读调优。通用音频和中文 ASR 仍默认推荐 legacy 版本。

### 运行要求与使用

- NVIDIA GPU 与 CUDA
- 兼容的 ExLlamaV3、PyTorch 与 CUDA 版本
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，其中集成了 TensorRT encoder 和 EXL3 decoder

```powershell
uv sync

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf transcribe .\audio.wav `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw-train-clean-100 `
  --language Chinese
```

这些权重基于 `Qwen/Qwen3-ASR-0.6B-hf`，以 Apache-2.0 发布。请保留上游归属和许可证说明。ExLlamaV3 与运行时依赖各自遵循其许可证。
