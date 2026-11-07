---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-0.6B-hf
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- tensorrt
- onnx
- bf16
- fp16
- int8
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 0.6B TensorRT Encoder

## English

ONNX exports of the audio encoder from [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf), prepared for TensorRT inference. Use an encoder export with a compatible Qwen3-ASR text decoder, such as the [EXL3 decoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder). The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) integrates the encoder and decoder for transcription.

### Files

| Precision | Files | Use |
| --- | --- | --- |
| BF16 | `encoder-window.bf16.onnx`, `encoder-window.bf16.onnx.data` | Reference export for GPUs with BF16 support |
| FP16 | `encoder-window.fp16.onnx`, `encoder-window.fp16.onnx.data` | Compatibility export for older NVIDIA GPUs |
| INT8 Q/DQ | `encoder-window.int8.onnx`, `encoder-window.int8.onnx.data` | Calibrated quantized export |

TensorRT `.engine` files are built locally for the target GPU and TensorRT environment.

### Precision and calibration

BF16 is the reference precision. FP16 supports NVIDIA GPUs without BF16. INT8 applies Q/DQ quantization to Conv and MatMul operations; remaining operations use floating-point precision. The INT8 artifact currently published in this model card was calibrated with 32 LibriSpeech test-clean audio files (52 encoder windows), which overlaps the benchmark set. A separate recalibration experiment now uses 32 speakers from train-clean-100 (59 windows); those replacement weights are not part of this model card revision. The 0.6B INT8 configuration showed a substantial WER regression and is not recommended.

### Requirements

- NVIDIA GPU and CUDA
- Reference environment: CUDA 12.8, TensorRT 11.4.0.106, Python 3.10–3.13
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which includes the encoder/decoder integration

Windows is tested and supported. Linux should work with compatible CUDA, TensorRT, PyTorch, and ExLlamaV3 packages.

### Build

```powershell
uv sync --extra quantization

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision bf16
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision fp16

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision int8
```

### License and citation

This export is derived from `Qwen/Qwen3-ASR-0.6B-hf` and is published under Apache-2.0. Retain the upstream attribution and license. Please also cite the Qwen3-ASR work and the runtime project.

## 中文

本仓库提供 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 音频编码器的 ONNX 导出，可用于 TensorRT 推理。使用时需配套兼容的 Qwen3-ASR 文本解码器，例如 [EXL3 解码器](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)已集成 encoder 和 decoder，可用于完整转写。

### 文件

| 精度 | 文件 | 用途 |
| --- | --- | --- |
| BF16 | `encoder-window.bf16.onnx`、`encoder-window.bf16.onnx.data` | 支持 BF16 的 GPU 基准导出 |
| FP16 | `encoder-window.fp16.onnx`、`encoder-window.fp16.onnx.data` | 兼容不支持 BF16 的较旧 NVIDIA GPU |
| INT8 Q/DQ | `encoder-window.int8.onnx`、`encoder-window.int8.onnx.data` | 经校准的量化导出 |

TensorRT `.engine` 文件需根据目标 GPU 和 TensorRT 环境在本地构建。

### 精度与校准

BF16 是基准精度，FP16 用于不支持 BF16 的 NVIDIA GPU。INT8 对 Conv 和 MatMul 运算应用 Q/DQ 量化，其余运算保留浮点精度。本模型卡当前发布的 INT8 权重使用 LibriSpeech test-clean 的 32 个音频文件校准，共生成 52 个 encoder window；这与基准评测集有重叠。新的独立校准实验改用 train-clean-100 中 32 位说话人的音频，共 59 个 window；重校准权重尚未包含在本次模型卡修订中。0.6B INT8 配置的 WER 明显退化，不建议使用。

### 运行要求

- NVIDIA GPU 与 CUDA
- 参考环境：CUDA 12.8、TensorRT 11.4.0.106、Python 3.10–3.13
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，其中已集成 encoder 和 decoder

项目已在 Windows 上测试并可用。Linux 环境在 CUDA、TensorRT、PyTorch 和 ExLlamaV3 软件包兼容时也可运行。

### 构建

```powershell
uv sync --extra quantization

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision bf16
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision fp16

uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf build --encoder-precision int8
```

### 许可证与引用

本导出基于 `Qwen/Qwen3-ASR-0.6B-hf`，以 Apache-2.0 发布。请保留上游归属说明和许可证，并引用 Qwen3-ASR 工作及本运行时项目。
