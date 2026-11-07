# Qwen3-ASR TensorRT

面向 Windows、基于 TensorRT 音频编码器和 ExLlamaV3 EXL3 文本解码器的 Qwen3-ASR 0.6B 与 1.7B 推理项目。本仓库集成 encoder 和 decoder，并提供 Python 命令行与 HTTP API。

[English](README.md) | [简体中文](README.zh-CN.md) | [模型文件说明](model-cards/)

## 运行环境

- Windows 10/11、NVIDIA GPU、CUDA 12.8
- Python 3.10–3.13 和 `uv`
- TensorRT 11.4.0.106
- 兼容的 PyTorch 和 ExLlamaV3 版本

Linux 环境在 CUDA、TensorRT、PyTorch 和 ExLlamaV3 软件包兼容时也可运行。

安装项目依赖：

```powershell
uv sync
```

## 模型

项目支持 [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 和 [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)，提供 BF16、FP16 TensorRT encoder、校准后的 INT8 Q/DQ encoder，以及 EXL3 3/4 bpw decoder。

已发布的模型文件：

- [TensorRT encoder：BF16、FP16 和 INT8 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)
- [EXL3 decoder：3 bpw 和 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder)
- [0.6B TensorRT encoder：BF16、FP16 和 INT8 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)
- [0.6B EXL3 decoder：3 bpw 和 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)

## 构建加速后端

构建 BF16 encoder engine 并提取文本 decoder：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build
```

为不支持 BF16 的 GPU 构建 FP16 encoder：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision fp16
```

构建校准后的 INT8 encoder：

```powershell
uv sync --extra quantization
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int8
```

实验性的 TensorRT INT4 权重专用 encoder 导出会按 64 个值一组量化符合条件的 MatMul 权重；卷积和激活仍使用 FP16。先导出 ONNX，再构建 engine：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder-int4
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int4
```

需要时，可从基础模型生成 EXL3 decoder：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 3 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-3bpw
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 4 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-4bpw
```

使用 0.6B 模型时，将命令中的模型 ID 替换为 `Qwen/Qwen3-ASR-0.6B-hf`，产物目录为 `artifacts/qwen3-asr-0.6b/`。

TensorRT engine 会根据目标 GPU 和本机 TensorRT 环境在本地构建。

## 转写音频

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

启动 HTTP API：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf serve --backend tensorrt-exllama
```

任何编程语言都可以通过 HTTP 提交音频数据：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=Chinese" `
  -H "Content-Type: audio/wav" --data-binary "@audio.wav"
```

API 默认只监听本机地址，支持 WAV、MP3、FLAC、OGG、M4A 和 AAC 音频。
