# Qwen3-ASR TensorRT

Windows-tested inference for Qwen3-ASR 0.6B and 1.7B using a TensorRT audio encoder and an ExLlamaV3 EXL3 text decoder. This repository integrates both stages and provides a Python CLI and HTTP API.

[English](README.md) | [简体中文](README.zh-CN.md) | [Model files](model-cards/)

## Requirements

- Windows 10/11, NVIDIA GPU, and CUDA 12.8
- Python 3.10–3.13 and `uv`
- TensorRT 11.4.0.106
- A compatible PyTorch and ExLlamaV3 build

Linux should work when compatible CUDA, TensorRT, PyTorch, and ExLlamaV3 packages are available.

Install the project dependencies:

```powershell
uv sync
```

## Models

The project supports [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) and [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), with BF16 and FP16 TensorRT encoder builds, calibrated INT8 Q/DQ encoder builds, and EXL3 3/4 bpw decoder weights.

Published model files:

- [TensorRT encoder: BF16, FP16, and INT8 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)
- [EXL3 decoder: 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder)
- [0.6B TensorRT encoder: BF16, FP16, and INT8 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)
- [0.6B EXL3 decoder: 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)

## Build the accelerated backend

Build the BF16 encoder engine and extract the text decoder:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build
```

Build an FP16 encoder for GPUs without BF16 support:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision fp16
```

Build a calibrated INT8 encoder:

```powershell
uv sync --extra quantization
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int8
```

Create an EXL3 decoder from the base checkpoint when needed:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 3 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-3bpw
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 4 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-4bpw
```

Use `--model Qwen/Qwen3-ASR-0.6B-hf` and `artifacts/qwen3-asr-0.6b/` for the 0.6B model.

TensorRT engines are generated locally for the target GPU and TensorRT environment.

## Transcribe audio

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

Start the HTTP API:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf serve --backend tensorrt-exllama
```

Clients can submit audio bytes from any programming language:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=Chinese" `
  -H "Content-Type: audio/wav" --data-binary "@audio.wav"
```

The API binds to loopback by default and accepts WAV, MP3, FLAC, OGG, M4A, and AAC uploads.
