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

The project supports [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) and [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), with BF16 and FP16 TensorRT encoder builds, optional local INT8 Q/DQ encoder calibration, and EXL3 3/4 bpw decoder weights.

Published model files:

- [TensorRT encoder: BF16 and FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)
- [EXL3 decoder: legacy 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder)
- [1.7B EXL3 3 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) (separate directory in the same decoder repository)
- [1.7B EXL3 4 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) (separate directory in the same decoder repository)
- [0.6B TensorRT encoder: BF16 and FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)
- [0.6B EXL3 decoder: legacy 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)
- [0.6B EXL3 3 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) (separate directory in the same decoder repository)
- [0.6B EXL3 4 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) (separate directory in the same decoder repository)
- [Qwen3-ForcedAligner 0.6B TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder)
- [Qwen3-ForcedAligner 0.6B EXL3 4 bpw decoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder)

The legacy 3 bpw and 4 bpw decoders were quantized with ExLlamaV3's default calibration data. The `train-clean-100` variants were calibrated on LibriSpeech `train-clean-100` and are tuned for LibriSpeech-style English read speech; for general-purpose audio and Chinese ASR the legacy decoders remain the default recommendation.

## Build the accelerated backend

Build the BF16 encoder engine and extract the text decoder:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build
```

Build an FP16 encoder for GPUs without BF16 support:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision fp16
```

Build an experimental local INT8 encoder (not included in the published model files):

```powershell
uv sync --extra quantization
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int8
```

An experimental TensorRT INT4 weight-only encoder export quantizes eligible MatMul weights in 64-value blocks. Convolutions and activations remain FP16; build its engine with the `int4` precision after exporting the ONNX:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder-int4
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int4
```

Create the 200-row calibration tensor from a local LibriSpeech `train-clean-100` corpus and Qwen3-ASR checkpoint, then quantize with that explicit training data. Omitting `--cal-data` uses ExLlamaV3's default calibration data, which is how the legacy decoders were produced:

```powershell
uv run python tools\prepare_exl3_calibration.py .\LibriSpeech\train-clean-100 .\models\Qwen3-ASR-1.7B-hf .\artifacts\train-clean-100.safetensors --rows 200 --columns 2048
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 3 --cal-data .\artifacts\train-clean-100.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-3bpw-train-clean-100
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 4 --cal-data .\artifacts\train-clean-100.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-4bpw-train-clean-100
```

For the separately published 0.6B variants, generate calibration data with the 0.6B checkpoint and use distinct output directories:

```powershell
uv run python tools\prepare_exl3_calibration.py .\LibriSpeech\train-clean-100 .\models\Qwen3-ASR-0.6B-hf .\artifacts\train-clean-100-0.6b.safetensors --rows 200 --columns 2048
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize --bits 3 --cal-data .\artifacts\train-clean-100-0.6b.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-3bpw-train-clean-100
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize --bits 4 --cal-data .\artifacts\train-clean-100-0.6b.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-4bpw-train-clean-100
```

TensorRT engines are generated locally for the target GPU and TensorRT environment.

Build the forced aligner encoder and export both decoder variants:

```powershell
# TensorRT encoder + TensorRT bf16/fp16 decoders
uv run qwen3-asr build-aligner

# optional EXL3 4 bpw decoder
uv run qwen3-asr --model Qwen/Qwen3-ForcedAligner-0.6B quantize --bits 4 --head-bits 16
```

## Transcribe audio

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

## Align audio

```powershell
uv run qwen3-asr align .\audio.wav --text-file .\transcript.txt --language English --format srt --output .\audio.srt
```

`align` uses the TensorRT encoder and a TensorRT fp16 aligner decoder by default.
It supports `--format json`, `--format srt`, and `--format txt`; `--text` can be
used instead of `--text-file`. `--decoder-precision bf16` selects the bf16 engine,
`--backend tensorrt-exllama` selects the EXL3 decoder, and `--engine` /
`--decoder-engine` point at specific builds.



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
