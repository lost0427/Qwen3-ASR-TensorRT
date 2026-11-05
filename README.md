# Qwen3-ASR TensorRT

Windows-first Qwen3-ASR inference for NVIDIA GPUs. The first implementation milestone provides a BF16 reference runtime and a language-neutral HTTP interface. TensorRT encoder and ExLlamaV3 decoder backends are not connected yet.

## Setup

Python 3.10-3.13 and `uv` are required. The project uses a dedicated CUDA 12.8 PyTorch wheel set so it does not modify the global Python environment.

```powershell
uv sync
```

## BF16 reference

Transcribe a local audio file:

```powershell
uv run qwen3-asr transcribe .\audio.wav --language Chinese
```

Start the local HTTP API:

```powershell
uv run qwen3-asr serve
```

Any language with an HTTP client can submit raw audio bytes:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=Chinese" `
  -H "Content-Type: audio/wav" --data-binary "@audio.wav"
```

The API accepts WAV, MP3, FLAC, OGG, M4A, and AAC content types, limits uploads to 128 MiB, and binds to loopback by default. Use `--host 0.0.0.0` only when remote clients should connect.

## Current scope

The reference runtime loads `Qwen/Qwen3-ASR-0.6B-hf` in BF16 on `cuda:0`. It establishes the transcription and API contract for later accelerated backends. TensorRT engine export/build, ExLlamaV3 integration, Linux packaging, and single-file packaging remain subsequent implementation work.

The initial smoke check ran on an NVIDIA GeForce RTX 4060 Laptop GPU with CUDA 12.8 and confirmed BF16 model loading and generation. It used one second of silence, so it validates the runtime path rather than recognition quality.

Run the HTTP contract tests with:

```powershell
uv run python -m unittest discover -s tests -v
```
