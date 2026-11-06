# Qwen3-ASR TensorRT

Windows-first Qwen3-ASR inference for NVIDIA GPUs. The project keeps a BF16 Transformers reference path and adds a TensorRT BF16 audio encoder with an ExLlamaV3 Qwen3 text decoder. The HTTP interface is language-neutral.

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

## TensorRT + ExLlamaV3

Build the BF16 TensorRT encoder engine and extract the text decoder weights into ExLlamaV3's Qwen3 layout:

```powershell
uv run qwen3-asr build
```

Run accelerated inference:

```powershell
uv run qwen3-asr transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

Create an EXL3 decoder at a target bitrate. A recipe file can assign different bitrates to different tensors for mixed precision:

```powershell
uv run qwen3-asr quantize --bits 4
uv run qwen3-asr quantize --bits 8 --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-8bpw
uv run qwen3-asr quantize --bits 4 --recipe .\recipes\mixed.yml --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-mixed
```

The encoder build currently provides BF16 and FP16 TensorRT engines. EXL3 provides 4/8 bpw and recipe-based mixed precision for the decoder; an INT8 TensorRT calibration path still needs a representative calibration set and is kept separate from EXL3 quantization.

Run the HTTP API with the accelerated backend:

```powershell
uv run qwen3-asr serve --backend tensorrt-exllama
```

The build uses the selected Hugging Face checkpoint. It writes ONNX, TensorRT engine, and extracted decoder files under `artifacts/qwen3-asr-0.6b` by default. Engine files are hardware and TensorRT-version specific and should be built on the target machine.

## Benchmark

Run the same audio repeatedly after model load and warmup:

```powershell
uv run qwen3-asr benchmark "path/to/sample.wav" --backend tensorrt-exllama --decoder-dir artifacts/qwen3-asr-0.6b/decoder-exl3-4bpw --warmups 2 --repeats 10
```

On the RTX 4060 desktop, both projects were measured on the same 23.86-second ASRbenchmark sample with 2 warmups and 10 repeats. The current run measured this project at 387.2 ms median (BF16 TensorRT encoder + EXL3 4 bpw decoder) and Qwen3-ASR-GGUF at 810.5 ms (DML + Vulkan, Q4_K decoder).

| Stage | This project | Qwen3-ASR-GGUF | Faster |
| --- | ---: | ---: | ---: |
| Encoder | 15.2 ms | 177.2 ms | 11.7x |
| Decoder prefill | included below | 78.3 ms | |
| Decoder token generation | included below | 541.5 ms | |
| Decoder total | 360.7 ms | 619.8 ms | 1.72x |
| Other / preprocessing | 11.4 ms | 10.9 ms | |
| End-to-end median | 387.2 ms | 810.5 ms | 2.09x |

The project decoder timing includes ExLlamaV3 prompt processing and generation. GGUF decoder timing is its measured LLM prefill plus token generation. This compares different model configurations: Qwen3-ASR 0.6B with an EXL3 4 bpw decoder versus Qwen3-ASR 1.7B with a Q4_K decoder; it is not a same-size model comparison. Measurements are hardware-specific. INT8 TensorRT export still requires a representative calibration set and explicit ONNX Q/DQ quantization for TensorRT 11.4. Single-file packaging is deferred.

The API accepts WAV, MP3, FLAC, OGG, M4A, and AAC content types, limits uploads to 128 MiB, and binds to loopback by default. Use `--host 0.0.0.0` only when remote clients should connect.

## Current scope

The reference runtime loads `Qwen/Qwen3-ASR-0.6B-hf` in BF16 on `cuda:0`. The accelerated runtime uses a BF16 TensorRT encoder engine; ExLlamaV3 runs the extracted Qwen3 text decoder using its supported FP16/EXL3 paths. The BF16 reference remains available for output comparison. Windows is the first target; the same Python API and CLI are intended for Linux where matching CUDA, TensorRT, and ExLlamaV3 packages are installed.

The initial smoke check ran on an NVIDIA GeForce RTX 4060 Laptop GPU with CUDA 12.8 and confirmed BF16 model loading and generation. It used one second of silence, so it validates the runtime path rather than recognition quality.

Run the HTTP contract tests with:

```powershell
uv run python -m unittest discover -s tests -v
```
