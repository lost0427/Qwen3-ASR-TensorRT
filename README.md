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

The encoder build supports BF16 and FP16 exports. To create a calibrated INT8 Q/DQ ONNX, install the optional quantization dependencies and provide a directory of representative audio files:

```powershell
uv sync --extra quantization
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision fp16
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int8
```

Calibration uses the selected model's audio processor and fixed encoder windows. It applies symmetric INT8 Q/DQ to Conv and MatMul operations, with other operations retaining FP16. The command writes the ONNX and external weights under the model artifact directory. A 32-file LibriSpeech calibration run produced 52 encoder windows. EXL3 provides 3/4 bpw and recipe-based mixed precision for the decoder.

Run the HTTP API with the accelerated backend:

```powershell
uv run qwen3-asr serve --backend tensorrt-exllama
```

The build uses the selected Hugging Face checkpoint. It writes ONNX, TensorRT engine, and extracted decoder files under `artifacts/qwen3-asr-0.6b` by default. Engine files are hardware and TensorRT-version specific and should be built on the target machine.

Pass `--model Qwen/Qwen3-ASR-1.7B-hf` before the subcommand to build the Transformers-compatible 1.7B checkpoint; its artifacts are kept separately under `artifacts/qwen3-asr-1.7b`:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build
```

The same model selection controls the default model-specific paths for `transcribe`, `benchmark`, and `quantize`.

## Benchmark

Run the same audio repeatedly after model load and warmup:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf benchmark "path/to/sample.wav" --backend tensorrt-exllama --decoder-dir artifacts/qwen3-asr-1.7b/decoder-exl3-3bpw --warmups 2 --repeats 10
```

### RTX 4060 Laptop comparison

Both backends used the same Qwen3-ASR 1.7B checkpoint and the same 23.86-second LibriSpeech test-clean sample. TensorRT results use the EXL3 3 bpw decoder; the INT8 path uses the calibrated encoder described above. Times are medians over 10 repetitions after 2 warmups. GGUF used DML+Vulkan. This is a laptop GPU result, not a desktop RTX 4060 result.

| Backend | Preprocess | Encoder | Decoder prefill | Decoder generation | End to end |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-ASR-GGUF | 10.5 ms other | 167.4 ms | 76.3 ms | 519.4 ms | 773.8 ms |
| TensorRT BF16 + EXL3 3 bpw | 8.1 ms | 19.7 ms | 54.1 ms | 468.8 ms | 552.4 ms |
| TensorRT INT8 + EXL3 3 bpw | 7.8 ms | 12.3 ms | 54.5 ms | 464.8 ms | 541.9 ms |

On this sample, BF16 and INT8 TensorRT paths were 28.6% and 30.0% faster than GGUF, respectively. A paired 100-utterance LibriSpeech test-clean evaluation measured 2.49% WER (45/1806 word errors) for BF16 encoder + EXL3 3 bpw and 2.33% (42/1806) for INT8 encoder + the same decoder. INT8 won 8 utterances, tied 84, and lost 8 against BF16. The measured speed advantage applies to this laptop, model, decoder bitrate, and audio set; desktop performance remains unmeasured. Single-file packaging is deferred.

The API accepts WAV, MP3, FLAC, OGG, M4A, and AAC content types, limits uploads to 128 MiB, and binds to loopback by default. Use `--host 0.0.0.0` only when remote clients should connect.

## Current scope

The reference runtime defaults to `Qwen/Qwen3-ASR-0.6B-hf` in BF16 on `cuda:0`. The accelerated runtime supports the 1.7B checkpoint with TensorRT BF16/FP16 or a calibrated INT8 encoder engine; ExLlamaV3 runs the extracted Qwen3 text decoder using its FP16/EXL3 paths. Pass `--model Qwen/Qwen3-ASR-1.7B-hf` when comparing against the 1.7B GGUF model. Windows is the first target; the same Python API and CLI are intended for Linux where matching CUDA, TensorRT, and ExLlamaV3 packages are installed.

The initial smoke check ran on an NVIDIA GeForce RTX 4060 Laptop GPU with CUDA 12.8 and confirmed BF16 model loading and generation. It used one second of silence, so it validates the runtime path rather than recognition quality.

Run the HTTP contract tests with:

```powershell
uv run python -m unittest discover -s tests -v
```
