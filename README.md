# Qwen3-ASR TensorRT

Windows-tested inference for Qwen3-ASR 0.6B and 1.7B using a TensorRT audio encoder and one of two text decoders: an ExLlamaV3 EXL3 decoder, or a fully TensorRT decoder with INT4 weight-only (W4A16) kernels. This repository integrates both stages and provides a Python CLI and HTTP API.

[English](README.md) | [简体中文](README.zh-CN.md) | [Model files](model-cards/)

## Requirements

- Windows 10/11, NVIDIA GPU, and CUDA 12.8
- Python 3.10–3.13 and `uv`
- TensorRT 11.4.0.106
- A compatible PyTorch build; ExLlamaV3 comes with the `exllama` extra

Linux should work when compatible CUDA, TensorRT, and PyTorch packages are available.

Install for the TensorRT backends:

```powershell
uv sync
```

Add the EXL3 backend or the encoder quantization path as extras:

```powershell
uv sync --extra exllama          # --backend tensorrt-exllama
uv sync --extra quantization     # quantize-encoder (INT8 Q/DQ)
```

The default install carries what inference needs and nothing else: `librosa` and its `numba`/`scipy`/`scikit-learn` chain are replaced by `soundfile` plus `soxr`, and ExLlamaV3 (with `triton-windows`) is optional.

## Models

The project supports [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) and [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf), with BF16 and FP16 TensorRT encoder builds, optional local INT8 Q/DQ encoder calibration, EXL3 3/4 bpw decoder weights, and a TensorRT W4A16 decoder (INT4 weights, FP16 activations, AWQ-scaled).

Published model files:

- [TensorRT encoder: BF16 and FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)
- [EXL3 decoder: legacy 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder)
- [1.7B EXL3 3 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) (separate directory in the same decoder repository)
- [1.7B EXL3 4 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) (separate directory in the same decoder repository)
- [1.7B W4A16 decoder, AWQ group 32](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-W4A16-Decoder)
- [1.7B W4A16 decoder, packed INT4, group 32](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-W4A16-Decoder-INT4) (smaller download, decoder only)
- [0.6B TensorRT encoder: BF16 and FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)
- [0.6B EXL3 decoder: legacy 3 bpw and 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)
- [0.6B EXL3 3 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100) (separate directory in the same decoder repository)
- [0.6B EXL3 4 bpw, recalibrated with train-clean-100](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100) (separate directory in the same decoder repository)
- [0.6B W4A16 decoder, AWQ group 32](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder)
- [0.6B W4A16 decoder, packed INT4, group 32](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder-INT4) (smaller download, decoder only)
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

Both commands give the encoder engine a 36-second input ceiling. Add `--max-frames` for longer audio, at 100 mel frames per second:

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --max-frames 16000
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

### W4A16 decoder

The alternative decoder keeps 4-bit weights in a 32-value group layout and runs the decode step with hand-written INT4 kernels, so nothing is dequantized into FP16. Quantization is activation-aware (AWQ scaling plus per-group clipping); the per-channel scales are folded into the adjacent normalization and projection weights, so prefill and decode stay numerically consistent without extra runtime scaling.

```powershell
# 128-utterance activation-aware quantisation from train-clean-100
uv run python tools\quantize_awq.py --group 32 --utterances 128 ^
  --manifest .\train-clean-100.manifest.jsonl --audio-root .\LibriSpeech ^
  --output models\Qwen3-ASR-0.6B-hf-awq32 --report artifacts\awq32-report.json
```

The resulting checkpoint is a plain FP16 model with folded scales, so it loads with `transformers` and reproduces the base model. The INT4 packing happens when the decoder is constructed (`W4A16ASRDecoder`). Build the 0.6B prefill engine as FP16 with the output projection inside the engine: on 2,620 utterances that measured 0.089 points better WER than an INT4 prefill engine and 1.7 ms faster. Moving the projection out of the 0.6B engine leaves the prefill time unchanged at 9.4 ms and costs 4 edits.

`--decoder-dir` selects the weights for `tensorrt-w4a16`: point it at a packed
decoder checkpoint (`packed_config.json` next to `model.safetensors`) and the
INT4 nibbles plus scales load as-is. Without the flag the CLI uses
`artifacts/<model>/decoder-int4-g32`. A directory without `packed_config.json`
is accepted too, but the decoder then quantizes the FP16 weights at load: that
prints a warning, costs startup time, and on 0.6B measured 0.4 points worse WER
than the published AWQ INT4 checkpoint.

The 1.7B prefill engine with the output projection inside is 3.29 GB. On an 8 GB card that leaves too little room next to the decoder weights, TensorRT re-reads its weight pages every step, and the full-corpus prefill median is 222 ms. Leave the projection out (`build-decoder-prefill` without `--keep-lm-head`): the engine drops to 2.69 GB, prefill to 20.6 ms, and the runtime applies the INT4 `lm_head` GEMV that the decode loop already uses. On 2,620 utterances the WER is 1.9876%, identical to the engine with the projection inside, and 0.015 points better than the INT4 prefill engine.

```powershell
uv run qwen3-asr --model .\models\Qwen3-ASR-1.7B-hf-awq32 build-decoder-prefill --max-seq 512 --opt-seq 256 `
  --output-dir artifacts\qwen3-asr-1.7b\decoder-prefill
uv run qwen3-asr --model .\models\Qwen3-ASR-1.7B-hf-awq32 transcribe .\audio.wav --backend tensorrt-w4a16 `
  --engine artifacts\qwen3-asr-1.7b\encoder-window.bf16.engine `
  --decoder-engine artifacts\qwen3-asr-1.7b\decoder-prefill\asr-prefill.fp16.engine
```

`--max-seq` is the longest prompt the prefill engine accepts and it has to cover the whole utterance, because the audio tokens dominate the prompt at roughly 13 per second of audio; the default 512 covers about 39 seconds. The encoder engine has its own ceiling (`--max-frames`, 3600 frames), and both have to cover the audio before a long file transcodes in one pass. `--max-cache` is a starting KV size rather than a reservation: the decoder grows past it per request, so it needs no tuning per file; the next section covers both knobs. For reference, the Transformers checkpoint accepts 65,536 positions.

A prompt longer than the profile raises an error naming the value to rebuild with instead of being truncated.

#### Packed INT4 decoder

`pack-decoder` writes a decoder-only checkpoint: the projections already hold INT4 nibbles plus one FP16 scale per 32 values, `embed_tokens` and the norms stay FP16, and the audio tower is not written because the TensorRT encoder replaces it. The runtime detects `packed_config.json` and loads the tensors as they are, so no FP16 weights are needed at run time.

```powershell
uv run qwen3-asr --model .\models\Qwen3-ASR-1.7B-hf-awq32 pack-decoder --group 32 --output-dir artifacts\qwen3-asr-1.7b\decoder-int4
uv run qwen3-asr --model artifacts\qwen3-asr-1.7b\decoder-int4 transcribe .\audio.wav --backend tensorrt-w4a16 `
  --engine artifacts\qwen3-asr-1.7b\encoder-window.bf16.engine `
  --decoder-engine artifacts\qwen3-asr-1.7b\decoder-prefill\asr-prefill.fp16.engine
```

The published packed repositories also carry `asr-prefill.onnx`, so the prefill engine can be built without
downloading the FP16 checkpoint:

```powershell
uv run qwen3-asr --model Retrieve0/Qwen3-ASR-1.7B-W4A16-Decoder-INT4 build-decoder-prefill `
  --onnx <packed-dir>\asr-prefill.onnx --max-seq 512 --opt-seq 256
```

The packed directory is 628 MB for 0.6B and 1528 MB for 1.7B, against 1492 MB and 3887 MB for the FP16 checkpoints. The nibbles come from the same kernel that packs at load time, so a packed checkpoint reproduces the FP16 path bit for bit. `--group` is fixed in the file; asking the loader for another value raises an error. The published repositories work directly as `--model Retrieve0/Qwen3-ASR-1.7B-W4A16-Decoder-INT4`.

#### Decoder parameters

Runtime flags, shared by `transcribe`, `benchmark` and `serve`:

| Flag | Backend | Default | Effect |
| --- | --- | ---: | --- |
| `--max-cache` | `tensorrt-w4a16` | 512 | Starting KV cache size, not a cap. The decoder starts at 512 slots and sizes each request as prompt plus `--max-new-tokens`, rounded up to 512 slots, growing in place when a longer request arrives: a 30 s file uses 1024 slots (112 MB) and a 145 s file grows to 2560 (280 MB). A slot costs 112 KB (28 layers × 8 KV heads × 128 × 2 values × FP16). |
| `--group` | `tensorrt-w4a16` | 32 | Packing granularity of the INT4 kernels: one FP16 scale per 32 or per 64 weights. The FP16 checkpoint is packed at load time, so `64` also runs, but it is coarser — on 0.6B it measured 0.16 points worse WER than 32. A packed checkpoint fixes the value in `packed_config.json`. |
| `--decoder-engine` | `tensorrt-w4a16` | `artifacts/<model>/decoder-prefill/asr-prefill.fp16.engine` | Prefill engine. Engines that emit `logits` and engines that emit the normed `hidden` both load. |
| `--max-context-tokens` | `tensorrt-exllama` | 8192 | KV budget of the EXL3 path. |
| `--engine` | all TensorRT backends | `artifacts/<model>/encoder-window.bf16.engine` | Audio encoder engine. |
| `--max-new-tokens` | all | 256 | Decode loop cap. |

Build flags for `build-decoder-prefill`:

| Flag | Default | Effect |
| --- | ---: | --- |
| `--precision` | `fp16` | Engine precision. |
| `--max-seq` | 512 | Longest prompt accepted, about 39 seconds of audio. Raise it for longer utterances. |
| `--opt-seq` | 256 | Length TensorRT tunes its tactics for. Set it near the typical prompt length of the workload. |
| `--workspace-gib` | 1 | Build-time workspace. |
| `--keep-lm-head` | off | Keep the output projection inside the engine. Off moves it to one INT4 GEMV in the runtime and saves 296 MB on 0.6B and 593 MB on 1.7B. |

`qwen3-asr build --max-frames N` sets the longest mel input the encoder engine accepts, in frames of 10 ms: the default 3600 is 36 seconds.

Both build commands derive their default output directory from the model argument, so an AWQ checkpoint named `Qwen3-ASR-1.7B-hf-awq32` writes to `artifacts/qwen3-asr-1.7b-hf-awq32/`, not to `artifacts/qwen3-asr-1.7b/`. Pass `--output-dir` to keep the layouts shown here.

Measured configurations on an 8 GB card:

| Model | `--max-seq` | `--keep-lm-head` | KV cache | Prefill engine | Result |
| --- | ---: | --- | --- | ---: | --- |
| 0.6B | 512 | yes | 2048 slots fixed | 1142 MB | 9.4 ms prefill, 72.2 ms median, 2.397% WER |
| 1.7B | 512 | no | 2048 slots fixed | 2694 MB | 20.6 ms prefill, 175.0 ms median, 1.988% WER |
| 1.7B, long audio | 2048 | no | 2560 slots allocated | 2694 MB | 145 s file in one pass, 14.9 s total |

The first two rows predate the per-request cache and used a fixed 2048-slot budget; the long-audio row uses the default ceiling.

The choice follows memory, not the model size: keep the projection inside the engine while the card still has room for it, and drop it when the engine plus the decoder weights leave too little. On 8 GB the 1.7B engine with the projection inside runs prefill at 222 ms, while the 1.14 GB 0.6B engine fits with no measurable cost.

#### Longer audio

Three limits decide how much audio one call takes, and on the shipped engines the encoder is the binding one:

| Limit | Knob | Shipped | Audio |
| --- | --- | ---: | ---: |
| Encoder input | `qwen3-asr build --max-frames` | 3600 frames | 36 s |
| Decoder prefill | `build-decoder-prefill --max-seq` | 512 tokens | 39 s |
| KV cache | `transcribe --max-cache` | starting size, 512 | 18 s, grows past it |

All three are memory, not compute. A 145 s file needs 14,500 mel frames and a 1,903-token prompt, so it fails on the encoder before the decoder sees it; rebuilding the encoder with `--max-frames 16000` gives a 610 MB engine, the same size as the 36 s one, and the file then transcribes in one pass: 14.9 s in total — 1.0 s preprocessing, 0.17 s encoder, 8.0 s prefill and 5.6 s decoding 477 tokens. Prefill is 12% of that pipeline on short utterances and 54% here, because attention grows with the square of the prompt. Either overrun names the knob and the value to rebuild with.

The decoder side costs next to nothing in engine size — 512, 1024 and 2048 all produce a 2693.9 MB 1.7B engine — but TensorRT allocates activation buffers at the maximum profile shape, so a larger profile holds more memory at load time: the 2048 profile costs 238 MB more than the 512 one. Prefill measured 22.7, 24.8 and 24.3 ms on the same 100 utterances at 512, 1024 and 2048, with identical transcripts.

The KV cache starts at `--max-cache` slots and grows past it when a request needs more, so the flag is a starting size, not a cap: both runtimes start at 512 slots and round `prompt + max_new_tokens` up to the next 512. Growth prints one line, keeps the written prefix and re-captures the CUDA graph once for the new size.

| Slots allocated | Memory | Audio |
| --- | ---: | ---: |
| 512 | 56 MB | 18 s |
| 1024 | 112 MB | 58 s |
| 1536 | 168 MB | 97 s |
| 2048 | 224 MB | 137 s |
| 2560 | 280 MB | 176 s |

A 30 s file asks for `prompt + 256` slots, which rounds to 1024; a 145 s file grows the buffer to 2560 in place, copying the written prefix and re-capturing the CUDA graph once for the new size. Nothing is released after a long request, so a session settles at the largest capacity it needed.

What remains is the prefill profile. **Chunked prefill** would remove it: the prefill graph currently takes no cache input, which is what fixes `--max-seq`. Giving it `cache_k`/`cache_v`/`cache_len` the way the decode graph already has them — with a mask that keeps the prefix visible and the chunk causal — lets the prompt arrive in blocks of the profile size while the cache accumulates. The arithmetic is the same as one-shot prefill, so any WER change would come from fp16 reduction order and the tactics TensorRT picks per sequence length, not from the model. It also caps the quadratic attention cost of a long prompt at the block size.

The absolute ceiling is the same as the official backend's: `max_position_embeddings` is 65,536, and 65,536 tokens of KV is about 7 GB, more than an 8 GB card holds next to the weights. Audio that does not fit has to be segmented at silences and transcribed piecewise, which drops the cross-segment context that a single pass keeps.

The audio encoder reproduces the reference inference windows (`n_window_infer`, 104 tokens) instead of attending over the whole utterance. Its output matches the Transformers reference at cosine 0.9995–0.9999 across one to five windows; corpus WER moved by ±0.04 points across decoders, a 20-edit scale on this corpus.


TensorRT engines are generated locally for the target GPU and TensorRT environment.

Build the forced aligner encoder and export both decoder variants:

```powershell
# 36 s / 1024-token / 128-word engines (default)
uv run qwen3-asr build-aligner

# 5 min / 8192-token / 1024-word engines
uv run qwen3-asr build-aligner --preset long

# optional EXL3 4 bpw decoder
uv run qwen3-asr --model Qwen/Qwen3-ForcedAligner-0.6B quantize --bits 4 --head-bits 16
```

`--max-frames`, `--max-seq` and `--max-timestamps` override the preset values;
the engine file names carry the preset (`aligner-encoder-long.fp16.engine`).

## Transcribe audio

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

Omitting `--language` lets the model detect it: the prompt carries no language
tag and the printed JSON adds the detected language next to the text, e.g.
`{"text": "...", "language": "English"}`. Audio of any sample rate is resampled
to 16 kHz on load, and multi-channel files are averaged to mono.

## Benchmark one file

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf benchmark .\audio.wav --backend tensorrt-w4a16 --warmups 2 --repeats 10
```

`--model` and `--max-new-tokens` are global flags and go before the subcommand; everything else follows it. `benchmark` prints mean, median, p90, min and max latency, the realtime factor, per-stage medians, the transcript, and a WER if an `<audio>.json` file next to the audio holds a `reference` field.

## Align audio

```powershell
uv run qwen3-asr align .\audio.wav --text-file .\transcript.txt --language English --format srt --output .\audio.srt
```

`align` uses the TensorRT encoder and a TensorRT bf16 aligner decoder by default.
It supports `--format json`, `--format srt`, and `--format txt`; `--text` can be
used instead of `--text-file`. `--decoder-precision fp16` selects the fp16 engine, which is about 8% faster,
`--backend tensorrt-exllama` selects the EXL3 decoder, and `--engine` /
`--decoder-engine` point at specific builds.

`--aligner-preset long` switches to the 5-minute engines. The limits are the
engine profiles: the default `window` set takes 36 s of audio and 128 words
(2 timestamp slots per word), the `long` set 300 s and 1024 words. Over-long
input is rejected before any buffer is touched, naming the preset to rebuild
with. The long set costs about 1.9 GB more VRAM at load because TensorRT sizes
activations for the maximum profile, even when the input is short.

The native runtime exposes the same thing as `qwen3-asr-native.exe --align`;
see [native/README.md](native/README.md#forced-alignment). Both servers can also
serve it: `POST /v1/align?language=English&text=<urlencoded>` with the audio as
the body returns `{"items": [{"text", "start_time", "end_time"}], "timings": {...}}`.
Enable it with `--aligner-model`/`--aligner-engine` on `serve` in Python, or
`--align-encoder`/`--align-decoder` in the native runtime.



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

## Disk budget

Inference needs the Python environment plus one encoder engine and one decoder artifact. Measured on this machine, in GB of 1024 MB:

| Backend | 0.6B | 1.7B |
| --- | ---: | ---: |
| `tensorrt-w4a16`, FP16 checkpoint | 10.6 | 14.7 |
| `tensorrt-w4a16`, packed INT4 decoder | 9.8 | 12.4 |
| `tensorrt-exllama` | 9.0 | 10.1 |

The environment is 7.7 GB for the TensorRT backends: torch 4.2 GB, `tensorrt_libs` 3.2 GB, and 0.3 GB of transformers, tokenizers, numpy, onnx and soxr. Torch's cu128 wheel carries its own CUDA libraries, so no `nvidia-*` packages are installed. `--extra exllama` adds 0.36 GB, mostly `triton-windows`.

| Artifact | 0.6B | 1.7B |
| --- | ---: | ---: |
| encoder engine | 359 MB | 610 MB |
| prefill engine | 1142 MB | 2694 MB |
| FP16 checkpoint | 1492 MB | 3887 MB |
| packed decoder | 628 MB | 1528 MB |
| EXL3 4 bpw decoder | 631 MB | 1502 MB |

The EXL3 backend needs no prefill engine and no FP16 weights, only the tokenizer and preprocessor files. Engines are built locally for the target GPU, so they cannot be downloaded; the ONNX files they are built from are not read at run time and can be deleted afterwards.

## Benchmarks

LibriSpeech `test-clean`, all 2,620 utterances, 53,120 reference words. Latencies are full-corpus medians; 30–200-utterance subsets run 20–40% higher because the first utterances pay for process and clock warm-up. Corpus WER is deterministic and reproduces across sessions; latency is only comparable within one session. The EXL3 4 bpw rows and the 0.6B W4A16 row were measured on 2026-11-12 with the windowed encoder, the 1.7B W4A16 row on 2026-11-14; re-running one 2026-11-12 entry in the later session moved its median from 208.8 ms to 211.5 ms. The 3 bpw and GGUF rows are the 2026-11-11 runs. The 2026-11-14 rows were re-run later with the per-request KV cache, which reproduced the WER to the digit.

| Model | Backend | Corpus WER | Median ms |
| --- | --- | ---: | ---: |
| 0.6B | Official Transformers BF16 | 2.321% | 787.1 |
| 0.6B | GGUF INT4 + Q4_K | 2.982% | 299.2 |
| 0.6B | TensorRT fp16 + W4A16 AWQ (g32) | 2.397% | **72.2** |
| 0.6B | TensorRT BF16 + EXL3 4 bpw | 2.383% | 99.5 |
| 0.6B | TensorRT BF16 + EXL3 4 bpw, train-clean-100 | **2.362%** | 99.8 |
| 1.7B | Official Transformers BF16 | 1.886% | 863.5 |
| 1.7B | GGUF INT4 + Q4_K | 2.090% | 391.8 |
| 1.7B | TensorRT fp16 prefill + W4A16 AWQ (g32) | 1.988% | **175.0** |
| 1.7B | TensorRT BF16 + EXL3 4 bpw | 1.984% | 208.8 |
| 1.7B | TensorRT BF16 + EXL3 4 bpw, train-clean-100 | **1.944%** | 208.1 |

![0.6B calibration vs GGUF and official](docs/windows-librispeech-qwen3-0.6b-calibration-vs-gguf.png)

![0.6B median latency by stage](docs/windows-librispeech-qwen3-0.6b-stage-latency.png)

![1.7B calibration vs GGUF and official](docs/windows-librispeech-qwen3-1.7b-calibration-vs-gguf.png)

![1.7B median latency by stage](docs/windows-librispeech-qwen3-1.7b-stage-latency.png)

Forced alignment on the same corpus: 25.06 ms median with the TensorRT bf16
aligner decoder and 39.52 ms with the EXL3 4 bpw aligner decoder, versus
93.13 ms for the official backend and 139.55 ms for GGUF. 99.2% of TensorRT
boundaries are exactly equal to the official output.

![Aligner comparison](docs/windows-librispeech-qwen3-aligner-comparison.png)

![Aligner median latency by stage](docs/windows-librispeech-qwen3-aligner-stage-latency.png)

## Pruning and speculative decoding

Branch `experiments/prune-spec` holds two attempts at making the 1.7B decoder
cheaper. Every calibration and tuning decision reads train-clean-100 only;
test-clean stays an evaluation set.

### Structured pruning

`tools/prune_decoder.py` ranks FFN channels by the mean absolute `down_proj`
input weighted by that channel's `down_proj` column norm, and ranks layers by the
KL divergence of the next-token distribution when a block is skipped. The
statistics come from real ASR prompts, audio tokens included, built from
train-clean-100 with the TensorRT encoder. Calibrating on text-only sentences
instead selected channels the audio positions do not use: that model answered
`<|im_end|>` for every clip even though its FP16 weights loaded cleanly.

Each variant is packed to INT4 at group 32 and gets its own prefill engine.

| Variant | Change | Packed | WER (2620) | Decode ms/token | Decode |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline | 28 layers, FFN 6144 | 1517 MB | 1.98% | 6.99 | — |
| ffn875 | 28 layers, FFN 5376 | 1457 MB | 2.21% | 6.57 | −5.9% |
| ffn75 | 28 layers, FFN 4608 | 1375 MB | 3.16% | 6.11 | −12.5% |
| depth26 | layers 11, 12 dropped | 1463 MB | 2.91% | 6.59 | −5.6% |
| depth26b | layers 3, 25 dropped | 1463 MB | 3.35% | 6.51 | −6.8% |

WER is the full 2,620-utterance corpus. Decode rates are medians of three
interleaved native passes over 200 utterances (`tools/compare_native_variants.py`);
the three passes agreed to within 0.3%, so the rates compare to each other but not
to the benchmark table above.

The exchange rate improves as the cut gets smaller, which is what the channel
ranking predicts: the last channels removed are the least important. Twelve and a
half percent of the FFN costs 0.23 WER points for 5.9% decode (0.039 points per
decode point); twenty-five percent costs 1.18 points for 12.5% (0.094). Depth
pruning costs about twice as much per decode point and buys no more.

`ffn875` is 0.08 points outside the +0.15 bound. Nothing else comes close, so
raising the bound to +0.25 would admit exactly one variant, on measured marginal
cost rather than on convenience; whether that trade is acceptable is a product
call, not a benchmarking one.

### Recovery

Three options were measured; all keep the tensor shapes, so the packed decoder
and the prefill engine come out of the same pipeline.

`tools/refit_pruned_ffn.py` re-solves each pruned FFN's output projection in
closed form so the layer reproduces the unpruned layer's output on train-clean-100
activations (256 audio prompts, ridge 1e-3):

| Variant | WER (2620) | vs unpruned | Decode |
| --- | ---: | ---: | ---: |
| ffn75 | 3.16% | +1.18 | −12.5% |
| ffn75 + refit | 2.78% | +0.80 | −12.5% |
| ffn875 | 2.21% | +0.23 | −5.9% |
| ffn875 + refit | 2.19% | +0.21 | −5.9% |

The refit recovers a third of the loss at a 25% cut and almost nothing at 12.5%.
At 12.5% the dropped channels carry so little of the layer's output that the
truncated sum is already near the least-squares optimum, so there is nothing to
re-solve; at 25% the error is systematic enough for the refit to help.

Self-distillation (`tools/distill_pruned.py`) did not work and is recorded as a
negative result. The student is the pruned FP16 decoder with rank-16 adapters on
its FFN projections; the teacher is the unpruned model's top-32 next-token
distribution on train-clean-100 prompts. The first attempt distilled every prompt
position and failed loudly: three quarters of those positions are audio
placeholders whose next token is arbitrary, so the loss was dominated by noise,
the adapters moved 18% of the weight, and WER collapsed to 92%. Restricting the
loss to the positions that predict the model's own transcript fixed the objective
(starting loss fell from 1.88 to 0.20 nats) but not the outcome: the adapters
still moved 3-14% of each projection and WER rose from 2.19% to 3.78%.

The lesson is that this model pair has almost no headroom on the next-token
distribution: the pruned model already reproduces the teacher on the tokens that
matter, so the remaining WER gap comes from compounding errors over a 30-token
transcript, which token-level distillation with a low-rank adapter does not
address. A production attempt would need the student to train on its own
trajectories (on-policy), or much larger data, and would still have to beat the
closed-form refit before it is worth the compute.

The measurement that would justify on-policy work is the per-token agreement on
the teacher's own transcripts: ffn875 + refit agrees with the teacher on 97.7% of
5,059 positions. That leaves 2.3% per token, which compounds to roughly a 38%
chance of a clean 42-token transcript, so the gap is partly per-token quality and
partly drift. On-policy training targets the drift half; whether that closes the
remaining 0.06 points to the acceptance bound is unknown and was not attempted.

### Speculative decoding

`tools/spec_decode_bench.py` drafts four tokens with the 0.6B decoder and verifies
them with the 1.7B decoder in a single batched pass. `W4A16ASRDecoder.verify`
replays the block through a captured CUDA graph, and `csrc/w4a16_gemm.cu` gives
each warp four output rows so the block's activations are loaded once per four
rows instead of once per row. Greedy verification is exact: 32 of 32 transcripts
are identical to the target's own greedy decode.

| | short audio (32 utts, 5.8 s median) | long audio (12 utts, 18.5 s median) |
| --- | ---: | ---: |
| Accepted draft tokens | 91.5% | 95.7% |
| Tokens per round | 4.66 | 4.83 |
| Decode | 115.9 / 152.4 ms (1.31x) | 319.6 / 328.0 ms (1.03x) |
| Decoder stage, prefill + decode | 126.8 / 161.5 ms (−21.5%) | 335.7 / 334.8 ms (+0.3%) |
| Full pipeline with the draft model | 162.2 / 161.6 ms (+0.4%) | 406.9 / 334.8 ms (+21.5%) |
| Transcripts identical to greedy | 32 / 32 | 12 / 12 |
| Peak GPU memory | 2.35 GiB | 2.37 GiB |

A round costs about 1.6 target-steps to verify and 1.9 to draft, and produces 4.66
tokens; the draft spends 2.39 ms per token against the target's 5.10, and the
verify 8.31 ms against 5.10. Three things make the verify cost more than the
single-row step it replaces: the batched GEMM runs at 1.5–1.7x the single-row
rate because it re-reads the block's activations once per output row, the layer
still needs about fifteen small kernel launches, and attention scans the live
length even though the kernel takes it from the device position.

Sweeping the draft length in one process, so clock drift hits every K equally,
puts the optimum at four:

| K | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Speedup | 1.00x | 1.00x | 1.18x | **1.28x** | 1.05x | 1.14x | 0.96x |

Past four the batch grows to six or more rows, the GEMM loses more to register
pressure than the extra candidates return, and acceptance per proposed token falls
from 0.91 to 0.63.

End to end the pairing only breaks even on short audio: the draft model needs its
own audio encoder and prefill, 22.8 ms per utterance, against the 35.7 ms the
pairing saves. The 0.6B draft cannot share the target's prompt embeddings, since
the two audio towers project to different widths, so a two-model pair always pays
two encoders. On long audio the gain shrinks as well, because a round issues nine
queries (four draft, five verify) against the target's 4.66, so the attention
share of the step grows with context even though acceptance improves.

Two follow-ups were measured and are recorded as negative results. **Self
speculation** (drafting with the target's first N layers, which would share the
encoder and the prompt) is not available: the hidden state after layer N pushed
through the final norm and the tied output projection agrees with the full
model's argmax on only 0.5% of transcript positions at N=8, 1.4% at N=12, 5.6% at
N=16 and 41.8% at N=20. Those layers were never trained to be predictive, so early
exit cannot draft here.

**Two resident models do not fit.** Of 8 GiB, 6.94 GiB is usable; the 1.7B
decoder, its prefill engine and its encoder already hold 5.4 GiB, and loading the
0.6B decoder and prefill leaves zero bytes free before its encoder is even
allocated. The benchmark above works only because it encodes every utterance in
two separate passes and releases the encoders before the timed loop. A single-shot
pipeline would have to overcommit and page.

#### Sharing the encoder with a projection adapter

The draft needs the 0.6B audio tower only to build its prompt embeddings, and that
tower is what pushed the pair out of memory. `tools/fit_draft_adapter.py` fits a
2048->1024 map from the 1.7B tower's audio embeddings to the 0.6B's, so the draft
can build its prompt from the target's single audio pass. Both towers emit the
same tokens for the same audio (no mismatches in 300 utterances), so the pairs
align and the fit is a closed-form ridge solve over 50,895 positions; it explains
83.6% of the 0.6B embedding energy.

| | two encoders | adapter |
| --- | ---: | ---: |
| Draft acceptance | 91.5% | 89.6% |
| Decode | 1.31x | 1.30x |
| Audio encoders resident | 2 | 1 |
| Full pipeline | 162.2 vs 161.6 ms (+0.4%) | 129.4 vs 147.0 ms (−12%) |
| Transcripts identical | 32 / 32 | 32 / 32 |

The adapter costs 1.9 points of acceptance and cannot change the transcript, since
the target still verifies every token. It is the first configuration where
speculation wins end to end, and the 147.0 ms baseline counts one encoder pass for
both paths.

It still does not fit cleanly: the target decoder, prefill engine and encoder take
5.39 GiB of the 6.94 GiB a fresh context sees, and the draft decoder plus prefill
take the remaining 1.55 GiB, leaving 0 MiB. Three ways to free the ~0.4 GiB were
checked. An INT4 prefill for the target would save 1.7 GiB, but this repo has no
supported path for quantizing the decoder prefill ONNX (only the encoder has
`quantize-encoder`). The INT8 encoder is 311 MB instead of 610, but the calibrated
width-pinned engine is limited to an 800-frame profile and rebuilding it for the
full 3,600 frames failed in the builder, so it saves 0.28 GiB at best. The pruned
`ffn875-refit` target saves 0.32 GiB and leaves 12 MiB free, which is the same
place: the card is fully committed.

So the adapter configuration needs roughly 0.4 GiB that this build cannot find; on
a 12 GB card it runs as measured, and on 8 GB it needs either a quantized decoder
prefill or the pruned target.

The native runtime now has the same batched verify (`Decoder::verify`, built from
`csrc/w4a16_gemm.cu` and `csrc/verify_ops.cu`); its self-check accepts 3 of 3 of
the target's own tokens, so the batched kernels agree with the single-token path.
Wiring it into a two-model driver is blocked by the memory above, not by the
kernels.