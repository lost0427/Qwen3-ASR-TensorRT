# Native C++ runtime

A C++20 reimplementation of the inference path: it reads the published engines,
the packed INT4 decoder and tokenizer assets, and prints the transcript without
PyTorch.

## Status

| Stage | State |
| --- | --- |
| TensorRT engine loading | works |
| audio decode (libsndfile) + 16 kHz | works |
| log-mel front end | matches the Python extractor to 1.6e-5 |
| prompt assembly (prefix + audio + language suffix) | identical to Python |
| encoder engine | bit-identical to the Python path |
| prefill engine + KV cache layout | bit-identical |
| decode loop (INT4 GEMV, RoPE, cache, attention) | works, transcripts match |

## Verified output

| Model | Same transcript as Python | WER |
| --- | --- | --- |
| 0.6B, first 100 utterances | 90 / 100 | 1.53% against 1.66% for the Python run |
| 1.7B, first 10 utterances | 9 / 10 | — |

The remaining differences are punctuation and borderline word choices: the mel
front end differs from `WhisperFeatureExtractor` by up to 1.6e-5 because it
computes the STFT with a DFT in double precision instead of torch's
single-precision FFT. Every stage before the decode loop reproduces Python bit
for bit when fed the same mel, and the decoder kernels match the Python kernels
exactly, so the mel rounding is the only source of divergence. Bit-identical
transcripts would require reproducing torch's FFT bit for bit.

## Build

```powershell
native\scripts\fetch_deps.ps1        # TensorRT headers + import library
cmake -S native -B native/build -G Ninja
cmake --build native/build
```

`fetch_deps.ps1` downloads `include/*.h` from
`github.com/NVIDIA/TensorRT` branch `release/11.4` (matching the 11.4.0.106
wheel) and builds `nvinfer_11.lib` from the `nvinfer_11.dll` that ships in
`tensorrt_libs`. `native/deps`, `native/build` and `native/assets` are ignored.

## Assets

```powershell
uv run python native/tools/gen_assets.py --model models\Qwen3-ASR-0.6B-hf-awq32-tc100 --out native\assets
```

`mel.bin` holds the mel filter bank and the Hann window taken from the Python
feature extractor; `tokenizer.bin` holds the byte-level vocabulary, the special
token ids, the prompt prefix and the per-language suffix.

## Run

```powershell
native\build\qwen3-asr-native.exe `
  --model artifacts\qwen3-asr-0.6b\decoder-int4-g32 `
  --encoder artifacts\qwen3-asr-0.6b\encoder-window-win.bf16.engine `
  --decoder artifacts\qwen3-asr-0.6b\decoder-trt-awq32-tc100\asr-prefill.fp16.engine `
  --assets native\assets --language English audio.flac
```

Set `QASR_DEBUG=1` to trace layer 0 of the first decode step, and
`QASR_DUMP=<dir>` to write the mel, encoder output, prompt ids and generated
tokens for comparison with the Python path.

## Not done yet

- One host sync per token; the Python runtime captures a CUDA graph and syncs
  every eight. Latency is therefore higher than Python's.
- Audio must be 16 kHz; the Python path resamples with soxr.
- `benchmark` and the HTTP server are Python-only.
