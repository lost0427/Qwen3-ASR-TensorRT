# Native C++ runtime

A C++20 reimplementation of the inference path: it reads the published engines,
the packed INT4 decoder and tokenizer assets, and transcribes without PyTorch.

## Status

| Stage | Implementation |
| --- | --- |
| TensorRT engine loading | C++ `nvinfer` API |
| audio decode | libsndfile, loaded at run time |
| log-mel | GPU kernel (windowed DFT, mel, log) or CPU `pocketfft` |
| prompt assembly | prefix + audio placeholders + language suffix, from `tokenizer.bin` |
| encoder + prefill | the published TensorRT engines |
| decode | hand-written INT4 kernels, captured into a CUDA graph |
| CLI | batch: one process, many files, engines loaded once |

## Verification

Corpus WER matches the Python run exactly, and most transcripts are identical
character for character:

| Model | 100 test-clean utterances | Identical transcripts | Median per utterance |
| --- | --- | --- | --- |
| 0.6B | WER 1.6624% (same as Python) | 85 / 100 | 84 ms |
| 1.7B | WER 1.1083% (same as Python) | 93 / 100 | 189 ms |

The CPU and GPU mel paths give identical transcripts to each other (100/100 on the same set).
The remaining differences against Python are punctuation and borderline words: the GPU mel
kernel differs from `WhisperFeatureExtractor` by about 1e-5 because it
accumulates the DFT in a different order, and a handful of positions are that
close. Every stage up to the decode loop reproduces Python bit for bit when fed
the same mel.

## Speed

Per-utterance medians from the same 100-utterance batch, 0.6B: mel 1.4 ms,
encoder 5.3 ms, prefill 13.8 ms, decode 63.6 ms. 1.7B: mel 1.4 ms, encoder
6.9 ms, prefill 21.5 ms, decode 158.3 ms. Python's full-corpus medians are
71.8 ms and 175.0 ms, so the two are at parity; the native runtime pays its
engine load once per process instead of once per file.

Mel cost alone, end to end:

| Frames | GPU | CPU (`pocketfft`) |
| --- | ---: | ---: |
| 400 (3.3 s audio) | 2.0 ms | 3.0 ms |
| 14,500 (145 s audio) | 31.6 ms | 107.6 ms |

`--mel-device cuda` (default) runs the whole front end on the GPU: one block per
frame, interleaved twiddle table, mel accumulation and log in the same kernel.

## Build

```powershell
native\scripts\fetch_deps.ps1        # TensorRT headers, import library, pocketfft
cmake -S native -B native/build -G Ninja
cmake --build native/build
```

`fetch_deps.ps1` downloads `include/*.h` from `github.com/NVIDIA/TensorRT`
branch `release/11.4` (matching the 11.4.0.106 wheel), the single-header
`pocketfft_hdronly.h`, and builds `nvinfer_11.lib` from the `nvinfer_11.dll` in
`tensorrt_libs`. `native/deps`, `native/build` and `native/assets` are ignored.

## Assets

```powershell
uv run python native/tools/gen_assets.py --model models\Qwen3-ASR-0.6B-hf-awq32-tc100 --out native\assets
```

`mel.bin` holds the mel filter bank and the Hann window from the Python feature
extractor; `tokenizer.bin` holds the byte-level vocabulary, special token ids,
the prompt prefix and the per-language suffix.

## Run

```powershell
native\build\qwen3-asr-native.exe `
  --model artifacts\qwen3-asr-0.6b\decoder-int4-g32 `
  --encoder artifacts\qwen3-asr-0.6b\encoder-window-win.bf16.engine `
  --decoder artifacts\qwen3-asr-0.6b\decoder-trt-awq32-tc100\asr-prefill.fp16.engine `
  --assets native\assets --language English audio1.flac audio2.flac ...
```

`--mel-device cpu` selects the pocketfft front end, `--max-new-tokens` and
`--max-cache` mirror the Python flags. `QASR_TIMING=1` prints per-stage
milliseconds per file.

## Not done yet

- Audio must be 16 kHz; the Python path resamples with soxr.
- `benchmark` and the HTTP server are Python-only.
