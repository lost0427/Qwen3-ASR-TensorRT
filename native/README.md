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
| HTTP API | `POST /v1/transcribe?language=...`, same protocol as the Python server |

## Verification

Corpus WER matches the Python run on the same 100 utterance sample, and most
transcripts are identical character for character:

| Model | 100 test-clean utterances | Identical transcripts | Median per utterance |
| --- | --- | --- | --- |
| 0.6B | WER 1.5345% (1.6624% with CPU mel, same as Python) | 92 / 100 | 84 ms |
| 1.7B | WER 1.1083% (same as Python) | 97 / 100 | 182 ms |

The CPU and GPU mel paths agree on 94/100 transcripts for 0.6B and 98/100 for
1.7B. The remaining differences against Python are punctuation and borderline
words: the GPU mel kernel differs from `WhisperFeatureExtractor` by about 1e-5
because it accumulates the DFT in a different order, and a handful of positions
are that close. Every stage up to the decode loop reproduces Python bit for bit
when fed the same mel.

## Speed

Per-utterance medians from the same 100-utterance batch. 0.6B: mel 1.3 ms,
encoder 4.9 ms, prefill 9.3 ms, decode 63.9 ms, total 83.5 ms; the Python run on
the same utterances is mel+load 6.3 ms, encoder 6.2 ms, prefill 10.2 ms, decode
62.4 ms, total 84.8 ms. Decode is at parity and the other stages are faster.
1.7B: prefill 17.3 ms, decode 157.7 ms, total 181.9 ms.

A micro-benchmark of one decode step (`QASR_BENCH=300`) puts the whole step at
3.16 ms, the three kernels this runtime adds on top of the Python set at 47 us
(argmax 22 us, rotary row gather 17 us, token commit 8 us), and the prefill KV
cache copy at 14 us for a typical prompt. That copy used to be 448 per-head
`cudaMemcpy2D` calls and cost 4.1 ms per utterance; it is now one kernel that
reads each row tile of `(seq, layers, kv_heads, head_dim)` once.

Cold start to first transcript, including reading the packed INT4 weights and
deserialising the engines: 2.3 s for 0.6B (0.6 GB of weights) and 5.7 s for
1.7B (1.5 GB).

Mel cost alone, end to end:

| Frames | GPU | CPU (`pocketfft`) |
| --- | ---: | ---: |
| 400 (3.3 s audio) | 1.0 ms | 0.8 ms |
| 14,500 (145 s audio) | 24.4 ms | 35.9 ms |

`--mel-device cuda` (default) runs the whole front end on the GPU: one block per
frame, interleaved twiddle table, one block-level max reduction, and mel
accumulation and log in the same kernel. The CPU path builds frames
contiguously, transposes them to `[n_mel][frames]` in cache-sized blocks, and
uses the triangular support of the mel filters instead of the full 80 x 201
product.

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
milliseconds per file, `QASR_BENCH=N` times the decode pieces.

Serve the same endpoint the Python runtime exposes:

```powershell
qwen3-asr-native.exe --model ... --encoder ... --decoder ... --serve --port 8000
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=English" `
  -H "Content-Type: audio/flac" --data-binary "@audio.flac"
```

Audio is decoded from the request body through libsndfile's virtual IO, so no
temporary files are written; bodies are capped at 128 MB and inference is
serialised, matching the Python server.

## Not done yet

- Audio must be 16 kHz; the Python path resamples with soxr.
- `benchmark` and the HTTP server are Python-only.
