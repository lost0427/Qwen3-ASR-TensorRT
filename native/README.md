# Native C++ runtime

A C++20 reimplementation of the inference path: it reads the published engines,
the packed INT4 decoder and tokenizer assets, and transcribes without PyTorch.

## Status

| Stage | Implementation |
| --- | --- |
| TensorRT engine loading | C++ `nvinfer` API |
| audio decode | libsndfile, loaded at run time, plus a Kaiser-windowed sinc resampler |
| log-mel | GPU kernel (windowed DFT, mel, log) or CPU `pocketfft` |
| prompt assembly | prefix + audio placeholders + language suffix, from `tokenizer.bin` |
| encoder + prefill | the published TensorRT engines |
| decode | hand-written INT4 kernels, captured into a CUDA graph |
| forced alignment | TensorRT audio encoder + a single-pass TensorRT decoder, own BPE tokenizer |
| CLI | batch: one process, many files, engines loaded once |
| HTTP API | `POST /v1/transcribe?language=...`, same protocol as the Python server |

## Verification

Corpus WER matches the Python run on the same 100 utterance sample, and most
transcripts are identical character for character:

| Model | 100 test-clean utterances | Identical transcripts | Median per utterance |
| --- | --- | --- | --- |
| 0.6B | WER 1.5345% (1.6624% with CPU mel, same as Python) | 92 / 100 | 79 ms |
| 1.7B | WER 1.1083% (same as Python) | 97 / 100 | 182 ms |

The CPU and GPU mel paths agree on 94/100 transcripts for 0.6B and 98/100 for
1.7B. The remaining differences against Python are punctuation and borderline
words: the GPU mel kernel differs from `WhisperFeatureExtractor` by about 1e-5
because it accumulates the DFT in a different order, and a handful of positions
are that close. Every stage up to the decode loop reproduces Python bit for bit
when fed the same mel.

## Speed

Per-utterance medians from the same 100-utterance batch. 0.6B: mel 1.3 ms,
encoder 4.4 ms, prefill 8.3 ms, decode 63.0 ms, total 78.5 ms; the Python run on
the same utterances is mel+load 6.3 ms, encoder 6.2 ms, prefill 10.2 ms, decode
62.4 ms, total 84.8 ms. Decode is at parity and the other stages are faster.
1.7B: prefill 16.6 ms, decode 158.0 ms, total 182.2 ms.

A micro-benchmark of one decode step (`QASR_BENCH=300`) puts the whole step at
3.16 ms, the three kernels this runtime adds on top of the Python set at 44 us
(argmax 21 us, rotary row gather 15 us, token commit 7 us), and the prefill KV
cache copy at 13 us for a typical prompt. That copy used to be 448 per-head
`cudaMemcpy2D` calls and cost 4.1 ms per utterance; it is now one kernel that
reads each row tile of `(seq, layers, kv_heads, head_dim)` once.

Cold start to first transcript, including reading the packed INT4 weights and
deserialising the engines: 2.3 s for 0.6B (0.6 GB of weights) and 5.7 s for
1.7B (1.5 GB).

Mel cost alone, end to end:

| Frames | GPU | CPU (`pocketfft`) |
| --- | ---: | ---: |
| 400 (3.3 s audio) | 0.8 ms | 1.0 ms |
| 14,500 (145 s audio) | 12.2 ms | 45.3 ms |

`--mel-device auto` (the default) uses `pocketfft` below ~600 frames, where the
GPU kernel is launch bound, and the GPU kernel above. `--mel-device cuda` runs
the whole front end on the GPU with four frames per block, so the twiddle table
and the mel filters are read once and reused across the four frames. The CPU
path builds frames contiguously, transposes them to `[n_mel][frames]` in
cache-sized blocks, and uses the triangular support of the mel filters instead of
the full 80 x 201 product.

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
the prompt prefix and the per-language suffix. The aligner adds
`aligner_tokenizer.bin` (151,643-entry BPE vocabulary, 151,387 merges, special
ids and the Unicode class table, 3.6 MB) and `aligner_embed.bin` (the FP16
152,064 x 1024 token embedding table, 297 MB):

```powershell
uv run python native\tools\gen_aligner_assets.py --model models\Qwen3-ForcedAligner-0.6B `
  --tokenizer artifacts\qwen3-forcedaligner-0.6b\decoder\tokenizer.json --out native\assets
```

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

`--language` defaults to `auto`, which leaves the language tag out of the prompt
and reports what the model detected (`{"text": ..., "language": "English"}`);
naming a language forces it. Anything that is not 16 kHz is resampled on load,
and multi-channel files are averaged to mono like the Python loader.

`--max-cache` is the starting size in both runtimes, and both grow past it
automatically: 512 slots (56 MB) in 512-slot steps. Sizing happens at prefill,
from `prompt + max_new_tokens`, so a longer utterance
reallocates once, prints `[cache] growing KV cache ...`, and re-captures the
decode graph. Only the new rotary rows are computed on a grow, so a single step
costs a few milliseconds and a 512 -> 2560 jump measured about 10 ms with the
graph re-capture included. Nothing is released afterwards, so a session settles
at the largest capacity it needed. A slot costs 112 KB, and test-clean needs 512
slots for 95% of its utterances (the longest file, 35 s, needs 1024).

## Input limits

| Engine set | Audio ceiling | Binding limit |
| --- | ---: | --- |
| `encoder-window-win` + any prefill | 36.0 s | encoder profile, 3600 frames |
| `encoder-long-160s` + `decoder-prefill-2048` | 156.2 s | prefill profile, 2048 tokens |
| `encoder-long-160s` + prefill rebuilt at `--max-seq 2112` | 160.0 s | encoder profile, 16000 frames |

Both runtimes compute the mel for the whole file, pad the mel axis to a multiple
of 100 frames and reject over-long input instead of truncating:

```text
error: audio of 4000 frames (40.0 s) exceeds this encoder's 3600-frame profile (36.0 s); rebuild it with `qwen3-asr build --max-frames 4000`
error: prompt of 2072 tokens (159.4 s of audio) exceeds this prefill engine's 2048-token profile; rebuild it with `qwen3-asr build-decoder-prefill --max-seq 2072`
```

The prompt is `9` fixed tokens + one `<|audio_pad|>` per encoder token, where
the audio tokens are `13` per second plus a short tail, and the suffix is
`<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage <L><asr_text>`; with
`--language auto` (the default) the suffix is 3 tokens shorter and the model's
`language X<asr_text>` answer prefix is split off and reported. The generated
text is capped by `--max-new-tokens` (256 by default), independent of the audio
length.

Serve the same endpoint the Python runtime exposes:

```powershell
qwen3-asr-native.exe --model ... --encoder ... --decoder ... --serve --port 8000
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=English" `
  -H "Content-Type: audio/flac" --data-binary "@audio.flac"
```

Audio is decoded from the request body through libsndfile's virtual IO, so no
temporary files are written; bodies are capped at 128 MB and inference is
serialised, matching the Python server.

Adding `--align-encoder` / `--align-decoder` also registers `POST /v1/align`,
which takes the transcript as a query parameter and returns the word timings:

```powershell
qwen3-asr-native.exe --model ... --serve --port 8000 `
  --align-encoder artifacts\qwen3-forcedaligner-0.6b\aligner-encoder-window.fp16.engine `
  --align-decoder artifacts\qwen3-forcedaligner-0.6b\aligner-decoder-window.fp16.engine
curl.exe -X POST "http://127.0.0.1:8000/v1/align?language=English&text=hello%20world" `
  -H "Content-Type: audio/flac" --data-binary "@audio.flac"
```

## Forced alignment

`--align` runs the same pipeline as `qwen3-asr align`: the prompt is
`<|audio_start|>` + one `<|audio_pad|>` per encoder token + `<|audio_end|>` +
words joined by `<timestamp><timestamp>`; one encoder pass and one decoder pass
emit the timestamp bins, and the monotonic repair is the reference algorithm.

```powershell
qwen3-asr-native.exe --align `
  --encoder artifacts\qwen3-forcedaligner-0.6b\aligner-encoder-window.fp16.engine `
  --decoder artifacts\qwen3-forcedaligner-0.6b\aligner-decoder-window.fp16.engine `
  --assets native\assets --language English `
  --text "Stuff it into you his belly counseled him" audio.flac
```

`--format srt|txt` and `--output FILE` mirror the Python flags and produce
byte-identical subtitles. The JSON carries `timings` with `preprocessingMs`,
`encoderMs`, `decoderMs`, `totalMs` plus `frames`, `tokens` and `words`.

The output is `{"items": [{"text", "start_time", "end_time"}], "timings": {...}}`,
free of PyTorch.

Engine profiles are built with `qwen3-asr build-aligner --preset window|long`:

| Preset | Encoder | Decoder tokens | Timestamp slots | Audio | Words |
| --- | ---: | ---: | ---: | ---: | ---: |
| `window` | 3600 frames | 1024 | 256 | 36 s | 128 |
| `long` | 30000 frames | 8192 | 2048 | 300 s | 1024 |

The two timestamp slots per word are what bind first on both presets. Both
limits are checked before any buffer is touched, so an over-long request names
the preset to rebuild with.

Memory, measured on the same 8 GB card (process baseline 1.08 GB):

| Set | Engine files | Load delta | Peak |
| --- | ---: | ---: | ---: |
| `window` | 1534 MB | +1748 MB | 2.85 GB |
| `long` | 1463 MB | +3622 MB | 4.84 GB (300 s input) |

TensorRT allocates activations at the maximum profile shape, so the long set
costs its 1.9 GB of extra activations up front regardless of input length.

Verified against the Python aligner: 54 words on three test-clean files match
exactly, and 168 words on a 70 s clip differ in four 80 ms boundaries. Per call
on a 3 s file the pipeline takes 11 ms (mel 0.3, encoder 2, decoder 8); Python
takes 17 ms steady state and 266 ms for a cold CLI run. A 70 s clip takes 94 ms
here against 281 ms in Python. Four frames per mel block cut the front end from
12 ms to 6 ms there; decoder build knobs (4 GiB workspace, opt 1024, dropping
strong typing) changed nothing measurable, so the decoder runs at its limit.

## Not done yet

- `benchmark` is Python-only.
- `--align` is CLI-only, and Japanese/Korean text falls back to per-character
  splitting (the Python path uses nagisa/soynlp when they are installed).
