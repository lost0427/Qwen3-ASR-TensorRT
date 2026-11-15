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
| prompt assembly (prefix + audio + language suffix) | matches Python exactly |
| encoder engine | bit-identical to the Python path |
| prefill engine + KV cache layout | bit-identical (cache rows compared) |
| first token | identical (86622 on the reference utterance) |
| decode step | **diverges at step 1**: 10 instead of 432, then repeats |

So the pipeline up to and including the first generated token is verified; the
per-token decode step still has a bug. The cache, embeddings and positions all
match, which leaves RMSNorm, RoPE, the attention launch, `silu_mul` and the
residual GEMV as suspects.

## Build

```powershell
# headers come from the NVIDIA TensorRT repository; import lib from the DLL
native\scripts\fetch_deps.ps1        # TODO: written once the runtime works
cmake -S native -B native/build -G Ninja
cmake --build native/build
```

Until the fetch script exists, the dependencies are:

- `native/deps/include/*.h` from `github.com/NVIDIA/TensorRT`, branch `release/11.4`
- `native/deps/lib/nvinfer_11.lib` generated with `dumpbin /exports` + `lib /def`
  from the `nvinfer_11.dll` in `tensorrt_libs`

`native/deps`, `native/build` and `native/assets` are ignored.

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
