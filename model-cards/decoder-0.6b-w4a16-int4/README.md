---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-0.6B-hf
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- w4a16
- awq
- int4
- packed
- tensorrt
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder"><img src="https://img.shields.io/badge/Hugging%20Face-FP16%20weights-yellow?logo=huggingface" alt="Hugging Face FP16 weights"></a>
</p>

# Qwen3-ASR 0.6B W4A16 Decoder (packed INT4)

## English

Decoder-only weights for [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf), stored with the INT4 packing already applied. Use with a compatible [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder); the [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) runs both stages.

### What is in here

`model.safetensors` holds, for every projection, the INT4 nibbles (`*.packed`, two values per byte along the contraction axis, low nibble first) and one FP16 scale per 32 values (`*.scales`, `max|w| / 7`). `embed_tokens` and the RMSNorm weights stay FP16. `packed_config.json` records the format, the group size, and the quantisation rule.

The audio tower is not included: transcription uses the TensorRT encoder instead.

This is the same quantisation the runtime performs when it packs the [FP16 checkpoint](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder) at load time, produced by the same kernel, so the transcripts are bit-identical. The only difference is what sits on disk: 0.61 GB instead of 1.47 GB.

### What it is not

`asr-prefill.onnx` (with its `.onnx.data`) is included so the prefill engine can be built without the FP16
checkpoint: `qwen3-asr --model <this repo> build-decoder-prefill --onnx asr-prefill.onnx`.

Not an AutoAWQ, GPTQ, or ExLlamaV3 checkpoint, and not loadable by `transformers`, vLLM, or llama.cpp as a quantized model. Only this project's runtime understands the layout. There are no FP16 weights here, so it cannot be re-quantised with a different group size — download the FP16 checkpoint for that.

### Accuracy and speed

Corpus WER and latency are identical to the FP16 checkpoint, because the weights are identical after packing: 2.397% and 72.2 ms on the 2,620 LibriSpeech `test-clean` utterances. The bench table lives in the [FP16 repository](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder).

### Use

```powershell
uv run qwen3-asr --model Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder-INT4 transcribe .\audio.wav --backend tensorrt-w4a16 `
  --engine artifacts\qwen3-asr-0.6b\encoder-window.bf16.engine `
  --decoder-engine artifacts\qwen3-asr-0.6b\decoder-prefill\asr-prefill.fp16.engine
```

The loader reads `packed_config.json` and needs no FP16 weights. `--group` is fixed at 32 here; passing another value raises an error.

### Settings

| Setting | Value | Notes |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | Selects this decoder. |
| `--group` | 32 (fixed) | Recorded in `packed_config.json`. |
| `--max-cache` | 4096 | Ceiling on the KV cache; allocated per request in 256-slot steps. |
| `--max-seq` at build | 512 | Longest prompt accepted, about 39 seconds of audio. The encoder engine has its own 3600-frame ceiling, raised with `qwen3-asr build --max-frames`. |

### Requirements

- NVIDIA GPU and CUDA, TensorRT 11.4
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which builds the engines and runs the decode kernels

## 中文

基于 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 的 decoder 权重，INT4 打包已完成。需配套 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)；完整转写由 [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)完成。

### 里面是什么

`model.safetensors` 为每个投影权重保存 INT4 半字节（`*.packed`，沿收缩轴每字节两个值，低半字节在前）和每 32 个权重一个 FP16 scale（`*.scales`，`max|w| / 7`）。`embed_tokens` 与 RMSNorm 权重仍是 FP16。`packed_config.json` 记录格式、分组大小和量化规则。

**不含 audio tower** —— 转写使用 TensorRT encoder。

这份权重就是运行时加载 [FP16 checkpoint](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder) 时打包出来的结果，由同一个 kernel 生成，因此转写结果逐位一致。唯一的区别是磁盘占用：0.61 GB 对 1.47 GB。

### 不是什么

仓库内含 `asr-prefill.onnx`（及其 `.onnx.data`），因此在没有 FP16 checkpoint 的情况下也能构建 prefill 引擎：
`qwen3-asr --model <本仓库> build-decoder-prefill --onnx asr-prefill.onnx`。

不是 AutoAWQ、GPTQ 或 ExLlamaV3 格式，`transformers`、vLLM、llama.cpp 都无法把它当作量化模型加载，只有本项目的运行时能读这种布局。这里没有 FP16 权重，因此无法改用其它分组重新量化 —— 需要那样做请下载 FP16 checkpoint。

### 精度与速度

语料 WER 与延迟和 FP16 checkpoint 完全相同（打包后权重一致）：在 LibriSpeech `test-clean` 全量 2,620 条上为 2.397% 与 72.2 ms。基准表见 [FP16 仓库](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder)。

### 使用

```powershell
uv run qwen3-asr --model Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder-INT4 transcribe .\audio.wav --backend tensorrt-w4a16 `
  --engine artifacts\qwen3-asr-0.6b\encoder-window.bf16.engine `
  --decoder-engine artifacts\qwen3-asr-0.6b\decoder-prefill\asr-prefill.fp16.engine
```

加载器读取 `packed_config.json`，不需要 FP16 权重。此处的 `--group` 固定为 32，传其它值会报错。

### 参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | 选中本解码器。 |
| `--group` | 32（固定） | 记录在 `packed_config.json` 中。 |
| `--max-cache` | 4096 | KV cache 上限，按请求以 256 槽为步长分配。 |
| 构建时的 `--max-seq` | 512 | 能接受的最长 prompt，约合 39 秒音频。encoder 引擎另有 3600 帧上限，用 `qwen3-asr build --max-frames` 调整。 |

### 环境要求

- NVIDIA GPU 与 CUDA，TensorRT 11.4
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，负责构建引擎并运行解码 kernel
