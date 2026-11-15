---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-1.7B-hf
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- w4a16
- awq
- int4
- quantized
- tensorrt
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-W4A16-Decoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 1.7B W4A16 Decoder

## English

INT4 weight-only text-decoder weights derived from [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf). Use with a compatible [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder). The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) integrates both parts for transcription.

### What this is, and what it is not

Weights are stored as packed INT4 (two values per byte along the contraction axis) with one FP16 scale per 32 values, symmetric, without zero points. The decode step runs on hand-written INT4 kernels.

This is **not** an AutoAWQ, GPTQ, or ExLlamaV3 checkpoint. `transformers`, vLLM, and other runtimes cannot load it as a quantized model — only this project's runtime consumes the INT4 layout. The repository ships the folded FP16 checkpoint (`model.safetensors`); the runtime packs it into INT4 when the decoder is constructed, so loading it with plain `transformers` simply reproduces the base model.

### Accuracy and speed

All numbers are the 2,620 LibriSpeech `test-clean` utterances, greedy decoding, RTX 4060 Laptop:

| Backend | Corpus WER | Median latency |
| --- | ---: | ---: |
| Official `qwen-asr` Transformers BF16 | 1.886% | 863.5 ms |
| EXL3 4 bpw, train-clean-100 | 1.944% | 208.1 ms |
| EXL3 4 bpw, legacy | 1.984% | 208.8 ms |
| **W4A16 AWQ, group 32** | **1.988%** | **175.0 ms** |
| GGUF INT4 + Q4_K | 2.090% | 391.8 ms |

This decoder runs 1.19× faster than both EXL3 4 bpw decoders, 2.24× faster than GGUF, and 4.93× faster than the official backend. Its corpus WER is 0.004 points above the legacy EXL3 decoder and 0.044 above the calibrated one; a 20-edit difference on this corpus corresponds to about 0.02 points.

This decoder was measured on 2026-11-14, the EXL3 rows on 2026-11-12, and the GGUF and official rows on 2026-11-11. Re-running the legacy EXL3 decoder on 2026-11-14 measured 211.5 ms against its 208.8 ms from 2026-11-12.

### Prefill

Build the prefill engine as FP16 **without** the output projection. With the projection inside the engine is 3.29 GB, so on an 8 GB card TensorRT re-reads its weight pages every step and the full-corpus prefill median is 222.0 ms. Without it the engine is 2.69 GB and prefill is 20.6 ms, and the output projection moves to the INT4 `lm_head` GEMV that the decode loop already uses. Corpus WER is unchanged against the engine with the projection inside.

`--max-seq` sets the longest prompt the engine accepts and has to cover the whole utterance at roughly 13 tokens per second of audio; 512 covers about 39 seconds. The encoder engine has its own limit — 3600 mel frames, 36 seconds — raised with `qwen3-asr build --max-frames`. Both have to cover the audio before a long file transcodes in one pass; either overrun names the knob to rebuild with.

### Calibration

Activation-aware quantisation in the AWQ sense: a per-input-channel scale is chosen by grid search over the AWQ exponent, then a per-group clipping factor is chosen by minimising the layer output error. Scales are folded into the adjacent normalisation and projection weights.

Calibration used 128 utterances from LibriSpeech `train-clean-100`. No `test-clean` sample was used. Calibration changes quantisation statistics; it is not fine-tuning.

### Requirements

- NVIDIA GPU and CUDA, TensorRT 11.4
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which builds the engines and runs the decode kernels

### Settings

| Setting | Value | Notes |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | Selects this decoder in `transcribe`, `benchmark`, and `serve`. |
| `--group` | 32 | Packing granularity of the INT4 kernels. The checkpoint ships as FP16 and is packed at load time, so `64` runs too but is coarser. All published numbers use 32. |
| `--max-cache` | 4096 | Ceiling on the KV cache. The decoder allocates prompt + `--max-new-tokens` in 256-slot steps, so short audio holds 768 slots (84 MB) and a 145 s file grows to 2560 (280 MB). 4096 slots is 448 MB and about 5.2 minutes of audio. |
| `--max-seq` at build | 512 | Longest prompt accepted, about 39 seconds of audio. Raise it for longer utterances. |
| output projection | outside the engine | The engine holds 2.69 GB instead of 3.29 GB, and prefill is 20.6 ms instead of 222 ms. Corpus WER is unchanged. |

## 中文

本仓库提供基于 [Qwen/Qwen3-ASR-1.7B-hf](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf) 的 INT4 weight-only 文本解码器权重。使用时需配套兼容的 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)集成了 encoder 和 decoder，可用于完整转写。

### 这是什么，不是什么

权重按收缩轴每字节两个值打包，每 32 个权重一个 FP16 scale，对称量化、无零点。decode 步骤由自写 INT4 kernel 完成。

这**不是** AutoAWQ、GPTQ 或 ExLlamaV3 格式的 checkpoint。`transformers`、vLLM 等运行时无法把它当作量化模型加载，只有本项目的运行时能消费这种 INT4 布局。仓库提供的是折叠过缩放的 FP16 checkpoint（`model.safetensors`），运行时在构造 decoder 时才会打包成 INT4 —— 因此用普通 `transformers` 加载它只会得到原模型。

### 精度与速度

以下为 LibriSpeech `test-clean` 全部 2,620 条、贪心解码、RTX 4060 Laptop：

| 后端 | 语料 WER | 中位延迟 |
| --- | ---: | ---: |
| 官方 `qwen-asr` Transformers BF16 | 1.886% | 863.5 ms |
| EXL3 4 bpw，train-clean-100 校准 | 1.944% | 208.1 ms |
| EXL3 4 bpw，旧版 | 1.984% | 208.8 ms |
| **W4A16 AWQ，group 32** | **1.988%** | **175.0 ms** |
| GGUF INT4 + Q4_K | 2.090% | 391.8 ms |

本解码器比两个 EXL3 4 bpw 解码器都快 1.19 倍、比 GGUF 快 2.24 倍、比官方后端快 4.93 倍。语料 WER 比旧版 EXL3 高 0.004 个百分点、比校准版高 0.044 个百分点；本语料上 20 个编辑的差异约对应 0.02 个百分点。

本解码器于 2026-11-14 测量，EXL3 各行于 2026-11-12，GGUF 与官方后端子集于 2026-11-11。把旧版 EXL3 解码器在 2026-11-14 重测，中位数为 211.5 ms，对应的 2026-11-12 值为 208.8 ms。

### prefill

这个模型请用 FP16 构建 prefill 引擎，并且**不带**输出投影。把投影放进引擎时引擎为 3.29 GB，在 8 GB 显存下 TensorRT 每一步都要重读权重页，全量语料 prefill 中位数为 222.0 ms。移出投影后引擎为 2.69 GB，prefill 为 20.6 ms，输出投影改由 decode 循环本来就在用的 INT4 `lm_head` GEMV 完成。语料 WER 与把投影放在引擎里时一致。

`--max-seq` 决定引擎能接受的最长 prompt，必须覆盖整条音频；按每秒音频约 13 个 token 计，512 约合 39 秒。encoder 引擎另有 3600 个 mel 帧（36 秒）的上限，用 `qwen3-asr build --max-frames` 调整。两者都覆盖住音频，长文件才能一次转完；任一上限溢出都会指出该重建哪个参数。

### 标定

AWQ 意义上的激活感知量化：先对 AWQ 指数做网格搜索得到逐输入通道缩放，再以最小化层输出误差为准则搜索逐组裁剪系数。缩放被折叠进相邻的归一化与投影权重。

标定使用 LibriSpeech `train-clean-100` 的 128 条音频，**未使用任何 `test-clean` 样本**。标定只改变量化统计量，不是微调。

### 环境要求

- NVIDIA GPU 与 CUDA，TensorRT 11.4
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，负责构建引擎并运行解码 kernel

### 参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | 在 `transcribe`、`benchmark`、`serve` 中选中本解码器。 |
| `--group` | 32 | INT4 kernel 的打包粒度。checkpoint 以 FP16 发布、加载时打包，所以 `64` 也能跑，但更粗。所有已发布数字都用 32。 |
| `--max-cache` | 4096 | KV cache 的上限。解码器按 prompt 加 `--max-new-tokens` 以 256 槽为步长分配，所以短音频只占 768 槽（84 MB），145 秒的文件扩到 2560 槽（280 MB）。4096 槽为 448 MB，约合 5.2 分钟音频。 |
| 构建时的 `--max-seq` | 512 | 能接受的最长 prompt，约合 39 秒音频；更长的音频要调大。 |
| 输出投影 | 放在引擎外 | 引擎为 2.69 GB 而非 3.29 GB，prefill 为 20.6 ms 而非 222 ms，语料 WER 不变。 |
