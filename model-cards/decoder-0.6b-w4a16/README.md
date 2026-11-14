---
license: apache-2.0
base_model: Qwen/Qwen3-ASR-0.6B-hf
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
  <a href="https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-W4A16-Decoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ASR 0.6B W4A16 Decoder

## English

INT4 weight-only text-decoder weights derived from [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf). Use with a compatible [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder). The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT) integrates both parts for transcription.

### What this is, and what it is not

Weights are stored as packed INT4 (two values per byte along the contraction axis) with one FP16 scale per 32 values, symmetric, without zero points. The decode step runs on hand-written INT4 kernels; the prefill step runs as an FP16 TensorRT engine.

This is **not** an AutoAWQ, GPTQ, or ExLlamaV3 checkpoint. `transformers`, vLLM, and other runtimes cannot load it as a quantized model — only this project's runtime consumes the INT4 layout. The repository ships the folded FP16 checkpoint (`model.safetensors`); the runtime packs it into INT4 when the decoder is constructed, so loading it with plain `transformers` simply reproduces the base model.

### Accuracy and speed

All numbers are the 2,620 LibriSpeech `test-clean` utterances, greedy decoding, RTX 4060 Laptop:

| Backend | Corpus WER | Median latency |
| --- | ---: | ---: |
| Official `qwen-asr` Transformers BF16 | 2.321% | 787.1 ms |
| EXL3 4 bpw, train-clean-100 | 2.362% | 99.8 ms |
| **W4A16 AWQ, group 32** | **2.397%** | **72.2 ms** |
| EXL3 4 bpw, legacy | 2.383% | 99.5 ms |
| GGUF INT4 + Q4_K | 2.982% | 299.2 ms |

This decoder runs 1.38× faster than both EXL3 4 bpw decoders. Its corpus WER is 0.014 points above the legacy decoder and 0.035 above the calibrated one; a 20-edit difference on this corpus corresponds to about 0.02 points, so the three sit at the same level.

### Calibration

Activation-aware quantisation in the AWQ sense: a per-input-channel scale is chosen by grid search over the AWQ exponent, then a per-group clipping factor is chosen by minimising the layer output error. Scales are folded into the adjacent normalisation and projection weights, so prefill and decode stay consistent without runtime scaling.

Calibration used 128 utterances from LibriSpeech `train-clean-100`. No `test-clean` sample was used. Calibration changes quantisation statistics; it is not fine-tuning.

### Requirements

- NVIDIA GPU and CUDA, TensorRT 11.4
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT), which builds the engines and runs the decode kernels
- Build the prefill engine as FP16 — quantising prefill costs accuracy and does not make it faster
- Keep the output projection inside the prefill engine. Leaving it out shrinks the engine from 1.14 GB to 0.85 GB but changes neither the prefill time (9.4 ms) nor the decode time, and it measured 4 edits worse on `test-clean`

Windows is tested and supported. Linux should work with compatible CUDA, TensorRT, and PyTorch packages.

### Settings

| Setting | Value | Notes |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | Selects this decoder in `transcribe`, `benchmark`, and `serve`. |
| `--group` | 32 | Matches the packed weights. `64` only matches a checkpoint packed that way. |
| `--max-cache` | 4096 | Ceiling on the KV cache. The decoder allocates prompt + `--max-new-tokens` in 256-slot steps, so short audio holds 768 slots (84 MB) and a 145 s file grows to 2560 (280 MB). 4096 slots is 448 MB and about 5.2 minutes of audio. |
| `--max-seq` at build | 512 | Longest prompt accepted, about 39 seconds of audio. Raise it for longer utterances; a longer prompt raises an error. |
| output projection | inside the engine | Removing it left the 9.4 ms prefill unchanged, so the 296 MB it saves buys no speed. |

## 中文

本仓库提供基于 [Qwen/Qwen3-ASR-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 的 INT4 weight-only 文本解码器权重。使用时需配套兼容的 [TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)。[Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)集成了 encoder 和 decoder，可用于完整转写。

### 这是什么，不是什么

权重按收缩轴每字节两个值打包，每 32 个权重一个 FP16 scale，对称量化、无零点。decode 步骤由自写 INT4 kernel 完成，prefill 步骤用 FP16 TensorRT 引擎。

这**不是** AutoAWQ、GPTQ 或 ExLlamaV3 格式的 checkpoint。`transformers`、vLLM 等运行时无法把它当作量化模型加载，只有本项目的运行时能消费这种 INT4 布局。仓库提供的是折叠过缩放的 FP16 checkpoint（`model.safetensors`），运行时在构造 decoder 时才会打包成 INT4 —— 因此用普通 `transformers` 加载它只会得到原模型。

### 精度与速度

以下为 LibriSpeech `test-clean` 全部 2,620 条、贪心解码、RTX 4060 Laptop：

| 后端 | 语料 WER | 中位延迟 |
| --- | ---: | ---: |
| 官方 `qwen-asr` Transformers BF16 | 2.321% | 787.1 ms |
| EXL3 4 bpw，train-clean-100 校准 | 2.362% | 99.8 ms |
| **W4A16 AWQ，group 32** | **2.397%** | **72.2 ms** |
| EXL3 4 bpw，旧版 | 2.383% | 99.5 ms |
| GGUF INT4 + Q4_K | 2.982% | 299.2 ms |

本解码器比两个 EXL3 4 bpw 解码器都快 1.38 倍。语料 WER 比旧版高 0.014 个百分点、比校准版高 0.035 个百分点；本语料上 20 个编辑的差异约对应 0.02 个百分点，三者处于同一水平。

### 标定

AWQ 意义上的激活感知量化：先对 AWQ 指数做网格搜索得到逐输入通道缩放，再以最小化层输出误差为准则搜索逐组裁剪系数。缩放被折叠进相邻的归一化与投影权重，因此 prefill 与 decode 数值天然一致，运行时无需额外缩放。

标定使用 LibriSpeech `train-clean-100` 的 128 条音频，**未使用任何 `test-clean` 样本**。标定只改变量化统计量，不是微调。

### 环境要求

- NVIDIA GPU 与 CUDA，TensorRT 11.4
- [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)，负责构建引擎并运行解码 kernel
- prefill 引擎请用 FP16 构建 —— 量化 prefill 既不更快，还会损失精度
- 输出投影请留在 prefill 引擎内。移出后引擎从 1.14 GB 降到 0.85 GB，但 prefill 时间（9.4 ms）和 decode 时间都不变，`test-clean` 上反而多 4 个编辑

### 参数

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `--backend` | `tensorrt-w4a16` | 在 `transcribe`、`benchmark`、`serve` 中选中本解码器。 |
| `--group` | 32 | 与打包权重一致。`64` 只适用于按该分组打包的 checkpoint。 |
| `--max-cache` | 4096 | KV cache 的上限。解码器按 prompt 加 `--max-new-tokens` 以 256 槽为步长分配，所以短音频只占 768 槽（84 MB），145 秒的文件扩到 2560 槽（280 MB）。4096 槽为 448 MB，约合 5.2 分钟音频。 |
| 构建时的 `--max-seq` | 512 | 能接受的最长 prompt，约合 39 秒音频；更长的音频要调大，否则直接报错。 |
| 输出投影 | 留在引擎内 | 移出后 9.4 ms 的 prefill 不变，省下的 296 MB 换不来速度。 |
