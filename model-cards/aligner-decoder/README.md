---
license: apache-2.0
base_model: Qwen/Qwen3-ForcedAligner-0.6B
library_name: exllamav3
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- qwen3-forcedaligner
- force-alignment
- exll3
- exllamav3
- quantized
- 4bpw
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ForcedAligner 0.6B EXL3 4 bpw Decoder

## English

EXL3 4 bpw text decoder for [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B). Use it with the [TensorRT aligner encoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder) and the [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT).

The aligner's classification head has 5,000 real rows. The exported decoder pads the LM head to 152,064 rows with zeros and keeps it at 16 bits, so the classification rows are not degraded by quantization. The decoder runs one full-prompt forward pass; the runtime takes the argmax at every `<timestamp>` position.

### Benchmark

On all 2,620 LibriSpeech `test-clean` utterances and 52,576 aligned items, the TensorRT BF16 encoder + EXL3 4 bpw decoder measured 39.52 ms median latency and 63.87 ms P90, versus 93.13 ms and 114.06 ms for the official Transformers BF16 backend and 139.55 ms and 182.02 ms for the GGUF INT4 + Q4_K backend. The token sequence matched on every utterance; 98.0% of TensorRT boundaries and 96.2% of GGUF boundaries are exactly equal to the official output, and the mean absolute boundary difference is 4.71 ms and 8.73 ms respectively.

The aligner decoder also runs as a plain TensorRT engine, since the aligner needs a single forward pass and no KV cache. With the TensorRT bf16 decoder in place of EXL3, the same pipeline reaches 25.06 ms median and 42.02 ms P90, with 99.2% exact boundaries and 1.50 ms mean absolute boundary difference.

![Aligner comparison](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder/resolve/main/qwen3-aligner-comparison.png)

![Aligner stage latency](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder/resolve/main/qwen3-aligner-stage-latency.png)

### Usage

```powershell
uv sync

uv run qwen3-asr align .\audio.wav `
  --text-file .\transcript.txt `
  --language English `
  --aligner-model Qwen/Qwen3-ForcedAligner-0.6B `
  --backend tensorrt-exllama `
  --decoder-dir .\decoder-exl3-4bpw `
  --format srt `
  --output .\audio.srt
```

## 中文

基于 [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的 EXL3 4 bpw 文本解码器。使用时需配套 [TensorRT aligner encoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder) 和 [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)。

aligner 的分类头只有 5,000 个真实行。导出时把 LM head 用零补齐到 152,064 行，并保持 16-bit，避免量化破坏分类头。运行时一次前向处理完整 prompt，在每个 `<timestamp>` 位置取 argmax。

在全部 2,620 条 LibriSpeech `test-clean`、52,576 个对齐项上，TensorRT BF16 encoder + EXL3 4 bpw decoder 的中位延迟为 39.52 ms，P90 为 63.87 ms；官方 Transformers BF16 后端分别为 93.13 ms 和 114.06 ms，GGUF INT4 + Q4_K 后端分别为 139.55 ms 和 186.88 ms。所有音频的 token 序列一致；98.0% 的 TensorRT 边界和 96.2% 的 GGUF 边界与官方输出完全相同，平均绝对边界差分别为 4.71 ms 和 8.73 ms。

这些权重基于 `Qwen/Qwen3-ForcedAligner-0.6B`，以 Apache-2.0 发布。请保留上游归属和许可证说明。
