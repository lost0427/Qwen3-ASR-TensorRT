---
license: apache-2.0
base_model: Qwen/Qwen3-ForcedAligner-0.6B
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- tensorrt
- onnx
- bf16
- fp16
- forced-alignment
---

# Qwen3-ForcedAligner 0.6B TensorRT Decoder

ONNX exports of the forced aligner text decoder from
[Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B),
prepared for TensorRT. The graph takes pre-built `inputs_embeds` (audio embeddings
already merged) and `ts_positions`, gathers the hidden states at the timestamp
positions and applies the 5,000-row classification head there, so the whole
decoder is a single forward pass with no KV cache.

Use it with the [aligner encoder export](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder)
and the [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT),
which integrates the pipeline.

## Files

| Precision | Files | Use |
| --- | --- | --- |
| BF16 | `aligner-decoder-window.bf16.onnx`, `aligner-decoder-window.bf16.onnx.data` | Reference export for GPUs with BF16 support |
| FP16 | `aligner-decoder-window.fp16.onnx`, `aligner-decoder-window.fp16.onnx.data` | Compatibility export for older NVIDIA GPUs |

The `window` profile accepts up to 1024 tokens and 256 timestamp slots, which is
36 s of audio and 128 words. TensorRT `.engine` files are not included because
they depend on the target GPU architecture and TensorRT build environment.

## Build

Engines are built locally for the target GPU. Without Python:

```powershell
build-engine.exe --onnx aligner-decoder-window.bf16.onnx --out aligner-decoder-window.bf16.engine
```

`build-engine.exe` is published in the project's `native-v*` releases and picks
the aligner decoder profile from the input names. With the Python runtime:

```powershell
uv run qwen3-asr build-aligner --preset window
```

## Requirements

- NVIDIA GPU and CUDA
- Reference environment: CUDA 12.8, TensorRT 11.4.0.106, Python 3.10–3.13
- [Qwen3-ASR-TensorRT](https://github.com/lost0427/Qwen3-ASR-TensorRT) for the
  encoder/decoder integration

Windows is tested and supported. Linux should work with compatible CUDA,
TensorRT, and PyTorch packages.

## License and citation

This export is derived from `Qwen/Qwen3-ForcedAligner-0.6B` and is published
under Apache-2.0. Retain the upstream attribution and license, and cite the
Qwen3-ASR work and the runtime project.

## 中文

本仓库提供 [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
强制对齐文本解码器的 ONNX 导出。输入是已经拼好音频嵌入的 `inputs_embeds` 和
`ts_positions`，图内只在时间戳位置取隐状态并接分类头，因此整条解码是一次前向、
没有 KV cache。

配套使用 [aligner encoder 导出](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder)
和 [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)。

`window` 档最多 1024 个 token、256 个时间戳槽位，对应 36 秒音频、128 个词。
本仓库不提供 `.engine`，因为它依赖目标 GPU 架构和 TensorRT 构建环境；请用
`build-engine.exe` 或 `qwen3-asr build-aligner --preset window` 本地构建。
