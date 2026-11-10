---
license: apache-2.0
base_model: Qwen/Qwen3-ForcedAligner-0.6B
library_name: onnxruntime
pipeline_tag: automatic-speech-recognition
tags:
- qwen3-asr
- qwen3-forcedaligner
- force-alignment
- tensorrt
- onnx
- bf16
---

<p align="center">
  <a href="https://github.com/lost0427/Qwen3-ASR-TensorRT"><img src="https://img.shields.io/badge/GitHub-source-181717?logo=github" alt="GitHub source"></a>
  <a href="https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder"><img src="https://img.shields.io/badge/Hugging_Face-model-yellow?logo=huggingface" alt="Hugging Face model"></a>
</p>

# Qwen3-ForcedAligner 0.6B TensorRT Encoder

## English

ONNX export of the audio encoder from [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B), prepared for TensorRT inference. Use it with the [EXL3 aligner decoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder) and the [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT).

### Files

| Precision | Files | Use |
| --- | --- | --- |
| BF16 | `aligner-encoder-window.bf16.onnx`, `aligner-encoder-window.bf16.onnx.data` | Reference export for GPUs with BF16 support |

The export applies the aligner encoder's `proj1/act/proj2` output projection inside the graph. TensorRT `.engine` files are built locally for the target GPU.

### Requirements

- NVIDIA GPU and CUDA
- Reference environment: CUDA 12.8, TensorRT 11.4.0.106, PyTorch 2.10.0
- The [Qwen3-ASR-TensorRT repository](https://github.com/lost0427/Qwen3-ASR-TensorRT)

### Build

```powershell
uv sync
uv run qwen3-asr build-aligner
```

## 中文

基于 [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) 的音频编码器 ONNX 导出，用于 TensorRT 推理。使用时需配套 [EXL3 aligner decoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder) 和 [Qwen3-ASR-TensorRT 仓库](https://github.com/lost0427/Qwen3-ASR-TensorRT)。

导出已经把 aligner encoder 的 `proj1/act/proj2` 输出投影包含在图内。TensorRT `.engine` 需在目标 GPU 上本地构建。

```powershell
uv sync
uv run qwen3-asr build-aligner
```

这些权重基于 `Qwen/Qwen3-ForcedAligner-0.6B`，以 Apache-2.0 发布。请保留上游归属和许可证说明。
