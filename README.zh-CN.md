# Qwen3-ASR TensorRT

面向 Windows、基于 TensorRT 音频编码器和两种文本解码器（ExLlamaV3 EXL3，或完全跑在 TensorRT 上的 INT4 weight-only W4A16 解码器）的 Qwen3-ASR 0.6B 与 1.7B 推理项目。本仓库集成 encoder 和 decoder，并提供 Python 命令行与 HTTP API。

[English](README.md) | [简体中文](README.zh-CN.md) | [模型文件说明](model-cards/)

## 运行环境

- Windows 10/11、NVIDIA GPU、CUDA 12.8
- Python 3.10–3.13 和 `uv`
- TensorRT 11.4.0.106
- 兼容的 PyTorch 和 ExLlamaV3 版本

Linux 环境在 CUDA、TensorRT、PyTorch 和 ExLlamaV3 软件包兼容时也可运行。

安装项目依赖：

```powershell
uv sync
```

## 模型

项目支持 [`Qwen/Qwen3-ASR-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-0.6B-hf) 和 [`Qwen/Qwen3-ASR-1.7B-hf`](https://huggingface.co/Qwen/Qwen3-ASR-1.7B-hf)，提供 BF16、FP16 TensorRT encoder、本地可选的 INT8 Q/DQ encoder 校准、EXL3 3/4 bpw decoder，以及 TensorRT W4A16 decoder（INT4 权重、FP16 激活、AWQ 缩放）。

已发布的模型文件：

- [TensorRT encoder：BF16 和 FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-TensorRT-Encoder)
- [EXL3 decoder：legacy 3 bpw 和 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder)
- [1.7B EXL3 3 bpw：使用 train-clean-100 重新校准](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100)（同一 decoder 仓库中的独立目录）
- [1.7B EXL3 4 bpw：使用 train-clean-100 重新校准](https://huggingface.co/Retrieve0/Qwen3-ASR-1.7B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100)（同一 decoder 仓库中的独立目录）
- [0.6B TensorRT encoder：BF16 和 FP16 ONNX](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-TensorRT-Encoder)
- [0.6B EXL3 decoder：legacy 3 bpw 和 4 bpw](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder)
- [0.6B EXL3 3 bpw：使用 train-clean-100 重新校准](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-3bpw-train-clean-100)（同一 decoder 仓库中的独立目录）
- [0.6B EXL3 4 bpw：使用 train-clean-100 重新校准](https://huggingface.co/Retrieve0/Qwen3-ASR-0.6B-EXL3-Decoder/tree/main/decoder-exl3-4bpw-train-clean-100)（同一 decoder 仓库中的独立目录）
- [Qwen3-ForcedAligner 0.6B TensorRT encoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-TensorRT-Encoder)
- [Qwen3-ForcedAligner 0.6B EXL3 4 bpw decoder](https://huggingface.co/Retrieve0/Qwen3-ForcedAligner-0.6B-EXL3-Decoder)

legacy 3 bpw 和 4 bpw 使用 ExLlamaV3 默认校准数据量化。`train-clean-100` 版本使用 LibriSpeech `train-clean-100` 校准，针对 LibriSpeech 风格的英语朗读调优；通用音频和中文 ASR 仍默认推荐 legacy 版本。

## 构建加速后端

构建 BF16 encoder engine 并提取文本 decoder：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build
```

为不支持 BF16 的 GPU 构建 FP16 encoder：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision fp16
```

在本地构建实验性 INT8 encoder（不包含在已发布模型文件中）：

```powershell
uv sync --extra quantization
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder .\calibration-audio --max-audio-files 32
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int8
```

实验性的 TensorRT INT4 权重专用 encoder 导出会按 64 个值一组量化符合条件的 MatMul 权重；卷积和激活仍使用 FP16。先导出 ONNX，再构建 engine：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize-encoder-int4
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf build --encoder-precision int4
```

准备本地 LibriSpeech `train-clean-100` 语料和 Qwen3-ASR 模型后，先生成 200 行校准数据，再显式使用训练数据进行量化。省略 `--cal-data` 时使用 ExLlamaV3 默认校准数据，legacy 版本就是这样生成的：

```powershell
uv run python tools\prepare_exl3_calibration.py .\LibriSpeech\train-clean-100 .\models\Qwen3-ASR-1.7B-hf .\artifacts\train-clean-100.safetensors --rows 200 --columns 2048
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 3 --cal-data .\artifacts\train-clean-100.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-3bpw-train-clean-100
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf quantize --bits 4 --cal-data .\artifacts\train-clean-100.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-1.7b/decoder-exl3-4bpw-train-clean-100
```

生成单独发布的 0.6B 版本时，使用 0.6B checkpoint 生成校准数据，并将权重输出到独立目录：

```powershell
uv run python tools\prepare_exl3_calibration.py .\LibriSpeech\train-clean-100 .\models\Qwen3-ASR-0.6B-hf .\artifacts\train-clean-100-0.6b.safetensors --rows 200 --columns 2048
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize --bits 3 --cal-data .\artifacts\train-clean-100-0.6b.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-3bpw-train-clean-100
uv run qwen3-asr --model Qwen/Qwen3-ASR-0.6B-hf quantize --bits 4 --cal-data .\artifacts\train-clean-100-0.6b.safetensors --cal-rows 200 --output-dir artifacts/qwen3-asr-0.6b/decoder-exl3-4bpw-train-clean-100
```

### W4A16 decoder

另一条 decoder 路线把 4-bit 权重按 32 个权重一组存放，decode 步骤用自写的 INT4 kernel 完成，全程不反量化到 FP16。量化是激活感知的（AWQ 缩放 + 逐组裁剪），逐通道缩放被折叠进相邻的归一化和投影权重里，因此 prefill 与 decode 数值天然一致，运行时不需要额外缩放。

```powershell
# 用 train-clean-100 的 128 条音频做激活感知量化
uv run python tools\quantize_awq.py --group 32 --utterances 128 ^
  --manifest .\train-clean-100.manifest.jsonl --audio-root .\LibriSpeech ^
  --output models\Qwen3-ASR-0.6B-hf-awq32 --report artifacts\awq32-report.json
```

得到的 checkpoint 是折叠过缩放的普通 FP16 模型，用 `transformers` 加载即可复现原模型；真正的 INT4 打包发生在构造 `W4A16ASRDecoder` 时。0.6B 的 prefill 引擎用 FP16 构建、并把输出投影留在引擎里：在 2,620 条语料上，它比 INT4 prefill 的 WER 好 0.089 个百分点、快 1.7 ms。把这个 0.6B 引擎的投影移出去，prefill 时间仍是 9.4 ms，但多 4 个编辑。

1.7B 则相反：把输出投影放进引擎时，FP16 prefill 引擎为 3.29 GB，在 8 GB 显存下与 decoder 权重放在一起只剩 1.24 GB 空闲；在这个水平上 TensorRT 每一步都要重读权重页，全量语料 prefill 中位数为 222 ms。用 `build-decoder-prefill`（不加 `--keep-lm-head`）把投影移出引擎后，引擎为 2.69 GB，prefill 为 20.6 ms，输出投影改由 decode 循环本来就在用的 INT4 `lm_head` GEMV 完成。在 2,620 条语料上 WER 为 1.9876%，与把投影放在引擎里时完全一致，比 INT4 prefill 引擎好 0.015 个百分点。

```powershell
uv run qwen3-asr --model .\models\Qwen3-ASR-1.7B-hf-awq32 build-decoder-prefill --max-seq 512 --opt-seq 256 `
  --output-dir artifacts\qwen3-asr-1.7b\decoder-prefill
uv run qwen3-asr --model .\models\Qwen3-ASR-1.7B-hf-awq32 transcribe .\audio.wav --backend tensorrt-w4a16 `
  --engine artifacts\qwen3-asr-1.7b\encoder-window.bf16.engine `
  --decoder-engine artifacts\qwen3-asr-1.7b\decoder-prefill\asr-prefill.fp16.engine
```

`--max-seq` 是 prefill 引擎能接受的最长 prompt，必须覆盖整条音频：音频 token 占满 prompt，约为每秒音频 13 个 token，默认的 512 覆盖约 39 秒。更长的 profile 会在运行时按序列长度线性占用显存；`--max-cache`（默认 2048）限制 KV cache；超过 profile 的 prompt 会直接报错，不会被截断。作为参照，Transformers checkpoint 支持 65,536 个位置。

#### Decoder 参数

`transcribe`、`benchmark`、`serve` 共用的运行期参数：

| 参数 | 适用后端 | 默认值 | 作用 |
| --- | --- | ---: | --- |
| `--max-cache` | `tensorrt-w4a16` | 4096 | KV cache 的**上限**，不是预留量。解码器按每个请求实际需要的量分配 —— prompt 加 `--max-new-tokens`，向上取整到 256 个槽位 —— 遇到更长的请求时原地扩容，所以 30 秒的文件只用 768 个槽位（84 MB），145 秒的文件扩到 2560 个（280 MB）。两个尺寸每个槽位都是 112 KB（28 层 × 8 个 KV head × 128 × key/value 两份 × FP16），4096 个槽位为 448 MB，约合 5.2 分钟音频。 |
| `--group` | `tensorrt-w4a16` | 32 | checkpoint 的 INT4 打包分组。已发布的 AWQ 权重是 group 32；`64` 只适用于按该分组打包的 checkpoint。 |
| `--decoder-engine` | `tensorrt-w4a16` | `artifacts/<model>/decoder-prefill/asr-prefill.fp16.engine` | prefill 引擎。输出 `logits` 的引擎和输出 norm 后 `hidden` 的引擎都能加载。 |
| `--max-context-tokens` | `tensorrt-exllama` | 8192 | EXL3 路径的 KV 预算。 |
| `--engine` | 所有 TensorRT 后端 | `artifacts/<model>/encoder-window.bf16.engine` | 音频编码器引擎。 |
| `--max-new-tokens` | 全部 | 256 | decode 循环上限。 |

`build-decoder-prefill` 的构建参数：

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--precision` | `fp16` | 引擎精度。 |
| `--max-seq` | 512 | 引擎能接受的最长 prompt，约合 39 秒音频。更长的音频要调大。 |
| `--opt-seq` | 256 | TensorRT 用于挑选 tactic 的序列长度，应接近实际负载的常见 prompt 长度。 |
| `--workspace-gib` | 1 | 构建期 workspace。 |
| `--keep-lm-head` | 关闭 | 把输出投影留在引擎内。关闭时改由运行时的 INT4 GEMV 完成，0.6B 省 296 MB、1.7B 省 593 MB。 |

`qwen3-asr build --max-frames N` 决定 encoder 引擎能接受的最长 mel 输入，单位是 10 ms 一帧：默认 3600 即 36 秒。

8 GB 卡上的实测配置：

| 模型 | `--max-seq` | `--keep-lm-head` | `--max-cache` | prefill 引擎 | 结果 |
| --- | ---: | --- | ---: | ---: | --- |
| 0.6B | 512 | 是 | 2048 | 1142 MB | prefill 9.4 ms，中位 72.2 ms，WER 2.397% |
| 1.7B | 512 | 否 | 2048 | 2694 MB | prefill 20.6 ms，中位 175.0 ms，WER 1.988% |
| 1.7B，长音频 | 2048 | 否 | 4096 | 2694 MB | 145 秒文件一次转完，共 14.9 秒 |

这个选择取决于显存，而不是模型大小：卡上还有余量就把投影留在引擎里，引擎加 decoder 权重之后余量太少才移出去。8 GB 卡上，1.7B 把投影放进引擎后只剩 1.24 GB，prefill 变成 222 ms；0.6B 的引擎只有 1.14 GB，放进去没有任何可测代价。

#### 更长的音频

一次调用能吃多长的音频由三个上限决定，而在已发布的引擎上卡住的是 encoder：

| 上限 | 参数 | 已发布值 | 音频长度 |
| --- | --- | ---: | ---: |
| encoder 输入 | `qwen3-asr build --max-frames` | 3600 帧 | 36 秒 |
| decoder prefill | `build-decoder-prefill --max-seq` | 512 token | 39 秒 |
| KV cache | `transcribe --max-cache` | 上限 4096 | 5.2 分钟 |

三者都是显存问题，不是算力问题。145 秒的文件需要 14,500 个 mel 帧和 1,903 个 token 的 prompt，因此在 decoder 之前就先被 encoder 挡住；用 `--max-frames 16000` 重建 encoder 得到的是 610 MB 的引擎，和 36 秒版本一样大，之后这个文件可以一次转完：总共 14.9 秒 —— 预处理 1.0 秒、encoder 0.17 秒、prefill 8.0 秒、解码 477 个 token 用 5.6 秒。这里 prefill 占大头，因为 attention 随 prompt 长度平方增长，而那时只剩约 0.5 GB 空闲显存。两个上限溢出时都会报错，并指出该改哪个参数、改成多少。

decoder 这一侧的代价几乎不在引擎体积上 —— 512、1024、2048 构建出的 1.7B 引擎都是 2693.9 MB —— 但 TensorRT 按最大 profile 形状分配激活缓冲，所以进程加载时会多占显存。1.7B 在构造完成后：512 profile 剩 990 MB、1024 剩 912 MB、2048 剩 752 MB；同样 100 条上 prefill 分别为 22.7、24.8、24.3 ms，转写结果完全相同。

KV cache 按请求分配，所以进程实际占用的不是上限值：

| 实际分配的槽位 | 显存 | 音频长度 |
| --- | ---: | ---: |
| 768 | 84 MB | 59 秒 |
| 2048 | 224 MB | 157 秒 |
| 2560 | 280 MB | 197 秒 |
| 4096 | 448 MB | 5.2 分钟 |
| 8192 | 896 MB | 10.5 分钟 |

30 秒的文件按 `prompt + 256` 申请、取整到 768 个槽位；145 秒的文件把缓冲原地扩到 2560，扩容时复制已写入的前缀，并为新尺寸重捕一次 CUDA Graph。超过上限的请求会报错并给出应设的数值。长请求之后不会释放，所以一个会话最终稳定在它用过的最大容量上。

剩下的限制只有 prefill profile。**分块 prefill** 可以去掉它：现在的 prefill 图不接受 cache 输入，这才是 `--max-seq` 成为硬上限的原因。像 decode 图那样给它 `cache_k` / `cache_v` / `cache_len`（mask 让前缀全部可见、块内因果），prompt 就能按块送达、KV cache 逐块累积。算术与一次性 prefill 相同，所以 WER 的差异只来自 fp16 归约顺序和 TensorRT 针对不同序列长度挑选的 tactic，而不是模型本身；它同时把长 prompt 的平方级 attention 开销限制在块大小内。

绝对上限和官方后端一样：`max_position_embeddings` 是 65,536，而 65,536 个 token 的 KV 约 7 GB，8 GB 卡放不下权重再加它。放不下的音频只能按静音切段分段转写，代价是失去单次推理保留的跨段上下文。

音频编码器按参考实现的推理窗口（`n_window_infer`，104 个 token）做分段注意力，不再对整段音频做全局注意力。其输出在 1–5 个窗口上与 Transformers 参考实现的余弦相似度为 0.9995–0.9999；corpus WER 在各解码器上的变化为 ±0.04 个百分点，在本语料上相当于 20 个编辑。


TensorRT engine 会根据目标 GPU 和本机 TensorRT 环境在本地构建。

构建强制对齐器 encoder，并导出两种 decoder：

```powershell
# TensorRT encoder + TensorRT bf16/fp16 decoder
uv run qwen3-asr build-aligner

# 可选：EXL3 4 bpw decoder
uv run qwen3-asr --model Qwen/Qwen3-ForcedAligner-0.6B quantize --bits 4 --head-bits 16
```

## 转写音频

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf transcribe .\audio.wav --backend tensorrt-exllama --language Chinese
```

## 对齐音频

```powershell
uv run qwen3-asr align .\audio.wav --text-file .\transcript.txt --language English --format srt --output .\audio.srt
```

`align` 默认使用 TensorRT encoder 和 TensorRT bf16 aligner decoder，支持 `--format json`、`--format srt`、`--format txt`；也可以用 `--text` 直接传入文本。`--decoder-precision fp16` 切换 fp16 engine（约快 8%，encoder 自动跟随），`--backend tensorrt-exllama` 切换到 EXL3 decoder，`--engine` / `--decoder-engine` 用于指定具体构建产物。


启动 HTTP API：

```powershell
uv run qwen3-asr --model Qwen/Qwen3-ASR-1.7B-hf serve --backend tensorrt-exllama
```

任何编程语言都可以通过 HTTP 提交音频数据：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/v1/transcribe?language=Chinese" `
  -H "Content-Type: audio/wav" --data-binary "@audio.wav"
```

API 默认只监听本机地址，支持 WAV、MP3、FLAC、OGG、M4A 和 AAC 音频。

## 基准测试

LibriSpeech `test-clean` 全量 2,620 条音频、53,120 个参考词。延迟为全量中位数；30–200 条的子集因前若干条要付进程与频率爬升的代价，会高 20–40%。Corpus WER 是确定性的、可跨 session 复现；延迟只在同一 session 内可比。EXL3 4 bpw 各行与 0.6B 的 W4A16 行于 2026-11-12 使用窗口化编码器测量，1.7B 的 W4A16 行于 2026-11-14 测量；把 2026-11-12 的一项在后者重测，中位数从 208.8 ms 变为 211.5 ms。3 bpw 与 GGUF 各行来自 2026-11-11 的运行。

| 模型 | 后端 | Corpus WER | 中位延迟 ms |
| --- | --- | ---: | ---: |
| 0.6B | 官方 Transformers BF16 | 2.321% | 787.1 |
| 0.6B | GGUF INT4 + Q4_K | 2.982% | 299.2 |
| 0.6B | TensorRT fp16 + W4A16 AWQ (g32) | 2.397% | **72.2** |
| 0.6B | TensorRT BF16 + EXL3 4 bpw | 2.383% | 99.5 |
| 0.6B | TensorRT BF16 + EXL3 4 bpw，train-clean-100 | **2.362%** | 99.8 |
| 1.7B | 官方 Transformers BF16 | 1.886% | 863.5 |
| 1.7B | GGUF INT4 + Q4_K | 2.090% | 391.8 |
| 1.7B | TensorRT fp16 prefill + W4A16 AWQ (g32) | 1.988% | **175.0** |
| 1.7B | TensorRT BF16 + EXL3 4 bpw | 1.984% | 208.8 |
| 1.7B | TensorRT BF16 + EXL3 4 bpw，train-clean-100 | **1.944%** | 208.1 |

![0.6B 校准对比与 GGUF、官方后端](docs/windows-librispeech-qwen3-0.6b-calibration-vs-gguf.png)

![0.6B 分阶段中位延迟](docs/windows-librispeech-qwen3-0.6b-stage-latency.png)

![1.7B 校准对比与 GGUF、官方后端](docs/windows-librispeech-qwen3-1.7b-calibration-vs-gguf.png)

![1.7B 分阶段中位延迟](docs/windows-librispeech-qwen3-1.7b-stage-latency.png)

强制对齐在同样语料上：TensorRT bf16 aligner decoder 25.06 ms 中位延迟，EXL3 4 bpw aligner decoder 39.52 ms；官方后端 93.13 ms，GGUF 139.55 ms。TensorRT 的边界有 99.2% 与官方完全相同。

![对齐器对比](docs/windows-librispeech-qwen3-aligner-comparison.png)

![对齐器分阶段中位延迟](docs/windows-librispeech-qwen3-aligner-stage-latency.png)