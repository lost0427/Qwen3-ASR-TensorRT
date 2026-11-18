"""Quantize the draft decoder's prefill graph to INT8 (Q/DQ) for a smaller engine.

The speculative draft needs a prefill engine of its own, and on an 8 GB card that
engine is what pushes the pair out of memory. Quantizing it only changes the
draft's KV cache and its first token, both of which the target re-verifies, so it
cannot change the transcript: the only risk is a lower draft acceptance rate.

    python tools/quantize_prefill.py --model models/Qwen3-ASR-0.6B-hf \\
        --packed artifacts/qwen3-asr-0.6b/decoder-int4-g32 \\
        --encoder artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine \\
        --work-dir artifacts/experiments/draft-prefill-int8 --calibration 48
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_decoder import calibration_files  # noqa: E402

DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))


class PromptReader:
    """Feeds the prefill graph the prompt distribution it sees at run time."""

    def __init__(self, samples: list[dict], input_names: list[str]) -> None:
        self.samples = samples
        self.input_names = input_names
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.samples):
            return None
        sample = self.samples[self.index]
        self.index += 1
        return {name: sample[name] for name in self.input_names if name in sample}

    def rewind(self) -> None:
        self.index = 0


def build_samples(model_dir: Path, packed: Path, encoder_path: Path, count: int, seed: int,
                  audio_token: int) -> list[dict]:
    from safetensors import safe_open
    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    processor = AutoProcessor.from_pretrained(str(model_dir))
    encoder = TensorRTAudioEncoder(encoder_path)
    embed = None
    with safe_open(str(packed / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            if key.endswith("embed_tokens.weight") or key == "embed_tokens.weight":
                embed = handle.get_tensor(key)
                break
    if embed is None:
        raise SystemExit(f"no embedding table in {packed / 'model.safetensors'}")

    samples = []
    for path in calibration_files(count, seed):
        inputs = processor.apply_transcription_request(audio=load_audio(path), language="English")
        features = inputs["input_features"]
        mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
        audio = encoder.encode(features, mask).to(torch.float16)
        ids = inputs["input_ids"]
        hidden = embed[ids].masked_scatter(
            (ids == audio_token).unsqueeze(-1), audio.reshape(-1, audio.shape[-1]).cpu()
        )
        samples.append({
            "inputs_embeds": hidden.float().numpy(),
            "position_ids": torch.arange(hidden.shape[1]).unsqueeze(0).numpy().astype(np.int64),
        })
    del encoder
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--packed", default="artifacts/qwen3-asr-0.6b/decoder-int4-g32")
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/experiments/draft-prefill-int8")
    parser.add_argument("--calibration", type=int, default=48)
    parser.add_argument("--calibration-method", choices=("minmax", "percentile"), default="minmax")
    parser.add_argument("--percentile", type=float, default=99.99)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--audio-token", type=int, default=151676)
    parser.add_argument("--skip-export", action="store_true")
    args = parser.parse_args()

    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    from qwen3_asr_trt.asr_decoder_trt import export_asr_decoder

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    # Keep the exporter's own naming: the graph points at `<name>.onnx.data`.
    fp32_onnx = work / "asr-prefill.onnx"
    int8_onnx = work / "asr-prefill.int8.onnx"

    if not args.skip_export or not fp32_onnx.is_file():
        # Q/DQ quantization needs a float32 graph; the fp16 export cannot be used.
        export_asr_decoder(args.model, work, "prefill", precision="fp32", include_lm_head=False)
    print(f"fp32 graph {fp32_onnx.stat().st_size / 2**20:.0f} MiB", flush=True)

    samples = build_samples(Path(args.model), Path(args.packed), Path(args.encoder),
                            args.calibration, args.seed, args.audio_token)
    print(f"{len(samples)} calibration prompts", flush=True)

    import onnx

    graph = onnx.load(str(fp32_onnx))
    input_names = [value.name for value in graph.graph.input]
    print("inputs", input_names, flush=True)

    # TensorRT rejects the Q/DQ patterns that ORT emits without shape inference and
    # a symbolic-shape pass, so run the documented pre-processing first.
    prepared = work / "asr-prefill.prepared.onnx"
    if not prepared.is_file():
        from onnxruntime.quantization.shape_inference import quant_pre_process

        # A 1.7B graph is over the 2 GB protobuf limit, so its tensors have to stay
        # in an external file.
        large = fp32_onnx.stat().st_size + (work / "asr-prefill.onnx.data").stat().st_size > 2**31
        quant_pre_process(
            str(fp32_onnx),
            str(prepared),
            skip_symbolic_shape=False,
            save_as_external_data=large,
            all_tensors_to_one_file=True,
            external_data_location="asr-prefill.prepared.onnx.data",
            external_data_size_threshold=1024,
        )
    print(f"prepared graph {prepared.stat().st_size / 2**20:.0f} MiB", flush=True)

    quantize_static(
        model_input=str(prepared),
        model_output=str(int8_onnx),
        calibration_data_reader=PromptReader(samples, input_names),
        quant_format=QuantFormat.QDQ,
        # TensorRT rejects asymmetric (zero point != 0) activation scales, so both
        # activations and weights use symmetric int8.
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=(CalibrationMethod.MinMax if args.calibration_method == "minmax"
                          else CalibrationMethod.Percentile),
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "CalibPercentile": args.percentile,
        },
        # Quantizing every op also quantizes the RMSNorm epsilon (1e-6), which INT8
        # turns into 0 and TensorRT rejects. Restrict to MatMul: the weights are
        # where the bytes are, and everything else stays float32.
        op_types_to_quantize=["MatMul"],
    )

    # TensorRT only accepts a quantized constant that feeds a DequantizeLinear, and
    # ORT emits `QuantizeLinear(constant) -> op` for a few small tensors. Drop those
    # pairs so the constant stays float32; the cost is a couple of fp32 tensors.
    model = onnx.load(str(int8_onnx))
    initializers = {tensor.name for tensor in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    drop, rewrites = set(), {}
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear" or node.input[0] not in initializers:
            continue
        users = consumers.get(node.output[0], [])
        if len(users) != 1 or users[0].op_type != "DequantizeLinear":
            continue
        dequant = users[0]
        rewrites[dequant.output[0]] = node.input[0]
        drop.add(id(node))
        drop.add(id(dequant))
    if drop:
        kept = []
        for node in model.graph.node:
            if id(node) in drop:
                continue
            for index, name in enumerate(node.input):
                if name in rewrites:
                    node.input[index] = rewrites[name]
            kept.append(node)
        del model.graph.node[:]
        model.graph.node.extend(kept)
        onnx.save(model, str(int8_onnx))
    print(f"bypassed {len(rewrites)} quantized constants", flush=True)
    print(json.dumps({
        "fp32MiB": round(fp32_onnx.stat().st_size / 2**20),
        "int8MiB": round(int8_onnx.stat().st_size / 2**20),
        "output": str(int8_onnx),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
