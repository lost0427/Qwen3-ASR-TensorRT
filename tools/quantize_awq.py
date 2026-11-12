"""AWQ-style activation-aware quantisation for the decoder.

Per-input-channel scales are folded into the preceding op so the runtime needs no
extra scaling: qkv scales go into ``input_layernorm``, gate/up into
``post_attention_layernorm``, ``o_proj`` into the V projection's output rows and
``down_proj`` into the up projection's output rows. Every fold is exact, so the
TensorRT prefill graph and the handwritten decode kernels both pick the change up
from the weights alone.

Usage:
    python tools/quantize_awq.py --model models/Qwen3-ASR-0.6B-hf \
        --output models/Qwen3-ASR-0.6B-hf-awq --utterances 48
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GROUP = 64
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
CLIPS = (1.0, 0.96, 0.92, 0.88, 0.84, 0.80, 0.75)
MAX_TOKENS = 1024


def quantize_dequant(weight: torch.Tensor, group: int, clip: float = 1.0) -> torch.Tensor:
    rows, cols = weight.shape
    blocks = weight.view(rows, cols // group, group)
    if clip < 1.0:
        peak = blocks.abs().amax(-1, keepdim=True)
        blocks = blocks.clamp(min=-peak * clip, max=peak * clip)
    scales = blocks.abs().amax(-1, keepdim=True) / 7.0
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    return (torch.round(blocks / scales).clamp(-7, 7) * scales).view(rows, cols)


def clip_to(weight: torch.Tensor, group: int, clip: float) -> None:
    """Clip in place so the plain max/7 quantiser reproduces the clipped grid."""
    if clip >= 1.0:
        return
    blocks = weight.view(weight.shape[0], weight.shape[1] // group, group)
    peak = blocks.abs().amax(-1, keepdim=True)
    blocks.clamp_(min=-peak * clip, max=peak * clip)


def search_scale(activations: torch.Tensor, weights: list[torch.Tensor], group: int):
    """Return the AWQ exponent, clip factor, scale and relative error for a group."""
    x = activations.float()
    magnitude = x.abs().mean(0)
    base = (magnitude / magnitude.max().clamp(min=1e-8)).clamp(min=1e-8)
    references = [weight.detach().float() @ x.t() for weight in weights]
    norm = sum((ref.pow(2).sum() for ref in references), torch.zeros((), device=x.device))

    def evaluate(alpha, clip):
        scale = base.pow(alpha) if alpha > 0 else torch.ones_like(base)
        scale = scale / torch.sqrt(scale.max() * scale.min())
        scaled_x = x / scale
        error = torch.zeros((), device=x.device)
        for weight, reference in zip(weights, references):
            approx = quantize_dequant(weight.float() * scale, group, clip) @ scaled_x.t()
            error = error + (reference - approx).pow(2).sum()
        return scale, float(error / norm)

    best = (0.0, 1.0, None, float("inf"))
    for alpha in ALPHAS:
        scale, ratio = evaluate(alpha, 1.0)
        if ratio < best[3]:
            best = (alpha, 1.0, scale, ratio)
    alpha = best[0]
    for clip in CLIPS[1:]:
        scale, ratio = evaluate(alpha, clip)
        if ratio < best[3]:
            best = (alpha, clip, scale, ratio)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--output", type=Path, default=ROOT / "models/Qwen3-ASR-0.6B-hf-awq")
    parser.add_argument("--manifest", type=Path, default=Path(r"C:\Users\z1\Documents\ASR-data\manifest.jsonl"))
    parser.add_argument("--audio-root", type=Path, default=Path(r"C:\Users\z1\Documents\ASR-data\ls"))
    parser.add_argument("--encoder", type=Path,
                        default=ROOT / "artifacts/qwen3-asr-0.6b/encoder-window.bf16.engine")
    parser.add_argument("--utterances", type=int, default=48)
    parser.add_argument("--group", type=int, default=64, choices=(32, 64))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()
    group = args.group

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
    encoder = TensorRTAudioEncoder(args.encoder)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(args.model), dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    language = model.model.language_model
    config = language.config
    head_dim = config.head_dim
    groups = config.num_attention_heads // config.num_key_value_heads
    kv_channels = config.num_key_value_heads * head_dim
    o_channels = config.num_attention_heads * head_dim

    captured: dict[tuple[int, str], torch.Tensor] = {}

    def hook_for(key):
        def hook(module, arguments, _kwargs=None):
            value = arguments[0].detach().reshape(-1, module.in_features).to(torch.float16)
            previous = captured.get(key)
            if previous is None:
                captured[key] = value[:MAX_TOKENS].clone()
            elif previous.shape[0] < MAX_TOKENS:
                captured[key] = torch.cat([previous, value], 0)[:MAX_TOKENS].clone()
        return hook

    handles = []
    for index, layer in enumerate(language.layers):
        handles.append(layer.self_attn.q_proj.register_forward_hook(hook_for((index, "qkv"))))
        handles.append(layer.self_attn.o_proj.register_forward_hook(hook_for((index, "o"))))
        handles.append(layer.mlp.gate_proj.register_forward_hook(hook_for((index, "gu"))))
        handles.append(layer.mlp.down_proj.register_forward_hook(hook_for((index, "down"))))

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (args.audio_root / row["flac"]).is_file()][: args.utterances]
    embed = language.embed_tokens.weight
    with torch.no_grad():
        for done, row in enumerate(rows, 1):
            inputs = processor.apply_transcription_request(
                audio=str(args.audio_root / row["flac"]), language="English"
            )
            ids = inputs["input_ids"].to("cuda")
            features = inputs["input_features"]
            mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
            audio = encoder.encode(features, mask).to(torch.float16)
            hidden = embed[ids].masked_scatter(
                (ids == 151676).unsqueeze(-1), audio.reshape(-1, audio.shape[-1])
            )
            language(inputs_embeds=hidden)
            if done % 12 == 0:
                print(f"  calibration {done}/{len(rows)}", flush=True)
    for handle in handles:
        handle.remove()

    position = torch.arange(o_channels, device="cuda")
    kv_index = (position // head_dim // groups) * head_dim + (position % head_dim)
    report = []
    with torch.no_grad():
        for index, layer in enumerate(language.layers):
            attn, mlp = layer.self_attn, layer.mlp
            scale_vectors = {}
            alpha, clip, scale, ratio = search_scale(
                captured[(index, "qkv")], [attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], group
            )
            for projection in (attn.q_proj, attn.k_proj, attn.v_proj):
                projection.weight.mul_(scale[None, :])
                clip_to(projection.weight, group, clip)
            layer.input_layernorm.weight.div_(scale)
            scale_vectors["qkv"] = {"alpha": alpha, "clip": clip, "relError": ratio}

            alpha_o, clip_o, scale_o, ratio_o = search_scale(
                captured[(index, "o")], [attn.o_proj.weight], group
            )
            # o_proj's inputs are GQA-expanded V heads, so the scale has to be
            # constant inside each expansion group for the fold to stay exact.
            log_mean = torch.zeros(kv_channels, device="cuda").index_add_(0, kv_index, scale_o.log())
            log_mean /= groups
            scale_kv = log_mean.exp()
            attn.o_proj.weight.mul_(scale_kv[kv_index][None, :])
            clip_to(attn.o_proj.weight, group, clip_o)
            attn.v_proj.weight.mul_(1.0 / scale_kv[:, None])
            scale_vectors["o"] = {"alpha": alpha_o, "clip": clip_o, "relError": ratio_o}

            alpha_g, clip_g, scale_g, ratio_g = search_scale(
                captured[(index, "gu")], [mlp.gate_proj.weight, mlp.up_proj.weight], group
            )
            for projection in (mlp.gate_proj, mlp.up_proj):
                projection.weight.mul_(scale_g[None, :])
                clip_to(projection.weight, group, clip_g)
            layer.post_attention_layernorm.weight.div_(scale_g)
            scale_vectors["gu"] = {"alpha": alpha_g, "clip": clip_g, "relError": ratio_g}

            alpha_d, clip_d, scale_d, ratio_d = search_scale(
                captured[(index, "down")], [mlp.down_proj.weight], group
            )
            mlp.down_proj.weight.mul_(scale_d[None, :])
            clip_to(mlp.down_proj.weight, group, clip_d)
            mlp.up_proj.weight.mul_(1.0 / scale_d[:, None])
            scale_vectors["down"] = {"alpha": alpha_d, "clip": clip_d, "relError": ratio_d}

            scale_vectors["layer"] = index
            report.append(scale_vectors)

    del captured
    torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output), safe_serialization=True)
    for name in ("chat_template.jinja", "processor_config.json", "tokenizer.json",
                 "tokenizer_config.json", "generation_config.json"):
        source = args.model / name
        if source.is_file():
            (args.output / name).write_bytes(source.read_bytes())
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    mean_error = sum(
        record[key]["relError"] for record in report for key in ("qkv", "o", "gu", "down")
    )
    print(f"group {group}, mean relative quantisation error: "
          f"{mean_error / (4 * len(report)) * 100:.3f}%")
    print("saved", args.output)


if __name__ == "__main__":
    main()
