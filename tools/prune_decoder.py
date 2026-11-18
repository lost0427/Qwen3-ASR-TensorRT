"""Structured pruning of the Qwen3-ASR decoder (FFN width and depth).

Channel saliency and layer redundancy are measured on real ASR prompts built
from LibriSpeech train-clean-100 audio: the activations that matter at inference
sit at audio-token positions, so calibrating on plain text picks the wrong
channels. The tool writes a pruned FP16 checkpoint that `pack-decoder` and
`build-decoder-prefill` consume directly.

    python tools/prune_decoder.py --model models/Qwen3-ASR-1.7B-hf-awq32 \
        --output artifacts/experiments/1.7b-ffn75/fp16 --keep-ffn 0.75
    python tools/prune_decoder.py --model models/Qwen3-ASR-1.7B-hf-awq32 \
        --output artifacts/experiments/1.7b-depth26/fp16 --drop-layers 2
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import statistics
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")


def calibration_files(limit: int, seed: int) -> list[Path]:
    root = DATA / "LibriSpeech" / "train-clean-100"
    files = sorted(root.glob("*/*/*.flac"))
    if not files:
        raise SystemExit(f"no train-clean-100 audio under {root}")
    random.Random(seed).shuffle(files)
    return files[:limit]


def build_prompts(model, processor, encoder, audio_files, audio_token_id: int):
    """Yields the exact decoder input distribution: text template plus audio.

    The saliency statistics have to see audio-token positions, which carry most
    of the prompt in ASR and behave nothing like text embeddings.
    """
    from qwen3_asr_trt.audio import load_audio

    for path in audio_files:
        inputs = processor.apply_transcription_request(audio=load_audio(path), language="English")
        features = inputs["input_features"]
        mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
        embeddings = encoder.encode(features, mask).to(torch.float16)
        ids = inputs["input_ids"].to("cuda")
        hidden = model.embed_tokens(ids).masked_scatter(
            (ids == audio_token_id).unsqueeze(-1), embeddings.reshape(-1, embeddings.shape[-1])
        )
        yield hidden


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="AWQ-scaled FP16 decoder directory")
    parser.add_argument("--output", required=True, help="pruned FP16 checkpoint directory")
    parser.add_argument("--keep-ffn", type=float, default=1.0)
    parser.add_argument("--drop-layers", type=int, default=0)
    parser.add_argument("--calibration", type=int, default=128,
                        help="train-clean-100 utterances used for saliency")
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
                        help="TensorRT audio encoder for the 1.7B checkpoint")
    parser.add_argument("--audio-token", type=int, default=151676)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3Config, Qwen3Model

    model_dir = Path(args.model)
    output = Path(args.output)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    # Two layouts exist: the original Qwen release nests under "thinker_config"
    # and prefixes weights with "thinker.model."; the transformers export puts
    # text_config at the top level and prefixes with "model.language_model.".
    if "thinker_config" in raw:
        config_key, layer_prefix = "thinker_config", "thinker.model."
    else:
        config_key, layer_prefix = None, "model.language_model."
    text_config = dict(raw[config_key]["text_config"] if config_key else raw["text_config"])

    torch.manual_seed(args.seed)
    config = Qwen3Config(**text_config)
    model = Qwen3Model(config).eval().to("cuda", torch.float16)

    state = {}
    with safe_open(str(model_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            if key.startswith(layer_prefix):
                state[key[len(layer_prefix):]] = handle.get_tensor(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise SystemExit(f"text weights did not load cleanly: missing {missing[:4]} unexpected {unexpected[:4]}")
    del state

    layers = len(model.layers)
    intermediate = config.intermediate_size
    sums = [torch.zeros(intermediate, device="cuda") for _ in range(layers)]
    counts = [0 for _ in range(layers)]
    influence: list[list[float]] = [[] for _ in range(layers)]

    def down_hook(index: int):
        def hook(module, inputs, output):
            x = inputs[0].detach().float()
            sums[index] += x.abs().sum(dim=(0, 1))
            counts[index] += x.shape[0] * x.shape[1]
        return hook

    def layer_hook(index: int):
        def hook(module, inputs, output):
            after = output if torch.is_tensor(output) else output[0]
            before = inputs[0].detach().float().reshape(-1)
            after = after.detach().float().reshape(-1)
            influence[index].append(float(torch.nn.functional.cosine_similarity(before, after, dim=0)))
        return hook

    handles = []
    for index, layer in enumerate(model.layers):
        handles.append(layer.mlp.down_proj.register_forward_hook(down_hook(index)))
        handles.append(layer.register_forward_hook(layer_hook(index)))

    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    processor = AutoProcessor.from_pretrained(str(model_dir))
    encoder = TensorRTAudioEncoder(Path(args.encoder))
    audio_files = calibration_files(args.calibration, args.seed)
    calls = list(build_prompts(model, processor, encoder, audio_files, args.audio_token))
    print(f"calibration prompts: {len(calls)} "
          f"(median {int(statistics.median(int(item.shape[1]) for item in calls))} tokens)", flush=True)
    del encoder
    torch.cuda.empty_cache()

    with torch.no_grad():
        for step, embeds in enumerate(calls, 1):
            model(inputs_embeds=embeds)
            if step % 20 == 0:
                print(f"  calibration {step}/{len(calls)} prompts", flush=True)
    for handle in handles:
        handle.remove()

    keep = max(1, int(round(intermediate * args.keep_ffn)))
    kept = []
    with safe_open(str(model_dir / "model.safetensors"), "pt", "cpu") as handle:
        for index in range(layers):
            score = sums[index] / max(1, counts[index])
            # Weight the activation by how strongly the channel drives the residual
            # stream, so AWQ-folded scale differences do not distort the ranking.
            weight = handle.get_tensor(f"{layer_prefix}layers.{index}.mlp.down_proj.weight")
            score = score * weight.to("cuda", torch.float32).norm(dim=0)
            order = torch.argsort(score, descending=True)[:keep].sort().values
            kept.append(order.cpu())
            print(f"  layer {index:2d} keep {keep}/{intermediate}  "
                  f"influence {statistics.fmean(influence[index]):.4f}", flush=True)

    dropped: set[int] = set()
    if args.drop_layers > 0:
        # Rank layers by how little the next-token distribution changes when the
        # block is skipped (mean KL over real ASR prompts from train-clean-100).
        lm_head = model.embed_tokens.weight.to(torch.float16)  # tie_word_embeddings
        tail = 8
        kl_prompts = calls[: min(16, len(calls))]

        def collect(skip: int | None):
            outputs = []
            patched = None
            if skip is not None:
                layer = model.layers[skip]
                patched = layer.forward
                layer.forward = lambda hidden, *rest, **kwargs: hidden
            try:
                with torch.no_grad():
                    for embeds in kl_prompts:
                        hidden = model(inputs_embeds=embeds).last_hidden_state[:, -tail:]
                        logits = model.norm(hidden).float() @ lm_head.float().t()
                        outputs.append(torch.log_softmax(logits, dim=-1))
            finally:
                if skip is not None:
                    model.layers[skip].forward = patched
            return outputs

        reference = collect(None)
        impact = []
        for index in range(layers):
            alternative = collect(index)
            kl = statistics.fmean(
                float((ref.exp() * (ref - alt)).sum(dim=-1).mean())
                for ref, alt in zip(reference, alternative)
            )
            impact.append((kl, index))
            print(f"  skip layer {index:2d}: KL {kl:.4f}", flush=True)
        impact.sort()
        dropped = {index for _, index in impact[: args.drop_layers]}
        print(f"dropping layers {sorted(dropped)} (lowest KL)")

    layer_map = {}
    for old in range(layers):
        if old not in dropped:
            layer_map[old] = len(layer_map)

    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(model_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            if key.startswith(layer_prefix + "layers."):
                parts = key.split(".")
                old = int(parts[3])
                if old in dropped:
                    continue
                rest = ".".join(parts[4:])
                value = handle.get_tensor(key)
                if rest in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
                    value = value[kept[old]][:, :].contiguous()
                elif rest == "mlp.down_proj.weight":
                    value = value[:, kept[old]].contiguous()
                tensors[f"{layer_prefix}layers.{layer_map[old]}.{rest}"] = value
            else:
                tensors[key] = handle.get_tensor(key)

    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"))
    del tensors
    # Keep the channel plan so a refit or a distillation run reproduces exactly
    # the same weights.
    save_file(
        {f"kept.{index}": kept[index].to(torch.int64).contiguous() for index in range(layers)},
        str(output / "prune_plan.safetensors"),
    )

    target_text = raw[config_key]["text_config"] if config_key else raw["text_config"]
    target_text["intermediate_size"] = int(keep)
    target_text["num_hidden_layers"] = len(layer_map)
    if isinstance(target_text.get("layer_types"), list):
        target_text["layer_types"] = ["full_attention"] * len(layer_map)
    if target_text.get("max_window_layers"):
        target_text["max_window_layers"] = min(int(target_text["max_window_layers"]), len(layer_map))
    (output / "config.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json", "preprocessor_config.json",
                 "processor_config.json", "chat_template.jinja", "generation_config.json",
                 "added_tokens.json", "special_tokens_map.json", "vocab.json", "merges.txt"):
        if (model_dir / name).is_file():
            shutil.copy2(model_dir / name, output / name)

    size = (output / "model.safetensors").stat().st_size / 2**30
    print(f"wrote {output} ({size:.2f} GiB, {len(layer_map)} layers, FFN {keep})")


if __name__ == "__main__":
    main()
