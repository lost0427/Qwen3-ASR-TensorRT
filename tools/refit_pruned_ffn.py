"""Refit a width-pruned FFN so its output matches the unpruned layer.

Pruning keeps the highest-saliency channels and drops the rest, which changes the
FFN output both by the missing channels and by the fact that the surviving
`down_proj` columns were fitted against all of them. This solves, per layer,

    W' = argmin_W || A' W^T - Y ||^2,   Y = A_full W_full^T

over calibration activations from train-clean-100 audio prompts, where A' holds
the kept channels. Shapes do not change, so `pack-decoder` and the prefill engine
work unchanged.

    python tools/refit_pruned_ffn.py --model models/Qwen3-ASR-1.7B-hf-awq32 \
        --pruned C:/temp/prune/ffn75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_decoder import build_prompts, calibration_files  # noqa: E402

LAYER_STRIDE = 14


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="unpruned AWQ FP16 checkpoint")
    parser.add_argument("--pruned", required=True, help="pruned checkpoint with prune_plan.safetensors")
    parser.add_argument("--calibration", type=int, default=256)
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, help="defaults to --pruned, overwritten in place")
    args = parser.parse_args()

    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoProcessor, Qwen3Config, Qwen3Model

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    model_dir = Path(args.model)
    pruned_dir = Path(args.pruned)
    output = Path(args.output) if args.output else pruned_dir

    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if "thinker_config" in raw:
        config_key, prefix = "thinker_config", "thinker.model."
    else:
        config_key, prefix = None, "model.language_model."
    text_config = dict(raw[config_key]["text_config"] if config_key else raw["text_config"])
    config = Qwen3Config(**text_config)

    model = Qwen3Model(config).eval().to("cuda", torch.float16)
    state = {}
    with safe_open(str(model_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = handle.get_tensor(key)
    model.load_state_dict(state, strict=False)
    del state

    plan = {}
    with safe_open(str(pruned_dir / "prune_plan.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            plan[int(key.split(".")[1])] = handle.get_tensor(key)
    layers = len(plan)
    keep = int(plan[0].numel())
    print(f"{layers} layers, {keep} kept channels", flush=True)

    processor = AutoProcessor.from_pretrained(str(model_dir))
    encoder = TensorRTAudioEncoder(Path(args.encoder))
    files = calibration_files(args.calibration, args.seed)
    prompts = list(build_prompts(model, processor, encoder, files, 151676))
    del encoder
    torch.cuda.empty_cache()
    print(f"{len(prompts)} calibration prompts", flush=True)

    hidden = config.hidden_size
    solved = {}
    for start in range(0, layers, LAYER_STRIDE):
        group = list(range(start, min(start + LAYER_STRIDE, layers)))
        acc_a = {index: torch.zeros((keep, keep), dtype=torch.float32, device="cuda") for index in group}
        acc_y = {index: torch.zeros((keep, hidden), dtype=torch.float32, device="cuda") for index in group}
        weights = {index: model.layers[index].mlp.down_proj.weight.detach() for index in group}

        def make_hook(index):
            def hook(module, inputs, output):
                full = inputs[0].detach().float().reshape(-1, module.in_features)
                kept = full.index_select(1, plan[index].to("cuda"))
                target = full @ weights[index].float().t()
                acc_a[index] += kept.t() @ kept
                acc_y[index] += kept.t() @ target
            return hook

        handles = [model.layers[index].mlp.down_proj.register_forward_hook(make_hook(index))
                   for index in group]
        with torch.no_grad():
            for prompt in prompts:
                model(inputs_embeds=prompt)
        for handle in handles:
            handle.remove()

        for index in group:
            a = acc_a[index]
            ridge = args.ridge * a.diagonal().mean()
            a = a + torch.eye(keep, device="cuda") * ridge
            solved[index] = torch.linalg.solve(a, acc_y[index]).t().contiguous()
        del acc_a, acc_y
        torch.cuda.empty_cache()
        print(f"  refit layers {group[0]}-{group[-1]}", flush=True)

    del model
    torch.cuda.empty_cache()

    tensors = {}
    base = len(prefix.rstrip(".").split("."))
    with safe_open(str(pruned_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            value = handle.get_tensor(key)
            if key.startswith(prefix + "layers.") and key.endswith(".mlp.down_proj.weight"):
                index = int(key.split(".")[base + 1])
                value = solved[index].to("cpu", torch.float16)
            tensors[key] = value
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"))
    size = (output / "model.safetensors").stat().st_size / 2**30
    print(f"wrote {output} ({size:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
