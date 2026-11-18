"""Recover a width-pruned decoder by distilling the unpruned model into it.

Stage 1 runs the unpruned teacher over train-clean-100 ASR prompts and stores its
top-k next-token distribution at every prompt position. Stage 2 trains low-rank
adapters on the pruned model's FFN against those distributions, merges them, and
writes a checkpoint that `pack-decoder` consumes unchanged.

The teacher logits are cached so the two 1.7B models never have to be resident at
the same time on an 8 GB card.

    python tools/distill_pruned.py --stage prepare --prompts 1200
    python tools/distill_pruned.py --stage train --prompts 1200 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_decoder import calibration_files  # noqa: E402

DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")
CACHE = ROOT / "artifacts/experiments/teacher-topk.pt"


def load_prompts(model_dir: Path, count: int, seed: int, encoder_path: Path):
    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    processor = AutoProcessor.from_pretrained(str(model_dir))
    encoder = TensorRTAudioEncoder(encoder_path)
    prompts = []
    for path in calibration_files(count, seed):
        inputs = processor.apply_transcription_request(audio=load_audio(path), language="English")
        features = inputs["input_features"]
        mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
        embeds = encoder.encode(features, mask).to(torch.float16)
        prompts.append({
            "ids": [int(value) for value in inputs["input_ids"][0].tolist()],
            "audio": embeds.reshape(-1, embeds.shape[-1]).half().cpu(),
        })
    del encoder
    torch.cuda.empty_cache()
    return processor, prompts


def embedding_table(model, record) -> torch.Tensor:
    ids = torch.tensor([record["ids"]], dtype=torch.long, device="cuda")
    weight = getattr(model, "embed", None)
    if weight is None:
        weight = model.embed_tokens.weight
    return weight[ids].masked_scatter(
        (ids == 151676).unsqueeze(-1), record["audio"].to("cuda")
    )


def full_embeddings(model, record, tokens) -> torch.Tensor:
    prompt = embedding_table(model, record)
    extra = model.embed_tokens.weight[torch.tensor([tokens], dtype=torch.long, device="cuda")]
    return torch.cat([prompt, extra], dim=1)


def prepare(args) -> None:
    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3Model

    from qwen3_asr_trt.w4a16_decoder import W4A16ASRDecoder

    model_dir = Path(args.model)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = dict(raw["thinker_config"]["text_config"] if "thinker_config" in raw else raw["text_config"])
    prefix = "thinker.model." if "thinker_config" in raw else "model.language_model."

    _, prompts = load_prompts(model_dir, args.prompts, args.seed, Path(args.encoder))

    # The transcripts come from the deployed INT4 decoder, so the student learns
    # the distribution of the model it replaces rather than an unreleased FP16 one.
    decoder = W4A16ASRDecoder(str(args.packed), Path(args.prefill), max_cache=1024, group=32)
    transcripts = []
    for index, record in enumerate(prompts, 1):
        decoder.prefill_prompt(embedding_table(decoder, record), reserve=args.max_new_tokens)
        transcripts.append(decoder.decode_loop(args.max_new_tokens))
        if index % 200 == 0:
            print(f"  transcript {index}/{len(prompts)}", flush=True)
    del decoder
    torch.cuda.empty_cache()

    model = Qwen3Model(Qwen3Config(**text)).eval().to("cuda", torch.float16)
    state = {}
    with safe_open(str(model_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            if key.startswith(prefix):
                state[key[len(prefix):]] = handle.get_tensor(key)
    model.load_state_dict(state, strict=False)
    del state

    head = model.embed_tokens.weight
    cache = []
    with torch.no_grad():
        for index, (record, tokens) in enumerate(zip(prompts, transcripts), 1):
            full = full_embeddings(model, record, tokens)
            hidden = model(inputs_embeds=full).last_hidden_state
            logits = model.norm(hidden).float() @ head.float().t()
            start = len(record["ids"]) - 1
            picked = logits[:, start:start + len(tokens)]
            values, indices = picked.topk(args.topk, dim=-1)
            cache.append({
                "prompt": record["ids"],
                "tokens": tokens,
                "values": values.half().cpu(),
                "indices": indices.to(torch.int32).cpu(),
            })
            if index % 200 == 0:
                print(f"  teacher {index}/{len(prompts)}", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, CACHE)
    print(f"wrote {CACHE} ({CACHE.stat().st_size / 2**20:.0f} MiB)", flush=True)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device,
                                           dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device,
                                          dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x.float() @ self.a.t()) @ self.b.t()
        return out + (delta * self.scale).to(x.dtype)

    def merged(self) -> torch.Tensor:
        weight = self.base.weight.detach().float()
        weight = weight + (self.b @ self.a) * self.scale
        return weight.to(torch.float16)


def train(args) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3Model

    pruned_dir = Path(args.pruned)
    output = Path(args.output) if args.output else pruned_dir
    raw = json.loads((pruned_dir / "config.json").read_text(encoding="utf-8"))
    text = dict(raw["thinker_config"]["text_config"] if "thinker_config" in raw else raw["text_config"])
    prefix = "thinker.model." if "thinker_config" in raw else "model.language_model."

    model = Qwen3Model(Qwen3Config(**text)).eval().to("cuda", torch.float16)
    tensors = {}
    with safe_open(str(pruned_dir / "model.safetensors"), "pt", "cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    state = {key[len(prefix):]: value for key, value in tensors.items() if key.startswith(prefix)}
    model.load_state_dict(state, strict=False)

    adapters = []
    for layer in model.layers:
        layer.mlp.down_proj = LoRALinear(layer.mlp.down_proj, args.rank, args.alpha)
        layer.mlp.gate_proj = LoRALinear(layer.mlp.gate_proj, args.rank, args.alpha)
        layer.mlp.up_proj = LoRALinear(layer.mlp.up_proj, args.rank, args.alpha)
        adapters.extend([layer.mlp.down_proj, layer.mlp.gate_proj, layer.mlp.up_proj])

    _, prompts = load_prompts(pruned_dir, args.prompts, args.seed, Path(args.encoder))
    cache = torch.load(CACHE, map_location="cpu")
    if len(cache) != len(prompts):
        raise SystemExit("teacher cache and prompt list disagree; re-run --stage prepare")
    print(f"{len(prompts)} prompts, "
          f"{sum(p.numel() for p in (a.a for a in adapters)) + sum(p.numel() for p in (a.b for a in adapters))} "
          f"adapter parameters", flush=True)

    head = model.embed_tokens.weight
    parameters = [p for adapter in adapters for p in (adapter.a, adapter.b)]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    order = list(range(len(prompts)))

    for epoch in range(args.epochs):
        total = 0.0
        for step, index in enumerate(order, 1):
            record = prompts[index]
            entry = cache[index]
            tokens = entry["tokens"]
            target = torch.softmax(entry["values"].float().to("cuda"), dim=-1)
            full = full_embeddings(model, record, tokens)
            hidden = model(inputs_embeds=full).last_hidden_state
            logits = model.norm(hidden).float() @ head.float().t()
            start = len(record["ids"]) - 1
            picked = logits[:, start:start + len(tokens)].gather(
                -1, entry["indices"].to("cuda").long()
            )
            loss = -(target * torch.log_softmax(picked, dim=-1)).sum(-1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total += float(loss.detach())
            if step % 200 == 0:
                print(f"  epoch {epoch + 1} {step}/{len(order)} loss {total / step:.4f}", flush=True)
        print(f"epoch {epoch + 1} mean loss {total / len(order):.4f}", flush=True)

    for layer_index, layer in enumerate(model.layers):
        for name in ("down_proj", "gate_proj", "up_proj"):
            adapter = getattr(layer.mlp, name)
            key = f"{prefix}layers.{layer_index}.mlp.{name}.weight"
            tensors[key] = adapter.merged().cpu()
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output / "model.safetensors"))
    if output.resolve() != pruned_dir.resolve():
        import shutil

        for candidate in pruned_dir.iterdir():
            if candidate.is_file() and candidate.name != "model.safetensors":
                shutil.copy2(candidate, output / candidate.name)
    print(f"wrote {output} ({(output / 'model.safetensors').stat().st_size / 2**30:.2f} GiB)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "train"), required=True)
    parser.add_argument("--model", help="unpruned checkpoint (prepare)")
    parser.add_argument("--pruned", help="pruned checkpoint (train)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompts", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--packed", default="artifacts/qwen3-asr-1.7b/decoder-int4-g32",
                        help="packed teacher decoder used to generate transcripts")
    parser.add_argument("--prefill", default="artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.stage == "prepare":
        prepare(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
