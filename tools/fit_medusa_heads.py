"""Fit residual prediction heads on the 1.7B decoder's final-norm hidden states.

Each head is ``h + W2 . silu(W1 h)`` and is scored through the *quantised* LM head
that inference uses, so training sees the same projection. Targets come from the
target model's own greedy transcripts collected with ``tools/collect_medusa.py``.

    python tools/fit_medusa_heads.py --data artifacts/experiments/medusa-data.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def dequantize_lm_head(packed: torch.Tensor, scales: torch.Tensor, group: int = 32,
                       chunk: int = 8192) -> torch.Tensor:
    """Rebuild the fp16 weights the INT4 kernel multiplies, nibbles included."""
    out = torch.empty((packed.shape[0], packed.shape[1] * 2), dtype=torch.float16,
                      device=packed.device)
    for start in range(0, packed.shape[0], chunk):
        rows = slice(start, start + chunk)
        nibbles = torch.empty((*packed[rows].shape, 2), dtype=torch.int16, device=packed.device)
        nibbles[..., 0] = packed[rows] & 0xF
        nibbles[..., 1] = (packed[rows] >> 4) & 0xF
        values = ((nibbles ^ 8) - 8).reshape(packed[rows].shape[0], -1).to(torch.float16)
        out[rows] = values * scales[rows].repeat_interleave(group, dim=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/experiments/medusa-data.pt")
    parser.add_argument("--packed", default="artifacts/qwen3-asr-1.7b/decoder-int4-g32")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/medusa-heads.pt")
    parser.add_argument("--heads", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--validation", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from safetensors.torch import load_file

    torch.manual_seed(args.seed)
    device = "cuda"

    data = torch.load(args.data, weights_only=False)
    hidden = data["hidden"].to(device)
    tokens = data["tokens"].to(device)
    starts = data["starts"]
    width = int(data["hiddenSize"])
    count = int(tokens.numel())

    # ``hidden`` at position i is the final-norm state that produced token i, so
    # head k predicts token i + k inside the same utterance.
    targets = torch.full((args.heads, count), -100, dtype=torch.long, device=device)
    for k in range(args.heads):
        targets[k, :count - k] = tokens[k:]
        for start, end in zip(starts[:-1].tolist(), starts[1:].tolist()):
            cut = end - k
            if cut < end:
                targets[k, cut:end] = -100
    # Positions whose segment is too short have already been masked above.

    utterances = len(starts) - 1
    split = int(utterances * (1 - args.validation))
    train_end = int(starts[split])
    train_rows = torch.arange(0, train_end, device=device)
    valid_rows = torch.arange(train_end, count, device=device)
    print(f"{count} positions, {utterances} utterances; train {train_rows.numel()}, "
          f"valid {valid_rows.numel()}", flush=True)

    state = load_file(str(ROOT / args.packed / "model.safetensors"))
    lm_head = dequantize_lm_head(state["lm_head.packed"].to(device), state["lm_head.scales"].to(device))
    lm_head.requires_grad_(False)
    del state
    torch.cuda.empty_cache()
    print(f"lm_head {tuple(lm_head.shape)} {lm_head.dtype}, "
          f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated", flush=True)

    heads = []
    for _ in range(args.heads):
        w1 = torch.empty((width, width), device=device)
        torch.nn.init.kaiming_uniform_(w1, a=math.sqrt(5))
        heads.append({
            "w1": w1.requires_grad_(True),
            "w2": torch.zeros((width, width), device=device, requires_grad=True),
        })
    parameters = [value for head in heads for value in head.values()]
    optimiser = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    steps = max(1, math.ceil(train_rows.numel() / args.batch)) * args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=args.lr, total_steps=steps,
                                                    pct_start=0.1)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def batch_loss(rows, head, target):
        h = hidden.index_select(0, rows).float()
        out = h + torch.nn.functional.silu(h @ head["w1"].t()) @ head["w2"].t()
        logits = out.to(torch.float16) @ lm_head.t()
        return torch.nn.functional.cross_entropy(logits.float(), target, ignore_index=-100)

    def evaluate(rows):
        accuracies = []
        with torch.no_grad():
            for k, head in enumerate(heads):
                pred_hits = 0
                total = 0
                for start in range(0, rows.numel(), 512):
                    chunk = rows[start:start + 512]
                    h = hidden.index_select(0, chunk).float()
                    out = h + torch.nn.functional.silu(h @ head["w1"].t()) @ head["w2"].t()
                    predicted = (out.to(torch.float16) @ lm_head.t()).argmax(-1)
                    target = targets[k].index_select(0, chunk)
                    keep = target >= 0
                    pred_hits += int((predicted[keep] == target[keep]).sum())
                    total += int(keep.sum())
                accuracies.append(pred_hits / max(1, total))
        return accuracies

    for epoch in range(args.epochs):
        order = train_rows[torch.randperm(train_rows.numel(), device=device)]
        running = 0.0
        seen = 0
        for start in range(0, order.numel(), args.batch):
            rows = order[start:start + args.batch]
            if rows.numel() < 8:
                continue
            loss = 0.0
            for k, head in enumerate(heads):
                loss = loss + batch_loss(rows, head, targets[k].index_select(0, rows))
            loss = loss / len(heads)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimiser.step()
            scheduler.step()
            running += float(loss.detach()) * rows.numel()
            seen += rows.numel()
        accuracy = evaluate(valid_rows)
        print(f"epoch {epoch + 1:2d}  loss {running / max(1, seen):.4f}  "
              f"valid acc {', '.join(f'{value * 100:.1f}%' for value in accuracy)}", flush=True)

    payload = {
        "w1": [head["w1"].detach().half().cpu() for head in heads],
        "w2": [head["w2"].detach().half().cpu() for head in heads],
        "accuracy": evaluate(valid_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"output": str(args.output), "validAccuracy": payload["accuracy"]}, indent=2))


if __name__ == "__main__":
    main()
