"""Measure how often an early-exit layer agrees with the full decoder.

If the hidden state after layer N, pushed through the final norm and the tied
output projection, picks the same token as the full model, then the first N
layers can draft for the rest without a second model, a second audio encoder or
a second prompt prefill.

    python tools/early_exit_acceptance.py --prompts 200
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

from distill_pruned import embedding_table, full_embeddings, load_prompts  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B-hf-awq32")
    parser.add_argument("--packed", default="artifacts/qwen3-asr-1.7b/decoder-int4-g32")
    parser.add_argument("--prefill", default="artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine")
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine")
    parser.add_argument("--prompts", type=int, default=200)
    parser.add_argument("--layers", nargs="*", type=int, default=[8, 12, 16, 20])
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    from safetensors import safe_open
    from transformers import Qwen3Config, Qwen3Model

    from qwen3_asr_trt.w4a16_decoder import W4A16ASRDecoder

    model_dir = Path(args.model)
    raw = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text = dict(raw["thinker_config"]["text_config"] if "thinker_config" in raw else raw["text_config"])
    prefix = "thinker.model." if "thinker_config" in raw else "model.language_model."

    _, prompts = load_prompts(model_dir, args.prompts, args.seed, Path(args.encoder))

    decoder = W4A16ASRDecoder(str(args.packed), Path(args.prefill), max_cache=1024, group=32)
    transcripts = []
    for record in prompts:
        decoder.prefill_prompt(embedding_table(decoder, record), reserve=128)
        transcripts.append(decoder.decode_loop(128))
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

    captured = {}
    handles = []
    for layer in args.layers:
        def hook(module, inputs, output, index=layer):
            captured[index] = output[0] if isinstance(output, tuple) else output
        handles.append(model.layers[layer - 1].register_forward_hook(hook))

    head = model.embed_tokens.weight
    agree = {layer: [0, 0] for layer in args.layers}
    with torch.no_grad():
        for step, (record, tokens) in enumerate(zip(prompts, transcripts), 1):
            if not tokens:
                continue
            full = full_embeddings(model, record, tokens)
            hidden = model(inputs_embeds=full).last_hidden_state
            end = len(record["ids"]) - 1 + len(tokens)
            target = (model.norm(hidden[:, :end]).float() @ head.float().t()).argmax(-1)
            start = len(record["ids"]) - 1
            for layer in args.layers:
                early = captured[layer][:, :end]
                predicted = (model.norm(early).float() @ head.float().t()).argmax(-1)
                same = (predicted[:, start:] == target[:, start:])
                agree[layer][0] += int(same.sum())
                agree[layer][1] += int(same.numel())
            if step % 50 == 0:
                print(f"  {step}/{len(prompts)}", flush=True)
    for handle in handles:
        handle.remove()

    print(json.dumps({
        "prompts": len(prompts),
        "agreement": {str(layer): round(count / max(1, total), 4)
                      for layer, (count, total) in agree.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
