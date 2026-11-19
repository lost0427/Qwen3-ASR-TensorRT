"""Collect (hidden, next tokens) pairs from the 1.7B decoder for the prediction heads.

Only train-clean-100 is used. Every generated position contributes its final-norm
hidden state and the three tokens that follow it in the target's own greedy
transcript.

    python tools/collect_medusa.py --count 600 --output artifacts/experiments/medusa-data.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_decoder import calibration_files  # noqa: E402

DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))
EOS = (151643, 151645)
AUDIO_TOKEN = 151676


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen3-ASR-1.7B-hf")
    parser.add_argument("--decoder-dir", default="artifacts/qwen3-asr-1.7b/decoder-int4-g32")
    parser.add_argument("--prefill", default="artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine")
    parser.add_argument("--encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine")
    parser.add_argument("--count", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--language", default="English")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/medusa-data.pt")
    args = parser.parse_args()

    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.features import GpuLogMel, PromptTemplate
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder, _post_cnn_tokens
    from qwen3_asr_trt.w4a16_decoder import W4A16ASRDecoder

    processor = AutoProcessor.from_pretrained(str(ROOT / args.model))
    mel = GpuLogMel(processor.feature_extractor)
    template = PromptTemplate(processor, args.language, AUDIO_TOKEN)
    encoder = TensorRTAudioEncoder(ROOT / args.encoder)
    decoder = W4A16ASRDecoder(str(ROOT / args.decoder_dir), ROOT / args.prefill,
                              max_cache=512, group=32)

    hidden_states, tokens, starts = [], [], [0]
    started = time.perf_counter()
    for index, path in enumerate(calibration_files(args.count, args.seed), 1):
        if args.verbose:
            print(f"  [{index}] {Path(path).name}", flush=True)
        features, mask, valid = mel(load_audio(path))
        ids = template(_post_cnn_tokens(valid))
        audio = encoder.encode(features, mask).to(torch.float16)
        if args.verbose:
            print(f"  [{index}] frames={int(valid)} prompt={int(ids.shape[1])}", flush=True)
        prompt = decoder.embed[ids].masked_scatter(
            (ids == AUDIO_TOKEN).unsqueeze(-1), audio.reshape(-1, audio.shape[-1])
        )
        decoder.prefill_prompt(prompt, reserve=args.max_new_tokens)
        decoder._generated.zero_()
        graph = decoder._graph()
        produced = []
        for _ in range(args.max_new_tokens):
            hidden_states.append(decoder._h_out.view(-1).clone())
            token = int(decoder._token.item())
            produced.append(token)
            if token in EOS:
                break
            graph.replay()
        tokens.extend(produced)
        starts.append(len(tokens))
        if index % 50 == 0:
            elapsed = time.perf_counter() - started
            print(f"  {index}/{args.count} utterances, {len(tokens)} positions, "
                  f"{elapsed:.0f}s", flush=True)

    payload = {
        "hidden": torch.stack(hidden_states).to(torch.float16),
        "tokens": torch.tensor(tokens, dtype=torch.int32),
        "starts": torch.tensor(starts, dtype=torch.int64),
        "hiddenSize": int(hidden_states[0].numel()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({
        "utterances": len(starts) - 1,
        "positions": int(payload["tokens"].numel()),
        "meanTokens": round(len(tokens) / max(1, len(starts) - 1), 1),
        "output": str(args.output),
        "elapsedSeconds": round(time.perf_counter() - started, 1),
    }, indent=2))


if __name__ == "__main__":
    main()
