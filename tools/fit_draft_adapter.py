"""Fit a 2048->1024 adapter that predicts the 0.6B audio embeddings.

Speculative decoding with the 0.6B draft needs the 0.6B decoder's prompt
embeddings, which normally come from its own audio tower. Those cost 1.2 GB and
about 6 ms per utterance. This adapter learns to predict them from the 1.7B
tower's embeddings instead, so only one encoder has to be resident and only one
audio pass is needed.

The fit is closed form: both towers emit the same number of tokens for the same
audio, so the pairs align and a ridge least-squares solve gives the map. No LLM
weight is touched, and a poor adapter can only lower speculative acceptance, never
the transcript.

    python tools/fit_draft_adapter.py --utterances 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prune_decoder import calibration_files  # noqa: E402

DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--utterances", type=int, default=300)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--big-encoder", default="artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine")
    parser.add_argument("--small-encoder", default="artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/draft-adapter.pt")
    args = parser.parse_args()

    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    big_model = ROOT / "models/Qwen3-ASR-1.7B-hf"
    small_model = ROOT / "models/Qwen3-ASR-0.6B-hf"
    big_processor = AutoProcessor.from_pretrained(str(big_model))
    small_processor = AutoProcessor.from_pretrained(str(small_model))
    big_encoder = TensorRTAudioEncoder(Path(args.big_encoder))
    small_encoder = TensorRTAudioEncoder(Path(args.small_encoder))

    files = calibration_files(args.utterances, args.seed)
    total_gram = None
    total_cross = None
    total_small_gram = None
    total_rows = 0
    mismatches = 0
    for index, path in enumerate(files, 1):
        audio = load_audio(path)
        big_inputs = big_processor.apply_transcription_request(audio=audio, language="English")
        small_inputs = small_processor.apply_transcription_request(audio=audio, language="English")
        big_features = big_inputs["input_features"]
        small_features = small_inputs["input_features"]
        big = big_encoder.encode(big_features, torch.ones((1, big_features.shape[-1]), dtype=torch.int32))
        small = small_encoder.encode(small_features, torch.ones((1, small_features.shape[-1]), dtype=torch.int32))
        big = big.reshape(-1, big.shape[-1]).float()
        small = small.reshape(-1, small.shape[-1]).float()
        count = min(big.shape[0], small.shape[0])
        if big.shape[0] != small.shape[0]:
            mismatches += 1
        big = big[:count]
        small = small[:count]
        total_gram = big.t() @ big if total_gram is None else total_gram + big.t() @ big
        total_cross = big.t() @ small if total_cross is None else total_cross + big.t() @ small
        total_small_gram = small.t() @ small if total_small_gram is None else total_small_gram + small.t() @ small
        total_rows += count
        if index % 50 == 0:
            print(f"  {index}/{len(files)}", flush=True)

    dim = total_gram.shape[0]
    ridge = args.ridge * total_gram.diagonal().mean()
    normal = total_gram + torch.eye(dim, device=total_gram.device) * ridge
    weight = torch.linalg.solve(normal, total_cross)
    residual = float(torch.trace(weight.t() @ total_gram @ weight)
                     - 2 * torch.trace(weight.t() @ total_cross)
                     + torch.trace(total_small_gram))
    energy = float(torch.trace(total_small_gram))
    torch.save({"weight": weight.to(torch.float16).cpu()}, args.output)
    print(json.dumps({
        "utterances": len(files),
        "positions": total_rows,
        "tokenCountMismatches": mismatches,
        "adapter": str(args.output),
        "relativeError": round((max(residual, 0.0) / energy) ** 0.5, 4),
        "explainedEnergy": round(1 - max(residual, 0.0) / energy, 4),
    }, indent=2))


if __name__ == "__main__":
    main()
