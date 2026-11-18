"""Evaluate a packed INT4 decoder variant on LibriSpeech test-clean.

This only measures; the variant is chosen on train-clean-100 statistics, so the
test set stays an evaluation set.

    python tools/eval_decoder_variant.py --experiment 1.7b-ffn75 --limit 200
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT.parent / "ASRbenchmark"
sys.path.insert(0, str(BENCH / "tools"))

DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))

SIZES = {
    "0.6b": {
        "model": ROOT / "models/Qwen3-ASR-0.6B-hf",
        "encoder": ROOT / "artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine",
    },
    "1.7b": {
        "model": ROOT / "models/Qwen3-ASR-1.7B-hf",
        "encoder": ROOT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
    },
}


def edit_distance(reference, hypothesis):
    left, right = reference.split(), hypothesis.split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (token != other)))
        row = current
    return row[-1], len(left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("0.6b", "1.7b"), default="1.7b")
    parser.add_argument("--experiment", help="name under artifacts/experiments")
    parser.add_argument("--packed", type=Path, help="packed decoder directory (overrides --experiment)")
    parser.add_argument("--prefill", type=Path, help="prefill engine (overrides --experiment)")
    parser.add_argument("--encoder", type=Path, help="audio encoder engine override")
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates every utterance")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    from windows_benchmark import normalize

    experiment = ROOT / "artifacts/experiments" / args.experiment if args.experiment else None
    packed = args.packed or experiment / "packed"
    prefill = args.prefill or experiment / "prefill/asr-prefill.fp16.engine"
    size = dict(SIZES[args.size])
    if args.encoder:
        size["encoder"] = args.encoder

    rows = [json.loads(line) for line in (BENCH / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()]
    if args.limit:
        rows = rows[: args.limit]

    from qwen3_asr_trt.runtime import TensorRTW4A16Transcriber

    started = time.perf_counter()
    transcriber = TensorRTW4A16Transcriber(
        str(size["model"]), prefill, size["encoder"],
        max_new_tokens=256, max_cache=512, decoder_dir=packed,
    )
    load_seconds = time.perf_counter() - started

    records = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        text = transcriber.transcribe(DATA / row["flac"], "English")
        elapsed = (time.perf_counter() - started) * 1000
        errors, words = edit_distance(normalize(row["reference"]), normalize(text))
        records.append({
            "flac": row["flac"],
            "reference": row["reference"],
            "hypothesis": text,
            "ms": elapsed,
            "errors": errors,
            "words": words,
            "timings": dict(transcriber.last_timings),
        })
        if index % 200 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    errors = sum(item["errors"] for item in records)
    words = sum(item["words"] for item in records)
    stages = {}
    for key in ("preprocessingMs", "encoderMs", "decoderPrefillMs", "decoderGenerationMs"):
        stages[key] = statistics.median(item["timings"][key] for item in records)
    summary = {
        "experiment": args.experiment or str(packed),
        "size": args.size,
        "utterances": len(records),
        "medianMs": statistics.median(item["ms"] for item in records),
        "p90Ms": statistics.quantiles([item["ms"] for item in records], n=10)[8],
        "corpusWER": errors / max(1, words),
        "errors": errors,
        "referenceWords": words,
        "medianStagesMs": stages,
        "decodeMsPerToken": sum(item["timings"]["decoderGenerationMs"] for item in records)
        / max(1, sum(item["timings"]["decodeTokens"] for item in records)),
        "packedMb": (packed / "model.safetensors").stat().st_size / 2**20,
        "engineMb": prefill.stat().st_size / 2**20,
        "loadSeconds": round(load_seconds, 2),
    }
    print(json.dumps(summary, indent=2))
    output = args.output or (experiment / "eval.json" if experiment else Path("eval.json"))
    output.write_text(json.dumps({"summary": summary, "records": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
