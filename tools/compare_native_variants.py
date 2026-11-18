"""Interleaved timing of native decoder variants on the same utterances.

Loads each variant through the C++ runtime, alternating variants across passes so
GPU clock drift hits them equally, and reports median per-stage times plus decode
milliseconds per generated token.

    python tools/compare_native_variants.py --limit 300
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT.parent / "ASRbenchmark"
DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))
BINARY = ROOT / "native/build/qwen3-asr-native.exe"

VARIANTS = {
    "baseline": (ROOT / "artifacts/qwen3-asr-1.7b/decoder-int4-g32",
                 ROOT / "artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine"),
    "ffn75": (ROOT / "artifacts/experiments/1.7b-ffn75/packed",
              ROOT / "artifacts/experiments/1.7b-ffn75/prefill/asr-prefill.fp16.engine"),
    "depth26": (ROOT / "artifacts/experiments/1.7b-depth26/packed",
                ROOT / "artifacts/experiments/1.7b-depth26/prefill/asr-prefill.fp16.engine"),
    "depth26b": (ROOT / "artifacts/experiments/1.7b-depth26b/packed",
                 ROOT / "artifacts/experiments/1.7b-depth26b/prefill/asr-prefill.fp16.engine"),
    "ffn875": (ROOT / "artifacts/experiments/1.7b-ffn875/packed",
               ROOT / "artifacts/experiments/1.7b-ffn875/prefill/asr-prefill.fp16.engine"),
}

TIME_RE = re.compile(
    r"mel ([\d.]+) encoder ([\d.]+) prefill ([\d.]+) decode ([\d.]+) total ([\d.]+) \| frames \d+ "
    r"prompt \d+ tokens (\d+)"
)


def normalize(text: str) -> str:
    keep = []
    for char in text.upper():
        if char.isalnum() or char == "'" or char == " ":
            keep.append(char)
        else:
            keep.append(" ")
    return " ".join("".join(keep).split())


def edit_distance(reference: str, hypothesis: str) -> tuple[int, int]:
    left, right = normalize(reference).split(), normalize(hypothesis).split()
    row = list(range(len(right) + 1))
    for i, token in enumerate(left):
        current = [i + 1]
        for j, other in enumerate(right):
            current.append(min(current[-1] + 1, row[j + 1] + 1, row[j] + (token != other)))
        row = current
    return row[-1], len(left)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--variants", nargs="*", default=["baseline", "ffn75", "depth26", "depth26b"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/native-variants.json")
    args = parser.parse_args()

    rows = [json.loads(line) for line in (BENCH / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()][: args.limit]
    paths = [str(DATA / row["flac"]) for row in rows]
    encoder = ROOT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine"

    results = {name: [] for name in args.variants}
    for pass_index in range(args.passes):
        for name in args.variants:
            packed, prefill = VARIANTS[name]
            run = subprocess.run(
                [str(BINARY), "--model", str(packed), "--encoder", str(encoder),
                 "--decoder", str(prefill), "--assets", str(ROOT / "native/assets"),
                 "--language", "English", *paths],
                capture_output=True, text=True, env={**os.environ, "QASR_TIMING": "1"},
            )
            texts = [json.loads(line)["text"] for line in run.stdout.splitlines() if line.startswith("{")]
            stages = []
            for line in run.stderr.splitlines():
                match = TIME_RE.search(line) if line.startswith("[time]") else None
                if match:
                    values = [float(match.group(i)) for i in range(1, 6)]
                    stages.append({
                        "mel": values[0], "encoder": values[1], "prefill": values[2],
                        "decode": values[3], "total": values[4], "tokens": int(match.group(6)),
                    })
            if len(texts) != len(rows) or len(stages) != len(rows):
                raise SystemExit(f"{name}: {len(texts)} texts / {len(stages)} timings for {len(rows)} inputs")
            errors = sum(edit_distance(row["reference"], text)[0] for row, text in zip(rows, texts))
            words = sum(edit_distance(row["reference"], text)[1] for row, text in zip(rows, texts))
            results[name].append({
                "pass": pass_index + 1,
                "medianMs": statistics.median(item["total"] for item in stages),
                "stages": {key: statistics.median(item[key] for item in stages)
                           for key in ("mel", "encoder", "prefill", "decode", "total")},
                "decodeMsPerToken": sum(item["decode"] for item in stages) / max(1, sum(item["tokens"] for item in stages)),
                "tokens": sum(item["tokens"] for item in stages),
                "corpusWER": errors / max(1, words),
            })
            print(f"pass {pass_index + 1} {name}: {results[name][-1]['medianMs']:.1f} ms, "
                  f"{results[name][-1]['decodeMsPerToken']:.3f} ms/token, "
                  f"WER {results[name][-1]['corpusWER'] * 100:.2f}%", flush=True)

    summary = {}
    for name, passes in results.items():
        summary[name] = {
            "medianMs": statistics.median(p["medianMs"] for p in passes),
            "decodeMsPerToken": statistics.median(p["decodeMsPerToken"] for p in passes),
            "corpusWER": statistics.median(p["corpusWER"] for p in passes),
            "stages": {key: statistics.median(p["stages"][key] for p in passes)
                       for key in ("mel", "encoder", "prefill", "decode", "total")},
        }
    print(json.dumps(summary, indent=2))
    args.output.write_text(json.dumps({"utterances": len(rows), "summary": summary, "passes": results}, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
