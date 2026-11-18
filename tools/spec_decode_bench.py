"""Greedy speculative decoding: 0.6B drafts, 1.7B verifies.

The verification pass feeds the draft block through the target in one batched
forward, so the INT4 weights are read once per round instead of once per token.
Under greedy decoding the output matches the target's own greedy transcript.

The audio encoders run first and are released, so only the two decoders and their
prefill engines stay resident during the timed loop.

    python tools/spec_decode_bench.py --limit 64 --draft-tokens 4
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT.parent / "ASRbenchmark"
sys.path.insert(0, str(BENCH / "tools"))

DATA = Path(r"C:\Users\z1\Documents\ASR-data\ls")

SIZES = {
    "0.6b": {
        "model": ROOT / "models/Qwen3-ASR-0.6B-hf",
        "packed": ROOT / "artifacts/qwen3-asr-0.6b/decoder-int4-g32",
        "encoder": ROOT / "artifacts/qwen3-asr-0.6b/encoder-window-win.bf16.engine",
        "prefill": ROOT / "artifacts/qwen3-asr-0.6b/decoder-trt-awq32-tc100/asr-prefill.fp16.engine",
    },
    "1.7b": {
        "model": ROOT / "models/Qwen3-ASR-1.7B-hf",
        "packed": ROOT / "artifacts/qwen3-asr-1.7b/decoder-int4-g32",
        "encoder": ROOT / "artifacts/qwen3-asr-1.7b/encoder-window-win.bf16.engine",
        "prefill": ROOT / "artifacts/qwen3-asr-1.7b/decoder-prefill-2048/asr-prefill.fp16.engine",
    },
}

EOS = (151643, 151645)


def encode_pass(size: str, rows) -> tuple[list[dict], float]:
    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder

    config = SIZES[size]
    processor = AutoProcessor.from_pretrained(str(config["model"]))
    encoder = TensorRTAudioEncoder(Path(config["encoder"]))
    started = time.perf_counter()
    outputs = []
    for row in rows:
        inputs = processor.apply_transcription_request(
            audio=load_audio(DATA / row["flac"]), language="English"
        )
        features = inputs["input_features"]
        mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
        embeds = encoder.encode(features, mask).to(torch.float16)
        outputs.append({
            "ids": [int(value) for value in inputs["input_ids"][0].tolist()],
            "audio": embeds.reshape(-1, embeds.shape[-1]).cpu(),
        })
    elapsed = (time.perf_counter() - started) * 1000 / max(1, len(rows))
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, elapsed


def build_decoder(size: str):
    from qwen3_asr_trt.w4a16_decoder import W4A16ASRDecoder

    config = SIZES[size]
    return W4A16ASRDecoder(str(config["packed"]), Path(config["prefill"]), max_cache=1024, group=32)


def prompt(decoder, record) -> torch.Tensor:
    ids = torch.tensor([record["ids"]], dtype=torch.long, device="cuda")
    return decoder.embed[ids].masked_scatter(
        (ids == 151676).unsqueeze(-1), record["audio"].to("cuda")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--baseline-limit", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/spec-decode.json")
    args = parser.parse_args()
    if not 1 <= args.draft_tokens <= 7:
        raise SystemExit("--draft-tokens must be 1..7 (the batched kernel handles 8 rows)")

    from windows_benchmark import normalize

    rows = [json.loads(line) for line in (BENCH / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()][: args.limit]

    started = time.perf_counter()
    drafts, draft_encoder_ms = encode_pass("0.6b", rows)
    targets, target_encoder_ms = encode_pass("1.7b", rows)
    print(f"encoder passes: {time.perf_counter() - started:.1f} s", flush=True)

    draft_decoder = build_decoder("0.6b")
    target_decoder = build_decoder("1.7b")
    print(f"resident: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    # Self-check: the target's own next tokens must all be accepted by `verify`.
    target_decoder.prefill_prompt(prompt(target_decoder, targets[0]), reserve=64)
    base = int(target_decoder._pos.item())
    first = int(target_decoder._token.item())
    expected = []
    for _ in range(3):
        target_decoder._step()
        expected.append(int(target_decoder._token.item()))
    target_decoder.rewind(base, first)
    matched, _ = target_decoder.verify(expected)
    print(f"self-check: {matched}/3 own tokens accepted", flush=True)
    if matched != 3:
        raise SystemExit("batched verify disagrees with the single-token path")

    def edit_distance(reference, hypothesis):
        left, right = reference.split(), hypothesis.split()
        row_values = list(range(len(right) + 1))
        for i, token in enumerate(left):
            current = [i + 1]
            for j, other in enumerate(right):
                current.append(min(current[-1] + 1, row_values[j + 1] + 1, row_values[j] + (token != other)))
            row_values = current
        return row_values[-1], len(left)

    results = []
    for index, (row, draft, target) in enumerate(zip(rows, drafts, targets), 1):
        draft_decoder.prefill_prompt(prompt(draft_decoder, draft), reserve=args.max_new_tokens)
        target_decoder.prefill_prompt(prompt(target_decoder, target), reserve=args.max_new_tokens)
        draft_first = int(draft_decoder._token.item())
        target_first = int(target_decoder._token.item())

        produced = [target_first]
        accepted = proposed = rounds = 0
        started = time.perf_counter()
        draft_graph = draft_decoder._graph()
        while len(produced) < args.max_new_tokens and produced[-1] not in EOS:
            # Mirror the target's state without a host round-trip.
            draft_decoder._pos.copy_(target_decoder._pos)
            draft_decoder._token.copy_(target_decoder._token)
            for _ in range(args.draft_tokens):
                draft_graph.replay()
            torch.cuda.synchronize()
            generated = int(draft_decoder._generated.item())
            candidates = draft_decoder._produced[generated - args.draft_tokens:generated].tolist()
            if not candidates:
                break
            matched, emitted = target_decoder.verify(candidates)
            produced.extend(emitted)
            accepted += matched
            proposed += len(candidates)
            rounds += 1
            # The target accepted everything, so it is one row ahead of what the
            # draft has written. Fill that row before rewinding.
            if matched == len(candidates):
                draft_decoder._pos.copy_(target_decoder._pos)
                draft_decoder._pos.sub_(1)
                draft_decoder._token.fill_(candidates[-1])
                draft_graph.replay()
            draft_decoder._pos.copy_(target_decoder._pos)
            draft_decoder._token.copy_(target_decoder._token)
        elapsed = (time.perf_counter() - started) * 1000
        results.append({
            "tokens": produced,
            "firstMatches": int(draft_first == target_first),
            "accepted": accepted,
            "proposed": proposed,
            "rounds": rounds,
            "specMs": elapsed,
            "reference": row["reference"],
        })
        if index % 10 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(str(SIZES["1.7b"]["model"])).tokenizer
    for record in results:
        tokens = record["tokens"]
        if 151704 in tokens:
            tokens = tokens[tokens.index(151704) + 1:]
        record["text"] = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        errors, words = edit_distance(normalize(record["reference"]), normalize(record["text"]))
        record["errors"] = errors
        record["words"] = words
        del record["tokens"]

    # Baseline: the target alone on the same prompts, to confirm the output is
    # the same greedy transcript and to time the unassisted path.
    baseline = []
    for row, target in list(zip(rows, targets))[: args.baseline_limit]:
        target_decoder.prefill_prompt(prompt(target_decoder, target), reserve=args.max_new_tokens)
        started = time.perf_counter()
        tokens = target_decoder.decode_loop(args.max_new_tokens)
        elapsed = (time.perf_counter() - started) * 1000
        if 151704 in tokens:
            tokens = tokens[tokens.index(151704) + 1:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        baseline.append({"text": text, "ms": elapsed, "tokens": len(tokens)})

    identical = sum(1 for record, base in zip(results, baseline) if record["text"] == base["text"])
    accepted = sum(item["accepted"] for item in results)
    proposed = sum(item["proposed"] for item in results)
    errors = sum(item["errors"] for item in results)
    words = sum(item["words"] for item in results)
    summary = {
        "utterances": len(results),
        "draftTokens": args.draft_tokens,
        "acceptanceRate": accepted / max(1, proposed),
        "meanAcceptedPerRound": accepted / max(1, sum(item["rounds"] for item in results)),
        "firstTokenMatches": sum(item["firstMatches"] for item in results),
        "medianSpecMs": statistics.median(item["specMs"] for item in results),
        "medianBaselineMs": statistics.median(item["ms"] for item in baseline) if baseline else None,
        "baselineUtterances": len(baseline),
        "identicalTranscripts": identical,
        "corpusWER": errors / max(1, words),
        "draftEncoderMs": round(draft_encoder_ms, 2),
        "targetEncoderMs": round(target_encoder_ms, 2),
        "peakGiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(summary, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": results, "baseline": baseline}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
