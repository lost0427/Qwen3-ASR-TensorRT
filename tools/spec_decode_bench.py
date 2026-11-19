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
import os
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BENCH = ROOT.parent / "ASRbenchmark"
sys.path.insert(0, str(BENCH / "tools"))

DATA = Path(os.environ.get("QASR_AUDIO_ROOT", r"C:\Users\z1\Documents\ASR-data\ls"))

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


def _clock_line(label: str) -> None:
    """Logs the SM clock: this laptop idles at 210 MHz and throttles near 83 C."""
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"clocks {label}: {out.stdout.strip()}", flush=True)
    except (OSError, subprocess.SubprocessError):
        pass


def encode_pass(size: str, rows) -> tuple[list[dict], list[dict]]:
    """Feature extraction and the audio encoder, timed separately.

    The mel front end is CPU work; the encoder is asynchronous until its own
    synchronize, so both legs are measured to the host.
    """
    from transformers import AutoProcessor

    from qwen3_asr_trt.audio import load_audio
    from qwen3_asr_trt.features import GpuLogMel, PromptTemplate
    from qwen3_asr_trt.tensorrt_encoder import TensorRTAudioEncoder, _post_cnn_tokens

    config = SIZES[size]
    processor = AutoProcessor.from_pretrained(str(config["model"]))
    encoder = TensorRTAudioEncoder(Path(config["encoder"]))
    mel = GpuLogMel(processor.feature_extractor)
    template = PromptTemplate(processor, "English")
    outputs = []
    timings = []
    for row in rows:
        started = time.perf_counter()
        features, mask, valid_frames = mel(load_audio(DATA / row["flac"]))
        ids = template(_post_cnn_tokens(valid_frames))
        torch.cuda.synchronize()
        preprocess_ms = (time.perf_counter() - started) * 1000
        torch.cuda.synchronize()
        encoding = time.perf_counter()
        embeds = encoder.encode(features, mask).to(torch.float16)
        torch.cuda.synchronize()
        encode_ms = (time.perf_counter() - encoding) * 1000
        timings.append({"preprocess": preprocess_ms, "encode": encode_ms})
        outputs.append({
            "ids": [int(value) for value in ids[0].tolist()],
            "audio": embeds.reshape(-1, embeds.shape[-1]).cpu(),
        })
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, timings


def build_decoder(size: str, prefill: Path | None = None):
    from qwen3_asr_trt.w4a16_decoder import W4A16ASRDecoder

    config = SIZES[size]
    return W4A16ASRDecoder(str(config["packed"]), Path(prefill or config["prefill"]),
                           max_cache=1024, group=32)


def prompt(decoder, record) -> torch.Tensor:
    ids = torch.tensor([record["ids"]], dtype=torch.long, device="cuda")
    return decoder.embed[ids].masked_scatter(
        (ids == 151676).unsqueeze(-1), record["audio"].to("cuda")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--min-seconds", type=float, default=0.0,
                        help="keep only utterances at least this long")
    parser.add_argument("--draft-prefill", type=Path,
                        help="override the draft prefill engine (for example an INT8 build)")
    parser.add_argument("--target-prefill", type=Path,
                        help="override the target prefill engine")
    parser.add_argument("--adapter", type=Path,
                        help="project the target's audio embeddings into the draft's space "
                             "instead of running the small encoder")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--baseline-limit", type=int, default=24)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/spec-decode.json")
    args = parser.parse_args()
    if not 1 <= args.draft_tokens <= 7:
        raise SystemExit("--draft-tokens must be 1..7 (the batched kernel handles 8 rows)")

    from windows_benchmark import normalize

    rows = [json.loads(line) for line in (BENCH / "data/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if (DATA / row["flac"]).is_file()]
    if args.min_seconds:
        rows = [row for row in rows if row.get("durationSeconds", 0) >= args.min_seconds]
    rows = rows[: args.limit]

    started = time.perf_counter()
    if args.adapter:
        targets, target_encoder = encode_pass("1.7b", rows)
        weight = torch.load(args.adapter, map_location="cpu")["weight"].float().to("cuda")
        drafts = []
        for record in targets:
            audio = record["audio"].to("cuda").float()
            drafts.append({
                "ids": record["ids"],
                "audio": (audio @ weight).half().cpu(),
            })
        draft_encoder = [{"preprocess": 0.0, "encode": 0.0} for _ in rows]
    else:
        drafts, draft_encoder = encode_pass("0.6b", rows)
        targets, target_encoder = encode_pass("1.7b", rows)
    print(f"encoder passes: {time.perf_counter() - started:.1f} s", flush=True)

    draft_decoder = build_decoder("0.6b", args.draft_prefill)
    target_decoder = build_decoder("1.7b", args.target_prefill)
    print(f"resident: {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

    # The laptop GPU idles at ~210 MHz and only reaches its boost clock after a
    # few seconds of work, so every measured stage is inflated without this.
    warm = torch.randn(4096, 2048, device="cuda", dtype=torch.float16)
    probe = torch.randn(512, 2048, device="cuda", dtype=torch.float16)
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        for _ in range(20):
            _ = probe @ warm.t()
        torch.cuda.synchronize()
    del warm, probe
    torch.cuda.empty_cache()
    _clock_line("after warm-up")

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

    def spec_pass(draft, target):
        started = time.perf_counter()
        draft_decoder.prefill_prompt(prompt(draft_decoder, draft), reserve=args.max_new_tokens)
        torch.cuda.synchronize()
        draft_prefill_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        target_decoder.prefill_prompt(prompt(target_decoder, target), reserve=args.max_new_tokens)
        torch.cuda.synchronize()
        target_prefill_ms = (time.perf_counter() - started) * 1000
        prefill_ms = draft_prefill_ms + target_prefill_ms
        draft_first = int(draft_decoder._token.item())
        target_first = int(target_decoder._token.item())

        produced = [target_first]
        accepted = proposed = rounds = 0
        started = time.perf_counter()
        while len(produced) < args.max_new_tokens and produced[-1] not in EOS:
            # The draft must cover whatever the target has produced so far, and a
            # grown cache invalidates its captured graph.
            draft_decoder.ensure_capacity(int(target_decoder._pos.item()) + args.draft_tokens + 2)
            draft_graph = draft_decoder._graph()
            # Mirror the target's state without a host round-trip.
            draft_decoder._pos.copy_(target_decoder._pos)
            draft_decoder._token.copy_(target_decoder._token)
            draft_decoder._generated.zero_()
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
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "tokens": produced,
            "firstMatches": int(draft_first == target_first),
            "accepted": accepted,
            "proposed": proposed,
            "rounds": rounds,
            "specMs": elapsed,
            "prefillMs": prefill_ms,
            "draftPrefillMs": draft_prefill_ms,
            "targetPrefillMs": target_prefill_ms,
        }

    def plain_pass(target):
        started = time.perf_counter()
        target_decoder.prefill_prompt(prompt(target_decoder, target), reserve=args.max_new_tokens)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        tokens = target_decoder.decode_loop(args.max_new_tokens)
        torch.cuda.synchronize()
        return {"tokens": tokens, "ms": (time.perf_counter() - started) * 1000,
                "prefillMs": prefill_ms}

    results = []
    baselines = []
    for index, (row, draft, target) in enumerate(zip(rows, drafts, targets)):
        record = {"reference": row["reference"]}
        # Alternate the order so neither path is always the one running on the
        # colder (or hotter) card.
        order = ("spec", "plain") if index % 2 else ("plain", "spec")
        baseline = None
        for stage in order:
            if stage == "plain":
                if index >= args.baseline_limit:
                    continue
                baseline = plain_pass(target)
            else:
                record.update(spec_pass(draft, target))
        results.append(record)
        if baseline is not None:
            baselines.append(baseline)
            record["baselineMs"] = baseline["ms"]
        if (index + 1) % 10 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(str(SIZES["1.7b"]["model"])).tokenizer

    def strip(tokens):
        if 151704 in tokens:
            tokens = tokens[tokens.index(151704) + 1:]
        return tokenizer.decode(tokens, skip_special_tokens=True).strip()

    for record, base in zip(results, baselines):
        record["text"] = strip(record["tokens"])
        errors, words = edit_distance(normalize(record["reference"]), normalize(record["text"]))
        record["errors"] = errors
        record["words"] = words
        del record["tokens"]
        base["text"] = strip(base["tokens"])
        base["tokens"] = len(base["tokens"]) - 1

    baseline = baselines
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
        "medianDraftEncodeMs": round(statistics.median(
            item["encode"] for item in draft_encoder), 2),
        "medianTargetEncodeMs": round(statistics.median(
            item["encode"] for item in target_encoder), 2),
        "medianDraftPreprocessMs": round(statistics.median(
            item["preprocess"] for item in draft_encoder), 2),
        "medianTargetPreprocessMs": round(statistics.median(
            item["preprocess"] for item in target_encoder), 2),
        "medianBaselinePrefillMs": round(statistics.median(
            base["prefillMs"] for base in baseline), 2) if baseline else None,
        "medianSpecPrefillMs": round(statistics.median(item["prefillMs"] for item in results), 2),
        "medianDraftPrefillMs": round(statistics.median(item["draftPrefillMs"] for item in results), 2),
        "medianTargetPrefillMs": round(statistics.median(item["targetPrefillMs"] for item in results), 2),
        "medianBaselinePipelineMs": round(statistics.median(
            base["prefillMs"] + base["ms"] for base in baseline), 1) if baseline else None,
        "medianSpecPipelineMs": round(statistics.median(
            item["prefillMs"] + item["specMs"] for item in results), 1),
        "medianSpecFullMs": round(statistics.median(
            draft_encoder[index]["encode"] + target_encoder[index]["encode"]
            + item["prefillMs"] + item["specMs"]
            for index, item in enumerate(results)), 1),
        "medianSpecEndToEndMs": round(statistics.median(
            target_encoder[index]["preprocess"] + draft_encoder[index]["encode"]
            + target_encoder[index]["encode"] + item["prefillMs"] + item["specMs"]
            for index, item in enumerate(results)), 1),
        "medianBaselineEndToEndMs": round(statistics.median(
            target_encoder[index]["preprocess"] + target_encoder[index]["encode"]
            + base["prefillMs"] + base["ms"]
            for index, base in enumerate(baseline)), 1) if baseline else None,
        "peakGiB": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "deviceFreeMiB": round(torch.cuda.mem_get_info()[0] / 2**20),
        "adapter": str(args.adapter) if args.adapter else None,
    }
    print(json.dumps(summary, indent=2))
    _clock_line("after the run")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"summary": summary, "records": results, "baseline": baseline}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
