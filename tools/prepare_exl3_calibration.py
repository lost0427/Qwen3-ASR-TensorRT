from __future__ import annotations

import argparse
from pathlib import Path


def transcripts(corpus: Path):
    for transcript_file in sorted(corpus.rglob("*.trans.txt")):
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            utterance_id, text = line.split(" ", 1)
            if text.strip():
                yield transcript_file.parent / f"{utterance_id}.flac", text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack LibriSpeech transcripts for ExLlamaV3 calibration.")
    parser.add_argument("corpus", type=Path, help="LibriSpeech/train-clean-100 directory")
    parser.add_argument("tokenizer", type=Path, help="Local Qwen3-ASR tokenizer directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=250)
    parser.add_argument("--columns", type=int, default=2048)
    args = parser.parse_args()
    if not args.corpus.is_dir() or not args.tokenizer.is_dir():
        parser.error("corpus and tokenizer must be existing directories")
    if args.rows < 1 or args.columns < 1:
        parser.error("rows and columns must be positive")

    try:
        import torch
        from safetensors.torch import save_file
        from transformers import AutoProcessor
    except ImportError as exc:
        raise SystemExit("Run with the Qwen3-ASR-TensorRT uv environment") from exc

    processor = AutoProcessor.from_pretrained(args.tokenizer, local_files_only=True)
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token or "<|im_end|>"
    encoded_rows = []
    current = []
    utterance_count = 0
    for audio_path, text in transcripts(args.corpus):
        if not audio_path.is_file():
            continue
        request = processor.apply_transcription_request(audio=str(audio_path), language="English")
        prompt_ids = request["input_ids"][0].tolist()
        response_ids = tokenizer.encode(text + eos, add_special_tokens=False)
        sample = prompt_ids + response_ids
        if len(sample) > args.columns:
            continue
        if current and len(current) + len(sample) > args.columns:
            encoded_rows.append(current)
            current = []
            if len(encoded_rows) == args.rows:
                break
        current.extend(sample)
        utterance_count += 1
    if current and len(encoded_rows) < args.rows:
        encoded_rows.append(current)
    if len(encoded_rows) < args.rows:
        raise RuntimeError(
            f"Only packed {len(encoded_rows)} rows; need {args.rows}. "
            "Increase the calibration corpus or reduce --rows/--columns."
        )

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    packed = torch.full((args.rows, args.columns), pad_id, dtype=torch.long)
    lengths = torch.empty((args.rows,), dtype=torch.int32)
    for index, row in enumerate(encoded_rows[: args.rows]):
        packed[index, : len(row)] = torch.tensor(row, dtype=torch.long)
        lengths[index] = len(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file({"input_ids": packed, "lengths": lengths}, str(args.output))
    print(
        f"rows={args.rows} columns={args.columns} packed_utterances={utterance_count} "
        f"tokens={int(lengths.sum())} output={args.output}"
    )


if __name__ == "__main__":
    main()
