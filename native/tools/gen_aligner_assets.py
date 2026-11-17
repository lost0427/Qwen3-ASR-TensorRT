"""Builds the forced-aligner assets the native runtime needs.

Writes `aligner_tokenizer.bin` (byte-level BPE vocabulary, merges, special ids
and a Unicode class table) and `aligner_embed.bin` (the FP16 token embedding
table) next to the ASR `mel.bin` / `tokenizer.bin`.
"""

from __future__ import annotations

import argparse
import json
import struct
import unicodedata
from pathlib import Path

import torch
from safetensors import safe_open

TOKENIZER_MAGIC = 0x4B544151  # "QATK"
EMBED_MAGIC = 0x424D4551  # "QEMB"


def byte_encoder() -> dict[int, str]:
    """GPT-2 byte-level mapping: raw byte -> printable unicode stand-in."""
    visible = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codes = visible[:]
    extra = 0
    for byte in range(256):
        if byte not in visible:
            visible.append(byte)
            codes.append(256 + extra)
            extra += 1
    return dict(zip(visible, (chr(code) for code in codes)))


def char_class(code: int) -> int:
    category = unicodedata.category(chr(code))
    if category in ("Lu", "Lt"):
        return 1
    if category == "Ll":
        return 2
    if category in ("Lm", "Lo"):
        return 3
    if category in ("Nd", "Nl", "No"):
        return 4
    return 0


def class_ranges() -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start = -1
    current = 0
    for code in range(0x110000):
        cls = char_class(code)
        if cls != current:
            if current:
                ranges.append((start, code - 1, current))
            current = cls
            start = code
    if current:
        ranges.append((start, 0x10FFFF, current))
    return ranges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Qwen3-ForcedAligner model directory")
    parser.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json (defaults to <model>/decoder)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_path = args.tokenizer
    if tokenizer_path is None:
        tokenizer_path = args.model / "decoder" / "tokenizer.json"
        if not tokenizer_path.is_file():
            tokenizer_path = args.model / "tokenizer.json"
    spec = json.loads(tokenizer_path.read_text(encoding="utf-8"))

    vocab = spec["model"]["vocab"]
    vocab_size = max(vocab.values()) + 1
    tokens = [""] * vocab_size
    for token, index in vocab.items():
        tokens[index] = token
    missing = [index for index, token in enumerate(tokens) if not token]
    if missing:
        raise RuntimeError(f"vocabulary has gaps at {missing[:8]}")

    mapping = byte_encoder()
    byte_ids = [vocab[mapping[byte]] for byte in range(256)]

    merges = []
    for entry in spec["model"]["merges"]:
        left, right = entry.split(" ") if isinstance(entry, str) else entry
        merged = left + right
        if merged not in vocab:
            raise RuntimeError(f"merge result missing from the vocabulary: {merged!r}")
        merges.append((vocab[left], vocab[right], vocab[merged]))

    added = spec["added_tokens"]
    specials = {}
    for entry in added:
        specials[entry["content"]] = entry["id"]
    wanted = {
        "audio_start": "<|audio_start|>",
        "audio_pad": "<|audio_pad|>",
        "audio_end": "<|audio_end|>",
        "timestamp": "<timestamp>",
        "asr_text": "<asr_text>",
    }
    special_ids = {}
    for key, content in wanted.items():
        if content not in specials:
            raise RuntimeError(f"missing added token {content}")
        special_ids[key] = specials[content]

    ranges = class_ranges()

    args.out.mkdir(parents=True, exist_ok=True)
    tokenizer_bin = args.out / "aligner_tokenizer.bin"
    with tokenizer_bin.open("wb") as handle:
        handle.write(struct.pack("<5I", TOKENIZER_MAGIC, 1, vocab_size, len(merges), len(ranges)))
        handle.write(
            struct.pack(
                "<5i",
                special_ids["audio_start"],
                special_ids["audio_pad"],
                special_ids["audio_end"],
                special_ids["timestamp"],
                special_ids["asr_text"],
            )
        )
        handle.write(struct.pack("<256I", *byte_ids))
        for token in tokens:
            raw = token.encode("utf-8")
            handle.write(struct.pack("<I", len(raw)))
            handle.write(raw)
        for left, right, merged in merges:
            handle.write(struct.pack("<3I", left, right, merged))
        for first, last, cls in ranges:
            handle.write(struct.pack("<3I", first, last, cls))

    safetensors = args.model / "model.safetensors"
    embed_bin = args.out / "aligner_embed.bin"
    with safe_open(str(safetensors), "pt", "cpu") as source:
        weight = source.get_tensor("thinker.model.embed_tokens.weight")
    table = weight.to(torch.float16).contiguous().numpy()
    if table.ndim != 2:
        raise RuntimeError(f"unexpected embedding shape {table.shape}")
    with embed_bin.open("wb") as handle:
        handle.write(struct.pack("<3I", EMBED_MAGIC, table.shape[0], table.shape[1]))
        handle.write(table.tobytes())

    print(
        f"{tokenizer_bin.name}: vocab {vocab_size}, merges {len(merges)}, ranges {len(ranges)}, "
        f"timestamp {special_ids['timestamp']} ({tokenizer_bin.stat().st_size / 1024**2:.1f} MB)"
    )
    print(
        f"{embed_bin.name}: {table.shape[0]} x {table.shape[1]} fp16 "
        f"({embed_bin.stat().st_size / 1024**2:.0f} MB)"
    )


if __name__ == "__main__":
    main()
