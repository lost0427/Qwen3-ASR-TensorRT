"""Generate the binary assets the native runtime needs.

mel.bin       : mel filter bank and Hann window, taken from the Python feature extractor
tokenizer.bin : byte-level vocabulary, special token ids, prompt prefix and per-language suffix
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np


def bytes_to_unicode() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(
        range(ord("®"), ord("ÿ") + 1)
    )
    cs = bs[:]
    n = 0
    for byte in range(256):
        if byte not in bs:
            bs.append(byte)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF checkpoint directory")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from transformers import AutoProcessor

    model = Path(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(str(model))
    fe = processor.feature_extractor

    filters = np.asarray(fe.mel_filters, dtype=np.float32).T.copy()  # [mel, freq]
    n_mel, n_freq = filters.shape
    n_fft = int(fe.n_fft)
    hop = int(fe.hop_length)
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)  # torch.hann_window(periodic=True)

    with (out / "mel.bin").open("wb") as handle:
        handle.write(struct.pack("<4I", n_mel, n_freq, n_fft, hop))
        handle.write(filters.astype("<f4").tobytes())
        handle.write(window.astype("<f4").tobytes())

    tokenizer = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    inv = bytes_to_unicode()
    vocab = tokenizer["model"]["vocab"]
    size = max(vocab.values()) + 1
    tokens = [b""] * size
    for text, index in vocab.items():
        tokens[index] = bytes(inv[ch] for ch in text)
    special = [t["id"] for t in tokenizer.get("added_tokens", []) if t.get("special")]

    import transformers.models.qwen3_asr.processing_qwen3_asr as proc

    names = sorted({name for name in proc.LANGUAGE_CODE_TO_NAME.values()})
    prefix = None
    suffix: dict[str, list[int]] = {}
    audio_token = None
    silence = np.zeros(16000, dtype=np.float32)
    for name in names:
        ids = processor.apply_transcription_request(audio=silence, language=name)["input_ids"][0].tolist()
        audio_token = ids[9] if audio_token is None else audio_token
        first = ids.index(audio_token)
        last = len(ids) - 1 - ids[::-1].index(audio_token)
        if prefix is None:
            prefix = ids[:first]
        elif prefix != ids[:first]:
            raise RuntimeError(f"prefix changed for {name}")
        suffix[name] = ids[last + 1 :]

    with (out / "tokenizer.bin").open("wb") as handle:
        handle.write(struct.pack("<I", size))
        for row in tokens:
            handle.write(struct.pack("<I", len(row)))
            handle.write(row)
        handle.write(struct.pack("<I", len(special)))
        handle.write(struct.pack(f"<{len(special)}I", *special))
        handle.write(struct.pack("<I", len(prefix or [])))
        handle.write(struct.pack(f"<{len(prefix or [])}I", *(prefix or [])))
        handle.write(struct.pack("<I", int(audio_token)))
        handle.write(struct.pack("<I", len(suffix)))
        for name, ids in suffix.items():
            raw = name.encode("utf-8")
            handle.write(struct.pack("<I", len(raw)))
            handle.write(raw)
            handle.write(struct.pack("<I", len(ids)))
            handle.write(struct.pack(f"<{len(ids)}I", *ids))

    print(
        f"mel {n_mel}x{n_freq} n_fft {n_fft} hop {hop}; "
        f"vocab {size}, special {len(special)}, languages {len(suffix)}, "
        f"prefix {len(prefix or [])}, audio token {audio_token}"
    )
    print(f"wrote {out/'mel.bin'} {np.round((out/'mel.bin').stat().st_size/2**20, 2)} MB")
    print(f"wrote {out/'tokenizer.bin'} {np.round((out/'tokenizer.bin').stat().st_size/2**20, 2)} MB")


if __name__ == "__main__":
    main()
