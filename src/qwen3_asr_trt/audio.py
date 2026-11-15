"""Audio loading without librosa: soundfile for decoding, soxr for resampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np

TARGET_RATE = 16000


def load_audio(path: str | Path) -> np.ndarray:
    """Read an audio file as mono float32 at 16 kHz."""
    import soundfile as sf

    pcm, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sample_rate != TARGET_RATE:
        import soxr

        pcm = soxr.resample(pcm, sample_rate, TARGET_RATE, quality="HQ")
    return np.ascontiguousarray(pcm, dtype=np.float32)
