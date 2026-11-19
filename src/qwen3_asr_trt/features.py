"""Log-mel front end on the GPU, matching ``Qwen3ASRFeatureExtractor``.

The Transformers feature extractor builds the spectrogram with ``torch.stft`` on
the CPU, which costs several milliseconds per utterance even though the GPU is
idle at that point. The arithmetic here is the same: a centred Hann STFT, the
power spectrum, the 128-band Slaney mel matrix, ``log10`` with the 1e-10 clamp,
the ``max - 8`` floor and the ``(x + 4) / 4`` rescale, then a right pad of the
mel axis to a multiple of ``2 * n_window``.
"""

from __future__ import annotations

import numpy as np


class GpuLogMel:
    """Turns mono 16 kHz float32 audio into the features the encoder expects."""

    def __init__(self, feature_extractor, device: str = "cuda") -> None:
        import torch

        self.torch = torch
        self.device = device
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.min_length = int(feature_extractor.min_length)
        self.multiple = int(feature_extractor.n_window) * 2
        self.window = torch.hann_window(self.n_fft, device=device)
        self.filters = torch.from_numpy(feature_extractor.mel_filters).to(
            device=device, dtype=torch.float32
        )
        # cuFFT caches one plan per transform length, so the length is rounded up
        # to a bucket. Zero padding at the tail only reaches the last frames, which
        # are cropped off again. Three seconds keeps the number of distinct lengths
        # small for speech while cutting the wasted transform work: a 30 s bucket
        # costs 1.0 ms per utterance, this one 0.66.
        self.bucket = 3 * int(feature_extractor.sampling_rate)
        # Build the plan for a common length once instead of on first speech.
        self(torch.zeros(self.bucket, dtype=torch.float32).numpy())

    def __call__(self, pcm: np.ndarray) -> tuple["torch.Tensor", "torch.Tensor", int]:
        """Returns ``features [1, mel, frames]``, an all-ones mask and the valid frame count."""
        torch = self.torch
        from torch.nn import functional as F

        total = max(int(pcm.shape[0]), self.min_length)
        transform = -(-total // self.bucket) * self.bucket
        valid_frames = total // self.hop_length

        waveform = torch.from_numpy(np.ascontiguousarray(pcm, dtype=np.float32))
        if total > pcm.shape[0]:
            waveform = F.pad(waveform, (0, total - pcm.shape[0]))
        waveform = waveform.to(self.device)
        # The reference uses torch.stft's default reflection at both ends, which
        # depends on the true last sample; pad it here, then zero fill to the
        # bucket so the transform length stays constant.
        waveform = waveform.view(1, 1, -1)
        waveform = F.pad(waveform, (self.n_fft // 2, self.n_fft // 2), mode="reflect").view(-1)
        if transform > total:
            waveform = F.pad(waveform, (0, transform - total))

        spectrum = torch.stft(
            waveform, self.n_fft, self.hop_length, window=self.window,
            center=False, return_complex=True,
        )
        # The last frame of a centred STFT is pure padding; the reference drops it.
        power = spectrum[..., :-1].abs() ** 2
        mel = self.filters.T @ power
        log = torch.clamp(mel, min=1e-10).log10()
        log = torch.maximum(log, log.max() - 8.0)
        log = (log + 4.0) / 4.0
        log = log[..., :valid_frames]

        padding = -valid_frames % self.multiple
        if padding:
            log = F.pad(log, (0, padding))
        features = log.unsqueeze(0).contiguous()
        mask = torch.ones((1, features.shape[-1]), dtype=torch.int32)
        return features, mask, valid_frames


class PromptTemplate:
    """Prompt ids around the audio placeholder run, built once per language.

    ``apply_transcription_request`` runs the tokenizer and the CPU mel front end
    on every call. The template only depends on the language, so the ids around
    the ``<|audio_pad|>`` run are cached and the run is expanded to the number of
    audio tokens this utterance produces.
    """

    def __init__(self, processor, language: str | None, audio_token: int = 151676) -> None:
        import numpy as np

        self.audio_token = audio_token
        sample = processor.apply_transcription_request(
            audio=np.zeros(16000, dtype=np.float32), language=language
        )
        ids = [int(value) for value in sample["input_ids"][0].tolist()]
        first = ids.index(audio_token)
        last = len(ids) - 1 - ids[::-1].index(audio_token)
        self.prefix = ids[:first]
        self.suffix = ids[last + 1:]

    def __call__(self, tokens: int, device: str = "cuda"):
        import torch

        ids = self.prefix + [self.audio_token] * int(tokens) + self.suffix
        return torch.tensor([ids], dtype=torch.long, device=device)
