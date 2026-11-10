from __future__ import annotations

import time
import unicodedata
from pathlib import Path

import numpy as np


def _is_kept_char(ch: str) -> bool:
    if ch == "'":
        return True
    return unicodedata.category(ch)[0] in ("L", "N")


def _clean_token(token: str) -> str:
    return "".join(ch for ch in token if _is_kept_char(ch))


def _is_cjk_char(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF
        or 0x3400 <= code <= 0x4DBF
        or 0x20000 <= code <= 0x2A6DF
        or 0x2A700 <= code <= 0x2B73F
        or 0x2B740 <= code <= 0x2B81F
        or 0x2B820 <= code <= 0x2CEAF
        or 0xF900 <= code <= 0xFAFF
    )


class AlignerProcessor:
    """Text normalization and timestamp parsing for Qwen3-ForcedAligner."""

    def tokenize(self, text: str, language: str | None = None) -> list[str]:
        language = str(language or "").lower()
        if language == "japanese":
            try:
                import nagisa

                words = nagisa.tagging(text).words
            except ImportError:
                return list(text)
            return [token for word in words if (token := _clean_token(word))]
        if language == "korean":
            try:
                from soynlp.tokenizer import LTokenizer

                words = LTokenizer().tokenize(text)
            except ImportError:
                return list(text)
            return [token for word in words if (token := _clean_token(word))]

        tokens: list[str] = []
        for segment in text.split():
            cleaned = _clean_token(segment)
            buffer: list[str] = []
            for ch in cleaned:
                if _is_cjk_char(ch):
                    if buffer:
                        tokens.append("".join(buffer))
                        buffer = []
                    tokens.append(ch)
                else:
                    buffer.append(ch)
            if buffer:
                tokens.append("".join(buffer))
        return tokens

    def encode_timestamp(self, text: str, language: str) -> tuple[list[str], str]:
        words = self.tokenize(text, language)
        body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
        return words, "<|audio_start|><|audio_pad|><|audio_end|>" + body

    def fix_timestamp(self, values: np.ndarray) -> list[int]:
        data = [int(value) for value in values.tolist()]
        if not data:
            return []
        n = len(data)
        dp = [1] * n
        parent = [-1] * n
        for i in range(1, n):
            for j in range(i):
                if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        max_idx = dp.index(max(dp))
        normal = [False] * n
        index = max_idx
        while index != -1:
            normal[index] = True
            index = parent[index]
        result = data.copy()
        i = 0
        while i < n:
            if normal[i]:
                i += 1
                continue
            j = i
            while j < n and not normal[j]:
                j += 1
            left = next((result[k] for k in range(i - 1, -1, -1) if normal[k]), None)
            right = next((result[k] for k in range(j, n) if normal[k]), None)
            count = j - i
            if count <= 2:
                for k in range(i, j):
                    if left is None:
                        result[k] = right
                    elif right is None:
                        result[k] = left
                    else:
                        result[k] = left if (k - i + 1) <= (j - k) else right
            else:
                if left is not None and right is not None:
                    step = (right - left) / (count + 1)
                    for k in range(i, j):
                        result[k] = int(left + step * (k - i + 1))
                elif left is not None:
                    result[i:j] = [left] * count
                elif right is not None:
                    result[i:j] = [right] * count
            i = j
        return result

    def parse_timestamp(
        self, words: list[str], timestamp_ms: np.ndarray, step_ms: float = 80.0
    ) -> list[dict]:
        fixed = self.fix_timestamp(timestamp_ms)
        items = []
        for index, word in enumerate(words):
            items.append(
                {
                    "text": word,
                    "start_time": round(fixed[index * 2] * step_ms / 1000, 3),
                    "end_time": round(fixed[index * 2 + 1] * step_ms / 1000, 3),
                }
            )
        return items


class TensorRTExLlamaAligner:
    """TensorRT audio encoder + EXL3 text decoder forced aligner."""

    def __init__(self, model_id, decoder_dir, engine_path, max_context_tokens: int = 8192) -> None:
        try:
            import torch
            from exllamav3 import Config, MMEmbedding, Model, Tokenizer
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the project environment with `uv sync`") from exc

        from .tensorrt_encoder import TensorRTAudioEncoder

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_id), local_files_only=Path(str(model_id)).is_dir()
        )
        self.encoder = TensorRTAudioEncoder(Path(engine_path))
        self.config = Config.from_directory(str(decoder_dir))
        self.model = Model.from_config(self.config)
        for _ in self.model.load_gen(device="cuda:0", progressbar=False):
            pass
        self.tokenizer = Tokenizer(self.config)
        self.mm_embedding_type = MMEmbedding
        self.processor_text = AlignerProcessor()
        self.timestamp_token_id = int(
            self.processor.tokenizer.convert_tokens_to_ids("<timestamp>")
        )
        self.audio_pad_id = int(
            self.processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        )
        self.timestamp_segment_time = 80.0
        self.last_timings: dict[str, float] = {}

    def align(self, audio_path: Path, text: str, language: str) -> list[dict]:
        import soundfile as sf

        torch = self.torch
        total_started = time.perf_counter()
        pcm, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        if sample_rate != 16000:
            import librosa

            pcm = librosa.resample(pcm, orig_sr=sample_rate, target_sr=16000)
        words, prompt_text = self.processor_text.encode_timestamp(text, language)
        inputs = self.processor(
            text=[prompt_text], audio=[np.asarray(pcm, dtype=np.float32)],
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"]
        mask = inputs.get("feature_attention_mask")
        if mask is None:
            mask = torch.ones(
                (1, inputs["input_features"].shape[-1]), dtype=torch.int32
            )
        preprocessing_ms = (time.perf_counter() - total_started) * 1000

        encoder_started = time.perf_counter()
        embeddings = self.encoder.encode(inputs["input_features"], mask).to(torch.bfloat16)
        torch.cuda.synchronize()
        encoder_ms = (time.perf_counter() - encoder_started) * 1000

        audio_tokens = int((input_ids[0] == self.audio_pad_id).sum())
        alias = "<|qwen_audio_alias|>"
        prompt = self.processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)
        prompt = prompt.replace("<|audio_pad|>" * audio_tokens, alias, 1)
        audio_embedding = self.mm_embedding_type(
            embeddings=embeddings,
            token_string=torch.full((1, embeddings.shape[0]), -1, dtype=torch.long),
            text_alias=alias,
        )
        token_ids = self.tokenizer.encode(
            prompt, encode_special_tokens=True, embeddings=[audio_embedding]
        )
        decoder_started = time.perf_counter()
        logits = self.model.forward(
            token_ids,
            params={
                "attn_mode": "flash_attn_nc",
                "indexed_embeddings": [audio_embedding],
            },
        )
        torch.cuda.synchronize()
        decoder_ms = (time.perf_counter() - decoder_started) * 1000

        positions = token_ids[0] == self.timestamp_token_id
        raw = logits[0, positions, :5000].argmax(-1).cpu().numpy()
        items = self.processor_text.parse_timestamp(
            words, raw, self.timestamp_segment_time
        )
        total_ms = (time.perf_counter() - total_started) * 1000
        self.last_timings = {
            "preprocessingMs": preprocessing_ms,
            "encoderMs": encoder_ms,
            "decoderMs": decoder_ms,
            "totalMs": total_ms,
        }
        return items


def _load_aligner(model_id: str, dtype):
    try:
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as exc:
        raise RuntimeError(
            "build-aligner requires the `qwen-asr` package; install it in a "
            "separate build environment so it does not replace the project's transformers version"
        ) from exc
    return Qwen3ForcedAligner.from_pretrained(str(model_id), device_map="cpu", dtype=dtype)


def export_aligner_encoder(model_id: str, output_path: Path, precision: str = "bf16") -> None:
    import torch

    from .tensorrt_encoder import StaticAudioWindow

    if precision != "bf16":
        raise ValueError("The aligner encoder export currently supports bf16 only")
    aligner = _load_aligner(model_id, torch.bfloat16)
    thinker = aligner.model.thinker
    wrapper = StaticAudioWindow(
        thinker.audio_tower,
        getattr(thinker, "multi_modal_projector", None),
        apply_output_projection=True,
    ).eval().to("cuda:0")
    frames = wrapper.window_frames
    features = torch.zeros((1, 128, frames), dtype=torch.float32, device="cuda")
    mask = torch.ones((1, frames), dtype=torch.int32, device="cuda")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (features, mask),
        str(output_path),
        input_names=["input_features", "input_features_mask"],
        output_names=["audio_embeddings", "valid_tokens"],
        opset_version=20,
        dynamo=True,
    )


def export_aligner_decoder(model_id: str, output_dir: Path) -> Path:
    import json

    import torch
    from safetensors.torch import save_file

    aligner = _load_aligner(model_id, torch.float32)
    thinker = aligner.model.thinker
    config = thinker.config.text_config.to_dict()
    vocab_size = 152064
    config.update(
        {
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "vocab_size": vocab_size,
            "tie_word_embeddings": False,
            "eos_token_id": [151643, 151645],
            "pad_token_id": 151645,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    weights = {}
    for key, value in thinker.state_dict().items():
        if key.startswith("model."):
            weights[key] = value.contiguous()
        elif key == "lm_head.weight":
            padded = torch.zeros((vocab_size, value.shape[1]), dtype=value.dtype)
            padded[: value.shape[0]] = value
            weights[key] = padded.contiguous()
    save_file(weights, str(output_dir / "model.safetensors"))
    aligner.processor.tokenizer.save_pretrained(str(output_dir))
    return output_dir
