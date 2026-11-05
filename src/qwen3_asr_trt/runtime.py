from pathlib import Path
from typing import Protocol


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path, language: str | None = None) -> str: ...


class Bf16ReferenceTranscriber:
    def __init__(self, model_id: str, max_new_tokens: int = 256) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the project environment with `uv sync` first") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("BF16 reference inference requires a CUDA GPU")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA GPU does not report BF16 support")

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
        )
        self.max_new_tokens = max_new_tokens

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        inputs = self.processor.apply_transcription_request(
            audio=str(audio_path), language=language
        ).to(self.model.device, self.model.dtype)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        result = self.processor.decode(generated_ids, return_format="transcription_only")
        return result[0]
