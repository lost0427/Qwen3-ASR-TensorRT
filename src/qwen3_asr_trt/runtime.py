from pathlib import Path
import time
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


class TensorRTExLlamaTranscriber:
    def __init__(
        self,
        model_id: str,
        decoder_dir: Path,
        engine_path: Path,
        max_new_tokens: int = 256,
        max_context_tokens: int = 8192,
        sampler=None,
    ) -> None:
        try:
            import torch
            from exllamav3 import Cache, Config, Generator, MMEmbedding, Model, Tokenizer
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the project environment with `uv sync`") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT and ExLlamaV3 inference requires a CUDA GPU")

        from .tensorrt_encoder import TensorRTAudioEncoder

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.encoder = TensorRTAudioEncoder(engine_path)
        self.config = Config.from_directory(str(decoder_dir))
        self.model = Model.from_config(self.config)
        self.model.caps["mrope"] = False
        self.cache = Cache(self.model, max_num_tokens=max_context_tokens, max_batch_size=1)
        for _ in self.model.load_gen(device="cuda:0", progressbar=True):
            pass
        self.tokenizer = Tokenizer(self.config)
        self.generator = Generator(self.model, self.cache, self.tokenizer, max_batch_size=1)
        self.mm_embedding_type = MMEmbedding
        self.max_new_tokens = max_new_tokens
        self.sampler = sampler
        self.last_timings: dict[str, float] = {}

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        torch = self.torch
        total_start = time.perf_counter()
        inputs = self.processor.apply_transcription_request(
            audio=str(audio_path), language=language
        )
        torch.cuda.synchronize()
        preprocessing_ms = (time.perf_counter() - total_start) * 1000

        encoder_start = time.perf_counter()
        embeddings = self.encoder.encode(
            inputs["input_features"], inputs["input_features_mask"]
        ).to(dtype=torch.float16)
        torch.cuda.synchronize()
        encoder_ms = (time.perf_counter() - encoder_start) * 1000

        audio_pad = self.processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        audio_positions = inputs["input_ids"][0] == audio_pad
        if int(audio_positions.sum()) != embeddings.shape[0]:
            raise RuntimeError(
                f"Encoder returned {embeddings.shape[0]} tokens for "
                f"{int(audio_positions.sum())} audio placeholders"
            )

        alias = "<$QWEN3_ASR_AUDIO$>"
        audio_embedding = self.mm_embedding_type(
            embeddings=embeddings,
            token_string=torch.full((1, embeddings.shape[0]), -1, dtype=torch.long),
            text_alias=alias,
        )
        prompt = self.processor.tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=False
        )
        prompt = prompt.replace("<|audio_pad|>" * embeddings.shape[0], alias, 1)
        decoder_start = time.perf_counter()
        output, generation = self.generator.generate(
            prompt,
            max_new_tokens=self.max_new_tokens,
            sampler=self.sampler,
            stop_conditions=[151643, 151645],
            encode_special_tokens=True,
            completion_only=True,
            embeddings=[audio_embedding],
            return_last_results=True,
        )
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - total_start) * 1000
        decoder_ms = (time.perf_counter() - decoder_start) * 1000
        self.last_timings = {
            "preprocessingMs": preprocessing_ms,
            "encoderMs": encoder_ms,
            "decoderMs": decoder_ms,
            "decoderPrefillMs": generation["time_prefill"] * 1000,
            "decoderGenerationMs": generation["time_generate"] * 1000,
            "promptTokens": generation["prompt_tokens"],
            "decodeTokens": generation["new_tokens"],
            "otherMs": max(0.0, total_ms - preprocessing_ms - encoder_ms - decoder_ms),
            "totalMs": total_ms,
        }
        return output.strip()
