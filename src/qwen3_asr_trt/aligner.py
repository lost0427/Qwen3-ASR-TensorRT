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


def _prepare_aligner_inputs(
    processor, processor_text, encoder, audio_path, text: str, language: str
):
    """Shared preprocess + TensorRT audio encoding for both aligner backends."""
    import soundfile as sf
    import torch

    started = time.perf_counter()
    pcm, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    if sample_rate != 16000:
        import librosa

        pcm = librosa.resample(pcm, orig_sr=sample_rate, target_sr=16000)
    words, prompt_text = processor_text.encode_timestamp(text, language)
    inputs = processor(
        text=[prompt_text],
        audio=[np.asarray(pcm, dtype=np.float32)],
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"]
    mask = inputs.get("feature_attention_mask")
    if mask is None:
        mask = torch.ones((1, inputs["input_features"].shape[-1]), dtype=torch.int32)
    preprocessing_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    embeddings = encoder.encode(inputs["input_features"], mask)
    torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - started) * 1000
    return words, input_ids, embeddings, preprocessing_ms, encoder_ms


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
        torch = self.torch
        total_started = time.perf_counter()
        words, input_ids, embeddings, preprocessing_ms, encoder_ms = _prepare_aligner_inputs(
            self.processor, self.processor_text, self.encoder, audio_path, text, language
        )
        embeddings = embeddings.to(torch.bfloat16)

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


def _load_token_embedding(model_id, dtype):
    """Token embedding table for the aligner decoder input (text tokens only)."""
    import torch
    from safetensors import safe_open

    path = Path(str(model_id)) / "model.safetensors"
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(str(model_id), "model.safetensors"))
    with safe_open(str(path), "pt") as handle:
        weight = handle.get_tensor("thinker.model.embed_tokens.weight")
    weight = weight.to(dtype=dtype, device="cuda")
    return torch.nn.Embedding.from_pretrained(weight)


class TensorRTAligner:
    """TensorRT audio encoder + TensorRT text decoder forced aligner."""

    def __init__(self, model_id, engine_path, decoder_engine, decoder_precision: str = "bf16") -> None:
        try:
            import torch
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install the project environment with `uv sync`") from exc

        from .tensorrt_encoder import TensorRTAudioEncoder

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            str(model_id), local_files_only=Path(str(model_id)).is_dir()
        )
        self.encoder = TensorRTAudioEncoder(Path(engine_path))
        self.processor_text = AlignerProcessor()
        self.timestamp_token_id = int(
            self.processor.tokenizer.convert_tokens_to_ids("<timestamp>")
        )
        self.audio_pad_id = int(
            self.processor.tokenizer.convert_tokens_to_ids("<|audio_pad|>")
        )
        self.timestamp_segment_time = 80.0
        self.last_timings: dict[str, float] = {}
        self.decoder = TensorRTAlignDecoder(Path(decoder_engine))
        self.decoder_precision = decoder_precision
        self.token_embedding = _load_token_embedding(model_id, self.decoder.dtype)

    def align(self, audio_path: Path, text: str, language: str) -> list[dict]:
        torch = self.torch
        total_started = time.perf_counter()
        words, input_ids, embeddings, preprocessing_ms, encoder_ms = _prepare_aligner_inputs(
            self.processor, self.processor_text, self.encoder, audio_path, text, language
        )

        decoder_started = time.perf_counter()
        token_ids = input_ids.to("cuda")
        hidden = self.token_embedding(token_ids)
        audio_mask = (token_ids == self.audio_pad_id).unsqueeze(-1)
        hidden = hidden.masked_scatter(
            audio_mask, embeddings.to(hidden.dtype).reshape(-1, hidden.shape[-1])
        )
        ts_positions = torch.nonzero(token_ids[0] == self.timestamp_token_id).flatten().unsqueeze(0)
        logits = self.decoder.forward(hidden, ts_positions)
        torch.cuda.synchronize()
        decoder_ms = (time.perf_counter() - decoder_started) * 1000

        raw = logits[0].argmax(-1).cpu().numpy()
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

    from .tensorrt_encoder import DynamicAudioWindow

    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be 'bf16', 'fp16' or 'fp32'")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    aligner = _load_aligner(model_id, dtype)
    thinker = aligner.model.thinker
    wrapper = DynamicAudioWindow(
        thinker.audio_tower,
        getattr(thinker, "multi_modal_projector", None),
        apply_output_projection=True,
    ).eval().to("cuda:0")
    frames = wrapper.frames_per_chunk * 8
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
        dynamic_shapes={
            "input_features": {2: torch.export.Dim("frames", min=100, max=3600)},
            "input_features_mask": {1: torch.export.Dim("masked", min=100, max=3600)},
        },
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


class _AlignerDecoderForExport:
    def __new__(cls, text_model, lm_head):
        import torch
        from torch import nn

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.text_model = text_model
                self.lm_head = lm_head

            def forward(self, inputs_embeds: "torch.Tensor", ts_positions: "torch.Tensor"):
                hidden = self.text_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=None,
                    use_cache=False,
                    return_dict=False,
                )[0]
                hidden = hidden.index_select(1, ts_positions.reshape(-1))
                return self.lm_head(hidden)

        return Module().eval()


def export_aligner_decoder_tensorrt(
    model_id: str, output_path: Path, precision: str = "bf16"
) -> None:
    """Export the aligner text decoder as a single-pass ONNX graph.

    The graph takes pre-built ``inputs_embeds`` (audio embeddings already merged),
    gathers the hidden states at the timestamp positions and applies the
    5,000-row classification head only there.
    """
    import torch

    if precision not in ("bf16", "fp16", "fp32"):
        raise ValueError("precision must be 'bf16', 'fp16' or 'fp32'")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    aligner = _load_aligner(model_id, dtype)
    aligner.model.to("cuda:0")
    thinker = aligner.model.thinker
    wrapper = _AlignerDecoderForExport(thinker.model, thinker.lm_head).to("cuda:0")
    hidden = thinker.config.text_config.hidden_size

    inputs_embeds = torch.zeros((1, 64, hidden), dtype=dtype, device="cuda:0")
    ts_positions = torch.zeros((1, 8), dtype=torch.int64, device="cuda:0")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (inputs_embeds, ts_positions),
        str(output_path),
        input_names=["inputs_embeds", "ts_positions"],
        output_names=["logits"],
        opset_version=20,
        dynamo=True,
        dynamic_shapes={
            "inputs_embeds": {1: torch.export.Dim("seq", min=8, max=1024)},
            "ts_positions": {1: torch.export.Dim("nts", min=2, max=256)},
        },
    )


def build_aligner_decoder_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: int = 2,
    seq_range: tuple[int, int, int] = (8, 256, 1024),
    ts_range: tuple[int, int, int] = (2, 64, 256),
    tf32: bool = True,
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the aligner decoder ONNX:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "inputs_embeds",
        (1, seq_range[0], 1024),
        (1, seq_range[1], 1024),
        (1, seq_range[2], 1024),
    )
    profile.set_shape(
        "ts_positions",
        (1, ts_range[0]),
        (1, ts_range[1]),
        (1, ts_range[2]),
    )
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the aligner decoder engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


class TensorRTAlignDecoder:
    """TensorRT engine for the aligner text decoder (single forward, no KV cache)."""

    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load aligner decoder engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        self.buffers: dict[tuple, "torch.Tensor"] = {}
        self.dtype = torch.float16
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            if name == "inputs_embeds":
                dtype = self.engine.get_tensor_dtype(name)
                if dtype == trt.DataType.BF16:
                    self.dtype = torch.bfloat16
                elif dtype == trt.DataType.FLOAT:
                    self.dtype = torch.float32
                else:
                    self.dtype = torch.float16

    def _buffer(self, name: str, shape: tuple, dtype):
        torch = self.torch
        key = (name, tuple(shape), dtype)
        buffer = self.buffers.get(key)
        if buffer is None:
            buffer = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers[key] = buffer
        return buffer

    def forward(self, inputs_embeds, ts_positions):
        torch = self.torch
        stream = torch.cuda.current_stream()
        sequence_length = int(inputs_embeds.shape[1])
        position_count = int(ts_positions.shape[1])
        self.context.set_input_shape("inputs_embeds", (1, sequence_length, inputs_embeds.shape[2]))
        self.context.set_input_shape("ts_positions", (1, position_count))
        embeddings = self._buffer(
            "inputs_embeds", (1, sequence_length, inputs_embeds.shape[2]), self.dtype
        )
        positions = self._buffer("ts_positions", (1, position_count), torch.int64)
        embeddings[:, :sequence_length].copy_(inputs_embeds)
        positions[:, :position_count].copy_(ts_positions)
        output_shape = tuple(self.context.get_tensor_shape("logits"))
        logits = self._buffer("logits", output_shape, self.dtype)
        self.context.set_tensor_address("inputs_embeds", embeddings.data_ptr())
        self.context.set_tensor_address("ts_positions", positions.data_ptr())
        self.context.set_tensor_address("logits", logits.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT aligner decoder execution failed")
        return logits
