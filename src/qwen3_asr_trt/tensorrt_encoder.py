from __future__ import annotations

from pathlib import Path


class DynamicAudioWindow:
    """Whole-utterance audio encoder export.

    The reference implementation (Transformers with the ``sdpa`` attention
    backend) ignores ``cu_seqlens`` and attends over every audio token, so the
    exported graph does the same: one pass over the whole utterance, with no
    inference windows.
    """

    def __new__(cls, audio_tower, projector, apply_output_projection: bool = False):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.audio_tower = audio_tower
                self.projector = projector if projector is not None else nn.Identity()
                self.apply_output_projection = apply_output_projection
                self.frames_per_chunk = audio_tower.config.n_window * 2
                self.num_heads = audio_tower.config.encoder_attention_heads
                self.head_dim = audio_tower.config.d_model // self.num_heads
                self.tokens_per_chunk = 13
                # Transformers splits post-CNN tokens into inference windows of
                # n_window_infer raw frames and attends inside each window only.
                self.window_tokens = self.tokens_per_chunk * (
                    audio_tower.config.n_window_infer // (audio_tower.config.n_window * 2)
                )

            def forward(self, input_features, input_features_mask):
                tower = self.audio_tower
                chunks = input_features.to(tower.conv2d1.weight.dtype).reshape(
                    1, 128, -1, self.frames_per_chunk
                ).permute(0, 2, 1, 3).reshape(-1, 1, 128, self.frames_per_chunk)
                hidden = F.gelu(tower.conv2d1(chunks))
                hidden = F.gelu(tower.conv2d2(hidden))
                hidden = F.gelu(tower.conv2d3(hidden))
                hidden = tower.conv_out(
                    hidden.permute(0, 3, 1, 2)
                    .contiguous()
                    .reshape(chunks.shape[0], self.tokens_per_chunk, -1)
                )
                hidden = hidden + tower.positional_embedding.positional_embedding[: self.tokens_per_chunk].to(
                    hidden.dtype
                )
                hidden = hidden.reshape(1, -1, hidden.shape[-1])

                chunk_lengths = input_features_mask.reshape(
                    1, -1, self.frames_per_chunk
                ).sum(-1).to(torch.long)
                for _ in range(3):
                    chunk_lengths = torch.where(
                        chunk_lengths > 0,
                        torch.div(chunk_lengths - 1, 2, rounding_mode="floor") + 1,
                        torch.zeros_like(chunk_lengths),
                    )
                valid_lengths = chunk_lengths.sum(-1)
                positions = torch.arange(hidden.shape[1], device=hidden.device)
                same_window = (positions[:, None] // self.window_tokens) == (
                    positions[None, :] // self.window_tokens
                )
                padded = positions[None, :] >= valid_lengths.reshape(-1, 1)
                key_mask = (~same_window).unsqueeze(0).unsqueeze(0) | padded[:, None, :]

                for layer in tower.layers:
                    residual = hidden
                    normalized = layer.self_attn_layer_norm(hidden)
                    attention = layer.self_attn
                    q = attention.q_proj(normalized).reshape(
                        1, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                    k = attention.k_proj(normalized).reshape(
                        1, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                    v = attention.v_proj(normalized).reshape(
                        1, -1, self.num_heads, self.head_dim
                    ).transpose(1, 2)
                    scores = torch.matmul(q, k.transpose(-1, -2)) * attention.scaling
                    scores = scores.masked_fill(key_mask, -1e4)
                    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
                    attended = torch.matmul(probs, v).transpose(1, 2).reshape(1, -1, self.num_heads * self.head_dim)
                    hidden = residual + attention.out_proj(attended)

                    residual = hidden
                    hidden = layer.final_layer_norm(hidden)
                    hidden = layer.fc1(hidden)
                    hidden = layer.activation_fn(hidden)
                    hidden = layer.fc2(hidden)
                    hidden = residual + hidden

                hidden = tower.ln_post(hidden)
                if self.apply_output_projection:
                    hidden = tower.proj2(tower.act(tower.proj1(hidden)))
                hidden = self.projector(hidden)
                return hidden, valid_lengths

        return Module()



def _post_cnn_tokens(frame_count: int, frames_per_chunk: int = 100) -> int:
    complete, remainder = divmod(frame_count, frames_per_chunk)
    if remainder == 0:
        return complete * 13
    for _ in range(3):
        remainder = (remainder + 1) // 2
    return complete * 13 + remainder


def export_encoder(model_id: str, output_path: Path, precision: str = "bf16") -> None:
    import torch
    from transformers import AutoModelForMultimodalLM

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16}
    if precision not in dtypes:
        raise ValueError(f"Unsupported encoder precision: {precision}")
    model, loading_info = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtypes[precision],
        device_map="cpu",
        output_loading_info=True,
    )
    model.eval()
    expected_audio = {
        key for key in model.state_dict()
        if ".audio_tower." in key or ".multi_modal_projector." in key
    }
    missing_audio = expected_audio.intersection(loading_info["missing_keys"])
    unexpected_audio = {
        key for key in loading_info["unexpected_keys"]
        if ".audio_tower." in key or ".multi_modal_projector." in key
    }
    if missing_audio or unexpected_audio:
        raise RuntimeError(
            "Checkpoint audio weights did not load correctly; "
            f"missing={sorted(missing_audio)[:5]}, "
            f"unexpected={sorted(unexpected_audio)[:5]}"
        )
    thinker = getattr(model, "thinker", getattr(model, "model", model))
    tower = thinker.audio_tower
    projector = getattr(thinker, "multi_modal_projector", None)
    wrapper = DynamicAudioWindow(tower, projector).eval().to("cuda:0")
    frames = wrapper.frames_per_chunk * 8
    sample_features = torch.zeros((1, 128, frames), dtype=torch.float32, device="cuda")
    sample_mask = torch.ones((1, frames), dtype=torch.int32, device="cuda")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (sample_features, sample_mask),
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
    del model
    torch.cuda.empty_cache()


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_gib: int = 2,
    frames_range: tuple[int, int, int] | None = (100, 800, 3600),
) -> None:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse the encoder ONNX model:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    names = {network.get_input(i).name for i in range(network.num_inputs)}
    if frames_range is not None and "input_features" in names:
        low, optimum, high = frames_range
        profile = builder.create_optimization_profile()
        profile.set_shape("input_features", (1, 128, low), (1, 128, optimum), (1, 128, high))
        if "input_features_mask" in names:
            profile.set_shape("input_features_mask", (1, low), (1, optimum), (1, high))
        config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the encoder engine")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)


class TensorRTAudioEncoder:
    def __init__(self, engine_path: Path) -> None:
        import tensorrt as trt
        import torch

        self.torch = torch
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not load TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) == trt.TensorIOMode.OUTPUT
        ]
        self.frames_per_chunk = 100
        window = self.engine.get_tensor_shape(self.input_names[0])[-1]
        if window > 0:
            self.frames_per_chunk = window
        self.embedding_shape = tuple(self.engine.get_tensor_shape(self.output_names[0]))
        self.embedding_dtype = {
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.HALF: torch.float16,
            trt.DataType.FLOAT: torch.float32,
        }[self.engine.get_tensor_dtype(self.output_names[0])]
        self.valid_dtype = torch.int64
        self.buffers: dict[tuple, "torch.Tensor"] = {}

    def _buffer(self, key, shape, dtype):
        buffer = self.buffers.get(key)
        if buffer is None or tuple(buffer.shape) != tuple(shape):
            buffer = self.torch.zeros(shape, device="cuda", dtype=dtype)
            self.buffers[key] = buffer
        return buffer

    def encode(self, input_features, input_features_mask):
        torch = self.torch
        frame_count = int(input_features_mask.sum().item())
        chunk = self.frames_per_chunk
        padded = max(chunk, ((max(frame_count, 1) + chunk - 1) // chunk) * chunk)
        available = min(frame_count, int(input_features.shape[-1]))

        features = self._buffer(("features", padded), (1, 128, padded), torch.float32)
        features.zero_()
        mask = self._buffer(("mask", padded), (1, padded), torch.int32)
        mask.zero_()
        if available > 0:
            features[..., :available] = input_features[..., :available].to("cuda", torch.float32)
            mask[..., :available] = input_features_mask[..., :available].to("cuda", torch.int32)

        embeddings = self._buffer(("out", padded), (1, padded // 2, self.embedding_shape[-1]), self.embedding_dtype)
        valid_tokens = self._buffer(("valid", 1), (1,), self.valid_dtype)

        self.context.set_input_shape("input_features", (1, 128, padded))
        self.context.set_input_shape("input_features_mask", (1, padded))
        for name, tensor in zip(self.input_names, (features, mask)):
            self.context.set_tensor_address(name, tensor.data_ptr())
        for name, tensor in zip(self.output_names, (embeddings, valid_tokens)):
            self.context.set_tensor_address(name, tensor.data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT encoder execution failed")
        stream.synchronize()

        token_count = _post_cnn_tokens(frame_count)
        if token_count <= 0:
            return torch.empty((0, self.embedding_shape[-1]), device="cuda", dtype=self.embedding_dtype)
        return embeddings[0, :token_count]
