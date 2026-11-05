from __future__ import annotations

from pathlib import Path


class StaticAudioWindow:
    def __new__(cls, audio_tower, projector):
        import torch
        from torch import nn
        from torch.nn import functional as F

        class Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.audio_tower = audio_tower
                self.projector = projector
                self.frames_per_chunk = audio_tower.config.n_window * 2
                self.window_frames = audio_tower.config.n_window_infer
                if self.window_frames % self.frames_per_chunk:
                    raise ValueError("Inference window must contain whole encoder chunks")
                self.num_chunks = self.window_frames // self.frames_per_chunk
                self.num_heads = audio_tower.config.encoder_attention_heads
                self.head_dim = audio_tower.config.d_model // self.num_heads
                self.tokens_per_chunk = 13

            def forward(self, input_features, input_features_mask):
                tower = self.audio_tower
                batch = input_features.to(tower.conv2d1.weight.dtype).reshape(
                    1, 128, self.num_chunks, self.frames_per_chunk
                ).permute(0, 2, 1, 3).reshape(self.num_chunks, 1, 128, self.frames_per_chunk)
                hidden = F.gelu(tower.conv2d1(batch))
                hidden = F.gelu(tower.conv2d2(hidden))
                hidden = F.gelu(tower.conv2d3(hidden))
                hidden = tower.conv_out(
                    hidden.permute(0, 3, 1, 2)
                    .contiguous()
                    .reshape(self.num_chunks, self.tokens_per_chunk, -1)
                )
                hidden = hidden + tower.positional_embedding.positional_embedding[: self.tokens_per_chunk].to(
                    hidden.dtype
                )
                hidden = hidden.reshape(1, self.num_chunks * self.tokens_per_chunk, -1)

                chunk_lengths = input_features_mask.reshape(
                    1, self.num_chunks, self.frames_per_chunk
                ).sum(-1).to(torch.long)
                for _ in range(3):
                    chunk_lengths = torch.where(
                        chunk_lengths > 0,
                        torch.div(chunk_lengths - 1, 2, rounding_mode="floor") + 1,
                        torch.zeros_like(chunk_lengths),
                    )
                valid_lengths = chunk_lengths.sum(-1)
                key_mask = torch.arange(
                    self.num_chunks * self.tokens_per_chunk,
                    device=hidden.device,
                ) >= valid_lengths.unsqueeze(-1)

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
                    scores = scores.masked_fill(key_mask[:, None, None, :], -1e4)
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
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        dtype=dtypes[precision],
        device_map="cuda:0",
    ).eval()
    tower = model.model.audio_tower
    wrapper = StaticAudioWindow(tower, model.model.multi_modal_projector).eval()
    frames = wrapper.window_frames
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
    )
    del model
    torch.cuda.empty_cache()


def build_engine(onnx_path: Path, engine_path: Path, workspace_gib: int = 2) -> None:
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
        self.window_frames = self.engine.get_tensor_shape(self.input_names[0])[-1]
        self.embedding_shape = tuple(self.engine.get_tensor_shape(self.output_names[0]))
        self.embedding_dtype = {
            trt.DataType.BF16: torch.bfloat16,
            trt.DataType.HALF: torch.float16,
        }[self.engine.get_tensor_dtype(self.output_names[0])]
        self.valid_dtype = torch.int64

    def encode(self, input_features, input_features_mask):
        torch = self.torch
        frames_per_window = self.window_frames
        frame_count = int(input_features_mask.sum().item())
        pieces = []
        for start in range(0, max(frame_count, 1), frames_per_window):
            valid = min(frame_count - start, frames_per_window)
            features = torch.zeros(
                (1, 128, frames_per_window), device="cuda", dtype=torch.float32
            )
            mask = torch.zeros((1, frames_per_window), device="cuda", dtype=torch.int32)
            available = min(valid, input_features.shape[-1] - start)
            if available > 0:
                features[..., :available] = input_features[..., start : start + available].to(
                    device="cuda", dtype=torch.float32
                )
                mask[..., :available] = input_features_mask[..., start : start + available].to(
                    device="cuda", dtype=torch.int32
                )
            embeddings = torch.empty(
                self.embedding_shape, device="cuda", dtype=self.embedding_dtype
            )
            valid_tokens = torch.empty((1,), device="cuda", dtype=self.valid_dtype)
            for name, tensor in zip(self.input_names, (features, mask)):
                self.context.set_tensor_address(name, tensor.data_ptr())
            for name, tensor in zip(self.output_names, (embeddings, valid_tokens)):
                self.context.set_tensor_address(name, tensor.data_ptr())
            stream = torch.cuda.current_stream()
            if not self.context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("TensorRT encoder execution failed")
            stream.synchronize()
            token_count = _post_cnn_tokens(valid)
            pieces.append(embeddings[0, :token_count])
        if not pieces:
            return torch.empty((0, self.embedding_shape[-1]), device="cuda", dtype=self.embedding_dtype)
        return torch.cat(pieces, dim=0)
