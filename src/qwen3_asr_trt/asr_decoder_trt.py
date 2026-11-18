"""TensorRT text decoder for Qwen3-ASR (prefill + incremental decode with KV cache).

The default runtime decodes with ExLlamaV3. This module exports the same Qwen3
text model as TensorRT graphs so the whole ASR path can run on TensorRT:

* prefill consumes the prompt embeddings and returns the KV cache for every
  prompt position plus the logits of the last position;
* decode consumes one token embedding and the cache, and returns the next-token
  logits plus the key/value slice for that position.

The cache is laid out as ``(layers, kv_heads, seq, head_dim)`` with a dynamic
sequence axis, so the runtime can hand TensorRT a prefix view of a preallocated
buffer instead of copying the cache every step.
"""
from __future__ import annotations

from pathlib import Path


def _make_graph(language_model, lm_head, mode: str, include_lm_head: bool = True):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

    class Graph(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = language_model
            self.lm_head = lm_head
            self.include_lm_head = include_lm_head
            self.layers = language_model.layers
            self.norm = language_model.norm
            self.rotary = language_model.rotary_emb
            self.num_heads = language_model.config.num_attention_heads
            self.num_kv_heads = language_model.config.num_key_value_heads
            self.head_dim = language_model.config.head_dim
            self.groups = self.num_heads // self.num_kv_heads

        def _attend(self, layer, hidden, cos, sin, mask, past_k, past_v):
            residual = hidden
            normed = layer.input_layernorm(hidden)
            attn = layer.self_attn
            batch, seq, _ = normed.shape
            q = attn.q_proj(normed).view(batch, seq, self.num_heads, self.head_dim).transpose(1, 2)
            k = attn.k_proj(normed).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = attn.v_proj(normed).view(batch, seq, self.num_kv_heads, self.head_dim).transpose(1, 2)
            q = attn.q_norm(q)
            k = attn.k_norm(k)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            if past_k is None:
                keys, values = k, v
            else:
                keys = torch.cat([past_k, k], dim=2)
                values = torch.cat([past_v, v], dim=2)
            out = F.scaled_dot_product_attention(
                q,
                repeat_kv(keys, self.groups),
                repeat_kv(values, self.groups),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
            out = out.transpose(1, 2).reshape(batch, seq, -1).contiguous()
            hidden = residual + attn.o_proj(out)
            residual = hidden
            hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
            return hidden, k, v

        def _logits(self, hidden):
            normed = self.norm(hidden[:, -1:, :])
            return self.lm_head(normed) if self.include_lm_head else normed

        def _run(self, hidden, cos, sin, mask, cache_k, cache_v):
            new_k, new_v = [], []
            for index, layer in enumerate(self.layers):
                hidden, k, v = self._attend(
                    layer,
                    hidden,
                    cos,
                    sin,
                    mask,
                    None
                    if cache_k is None
                    else cache_k[:, index].permute(1, 0, 2).unsqueeze(0),
                    None
                    if cache_v is None
                    else cache_v[:, index].permute(1, 0, 2).unsqueeze(0),
                )
                new_k.append(k)
                new_v.append(v)
            logits = self._logits(hidden)
            if cache_k is None:
                # (seq, layers, kv_heads, head_dim)
                keys = torch.stack([item.squeeze(0) for item in new_k], dim=0).permute(2, 0, 1, 3)
                values = torch.stack([item.squeeze(0) for item in new_v], dim=0).permute(2, 0, 1, 3)
            else:
                # (1, layers, kv_heads, head_dim): the runtime writes this slice
                # into the preallocated cache buffer
                keys = torch.stack([item[:, :, -1:, :].squeeze(0) for item in new_k], dim=0).permute(2, 0, 1, 3)
                values = torch.stack([item[:, :, -1:, :].squeeze(0) for item in new_v], dim=0).permute(2, 0, 1, 3)
            return logits, keys, values

    class PrefillGraph(Graph):
        def forward(self, inputs_embeds, position_ids):
            seq = inputs_embeds.shape[1]
            cos, sin = self.rotary(inputs_embeds, position_ids)
            causal = torch.ones(seq, seq, dtype=torch.bool, device=inputs_embeds.device).tril()
            mask = torch.zeros(1, 1, seq, seq, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            mask = mask.masked_fill(~causal, torch.finfo(inputs_embeds.dtype).min)
            return self._run(inputs_embeds, cos, sin, mask, None, None)

    class DecodeGraph(Graph):
        def forward(self, inputs_embeds, position_ids, cache_k, cache_v, cache_len):
            total = cache_k.shape[0] + 1
            cos, sin = self.rotary(inputs_embeds, position_ids)
            valid = torch.arange(total, device=inputs_embeds.device) < (cache_len.reshape(()) + 1)
            mask = torch.zeros(1, 1, 1, total, dtype=inputs_embeds.dtype, device=inputs_embeds.device)
            mask = mask.masked_fill(~valid.view(1, 1, 1, total), torch.finfo(inputs_embeds.dtype).min)
            return self._run(inputs_embeds, cos, sin, mask, cache_k, cache_v)

    return (PrefillGraph() if mode == "prefill" else DecodeGraph()).eval()


def export_asr_decoder(
    model_id: str,
    output_dir: Path,
    mode: str,
    precision: str = "fp16",
    include_lm_head: bool = True,
) -> None:
    import torch
    from transformers import AutoModelForMultimodalLM

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True
    )
    model.eval()
    language_model = model.model.language_model
    graph = _make_graph(language_model, model.lm_head, mode, include_lm_head).to("cuda").to(dtype)
    config = language_model.config
    hidden = config.hidden_size
    layers = len(language_model.layers)
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "prefill":
        embeds = torch.zeros((1, 32, hidden), dtype=dtype, device="cuda")
        positions = torch.arange(32, device="cuda").unsqueeze(0)
        torch.onnx.export(
            graph,
            (embeds, positions),
            str(output_dir / "asr-prefill.onnx"),
            input_names=["inputs_embeds", "position_ids"],
            output_names=["logits" if include_lm_head else "hidden", "cache_k", "cache_v"],
            opset_version=20,
            dynamo=True,
            dynamic_shapes={
                "inputs_embeds": {1: torch.export.Dim("seq", min=1, max=2048)},
                "position_ids": {1: torch.export.Dim("pos", min=1, max=2048)},
            },
        )
    else:
        embeds = torch.zeros((1, 1, hidden), dtype=dtype, device="cuda")
        positions = torch.zeros((1, 1), dtype=torch.long, device="cuda")
        cache_k = torch.zeros((64, layers, kv_heads, head_dim), dtype=dtype, device="cuda")
        cache_v = torch.zeros((64, layers, kv_heads, head_dim), dtype=dtype, device="cuda")
        cache_len = torch.tensor([63], dtype=torch.int32, device="cuda")
        torch.onnx.export(
            graph,
            (embeds, positions, cache_k, cache_v, cache_len),
            str(output_dir / "asr-decode.onnx"),
            input_names=["inputs_embeds", "position_ids", "cache_k", "cache_v", "cache_len"],
            output_names=["logits", "out_k", "out_v"],
            opset_version=20,
            dynamo=True,
            dynamic_shapes={
                "inputs_embeds": {},
                "position_ids": {},
                "cache_k": {0: torch.export.Dim("cache_k", min=1, max=2048)},
                "cache_v": {0: torch.export.Dim("cache_v", min=1, max=2048)},
                "cache_len": {},
            },
        )

def build_asr_decoder_engine(
    onnx_path: Path,
    engine_path: Path,
    profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    workspace_gib: int = 2,
) -> None:
    """Build a TensorRT engine for one of the exported graphs."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path.name}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib << 30)
    profile = builder.create_optimization_profile()
    for name, (low, optimum, high) in profiles.items():
        profile.set_shape(name, low, optimum, high)
    config.add_optimization_profile(profile)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(serialized)

class TensorRTASRDecoder:
    """Greedy TensorRT decoder with an explicit KV cache (prefill + decode engines)."""

    def __init__(
        self,
        model_id: str,
        prefill_engine: Path,
        decode_engine: Path,
        max_cache: int = 1024,
    ) -> None:
        import tensorrt as trt
        import torch
        from safetensors import safe_open
        from transformers import AutoModelForMultimodalLM

        self.torch = torch
        self.max_cache = max_cache
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.prefill = self._load(prefill_engine)
        self.prefill_ctx = self.prefill.create_execution_context()
        self.decode = self._load(decode_engine)
        self.decode_ctx = self.decode.create_execution_context()

        path = Path(str(model_id))
        weights = path / "model.safetensors"
        with safe_open(str(weights), "pt") as handle:
            keys = [key for key in handle.keys() if key.endswith("embed_tokens.weight")]
            if len(keys) != 1:
                raise RuntimeError(f"could not find the embedding table in {weights}")
            self.embed = handle.get_tensor(keys[0]).to(torch.float16).to("cuda")

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_id), dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
        )
        language_model = model.model.language_model
        config = language_model.config
        self.layers = len(language_model.layers)
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        del model
        torch.cuda.empty_cache()

        size = (self.max_cache, self.layers, self.kv_heads, self.head_dim)
        self.cache_k = torch.zeros(size, dtype=torch.float16, device="cuda")
        self.cache_v = torch.zeros(size, dtype=torch.float16, device="cuda")
        self.embeds_buffer = torch.zeros((1, 1, config.hidden_size), dtype=torch.float16, device="cuda")
        self.positions_buffer = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
        self.cache_len_buffer = torch.zeros((1,), dtype=torch.int32, device="cuda")
        self.last_timings: dict[str, float] = {}
        self._token_buffer = None
        self._logits = None
        self._delta_k = None
        self._delta_v = None

    def _load(self, engine_path: Path):
        engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if engine is None:
            raise RuntimeError(f"could not load {engine_path}")
        return engine

    def _bind(self, context, engine, tensors):
        for index in range(engine.num_io_tensors):
            name = engine.get_tensor_name(index)
            context.set_tensor_address(name, tensors[name].data_ptr())

    def prefill_prompt(self, inputs_embeds):
        torch = self.torch
        seq = int(inputs_embeds.shape[1])
        if seq > self.max_cache:
            raise ValueError(f"prompt of {seq} tokens exceeds the {self.max_cache} cache budget")
        embeds = inputs_embeds.contiguous()
        positions = torch.arange(seq, device="cuda").unsqueeze(0)
        logits = torch.empty((1, 1, self.embed.shape[0]), dtype=torch.float16, device="cuda")
        cache_k = torch.empty((seq, self.layers, self.kv_heads, self.head_dim), dtype=torch.float16, device="cuda")
        cache_v = torch.empty_like(cache_k)
        self.prefill_ctx.set_input_shape("inputs_embeds", (1, seq, self.embed.shape[1]))
        self.prefill_ctx.set_input_shape("position_ids", (1, seq))
        self._bind(self.prefill_ctx, self.prefill, {
            "inputs_embeds": embeds,
            "position_ids": positions,
            "logits": logits,
            "cache_k": cache_k,
            "cache_v": cache_v,
        })
        stream = torch.cuda.current_stream()
        if not self.prefill_ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("prefill execution failed")
        self.cache_k[:seq] = cache_k
        self.cache_v[:seq] = cache_v
        return logits, seq

    def decode_step(self, token_id, position: int):
        torch = self.torch
        if self._token_buffer is None:
            self._token_buffer = torch.zeros((1, 1), dtype=torch.long, device="cuda")
            self._logits = torch.empty((1, 1, self.embed.shape[0]), dtype=torch.float16, device="cuda")
            self._delta_k = torch.empty((1, self.layers, self.kv_heads, self.head_dim), dtype=torch.float16, device="cuda")
            self._delta_v = torch.empty_like(self._delta_k)
        if isinstance(token_id, int):
            self._token_buffer.fill_(token_id)
        else:
            self._token_buffer.copy_(token_id.view(1, 1).long())
        self.embeds_buffer.copy_(self.embed[self._token_buffer])
        self.positions_buffer.fill_(position)
        self.cache_len_buffer.fill_(position)
        logits = self._logits
        delta_k = self._delta_k
        delta_v = self._delta_v
        self.decode_ctx.set_input_shape("cache_k", (position, self.layers, self.kv_heads, self.head_dim))
        self.decode_ctx.set_input_shape("cache_v", (position, self.layers, self.kv_heads, self.head_dim))
        self._bind(self.decode_ctx, self.decode, {
            "inputs_embeds": self.embeds_buffer,
            "position_ids": self.positions_buffer,
            "cache_k": self.cache_k,
            "cache_v": self.cache_v,
            "cache_len": self.cache_len_buffer,
            "logits": logits,
            "out_k": delta_k,
            "out_v": delta_v,
        })
        stream = torch.cuda.current_stream()
        if not self.decode_ctx.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("decode execution failed")
        self.cache_k[position:position + 1] = delta_k
        self.cache_v[position:position + 1] = delta_v
        return logits

    def generate(self, inputs_embeds, max_new_tokens: int = 128, chunk: int = 1):
        """Greedy decode with the argmax kept on the GPU.

        The token feedback stays on the device, so the loop only synchronises once
        per ``chunk`` steps (to check for the end-of-sequence token) instead of
        once per step.
        """
        torch = self.torch
        logits, position = self.prefill_prompt(inputs_embeds)
        token = logits[:, -1, :].argmax(-1).view(1, 1)
        produced = []
        while len(produced) < max_new_tokens:
            pending = []
            steps = min(chunk, max_new_tokens - len(produced))
            for _ in range(steps):
                pending.append(token)
                if position >= self.max_cache:
                    break
                logits = self.decode_step(token, position)
                position += 1
                token = logits[:, -1, :].argmax(-1).view(1, 1)
            chunk_ids = torch.cat([item.view(1) for item in pending]).cpu().tolist()
            for value in chunk_ids:
                if value in (151643, 151645):
                    return produced
                produced.append(value)
        return produced
