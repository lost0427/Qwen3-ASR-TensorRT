"""TensorRT prefill + W4A16 GEMV decode.

Decode runs as plain PyTorch on top of the packed INT4 kernels, so the whole step
can be captured into a CUDA graph: shapes are static, the KV cache is
preallocated, the position lives on the device and the token is fed back without
touching the host. That removes both the dynamic-shape penalty and the per-step
synchronisation that made the all-TensorRT path slow.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .w4a16 import extension


class _RMSNorm:
    def __init__(self, weight: Tensor, eps: float):
        self.weight = weight.detach().to(device="cuda", dtype=torch.float16).contiguous()
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _PackedLinear:
    """Bias-free projection evaluated one token at a time."""

    def __init__(self, weight: Tensor):
        weight = weight.detach().to(device="cuda", dtype=torch.float16).contiguous()
        self.packed = extension().pack(weight)
        self.scales = extension().scale(weight)
        self.out_features = weight.shape[0]

    def __call__(self, x: Tensor, residual: Tensor | None = None) -> Tensor:
        if residual is None:
            out = extension().gemv(x.reshape(1, -1).contiguous(), self.packed, self.scales)
        else:
            out = extension().gemv_residual(
                x.reshape(1, -1).contiguous(), self.packed, self.scales, residual.reshape(1, -1)
            )
        return out.view(*x.shape[:-1], self.out_features)


class _TRTPrefill:
    """Loads only the prefill engine and returns its KV cache."""

    def __init__(self, engine_path: Path):
        import tensorrt as trt

        self.runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if engine is None:
            raise RuntimeError(f"could not load {engine_path}")
        self.engine = engine
        self.context = engine.create_execution_context()

    def run(self, inputs_embeds: Tensor, layers: int, kv_heads: int, head_dim: int, vocab: int):
        seq = int(inputs_embeds.shape[1])
        embeds = inputs_embeds.contiguous()
        positions = torch.arange(seq, device="cuda").unsqueeze(0)
        logits = torch.empty((1, 1, vocab), dtype=torch.float16, device="cuda")
        cache_k = torch.empty((seq, layers, kv_heads, head_dim), dtype=torch.float16, device="cuda")
        cache_v = torch.empty_like(cache_k)
        self.context.set_input_shape("inputs_embeds", (1, seq, embeds.shape[-1]))
        self.context.set_input_shape("position_ids", (1, seq))
        tensors = {
            "inputs_embeds": embeds,
            "position_ids": positions,
            "logits": logits,
            "cache_k": cache_k,
            "cache_v": cache_v,
        }
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.context.set_tensor_address(name, tensors[name].data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("prefill execution failed")
        return logits, cache_k, cache_v


class W4A16ASRDecoder:
    """Greedy decoder: TensorRT prefill, packed-INT4 decode."""

    def __init__(
        self,
        model_id: str,
        prefill_engine: Path,
        max_cache: int = 2048,
    ):
        from transformers import AutoConfig, AutoModelForMultimodalLM

        self.torch = torch
        self.max_cache = max_cache
        config = AutoConfig.from_pretrained(str(model_id), local_files_only=True)
        text = getattr(config, "text_config", config)

        self.layers = text.num_hidden_layers
        self.heads = text.num_attention_heads
        self.kv_heads = text.num_key_value_heads
        self.head_dim = text.head_dim
        self.groups = self.heads // self.kv_heads
        self.hidden = text.hidden_size
        self.intermediate = text.intermediate_size
        self.vocab = text.vocab_size
        self.eps = text.rms_norm_eps
        rope = getattr(text, "rope_parameters", None) or getattr(text, "rope_scaling", None) or {}
        self.rope_theta = rope.get("rope_theta", getattr(text, "rope_theta", 1000000.0))

        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_id), dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
        )
        language = model.model.language_model
        cuda = torch.device("cuda")

        self.embed = language.embed_tokens.weight.detach().to(cuda, torch.float16).contiguous()
        self.lm_head = _PackedLinear(model.lm_head.weight)
        self.final_norm = _RMSNorm(language.norm.weight, self.eps)

        self.qkv, self.o_proj, self.gu, self.down = [], [], [], []
        self.q_norm, self.k_norm = [], []
        self.in_norm, self.post_norm = [], []
        for layer in language.layers:
            attn = layer.self_attn
            self.qkv.append(_PackedLinear(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0)))
            self.o_proj.append(_PackedLinear(attn.o_proj.weight))
            self.gu.append(_PackedLinear(torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], 0)))
            self.down.append(_PackedLinear(layer.mlp.down_proj.weight))
            self.q_norm.append(_RMSNorm(attn.q_norm.weight, self.eps))
            self.k_norm.append(_RMSNorm(attn.k_norm.weight, self.eps))
            self.in_norm.append(_RMSNorm(layer.input_layernorm.weight, self.eps))
            self.post_norm.append(_RMSNorm(layer.post_attention_layernorm.weight, self.eps))
        del model

        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        freqs = torch.outer(torch.arange(max_cache, dtype=torch.float32), inv_freq)
        table = torch.cat([freqs, freqs], dim=-1)
        self.cos_table = table.cos().to(cuda, torch.float16).contiguous()
        self.sin_table = table.sin().to(cuda, torch.float16).contiguous()

        size = (1, self.kv_heads, max_cache, self.head_dim)
        self.cache_k = [torch.zeros(size, dtype=torch.float16, device=cuda) for _ in range(self.layers)]
        self.cache_v = [torch.zeros(size, dtype=torch.float16, device=cuda) for _ in range(self.layers)]

        self.prefill = _TRTPrefill(prefill_engine)

        self._token = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._pos = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._produced = torch.zeros((max_cache,), dtype=torch.long, device=cuda)
        self._generated = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._logits = torch.empty((1, self.vocab), dtype=torch.float16, device=cuda)
        self._attn_out = torch.empty((1, self.heads * self.head_dim), dtype=torch.float16, device=cuda)
        self._captured: torch.cuda.CUDAGraph | None = None

    # ---------------------------------------------------------------- forward

    def _layer(self, index: int, hidden: Tensor, pos: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        normed = self.in_norm[index](hidden)
        qkv = self.qkv[index](normed)
        q_dim = self.heads * self.head_dim
        k_dim = self.kv_heads * self.head_dim
        q = qkv[:, :, :q_dim]
        k = qkv[:, :, q_dim:q_dim + k_dim]
        v = qkv[:, :, q_dim + k_dim:]
        extension().qk_norm_rope(q, k, self.q_norm[index].weight, self.k_norm[index].weight,
                                 cos, sin, self.eps)

        cache_k = self.cache_k[index]
        cache_v = self.cache_v[index]
        extension().cache_write(cache_k, cache_v, k, v, pos)

        attended = self._attn_out
        extension().attention(q, cache_k, cache_v, pos, attended, self.groups)
        hidden = self.o_proj[index](attended, hidden)

        normed = self.post_norm[index](hidden)
        gate_up = self.gu[index](normed)
        gated = extension().silu_mul(gate_up, self.intermediate)
        return self.down[index](gated, hidden)

    def _step(self) -> None:
        torch = self.torch
        pos = self._pos
        cos = self.cos_table.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        sin = self.sin_table.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        hidden = F.embedding(self._token.view(1, 1), self.embed)
        for index in range(self.layers):
            hidden = self._layer(index, hidden, pos, cos, sin)
        logits = self.lm_head(self.final_norm(hidden))
        self._logits.copy_(logits.view(1, -1))
        token = logits.argmax(-1).view(1)
        self._token.copy_(token)
        self._produced.index_copy_(0, self._generated, token)
        self._generated.add_(1)
        self._pos.add_(1)

    def _snapshot(self):
        return int(self._pos.item()), self._token.clone(), int(self._generated.item())

    def _restore(self, state) -> None:
        position, token, generated = state
        self._pos.fill_(position)
        self._token.copy_(token)
        self._generated.fill_(generated)

    def _graph(self) -> torch.cuda.CUDAGraph:
        if self._captured is not None:
            return self._captured
        torch = self.torch
        # Both the warm-up run and the capture itself execute the step, so the
        # decoder state has to be rolled back around them.
        state = self._snapshot()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self._step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self._restore(state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._step()
        self._restore(state)
        self._captured = graph
        return graph

    # -------------------------------------------------------------------- api

    def prefill_prompt(self, inputs_embeds: Tensor) -> tuple[Tensor, int]:
        logits, cache_k, cache_v = self.prefill.run(
            inputs_embeds, self.layers, self.kv_heads, self.head_dim, self.vocab
        )
        seq = cache_k.shape[0]
        for index in range(self.layers):
            self.cache_k[index][:, :, :seq] = cache_k[:, index].permute(1, 0, 2).unsqueeze(0)
            self.cache_v[index][:, :, :seq] = cache_v[:, index].permute(1, 0, 2).unsqueeze(0)
        self._pos.fill_(seq)
        self._token.copy_(logits[:, -1, :].argmax(-1))
        return logits, seq

    def generate(self, inputs_embeds: Tensor, max_new_tokens: int = 128, chunk: int = 8) -> list[int]:
        _, _ = self.prefill_prompt(inputs_embeds)
        return self.decode_loop(max_new_tokens, chunk)

    def decode_loop(self, max_new_tokens: int = 128, chunk: int = 8) -> list[int]:
        """Greedy decode from the current prefill state (no prompt re-encoding)."""
        torch = self.torch
        eos = (151643, 151645)
        position = int(self._pos.item())
        # The argmax of the prefill logits is the first generated token; the step
        # graph produces the one after it.
        self._produced[0].copy_(self._token[0])
        self._generated.fill_(1)
        produced = 1
        tokens = [int(self._token.item())]
        if tokens[0] in eos or max_new_tokens <= 1:
            return tokens[:max_new_tokens]
        while produced < max_new_tokens and position < self.max_cache - 1:
            graph = self._graph()
            steps = min(chunk, max_new_tokens - produced, self.max_cache - 1 - position)
            for _ in range(steps):
                graph.replay()
            torch.cuda.synchronize()
            position += steps
            produced = int(self._generated.item())
            tokens = self._produced[:produced].tolist()
            stop = next((i for i, value in enumerate(tokens) if value in eos), None)
            if stop is not None:
                return tokens[:stop]
        return self._produced[:produced].tolist()
