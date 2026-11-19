"""TensorRT prefill + W4A16 GEMV decode.

Decode runs as plain PyTorch on top of the packed INT4 kernels, so the whole step
can be captured into a CUDA graph: shapes are static, the KV cache is
preallocated, the position lives on the device and the token is fed back without
touching the host. That removes both the dynamic-shape penalty and the per-step
synchronisation that made the all-TensorRT path slow.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .w4a16 import extension

logger = logging.getLogger(__name__)

PACKED_FORMAT = "qwen3-asr-w4a16"
PACKED_LAYOUT = (
    "nibbles along the contraction axis, low nibble holds the even column; "
    "value = round(w / scale) clipped to [-7, 7], scale = max|w| / 7 per group"
)
COPIED_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def read_packed_config(directory: str | Path) -> dict | None:
    """Return the descriptor of a packed decoder directory, or None."""
    path = Path(directory) / "packed_config.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_packed(model_id: str) -> tuple[dict, Path] | None:
    """Find a packed checkpoint, locally or in a Hub repository."""
    local = Path(str(model_id))
    if (local / "packed_config.json").is_file():
        return read_packed_config(local), local / "model.safetensors"
    if local.exists():
        return None
    try:
        from huggingface_hub import hf_hub_download

        descriptor = json.loads(
            Path(hf_hub_download(str(model_id), "packed_config.json")).read_text(encoding="utf-8")
        )
        weights = Path(hf_hub_download(str(model_id), "model.safetensors"))
    except Exception:
        return None
    return descriptor, weights


def pack_decoder(model_id: str, output_dir: str | Path, group: int = 32) -> dict:
    """Write a decoder-only INT4 checkpoint: packed projections, no audio tower."""
    from safetensors.torch import save_file
    from transformers import AutoModelForMultimodalLM

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        str(model_id), dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
    )
    language = model.model.language_model
    kernels = extension()
    tensors: dict[str, Tensor] = {}

    def keep(weight: Tensor) -> Tensor:
        return weight.detach().to(torch.float16).contiguous()

    def pack(name: str, weight: Tensor) -> None:
        weight = weight.detach().to("cuda", torch.float16).contiguous()
        tensors[f"{name}.packed"] = kernels.pack(weight, group).cpu()
        tensors[f"{name}.scales"] = kernels.scale(weight, group).cpu()

    tensors["embed_tokens.weight"] = keep(language.embed_tokens.weight)
    tensors["norm.weight"] = keep(language.norm.weight)
    for index, layer in enumerate(language.layers):
        prefix = f"layers.{index}"
        attn = layer.self_attn
        tensors[f"{prefix}.input_layernorm.weight"] = keep(layer.input_layernorm.weight)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = keep(
            layer.post_attention_layernorm.weight
        )
        tensors[f"{prefix}.q_norm.weight"] = keep(attn.q_norm.weight)
        tensors[f"{prefix}.k_norm.weight"] = keep(attn.k_norm.weight)
        pack(
            f"{prefix}.qkv",
            torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0),
        )
        pack(f"{prefix}.o", attn.o_proj.weight)
        pack(f"{prefix}.gu", torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], 0))
        pack(f"{prefix}.down", layer.mlp.down_proj.weight)
    pack("lm_head", model.lm_head.weight)
    del model
    torch.cuda.empty_cache()

    save_file(tensors, str(target / "model.safetensors"))
    del tensors
    source = Path(model_id)
    for name in COPIED_FILES:
        candidate = source / name
        if candidate.is_file():
            shutil.copy2(candidate, target / name)
    (target / "packed_config.json").write_text(
        json.dumps(
            {"format": PACKED_FORMAT, "version": 1, "group": group, "layout": PACKED_LAYOUT},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "output": str(target),
        "group": group,
        "sizeMb": round(sum(p.stat().st_size for p in target.iterdir()) / 2**20, 1),
    }


class _RMSNorm:
    def __init__(self, weight: Tensor, eps: float):
        self.weight = weight.detach().to(device="cuda", dtype=torch.float16).contiguous()
        self.eps = eps

    def __call__(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class _PackedLinear:
    """Bias-free projection evaluated one token at a time.

    ``keep_fp16`` is used by the mixed-precision probe: the projection then runs
    as a plain fp16 matmul instead of the INT4 GEMV.
    """

    def __init__(self, weight: Tensor, group: int = 32, keep_fp16: bool = False):
        weight = weight.detach().to(device="cuda", dtype=torch.float16).contiguous()
        self.out_features = weight.shape[0]
        self.fp16 = weight if keep_fp16 else None
        self.packed = None if keep_fp16 else extension().pack(weight, group)
        self.scales = None if keep_fp16 else extension().scale(weight, group)

    @classmethod
    def from_packed(cls, packed: Tensor, scales: Tensor) -> "_PackedLinear":
        """Build from a checkpoint that already holds INT4 nibbles and scales."""
        linear = cls.__new__(cls)
        linear.out_features = int(packed.shape[0])
        linear.fp16 = None
        linear.packed = packed.to(device="cuda").contiguous()
        linear.scales = scales.to(device="cuda", dtype=torch.float16).contiguous()
        return linear

    def gemm(self, x: Tensor, residual: Tensor | None = None) -> Tensor:
        """Multi-row projection: the weights are read once for M rows."""
        if self.packed is None:
            out = x.reshape(-1, x.shape[-1]) @ self.fp16.T
            out = out.view(*x.shape[:-1], self.out_features)
            return out + residual if residual is not None else out
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        if residual is None:
            out = extension().gemm(flat, self.packed, self.scales, flat.new_empty((0,)))
        else:
            out = extension().gemm(
                flat, self.packed, self.scales,
                residual.reshape(-1, self.out_features).contiguous(),
            )
        return out.view(*x.shape[:-1], self.out_features)

    def __call__(self, x: Tensor, residual: Tensor | None = None) -> Tensor:
        if self.packed is None:
            out = x.reshape(1, -1) @ self.fp16.T
        elif residual is None:
            out = extension().gemv(x.reshape(1, -1).contiguous(), self.packed, self.scales)
        else:
            out = extension().gemv_residual(
                x.reshape(1, -1).contiguous(), self.packed, self.scales, residual.reshape(1, -1)
            )
        out = out.view(*x.shape[:-1], self.out_features)
        if residual is not None and self.packed is None:
            out = out + residual
        return out


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
        outputs = {
            engine.get_tensor_name(index)
            for index in range(engine.num_io_tensors)
            if engine.get_tensor_mode(engine.get_tensor_name(index)) == trt.TensorIOMode.OUTPUT
        }
        # Engines exported with include_lm_head=False emit the final normed hidden
        # state and the runtime applies the INT4 output projection itself.
        self.emits_logits = "logits" in outputs
        # A graph quantized for Q/DQ keeps float32 activations, so its bindings can
        # differ from the fp16 the decoder carries.
        self.torch_dtypes = {
            trt.DataType.FLOAT: torch.float32,
            trt.DataType.HALF: torch.float16,
            trt.DataType.BF16: torch.bfloat16,
        }
        self.input_dtype = self.torch_dtypes.get(
            engine.get_tensor_dtype("inputs_embeds"), torch.float16
        )
        self.output_dtype = self.torch_dtypes.get(
            engine.get_tensor_dtype("logits" if self.emits_logits else "hidden"), torch.float16
        )
        profile = engine.get_tensor_profile_shape("inputs_embeds", 0)
        self.max_seq = int(profile[2][1]) if profile else 0

    def run(self, inputs_embeds: Tensor, layers: int, kv_heads: int, head_dim: int,
            vocab: int, hidden_size: int | None = None):
        seq = int(inputs_embeds.shape[1])
        if self.max_seq and seq > self.max_seq:
            raise ValueError(
                f"prompt of {seq} tokens ({seq / 13:.1f} s of audio) exceeds this prefill engine's "
                f"{self.max_seq}-token profile; rebuild it with "
                f"`qwen3-asr build-decoder-prefill --max-seq {seq}`"
            )
        embeds = inputs_embeds.to(self.input_dtype).contiguous()
        positions = torch.arange(seq, device="cuda").unsqueeze(0)
        head_name = "logits" if self.emits_logits else "hidden"
        head_width = vocab if self.emits_logits else hidden_size
        head = torch.empty((1, 1, head_width), dtype=self.output_dtype, device="cuda")
        cache_k = torch.empty((seq, layers, kv_heads, head_dim), dtype=self.output_dtype,
                              device="cuda")
        cache_v = torch.empty_like(cache_k)
        self.context.set_input_shape("inputs_embeds", (1, seq, embeds.shape[-1]))
        self.context.set_input_shape("position_ids", (1, seq))
        tensors = {
            "inputs_embeds": embeds,
            "position_ids": positions,
            head_name: head,
            "cache_k": cache_k,
            "cache_v": cache_v,
        }
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            self.context.set_tensor_address(name, tensors[name].data_ptr())
        stream = torch.cuda.current_stream()
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("prefill execution failed")
        return head, cache_k, cache_v


class W4A16ASRDecoder:
    """Greedy decoder: TensorRT prefill, packed-INT4 decode.

    The KV cache is allocated per request: ``max_cache`` is a floor, the buffer
    grows in ``growth_stride`` steps to cover the actual prompt plus transcript,
    and the CUDA graph is re-captured once per size.
    """

    def __init__(
        self,
        model_id: str,
        prefill_engine: Path,
        max_cache: int = 512,
        group: int = 32,
        keep_fp16: frozenset[str] = frozenset(),
        growth_stride: int = 512,
    ):
        from transformers import AutoConfig, AutoModelForMultimodalLM

        self.torch = torch
        self.max_cache = max_cache
        self.growth_stride = max(1, growth_stride)
        self.capacity = 0
        self.group = group
        self.keep_fp16 = set(keep_fp16)
        config = AutoConfig.from_pretrained(str(model_id))
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

        cuda = torch.device("cuda")
        packed = resolve_packed(str(model_id))
        descriptor = packed[0] if packed else None
        if descriptor is None:
            logger.warning(
                "%s is not a packed checkpoint; quantizing FP16 weights at load. "
                "Pass --decoder-dir with a decoder-int4-g32 directory for the published "
                "AWQ INT4 decoder (better WER, faster startup).",
                model_id,
            )
        self.qkv, self.o_proj, self.gu, self.down = [], [], [], []
        self.q_norm, self.k_norm = [], []
        self.in_norm, self.post_norm = [], []

        if descriptor is not None:
            # A decoder-only checkpoint: INT4 nibbles and scales already on disk,
            # no audio tower and no FP16 weights.
            if self.keep_fp16:
                raise ValueError("keep_fp16 cannot be used with a packed checkpoint")
            if group != descriptor["group"]:
                raise ValueError(
                    f"this checkpoint is packed with group {descriptor['group']}, not {group}"
                )
            from safetensors.torch import load_file

            data = load_file(str(packed[1]))
            self.embed = data["embed_tokens.weight"].to(cuda, torch.float16).contiguous()
            self.lm_head = _PackedLinear.from_packed(data["lm_head.packed"], data["lm_head.scales"])
            self.final_norm = _RMSNorm(data["norm.weight"], self.eps)
            for index in range(self.layers):
                prefix = f"layers.{index}"
                self.qkv.append(
                    _PackedLinear.from_packed(data[f"{prefix}.qkv.packed"], data[f"{prefix}.qkv.scales"])
                )
                self.o_proj.append(
                    _PackedLinear.from_packed(data[f"{prefix}.o.packed"], data[f"{prefix}.o.scales"])
                )
                self.gu.append(
                    _PackedLinear.from_packed(data[f"{prefix}.gu.packed"], data[f"{prefix}.gu.scales"])
                )
                self.down.append(
                    _PackedLinear.from_packed(data[f"{prefix}.down.packed"], data[f"{prefix}.down.scales"])
                )
                self.q_norm.append(_RMSNorm(data[f"{prefix}.q_norm.weight"], self.eps))
                self.k_norm.append(_RMSNorm(data[f"{prefix}.k_norm.weight"], self.eps))
                self.in_norm.append(_RMSNorm(data[f"{prefix}.input_layernorm.weight"], self.eps))
                self.post_norm.append(_RMSNorm(data[f"{prefix}.post_attention_layernorm.weight"], self.eps))
            del data
        else:
            model = AutoModelForMultimodalLM.from_pretrained(
                str(model_id), dtype=torch.float16, device_map="cpu", low_cpu_mem_usage=True
            )
            language = model.model.language_model

            self.embed = language.embed_tokens.weight.detach().to(cuda, torch.float16).contiguous()
            self.lm_head = _PackedLinear(model.lm_head.weight, group)
            self.final_norm = _RMSNorm(language.norm.weight, self.eps)

            for layer in language.layers:
                attn = layer.self_attn
                self.qkv.append(_PackedLinear(torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], 0), group, "qkv" in self.keep_fp16))
                self.o_proj.append(_PackedLinear(attn.o_proj.weight, group, "o" in self.keep_fp16))
                self.gu.append(_PackedLinear(torch.cat([layer.mlp.gate_proj.weight, layer.mlp.up_proj.weight], 0), group, "gu" in self.keep_fp16))
                self.down.append(_PackedLinear(layer.mlp.down_proj.weight, group, "down" in self.keep_fp16))
                self.q_norm.append(_RMSNorm(attn.q_norm.weight, self.eps))
                self.k_norm.append(_RMSNorm(attn.k_norm.weight, self.eps))
                self.in_norm.append(_RMSNorm(layer.input_layernorm.weight, self.eps))
                self.post_norm.append(_RMSNorm(layer.post_attention_layernorm.weight, self.eps))
            del model

        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        # Kept on the host: the rope table is built there in FP32, which is the
        # arithmetic the reference checkpoints and the earlier runs used.
        self.inv_freq = inv_freq
        self.cache_k: list = []
        self.cache_v: list = []

        self.prefill = _TRTPrefill(prefill_engine)

        self._token = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._pos = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._generated = torch.zeros((1,), dtype=torch.long, device=cuda)
        self._logits = torch.empty((1, self.vocab), dtype=torch.float16, device=cuda)
        self._attn_out = torch.empty((1, self.heads * self.head_dim), dtype=torch.float16, device=cuda)
        self._h_out = torch.zeros((1, self.hidden), dtype=torch.float16, device=cuda)
        self.heads_w1: list = []
        self.heads_w2: list = []
        self._captured: torch.cuda.CUDAGraph | None = None
        # Speculative verification: one batched step per draft width.
        self._verify_graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._v_ids = torch.zeros((1, 8), dtype=torch.long, device=cuda)
        self._v_predicted = torch.zeros((1, 8), dtype=torch.long, device=cuda)
        self._v_hidden = torch.zeros((1, 8, self.hidden), dtype=torch.float16, device=cuda)
        # Same starting capacity as the native runtime (512 slots, 56 MB).
        self._allocate(max(512, max_cache))
        # Packing the projections allocates large FP32 temporaries; return them to
        # the driver so the TensorRT prefill engine has room for its buffers.
        torch.cuda.empty_cache()

    # ------------------------------------------------------------- kv capacity

    def _allocate(self, capacity: int) -> None:
        """Resize the KV cache to ``capacity`` slots, keeping the written prefix."""
        torch = self.torch
        keep = min(int(self._pos.item()), self.capacity) if self.capacity else 0
        previous_k, previous_v = self.cache_k, self.cache_v
        size = (1, self.kv_heads, capacity, self.head_dim)
        self.cache_k = [
            torch.zeros(size, dtype=torch.float16, device="cuda") for _ in range(self.layers)
        ]
        self.cache_v = [
            torch.zeros(size, dtype=torch.float16, device="cuda") for _ in range(self.layers)
        ]
        for index in range(self.layers):
            if keep:
                self.cache_k[index][:, :, :keep] = previous_k[index][:, :, :keep]
                self.cache_v[index][:, :, :keep] = previous_v[index][:, :, :keep]

        freqs = torch.outer(torch.arange(capacity, dtype=torch.float32), self.inv_freq)
        table = torch.cat([freqs, freqs], dim=-1)
        self.cos_table = table.cos().to("cuda", torch.float16).contiguous()
        self.sin_table = table.sin().to("cuda", torch.float16).contiguous()
        self._produced = torch.zeros((capacity,), dtype=torch.long, device="cuda")
        self.capacity = capacity
        # The graph holds the buffer pointers it was captured with.
        self._captured = None
        self._verify_graphs.clear()

    def ensure_capacity(self, tokens: int) -> None:
        """Grow the cache to cover ``tokens`` slots; ``max_cache`` is a floor."""
        if tokens <= self.capacity:
            return
        step = self.growth_stride
        target = ((tokens + step - 1) // step) * step
        print(
            f"[cache] growing KV cache {self.capacity} -> {target} slots for a {tokens}-token request",
            file=sys.stderr,
        )
        self.max_cache = max(self.max_cache, target)
        self._allocate(target)

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

    def candidate_tokens(self, hidden: Tensor, count: int) -> list[int]:
        """Proposes tokens from a post-norm hidden state with the fitted heads.

        ``hidden`` is the final-norm output of the last accepted position; each
        head adds a residual MLP to it and the shared packetised LM head scores
        the results in one pass.
        """
        torch = self.torch
        if not self.heads_w1:
            raise RuntimeError("no prediction heads attached")
        x = hidden.view(1, -1)
        rows = []
        for w1, w2 in zip(self.heads_w1, self.heads_w2):
            rows.append(x + F.silu(x @ w1.t()) @ w2.t())
        stacked = torch.cat(rows, 0)
        logits = self.lm_head.gemm(stacked)
        return [int(value) for value in logits.argmax(-1).tolist()[:count]]

    def load_heads(self, path: str | Path) -> None:
        torch = self.torch
        state = torch.load(str(path), map_location="cpu")
        self.heads_w1 = [value.to("cuda", torch.float16).contiguous() for value in state["w1"]]
        self.heads_w2 = [value.to("cuda", torch.float16).contiguous() for value in state["w2"]]

    def _step(self) -> None:
        torch = self.torch
        pos = self._pos
        cos = self.cos_table.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        sin = self.sin_table.index_select(0, pos).unsqueeze(0).unsqueeze(0)
        hidden = F.embedding(self._token.view(1, 1), self.embed)
        for index in range(self.layers):
            hidden = self._layer(index, hidden, pos, cos, sin)
        normed = self.final_norm(hidden)
        self._h_out.copy_(normed.view(1, -1))
        logits = self.lm_head(normed)
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

    def prefill_prompt(self, inputs_embeds: Tensor, reserve: int = 0) -> tuple[Tensor, int]:
        prompt = int(inputs_embeds.shape[1])
        # A new request starts at position 0, so a grow here has no prefix to keep.
        self._pos.fill_(0)
        self.ensure_capacity(prompt + max(0, reserve))
        head, cache_k, cache_v = self.prefill.run(
            inputs_embeds, self.layers, self.kv_heads, self.head_dim, self.vocab,
            hidden_size=self.hidden,
        )
        if head.dtype != torch.float16:
            head = head.to(torch.float16)
        if self.prefill.emits_logits:
            logits = head
        else:
            normed = self.final_norm(head)
            self._h_out.copy_(normed.view(1, -1))
            logits = self.lm_head(normed)
        seq = cache_k.shape[0]
        for index in range(self.layers):
            self.cache_k[index][:, :, :seq] = cache_k[:, index].permute(1, 0, 2).unsqueeze(0)
            self.cache_v[index][:, :, :seq] = cache_v[:, index].permute(1, 0, 2).unsqueeze(0)
        self._pos.fill_(seq)
        self._token.copy_(logits[:, -1, :].argmax(-1))
        return logits, seq

    def generate(self, inputs_embeds: Tensor, max_new_tokens: int = 128, chunk: int = 8) -> list[int]:
        # Reserve the whole transcript up front so any growth happens before the
        # prefill writes the prompt into the cache.
        _, _ = self.prefill_prompt(inputs_embeds, reserve=max_new_tokens)
        return self.decode_loop(max_new_tokens, chunk)

    def decode_loop(self, max_new_tokens: int = 128, chunk: int = 8) -> list[int]:
        """Greedy decode from the current prefill state (no prompt re-encoding)."""
        torch = self.torch
        eos = (151643, 151645)
        position = int(self._pos.item())
        self.ensure_capacity(position + max_new_tokens)
        # The argmax of the prefill logits is the first generated token; the step
        # graph produces the one after it.
        self._produced[0].copy_(self._token[0])
        self._generated.fill_(1)
        produced = 1
        tokens = [int(self._token.item())]
        if tokens[0] in eos or max_new_tokens <= 1:
            return tokens[:max_new_tokens]
        while produced < max_new_tokens and position < self.capacity:
            graph = self._graph()
            steps = min(chunk, max_new_tokens - produced, self.capacity - position)
            for _ in range(steps):
                graph.replay()
            torch.cuda.synchronize()
            position += steps
            produced = int(self._generated.item())
            tokens = self._produced[:produced].tolist()
            stop = next((i for i, value in enumerate(tokens) if value in eos), None)
            if stop is not None:
                return tokens[:stop]
        if produced < max_new_tokens:
            raise RuntimeError(
                f"transcript stopped at the {self.capacity}-token KV cache after {produced} tokens"
            )
        return self._produced[:produced].tolist()

    # ------------------------------------------------------- speculative use

    def rewind(self, position: int, token: int) -> None:
        """Reset the cache length and the next input token.

        Cache rows at or after `position` are stale; the next step overwrites
        them, so a rejected speculative block costs no extra copy. The produced
        counter restarts too, so a draft that runs ahead never overruns its
        token buffer.
        """
        self._pos.fill_(position)
        self._token.fill_(token)
        self._generated.zero_()

    def _rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        half = self.head_dim // 2
        rotated = self.torch.cat((-x[..., half:], x[..., :half]), dim=-1)
        return x * cos + rotated * sin

    def _expand_kv(self, tensor: Tensor) -> Tensor:
        return tensor[:, :, None].expand(-1, -1, self.groups, -1, -1).reshape(
            1, self.heads, tensor.shape[2], tensor.shape[3]
        )

    def _verify_step(self, rows: int) -> None:
        """One batched target step over ``rows`` tokens."""
        torch = self.torch
        ext = extension()
        positions = self._pos + torch.arange(rows, device="cuda", dtype=torch.long)
        cos = self.cos_table.index_select(0, positions)
        sin = self.sin_table.index_select(0, positions)

        hidden = self.embed[self._v_ids[:, :rows]]
        q_dim = self.heads * self.head_dim
        k_dim = self.kv_heads * self.head_dim
        qkv_dim = q_dim + 2 * k_dim

        for index in range(self.layers):
            normed = self.in_norm[index](hidden)
            qkv = self.qkv[index].gemm(normed)
            ext.qk_norm_rope_rows(
                qkv, self.q_norm[index].weight, self.k_norm[index].weight, cos, sin,
                rows, q_dim, k_dim, self.heads, self.kv_heads, self.eps,
            )
            k = qkv[..., q_dim:q_dim + k_dim].view(1, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            v = qkv[..., q_dim + k_dim:].view(1, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            self.cache_k[index].index_copy_(2, positions, k)
            self.cache_v[index].index_copy_(2, positions, v)
            # The kernel reads the sequence length from `_pos`, so one graph
            # serves every position and no length bucketing is needed.
            attended = ext.verify_attention(
                qkv, qkv_dim, self.cache_k[index], self.cache_v[index], self._pos,
                rows, self.heads, self.head_dim, self.groups,
            )
            hidden = self.o_proj[index].gemm(attended, hidden)
            normed = self.post_norm[index](hidden)
            gate_up = self.gu[index].gemm(normed)
            gated = ext.silu_mul_rows(gate_up, rows, self.intermediate)
            hidden = self.down[index].gemm(gated, hidden)

        normed = self.final_norm(hidden)
        self._v_hidden[:, :rows].copy_(normed)
        logits = self.lm_head.gemm(normed)
        self._v_predicted[:, :rows].copy_(logits.argmax(-1))

    def _verify_graph(self, rows: int) -> "torch.cuda.CUDAGraph":
        graph = self._verify_graphs.get(rows)
        if graph is None:
            # Warm-up writes the same cache rows the capture then overwrites, and
            # neither run moves the position, so no state has to be restored.
            self._verify_step(rows)
            graph = self.torch.cuda.CUDAGraph()
            with self.torch.cuda.graph(graph):
                self._verify_step(rows)
            self._verify_graphs[rows] = graph
        return graph

    def verify(self, candidates: list[int]) -> tuple[int, list[int]]:
        """Runs the target over a draft block in one pass.

        Feeds the current token plus `candidates`, so the pass predicts every
        candidate position and the one after the block. Returns how many
        candidates matched and the tokens to append (the accepted prefix plus
        the target's correction, or its bonus token when everything matched).
        """
        torch = self.torch
        base = int(self._pos.item())
        count = len(candidates)
        rows = count + 1
        self.ensure_capacity(base + rows + 2)

        self._v_ids[0, 0].copy_(self._token[0])
        self._v_ids[0, 1:rows].copy_(
            torch.tensor(candidates, dtype=torch.long, device="cuda")
        )
        self._verify_graph(rows).replay()
        predicted = [int(value) for value in self._v_predicted[0, :rows].tolist()]

        matched = 0
        while matched < count and predicted[matched] == candidates[matched]:
            matched += 1
        emitted = list(candidates[:matched]) + [predicted[matched]]
        for position, token in enumerate(emitted):
            if token in (151643, 151645):
                emitted = emitted[: position + 1]
                break
        self._pos.add_(matched + 1)
        self._token.fill_(emitted[-1])
        return matched, emitted
