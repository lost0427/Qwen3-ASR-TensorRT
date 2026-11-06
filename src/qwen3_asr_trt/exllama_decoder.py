from __future__ import annotations

import json
import shutil
from pathlib import Path


def export_text_model(source: str, output_dir: Path) -> Path:
    """Extract Qwen3 text weights into the standalone layout ExLlamaV3 expects."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    source_dir = Path(source)
    if not source_dir.is_dir():
        source_dir = Path(snapshot_download(source))

    with (source_dir / "config.json").open(encoding="utf-8") as stream:
        source_config = json.load(stream)
    if source_config.get("model_type") != "qwen3_asr":
        raise ValueError(f"Expected a Qwen3-ASR checkpoint, got {source_config.get('model_type')!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    thinker_config = source_config.get("thinker_config", source_config)
    config = dict(thinker_config.get("text_config", source_config.get("text_config", {})))
    if not config:
        raise ValueError("Checkpoint does not contain a Qwen3 text configuration")
    config["architectures"] = ["Qwen3ForCausalLM"]
    config["model_type"] = "qwen3"
    config["tie_word_embeddings"] = thinker_config.get("tie_word_embeddings", source_config.get("tie_word_embeddings", True))
    config["eos_token_id"] = thinker_config.get("eos_token_id", source_config.get("eos_token_id", [151643, 151645]))
    config["pad_token_id"] = thinker_config.get("pad_token_id", source_config.get("pad_token_id", 151645))
    with (output_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2)

    weights: dict[str, object] = {}
    shard_files = sorted(source_dir.glob("*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No safetensors weights found in {source_dir}")
    for shard in shard_files:
        with safe_open(shard, framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                for prefix in ("model.language_model.", "thinker.model."):
                    if key.startswith(prefix):
                        weights["model." + key[len(prefix):]] = checkpoint.get_tensor(key).contiguous()
                        break
                else:
                    if key == "thinker.lm_head.weight":
                        weights["lm_head.weight"] = checkpoint.get_tensor(key).contiguous()
    if not weights:
        raise ValueError(f"No Qwen3 language-model weights found in {source_dir}")
    save_file(weights, str(output_dir / "model.safetensors"))

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        source_file = source_dir / name
        if source_file.is_file():
            shutil.copy2(source_file, output_dir / name)
    return output_dir
