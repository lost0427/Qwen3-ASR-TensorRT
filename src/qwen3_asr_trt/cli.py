import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys

from .runtime import Bf16ReferenceTranscriber
from .runtime import TensorRTExLlamaTranscriber
from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(prog="qwen3-asr")
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-0.6B-hf")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser("transcribe")
    transcribe.add_argument("audio", type=Path)
    transcribe.add_argument("--language")
    transcribe.add_argument("--backend", choices=("bf16", "tensorrt-exllama"), default="bf16")
    transcribe.add_argument("--decoder-dir", type=Path, default=Path("artifacts/qwen3-asr-0.6b/decoder"))
    transcribe.add_argument("--engine", type=Path, default=Path("artifacts/qwen3-asr-0.6b/encoder-window.bf16.engine"))

    api = commands.add_parser("serve")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--backend", choices=("bf16", "tensorrt-exllama"), default="bf16")
    api.add_argument("--decoder-dir", type=Path, default=Path("artifacts/qwen3-asr-0.6b/decoder"))
    api.add_argument("--engine", type=Path, default=Path("artifacts/qwen3-asr-0.6b/encoder-window.bf16.engine"))

    build = commands.add_parser("build")
    build.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3-asr-0.6b"))
    build.add_argument("--workspace-gib", type=int, default=2)
    build.add_argument("--encoder-precision", choices=("bf16", "fp16"), default="bf16")

    quantize = commands.add_parser("quantize")
    quantize.add_argument("--input-dir", type=Path, default=Path("artifacts/qwen3-asr-0.6b/decoder"))
    quantize.add_argument("--output-dir", type=Path, default=Path("artifacts/qwen3-asr-0.6b/decoder-exl3-4bpw"))
    quantize.add_argument("--bits", type=float, default=4.0)
    quantize.add_argument("--recipe", type=Path)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.command == "build":
        from .exllama_decoder import export_text_model
        from .tensorrt_encoder import build_engine, export_encoder

        precision = args.encoder_precision
        onnx_path = args.output_dir / f"encoder-window.{precision}.onnx"
        engine_path = args.output_dir / f"encoder-window.{precision}.engine"
        export_encoder(args.model, onnx_path, precision)
        build_engine(onnx_path, engine_path, args.workspace_gib)
        export_text_model(args.model, args.output_dir / "decoder")
        return

    if args.command == "quantize":
        command = [
            sys.executable,
            "-m",
            "exllamav3.conversion.convert_model",
            "--in_dir",
            str(args.input_dir),
            "--out_dir",
            str(args.output_dir),
            "--bits",
            str(args.bits),
        ]
        if args.recipe:
            command.extend(("--recipe", str(args.recipe)))
        subprocess.run(command, check=True)
        return

    if args.backend == "tensorrt-exllama":
        transcriber = TensorRTExLlamaTranscriber(
            args.model,
            args.decoder_dir,
            args.engine,
            args.max_new_tokens,
        )
    else:
        transcriber = Bf16ReferenceTranscriber(args.model, args.max_new_tokens)
    if args.command == "transcribe":
        if not args.audio.is_file():
            parser.error(f"audio file not found: {args.audio}")
        print(json.dumps({"text": transcriber.transcribe(args.audio, args.language)}, ensure_ascii=False))
    else:
        serve(transcriber, args.host, args.port)
