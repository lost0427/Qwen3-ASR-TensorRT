import argparse
import json
import logging
from pathlib import Path

from .runtime import Bf16ReferenceTranscriber
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

    api = commands.add_parser("serve")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    transcriber = Bf16ReferenceTranscriber(args.model, args.max_new_tokens)
    if args.command == "transcribe":
        if not args.audio.is_file():
            parser.error(f"audio file not found: {args.audio}")
        print(json.dumps({"text": transcriber.transcribe(args.audio, args.language)}, ensure_ascii=False))
    else:
        serve(transcriber, args.host, args.port)
