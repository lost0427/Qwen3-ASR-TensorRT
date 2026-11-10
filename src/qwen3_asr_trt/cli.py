import argparse
import json
import logging
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

from .runtime import Bf16ReferenceTranscriber
from .runtime import TensorRTExLlamaTranscriber
from .server import serve


def _artifact_dir(model_id: str) -> Path:
    model_name = Path(model_id.rstrip("/\\")).name.lower()
    if model_name.endswith("-hf"):
        model_name = model_name[:-3]
    return Path("artifacts") / model_name


def _wer(reference: str, hypothesis: str) -> float:
    normalize = lambda text: re.sub(r"\s+", " ", re.sub(r"[^A-Z0-9 ]", " ", text.upper())).strip().split()
    expected, actual = normalize(reference), normalize(hypothesis)
    previous = list(range(len(actual) + 1))
    for i, token in enumerate(expected):
        current = [i + 1]
        for j, other in enumerate(actual):
            current.append(min(current[-1] + 1, previous[j + 1] + 1, previous[j] + (token != other)))
        previous = current
    return previous[-1] / max(1, len(expected))


def _srt_time(seconds: float) -> str:
    total_ms = max(0, round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_srt(items: list[dict]) -> str:
    return "\n".join(
        f"{index}\n{_srt_time(item['start_time'])} --> {_srt_time(item['end_time'])}\n{item['text']}\n"
        for index, item in enumerate(items, 1)
    ).rstrip()


def _format_txt(items: list[dict]) -> str:
    return "\n".join(
        f"{item['start_time']:.3f}\t{item['end_time']:.3f}\t{item['text']}"
        for item in items
    )


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
    transcribe.add_argument("--decoder-dir", type=Path)
    transcribe.add_argument("--engine", type=Path)

    align = commands.add_parser("align")
    align.add_argument("audio", type=Path)
    align.add_argument("--text")
    align.add_argument("--text-file", type=Path)
    align.add_argument("--language", required=True)
    align.add_argument("--backend", choices=("tensorrt-exllama",), default="tensorrt-exllama")
    align.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    align.add_argument("--decoder-dir", type=Path)
    align.add_argument("--engine", type=Path)
    align.add_argument("--format", choices=("json", "srt", "txt"), default="json")
    align.add_argument("--output", type=Path)

    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("audio", type=Path)
    benchmark.add_argument("--language", default="English")
    benchmark.add_argument("--backend", choices=("bf16", "tensorrt-exllama"), default="tensorrt-exllama")
    benchmark.add_argument("--decoder-dir", type=Path)
    benchmark.add_argument("--engine", type=Path)
    benchmark.add_argument("--warmups", type=int, default=2)
    benchmark.add_argument("--repeats", type=int, default=10)

    api = commands.add_parser("serve")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--backend", choices=("bf16", "tensorrt-exllama"), default="bf16")
    api.add_argument("--decoder-dir", type=Path)
    api.add_argument("--engine", type=Path)

    build = commands.add_parser("build")
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--workspace-gib", type=int, default=2)
    build.add_argument("--encoder-precision", choices=("bf16", "fp16", "int8", "int4"), default="bf16")

    build_aligner = commands.add_parser("build-aligner")
    build_aligner.add_argument("--aligner-model", default="Qwen/Qwen3-ForcedAligner-0.6B")
    build_aligner.add_argument("--output-dir", type=Path)
    build_aligner.add_argument("--workspace-gib", type=int, default=2)

    quantize = commands.add_parser("quantize")
    quantize.add_argument("--input-dir", type=Path)
    quantize.add_argument("--output-dir", type=Path)
    quantize.add_argument("--work-dir", type=Path)
    quantize.add_argument("--bits", type=float, default=4.0)
    quantize.add_argument("--recipe", type=Path)
    quantize.add_argument("--cal-data", type=Path)
    quantize.add_argument("--cal-rows", type=int, default=250)
    quantize.add_argument("--head-bits", type=int, default=6)

    quantize_encoder = commands.add_parser("quantize-encoder")
    quantize_encoder.add_argument("calibration_audio_dir", type=Path)
    quantize_encoder.add_argument("--input-onnx", type=Path)
    quantize_encoder.add_argument("--output-onnx", type=Path)
    quantize_encoder.add_argument("--max-audio-files", type=int, default=32)
    quantize_encoder.add_argument("--language", default="English")

    quantize_encoder_int4 = commands.add_parser("quantize-encoder-int4")
    quantize_encoder_int4.add_argument("--input-onnx", type=Path)
    quantize_encoder_int4.add_argument("--output-onnx", type=Path)

    args = parser.parse_args()
    artifact_dir = _artifact_dir(args.model)
    if args.command == "build":
        args.output_dir = args.output_dir or artifact_dir
    elif args.command == "build-aligner":
        args.output_dir = args.output_dir or _artifact_dir(args.aligner_model)
    elif args.command == "quantize":
        args.input_dir = args.input_dir or artifact_dir / "decoder"
        args.output_dir = args.output_dir or artifact_dir / "decoder-exl3-4bpw"
    elif args.command == "quantize-encoder":
        args.input_onnx = args.input_onnx or artifact_dir / "encoder-window.fp16.onnx"
        args.output_onnx = args.output_onnx or artifact_dir / "encoder-window.int8.onnx"
    elif args.command == "quantize-encoder-int4":
        args.input_onnx = args.input_onnx or artifact_dir / "encoder-window.fp16.onnx"
        args.output_onnx = args.output_onnx or artifact_dir / "encoder-window.int4.onnx"
    elif args.command == "align":
        aligner_dir = _artifact_dir(args.aligner_model)
        args.decoder_dir = args.decoder_dir or aligner_dir / "decoder-exl3-4bpw"
        args.engine = args.engine or aligner_dir / "aligner-encoder-window.bf16.engine"
    else:
        args.decoder_dir = args.decoder_dir or artifact_dir / "decoder"
        args.engine = args.engine or artifact_dir / "encoder-window.bf16.engine"
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
        if precision in ("int8", "int4"):
            if not onnx_path.is_file():
                parser.error(f"{precision.upper()} encoder ONNX not found: {onnx_path}")
        else:
            export_encoder(args.model, onnx_path, precision)
        build_engine(onnx_path, engine_path, args.workspace_gib)
        export_text_model(args.model, args.output_dir / "decoder")
        return

    if args.command == "build-aligner":
        from .aligner import export_aligner_decoder, export_aligner_encoder
        from .tensorrt_encoder import build_engine

        onnx_path = args.output_dir / "aligner-encoder-window.bf16.onnx"
        engine_path = args.output_dir / "aligner-encoder-window.bf16.engine"
        export_aligner_encoder(args.aligner_model, onnx_path)
        build_engine(onnx_path, engine_path, args.workspace_gib)
        export_aligner_decoder(args.aligner_model, args.output_dir / "decoder")
        return

    if args.command == "align":
        from .aligner import TensorRTExLlamaAligner

        if args.text is None and args.text_file is None:
            parser.error("align requires --text or --text-file")
        if args.text is not None and args.text_file is not None:
            parser.error("use either --text or --text-file")
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        aligner = TensorRTExLlamaAligner(
            args.aligner_model, args.decoder_dir, args.engine
        )
        items = aligner.align(args.audio, text.strip(), args.language)
        if args.format == "json":
            payload = json.dumps(
                {"items": items, "timings": aligner.last_timings},
                indent=2,
                ensure_ascii=False,
            )
        elif args.format == "srt":
            payload = _format_srt(items)
        else:
            payload = _format_txt(items)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
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
            "--work_dir",
            str(args.work_dir or args.output_dir.with_name(args.output_dir.name + ".work")),
            "--bits",
            str(args.bits),
            "--head_bits",
            str(args.head_bits),
        ]
        if args.recipe:
            command.extend(("--recipe", str(args.recipe)))
        if args.cal_data:
            command.extend(("--cal_data", str(args.cal_data), "--cal_rows", str(args.cal_rows)))
        subprocess.run(command, check=True)
        return

    if args.command == "quantize-encoder":
        from .encoder_quantization import quantize_encoder

        if not args.calibration_audio_dir.is_dir():
            parser.error(f"calibration audio directory not found: {args.calibration_audio_dir}")
        if args.max_audio_files < 1:
            parser.error("max-audio-files must be positive")
        report = quantize_encoder(
            args.model,
            args.input_onnx,
            args.output_onnx,
            args.calibration_audio_dir,
            args.max_audio_files,
            args.language,
        )
        manifest_path = args.output_onnx.with_suffix(args.output_onnx.suffix + ".calibration.json")
        manifest_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report["calibrationManifest"] = str(manifest_path)
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.command == "quantize-encoder-int4":
        from .encoder_quantization import quantize_encoder_int4

        report = quantize_encoder_int4(args.input_onnx, args.output_onnx)
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.backend == "tensorrt-exllama":
        load_start = time.perf_counter()
        transcriber = TensorRTExLlamaTranscriber(
            args.model,
            args.decoder_dir,
            args.engine,
            args.max_new_tokens,
        )
    else:
        load_start = time.perf_counter()
        transcriber = Bf16ReferenceTranscriber(args.model, args.max_new_tokens)
    load_ms = (time.perf_counter() - load_start) * 1000

    if args.command == "benchmark":
        if not args.audio.is_file():
            parser.error(f"audio file not found: {args.audio}")
        if args.warmups < 0 or args.repeats < 1:
            parser.error("warmups must be nonnegative and repeats must be positive")
        import soundfile

        audio_seconds = soundfile.info(args.audio).duration

        def run_once() -> tuple[float, str]:
            started = time.perf_counter()
            result = transcriber.transcribe(args.audio, args.language)
            return (time.perf_counter() - started) * 1000, result

        for _ in range(args.warmups):
            run_once()
        samples = []
        stage_samples = {}
        for _ in range(args.repeats):
            elapsed, result = run_once()
            samples.append((elapsed, result))
            for name, value in getattr(transcriber, "last_timings", {}).items():
                stage_samples.setdefault(name, []).append(value)
        times = sorted(sample[0] for sample in samples)
        p90_index = (len(times) - 1) * 0.9
        p90_low = math.floor(p90_index)
        p90 = times[p90_low] + (times[math.ceil(p90_index)] - times[p90_low]) * (p90_index - p90_low)
        report = {
            "engine": args.backend,
            "model": args.model,
            "audioSeconds": audio_seconds,
            "loadMs": load_ms,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "meanMs": statistics.mean(times),
            "medianMs": statistics.median(times),
            "p90Ms": p90,
            "minMs": times[0],
            "maxMs": times[-1],
            "rtf": statistics.median(times) / (audio_seconds * 1000),
            "text": samples[-1][1],
        }
        reference_path = args.audio.with_suffix(".json")
        if reference_path.is_file():
            reference = json.loads(reference_path.read_text(encoding="utf-8")).get("reference")
            if reference:
                report["wer"] = _wer(reference, samples[-1][1])
        if stage_samples:
            report["stages"] = {
                name: statistics.median(values) for name, values in stage_samples.items()
            }
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.command == "transcribe":
        if not args.audio.is_file():
            parser.error(f"audio file not found: {args.audio}")
        print(json.dumps({"text": transcriber.transcribe(args.audio, args.language)}, ensure_ascii=False))
    else:
        serve(transcriber, args.host, args.port)
