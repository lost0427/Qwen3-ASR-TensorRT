import json
import logging
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .runtime import Transcriber

logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 128 * 1024 * 1024
MIME_SUFFIXES = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
}


def make_handler(
    transcriber: Transcriber, aligner: object | None = None
) -> type[BaseHTTPRequestHandler]:
    inference_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict[str, str]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _align(self, parsed, suffix: str) -> None:
            if aligner is None:
                self._json(404, {"error": "not found"})
                return
            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(400, {"error": "valid Content-Length is required"})
                return
            if content_length <= 0:
                self._json(400, {"error": "audio body is empty"})
                return
            if content_length > MAX_AUDIO_BYTES:
                self._json(413, {"error": "audio exceeds the 128 MiB limit"})
                return
            query = parse_qs(parsed.query, keep_blank_values=True)
            text = query.get("text", [""])[0]
            if not text.strip():
                self._json(400, {"error": "the text query parameter is required"})
                return
            language = query.get("language", ["English"])[0]
            if len(language) > 40:
                self._json(400, {"error": "language is too long"})
                return
            audio = self.rfile.read(content_length)
            if len(audio) != content_length:
                self._json(400, {"error": "incomplete audio body"})
                return
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(audio)
                with inference_lock:
                    items = aligner.align(temporary_path, text, language)
                self._json(200, {"items": items, "timings": aligner.last_timings})
            except Exception as error:
                logger.exception("Alignment request failed")
                self._json(400, {"error": str(error)})
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        def do_POST(self) -> None:
            parsed = urlsplit(self.path)
            if parsed.path not in ("/v1/transcribe", "/v1/align"):
                self._json(404, {"error": "not found"})
                return

            content_type = self.headers.get_content_type().lower()
            suffix = MIME_SUFFIXES.get(content_type)
            if suffix is None:
                self._json(415, {"error": "unsupported audio content type"})
                return
            if parsed.path == "/v1/align":
                self._align(parsed, suffix)
                return

            try:
                content_length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._json(400, {"error": "valid Content-Length is required"})
                return
            if content_length <= 0:
                self._json(400, {"error": "audio body is empty"})
                return
            if content_length > MAX_AUDIO_BYTES:
                self._json(413, {"error": "audio exceeds the 128 MiB limit"})
                return

            query = parse_qs(parsed.query, keep_blank_values=True)
            language = query.get("language", [None])[0]
            if language is not None and len(language) > 40:
                self._json(400, {"error": "language is too long"})
                return

            audio = self.rfile.read(content_length)
            if len(audio) != content_length:
                self._json(400, {"error": "incomplete audio body"})
                return

            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(audio)
                with inference_lock:
                    text = transcriber.transcribe(temporary_path, language)
                    detected = getattr(transcriber, "last_language", None)
                payload = {"text": text}
                if detected:
                    payload["language"] = detected
                self._json(200, payload)
            except Exception:
                logger.exception("Transcription request failed")
                self._json(500, {"error": "transcription failed"})
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)

        def log_message(self, format: str, *args: object) -> None:
            logger.info("%s - %s", self.address_string(), format % args)

    return Handler


def serve(transcriber: Transcriber, host: str, port: int, aligner: object | None = None) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(transcriber, aligner))
    logger.info("Listening on http://%s:%d", host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
