import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from qwen3_asr_trt.server import make_handler


class FakeTranscriber:
    def __init__(self) -> None:
        self.call: tuple[bytes, str | None] | None = None

    def transcribe(self, audio_path: Path, language: str | None = None) -> str:
        self.call = (audio_path.read_bytes(), language)
        return "recognized"


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.transcriber = FakeTranscriber()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.transcriber))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.connection = HTTPConnection("127.0.0.1", self.server.server_port)

    def tearDown(self) -> None:
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_transcribe_accepts_raw_audio_and_language(self) -> None:
        self.connection.request(
            "POST",
            "/v1/transcribe?language=Chinese",
            body=b"audio-bytes",
            headers={"Content-Type": "audio/wav"},
        )
        response = self.connection.getresponse()
        payload = json.loads(response.read())

        self.assertEqual(response.status, 200)
        self.assertEqual(payload, {"text": "recognized"})
        self.assertEqual(self.transcriber.call, (b"audio-bytes", "Chinese"))

    def test_rejects_unsupported_content_type(self) -> None:
        self.connection.request(
            "POST", "/v1/transcribe", body=b"audio", headers={"Content-Type": "text/plain"}
        )
        response = self.connection.getresponse()

        self.assertEqual(response.status, 415)
        self.assertIsNone(self.transcriber.call)


if __name__ == "__main__":
    unittest.main()
