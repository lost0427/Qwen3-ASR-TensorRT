import unittest
import tempfile
from pathlib import Path

from qwen3_asr_trt.encoder_quantization import _calibration_files
from qwen3_asr_trt.tensorrt_encoder import _post_cnn_tokens


class EncoderLengthTests(unittest.TestCase):
    def test_chunk_boundaries_match_qwen_conv_lengths(self) -> None:
        expected = {
            0: 0,
            1: 1,
            50: 7,
            99: 13,
            100: 13,
            101: 14,
            800: 104,
            801: 105,
            1600: 208,
        }
        for frames, tokens in expected.items():
            with self.subTest(frames=frames):
                self.assertEqual(_post_cnn_tokens(frames), tokens)


class CalibrationFileTests(unittest.TestCase):
    def test_selects_spread_of_supported_audio_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_dir = Path(temporary_dir)
            for name in ("a.wav", "b.flac", "c.mp3", "ignored.txt"):
                (audio_dir / name).touch()

            selected = _calibration_files(audio_dir, 2)

            self.assertEqual([path.name for path in selected], ["a.wav", "c.mp3"])

    def test_requires_supported_audio_and_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            audio_dir = Path(temporary_dir)
            with self.assertRaisesRegex(ValueError, "No supported audio"):
                _calibration_files(audio_dir, 1)
            with self.assertRaisesRegex(ValueError, "must be positive"):
                _calibration_files(audio_dir, 0)


if __name__ == "__main__":
    unittest.main()
