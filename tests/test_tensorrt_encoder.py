import unittest

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


if __name__ == "__main__":
    unittest.main()
