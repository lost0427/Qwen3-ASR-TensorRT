"""Micro-benchmark for the batched W4A16 GEMM.

Reports, per projection shape, the single-row and five-row cost, the implied
bandwidth, and whether the batched result still matches the single-row kernel.

    python tools/bench_w4a16_gemm.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHAPES = (
    ("lm_head", 151936, 2048),
    ("gu", 12288, 2048),
    ("qkv", 6144, 2048),
    ("o", 2048, 2048),
    ("down", 2048, 6144),
)


def bench(fn, n: int = 40) -> float:
    fn()
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started) / n * 1000


def main() -> None:
    from qwen3_asr_trt.w4a16 import extension

    ext = extension()
    torch.manual_seed(0)
    empty = torch.empty(0, device="cuda", dtype=torch.float16)

    # The card idles at ~200 MHz and only boosts under sustained load, so the first
    # shape measured would otherwise look 2-3x slower than the rest.
    warm_weight = torch.randn(151936, 2048, device="cuda", dtype=torch.float16) * 0.02
    warm_packed = ext.pack(warm_weight, 32)
    warm_scales = ext.scale(warm_weight, 32)
    warm_x = torch.randn(5, 2048, device="cuda", dtype=torch.float16)
    deadline = time.perf_counter() + 3.0
    while time.perf_counter() < deadline:
        for _ in range(20):
            ext.gemm(warm_x, warm_packed, warm_scales, empty)
        torch.cuda.synchronize()
    del warm_weight, warm_packed, warm_scales
    torch.cuda.empty_cache()

    print(f"{'shape':>8} {'rows':>7} {'k':>5} {'m1 ms':>7} {'m5 ms':>7} {'ratio':>6} "
          f"{'m5 GB/s':>8} {'max|diff|':>9}")
    for name, rows, k in SHAPES:
        weight = torch.randn(rows, k, device="cuda", dtype=torch.float16) * 0.02
        packed = ext.pack(weight, 32)
        scales = ext.scale(weight, 32)
        x1 = torch.randn(1, k, device="cuda", dtype=torch.float16)
        x5 = torch.randn(5, k, device="cuda", dtype=torch.float16)
        one = bench(lambda: ext.gemm(x1, packed, scales, empty))
        five = bench(lambda: ext.gemm(x5, packed, scales, empty))
        got = ext.gemm(x5, packed, scales, empty)
        ref = torch.stack([ext.gemv(x5[i:i + 1].contiguous(), packed, scales)[0] for i in range(5)])
        diff = float((got.float() - ref.float()).abs().max())
        weight_mb = rows * (k // 2) / 2**20
        print(f"{name:>8} {rows:>7} {k:>5} {one:7.3f} {five:7.3f} {five / one:6.2f} "
              f"{weight_mb / five:8.1f} {diff:9.5f}")
        del weight, packed, scales
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
