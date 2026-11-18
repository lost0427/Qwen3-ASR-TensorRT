"""W4A16 GEMV path for the single-token decode step.

Weights are stored as packed INT4 (two values per byte along K) plus FP16 scales
per 64-element group. The kernel is bandwidth bound, which is what the decode
step needs: M = 1, K = 1024/2048/3072.

The extension is compiled on first use and cached by torch. On Windows it needs
`cl.exe` on PATH, i.e. run from a "x64 Native Tools" prompt or call
`vcvars64.bat` first.
"""

from __future__ import annotations

from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
_SOURCES = [
    _ROOT / "csrc" / "w4a16_kernel.cu",
    _ROOT / "csrc" / "w4a16_gemm.cu",
    _ROOT / "csrc" / "verify_ops.cu",
    _ROOT / "csrc" / "decode_fusions.cu",
    _ROOT / "csrc" / "decode_attention.cu",
    _ROOT / "csrc" / "w4a16_binding.cpp",
]
_extension = None


def _arch_flags() -> list[str]:
    """Match the kernels to the GPU they will run on."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return []
    arch = f"{major}{minor}"
    return [f"--generate-code=arch=compute_{arch},code=sm_{arch}"]


def extension():
    """Compile (once) and return the CUDA extension."""
    global _extension
    if _extension is not None:
        return _extension
    from torch.utils.cpp_extension import load

    _extension = load(
        name="w4a16_gemv",
        sources=[str(path) for path in _SOURCES],
        extra_cuda_cflags=["-O3", "--use_fast_math", *_arch_flags()],
        verbose=False,
    )
    return _extension

