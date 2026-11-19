"""Collapse per-channel weight scales in a Q/DQ prefill graph to per-tensor.

``onnxruntime.quantization`` emits per-output-channel scales for the MatMul
weights by default. TensorRT's strongly-typed parser does not turn that pattern
into a quantized kernel: it dequantizes the weights and runs the matmul in
float32, which is what made the INT8 draft prefill as slow as an fp16 one twice
its size.

The int8 weights are requantized onto a single scale per tensor, so the graph
keeps int8 storage and TRT can pick an INT8 matmul. Requantizing on the way is
equivalent to calibrating the tensor with a per-tensor scale in the first place,
apart from the rounding error of the second rounding step.

    python tools/collapse_weight_scales.py --onnx in.onnx --output out.onnx
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def collapse(model: onnx.ModelProto, percentile: float) -> int:
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    changed = 0
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        weight, scale = node.input[0], node.input[1]
        if weight not in initializers or scale not in initializers:
            continue
        scale_tensor = initializers[scale]
        if len(scale_tensor.dims) != 1:
            continue
        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
        values = numpy_helper.to_array(scale_tensor).astype(np.float32)
        target = float(np.percentile(values, percentile))
        weight_tensor = initializers[weight]
        quantized = numpy_helper.to_array(weight_tensor).astype(np.float32)
        shape = [1] * quantized.ndim
        shape[axis if 0 <= axis < quantized.ndim else 0] = values.size
        factor = (values / target).reshape(shape)
        requantized = np.rint(quantized * factor)
        np.clip(requantized, -127, 127, out=requantized)
        weight_tensor.CopyFrom(numpy_helper.from_array(
            requantized.astype(np.int8), name=weight_tensor.name))
        scale_tensor.CopyFrom(numpy_helper.from_array(
            np.array(target, dtype=np.float32), name=scale_tensor.name))
        # A per-channel zero point would break the same size check on parse; a
        # symmetric quantized weight only ever carries zeros here.
        if len(node.input) > 2 and node.input[2] in initializers:
            zero_tensor = initializers[node.input[2]]
            zeros = numpy_helper.to_array(zero_tensor)
            if len(zero_tensor.dims) == 1 and not zeros.any():
                zero_tensor.CopyFrom(numpy_helper.from_array(
                    np.array(0, dtype=np.int8), name=zero_tensor.name))
        changed += 1
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=100.0,
                        help="per-tensor scale is this percentile of the channel scales")
    args = parser.parse_args()
    model = onnx.load(str(args.onnx))
    changed = collapse(model, args.percentile)
    if changed == 0:
        raise SystemExit("no per-channel weight scales found")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.output))
    print(f"collapsed {changed} weight tensors into per-tensor scales")


if __name__ == "__main__":
    main()
