from __future__ import annotations

from pathlib import Path

from .audio import load_audio


_AUDIO_SUFFIXES = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav"}


def _calibration_files(audio_dir: Path, limit: int) -> list[Path]:
    if limit < 1:
        raise ValueError("max_audio_files must be positive")
    files = sorted(
        path for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in _AUDIO_SUFFIXES
    )
    if not files:
        raise ValueError(f"No supported audio files found in {audio_dir}")
    count = min(limit, len(files))
    if count == len(files):
        return files
    if count == 1:
        return [files[0]]
    return [files[index * (len(files) - 1) // (count - 1)] for index in range(count)]


def quantize_encoder(
    model_id: str,
    input_onnx: Path,
    output_onnx: Path,
    audio_dir: Path,
    max_audio_files: int = 32,
    language: str = "English",
) -> dict:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
        from onnxruntime.quantization import (
            CalibrationDataReader,
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError(
            "Install encoder quantization dependencies with `uv sync --extra quantization`"
        ) from exc

    if not input_onnx.is_file():
        raise FileNotFoundError(f"FP16 encoder ONNX not found: {input_onnx}")
    if input_onnx.resolve() == output_onnx.resolve():
        raise ValueError("Input and output ONNX paths must be different")
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CUDA execution provider is required for calibration")

    model = onnx.load(str(input_onnx), load_external_data=False)
    input_shapes = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in model.graph.input
    }
    feature_shape = input_shapes.get("input_features", [])
    mask_shape = input_shapes.get("input_features_mask", [])
    if len(feature_shape) != 3 or feature_shape[:2] != [1, 128] or feature_shape[2] < 1:
        raise ValueError(f"Unsupported encoder input shape: {feature_shape}")
    window_frames = feature_shape[2]
    if mask_shape != [1, window_frames]:
        raise ValueError(f"Unsupported encoder mask shape: {mask_shape}")

    files = _calibration_files(audio_dir, max_audio_files)
    processor = AutoProcessor.from_pretrained(model_id)
    samples = []
    for audio_path in files:
        inputs = processor.apply_transcription_request(audio=load_audio(audio_path), language=language)
        features = inputs["input_features"].detach().to(device="cpu", dtype=torch.float32).numpy()
        mask = inputs["input_features_mask"].detach().to(device="cpu").numpy()
        frame_count = int(mask.sum())
        if features.ndim != 3 or features.shape[:2] != (1, 128) or mask.shape != (1, features.shape[-1]):
            raise ValueError(f"Unexpected audio feature shapes for {audio_path}")
        if frame_count < 1:
            raise ValueError(f"No valid audio frames in {audio_path}")
        for start in range(0, frame_count, window_frames):
            valid = min(frame_count - start, window_frames)
            available = min(valid, features.shape[-1] - start)
            window = np.zeros((1, 128, window_frames), dtype=np.float32)
            window_mask = np.zeros((1, window_frames), dtype=np.int32)
            if available > 0:
                window[..., :available] = features[..., start : start + available]
                window_mask[..., :available] = mask[..., start : start + available].astype(np.int32)
            samples.append({"input_features": window, "input_features_mask": window_mask})

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.rewind()

        def get_next(self):
            return next(self._samples, None)

        def rewind(self) -> None:
            self._samples = iter(samples)

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    quantize_static(
        str(input_onnx),
        str(output_onnx),
        Reader(),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=["Conv", "MatMul"],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        calibration_providers=["CUDAExecutionProvider"],
        use_external_data_format=True,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "QuantizeBias": False,
        },
    )
    onnx.checker.check_model(str(output_onnx))
    quantized = onnx.load(str(output_onnx), load_external_data=False)
    qdq_nodes = sum(node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in quantized.graph.node)
    return {
        "inputOnnx": str(input_onnx),
        "outputOnnx": str(output_onnx),
        "calibrationFiles": len(files),
        "calibrationAudioFiles": [str(path.relative_to(audio_dir)) for path in files],
        "calibrationWindows": len(samples),
        "quantFormat": "QDQ",
        "quantizedOperators": ["Conv", "MatMul"],
        "qdqNodes": qdq_nodes,
        "calibrationProvider": "CUDAExecutionProvider",
    }


def quantize_encoder_int4(input_onnx: Path, output_onnx: Path, block_size: int = 64) -> dict:
    """Quantize constant MatMul weights to block-wise INT4 for TensorRT weight-only GEMMs."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    if not input_onnx.is_file():
        raise FileNotFoundError(f"FP16 encoder ONNX not found: {input_onnx}")
    if input_onnx.resolve() == output_onnx.resolve():
        raise ValueError("Input and output ONNX paths must be different")
    if block_size not in (32, 64, 128):
        raise ValueError("TensorRT weight-only quantization supports 32, 64 or 128-value blocks")

    model = onnx.load(str(input_onnx), load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}
    quantized_by_weight = {}
    dequantize_nodes = {}
    added_initializers = []
    quantized_matmuls = 0
    for node in model.graph.node:
        if node.op_type != "MatMul" or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        weight = initializers[node.input[1]]
        if weight.name in quantized_by_weight:
            node.input[1] = quantized_by_weight[weight.name]
            continue
        if weight.data_type not in (TensorProto.FLOAT16, TensorProto.FLOAT):
            continue
        values = numpy_helper.to_array(weight).astype(np.float32, copy=False)
        if values.ndim != 2 or values.shape[0] % block_size:
            continue
        rows, columns = values.shape
        blocks = values.reshape(rows // block_size, block_size, columns)
        scales = np.max(np.abs(blocks), axis=1) / 7.0
        scales = np.where(scales > 0, scales, 1.0).astype(np.float16)
        quantized = np.rint(blocks / scales.astype(np.float32)[:, None, :])
        quantized = np.clip(quantized, -7, 7).astype(np.int8).reshape(rows, columns)

        quantized_name = f"{weight.name}.int4"
        scale_name = f"{weight.name}.int4_scale"
        output_name = f"{weight.name}.dequantized"
        added_initializers.append(helper.make_tensor(
            quantized_name, TensorProto.INT4, [rows, columns], quantized.reshape(-1).tolist()
        ))
        added_initializers.append(numpy_helper.from_array(scales, name=scale_name))
        dequantize_nodes[output_name] = helper.make_node(
            "DequantizeLinear",
            [quantized_name, scale_name],
            [output_name],
            name=f"{node.name}.int4_dequantize",
            axis=0,
            block_size=block_size,
        )
        node.input[1] = output_name
        quantized_by_weight[weight.name] = output_name
        quantized_matmuls += 1

    if not quantized_matmuls:
        raise RuntimeError("No eligible constant MatMul weights were found for INT4 quantization")

    kept = [item for item in model.graph.initializer if item.name not in quantized_by_weight]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept + added_initializers)
    nodes = list(model.graph.node)
    del model.graph.node[:]
    inserted_dequantizers = set()
    for node in nodes:
        if node.op_type == "MatMul" and len(node.input) > 1 and node.input[1] in dequantize_nodes:
            if node.input[1] not in inserted_dequantizers:
                model.graph.node.append(dequantize_nodes[node.input[1]])
                inserted_dequantizers.add(node.input[1])
        model.graph.node.append(node)

    default_opset = next((opset for opset in model.opset_import if not opset.domain), None)
    if default_opset is None:
        raise RuntimeError("ONNX model has no default-domain opset")
    if default_opset.version < 21:
        default_opset.version = 21
    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(output_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output_onnx.name + ".data",
        size_threshold=1024,
    )
    onnx.checker.check_model(str(output_onnx))
    return {
        "inputOnnx": str(input_onnx),
        "outputOnnx": str(output_onnx),
        "quantFormat": "INT4 weight-only QDQ",
        "blockSize": block_size,
        "quantizedMatMulWeights": quantized_matmuls,
        "activationAndConvPrecision": "FP16",
    }
