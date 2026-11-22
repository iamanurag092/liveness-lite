"""Export to ONNX, quantise to INT8, verify parity, benchmark, and write a model card.

The exported graph takes float32 RGB in [0, 1] of shape (N, 3, H, W) and returns the spoof
probability, shape (N,). Normalisation and the sigmoid are inside the graph.
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from liveness.data import make_loader
from liveness.evaluate import onnx_scores
from liveness.metrics import eer_and_threshold
from liveness.model import count_flops, count_params, load_model
from liveness.utils import load_config, save_json


class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))


def export_onnx(model: nn.Module, img_size: int, path: Path, opset: int = 17) -> None:
    wrapper = ExportWrapper(copy.deepcopy(model).cpu().eval())
    dummy = torch.rand(1, 3, img_size, img_size)
    kwargs = {
        "input_names": ["image"],
        "output_names": ["spoof_prob"],
        "dynamic_axes": {"image": {0: "batch"}, "spoof_prob": {0: "batch"}},
        "opset_version": opset,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:  # newer torch defaults to the dynamo exporter; the legacy one is the stable choice here
        torch.onnx.export(wrapper, dummy, str(path), dynamo=False, **kwargs)
    except TypeError:  # older torch has no `dynamo` argument
        torch.onnx.export(wrapper, dummy, str(path), **kwargs)


class DevCalibrationReader:
    """Feeds dev-split batches (never test) to the ONNX Runtime calibrator."""

    def __init__(self, loader, input_name: str, max_batches: int):
        self.input_name = input_name
        self.batches = iter(loader)
        self.remaining = max_batches

    def get_next(self):
        if self.remaining <= 0:
            return None
        batch = next(self.batches, None)
        if batch is None:
            return None
        self.remaining -= 1
        return {self.input_name: batch[0].numpy().astype(np.float32)}


def quantize_int8(
    fp32_path: Path,
    int8_path: Path,
    cfg: dict | None = None,
    max_batches: int = 20,
    reader=None,
    method: str = "minmax",
    conv_only: bool = False,
    reduce_range: bool = False,
) -> None:
    """Static INT8 (QDQ) quantisation.

    method: minmax | percentile | entropy (activation range calibration).
    conv_only: quantise only Conv/Gemm/MatMul; leave hard-swish, squeeze-excite products and
    additions in float (often needed for MobileNetV3).
    reduce_range: 7-bit weights/activations, avoids int8 saturation on CPUs without VNNI.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    methods = {
        "minmax": CalibrationMethod.MinMax,
        "percentile": CalibrationMethod.Percentile,
        "entropy": CalibrationMethod.Entropy,
    }
    if method not in methods:
        raise ValueError(f"Unknown calibration method {method!r}; choose from {sorted(methods)}")
    if reader is None:
        reader = DevCalibrationReader(make_loader(cfg, "dev", train=False), "image", max_batches)

    class _Reader(CalibrationDataReader):
        def get_next(self):
            return reader.get_next()

    quantize_static(
        str(fp32_path),
        str(int8_path),
        _Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=reduce_range,
        calibrate_method=methods[method],
        op_types_to_quantize=["Conv", "Gemm", "MatMul"] if conv_only else None,
    )


def verify_parity(model: nn.Module, onnx_path: Path, img_size: int, tol: float = 1e-3) -> float:
    import onnxruntime as ort

    model = copy.deepcopy(model).cpu().eval()
    x = torch.rand(4, 3, img_size, img_size)
    with torch.no_grad():
        expected = torch.sigmoid(model(x)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": x.numpy()})[0].reshape(-1)
    max_diff = float(np.max(np.abs(expected - got)))
    if max_diff > tol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {max_diff:.2e} (tol {tol})")
    return max_diff


def benchmark_onnx(path: Path, img_size: int, runs: int = 100, warmup: int = 10) -> dict:
    """Single-thread CPU latency, batch size 1."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"image": x})
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run(None, {"image": x})
        times.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": round(float(np.percentile(times, 50)), 2),
        "p95_ms": round(float(np.percentile(times, 95)), 2),
        "mean_ms": round(float(np.mean(times)), 2),
    }


def check_budget(card: dict, budget: dict) -> list[str]:
    checks = [
        ("params_m", "max_params_m", card["params_m"]),
        ("fp32_onnx_mb", "max_fp32_mb", card["fp32_onnx_mb"]),
        ("int8_onnx_mb", "max_int8_mb", card.get("int8_onnx_mb")),
        ("fp32_p50_ms", "max_p50_ms", card["fp32_latency"]["p50_ms"]),
    ]
    violations = []
    for name, limit_key, value in checks:
        if value is not None and limit_key in budget and value > budget[limit_key]:
            violations.append(f"{name}={value} exceeds {limit_key}={budget[limit_key]}")
    return violations


def dev_threshold(onnx_path: Path, cfg: dict) -> tuple[float, float]:
    """Dev-EER operating point of an exported model: (threshold, dev EER)."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    labels, scores = onnx_scores(session, make_loader(cfg, "dev", train=False))
    eer, threshold = eer_and_threshold(labels, scores)
    return threshold, eer


def run(
    cfg: dict,
    checkpoint: str,
    out_dir: str,
    quantize: bool = True,
    quant_options: dict | None = None,
) -> dict:
    out = Path(out_dir)
    model, ckpt = load_model(checkpoint, "cpu")
    img_size = int(ckpt["img_size"])
    cfg = {**cfg, "data": {**cfg["data"], "img_size": img_size}}  # match the trained size

    fp32_path = out / "model_fp32.onnx"
    export_onnx(model, img_size, fp32_path)
    parity = verify_parity(model, fp32_path, img_size)
    threshold, dev_eer = dev_threshold(fp32_path, cfg)

    card: dict = {
        "backbone": ckpt["model_cfg"]["name"],
        "img_size": img_size,
        "params_m": round(count_params(model) / 1e6, 3),
        "gmacs": round(count_flops(model, img_size) / 2e9, 4),
        "fp32_onnx_mb": round(fp32_path.stat().st_size / 1e6, 2),
        "fp32_latency": benchmark_onnx(fp32_path, img_size),
        "fp32_max_abs_diff_vs_torch": parity,
        "threshold": round(threshold, 4),
        "dev_eer": round(dev_eer, 4),
        "input": "float32 RGB in [0,1], shape (N,3,H,W)",
        "output": "spoof probability, shape (N,); label 1 = spoof; spoof if score >= threshold",
    }
    if quantize:
        int8_path = out / "model_int8.onnx"
        quantize_int8(fp32_path, int8_path, cfg, **(quant_options or {}))
        card["int8_onnx_mb"] = round(int8_path.stat().st_size / 1e6, 2)
        card["int8_latency"] = benchmark_onnx(int8_path, img_size)
        card["int8_options"] = quant_options or {}
    save_json(card, out / "model_card.json")
    return card


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument(
        "--quant-method", default="minmax", choices=["minmax", "percentile", "entropy"]
    )
    parser.add_argument("--conv-only", action="store_true", help="quantise only Conv/Gemm/MatMul")
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--enforce-budget", action="store_true", help="exit 1 if budget exceeded")
    parser.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    options = {
        "method": args.quant_method,
        "conv_only": args.conv_only,
        "reduce_range": args.reduce_range,
    }
    card = run(
        cfg, args.checkpoint, args.out_dir, quantize=not args.no_quant, quant_options=options
    )
    for key, value in card.items():
        print(f"{key}: {value}")

    violations = check_budget(card, cfg.get("budget", {}))
    for violation in violations:
        print(f"BUDGET VIOLATION: {violation}")
    if violations and args.enforce_budget:
        sys.exit(1)


if __name__ == "__main__":
    main()
