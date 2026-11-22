"""Try several INT8 quantisation recipes on the FP32 ONNX model and judge them on DEV only.

    python scripts/quant_sweep.py --config configs/default.yaml --onnx models/model_fp32.onnx

Static INT8 with default settings collapsed MobileNetV3 accuracy in the first attempt, so this
compares calibration methods and "quantise only the convolutions" variants. Nothing here touches
the test split. Rule: a recipe is acceptable if its dev EER is at most ``--max-eer-increase``
above FP32 and its dev AUC at most 0.005 below. The smallest acceptable file is recommended.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from liveness.data import make_loader
from liveness.export import benchmark_onnx, quantize_int8
from liveness.metrics import eer_and_threshold, roc_auc
from liveness.utils import load_config

RECIPES = [
    ("minmax_all", {"method": "minmax", "conv_only": False, "reduce_range": False}),
    ("minmax_conv", {"method": "minmax", "conv_only": True, "reduce_range": False}),
    ("percentile_conv", {"method": "percentile", "conv_only": True, "reduce_range": False}),
    ("entropy_conv", {"method": "entropy", "conv_only": True, "reduce_range": False}),
    ("percentile_conv_rr", {"method": "percentile", "conv_only": True, "reduce_range": True}),
]


class CachedReader:
    def __init__(self, images: torch.Tensor, batch: int, max_batches: int):
        self.images, self.batch, self.max_batches, self.index = images, batch, max_batches, 0

    def get_next(self):
        start = self.index * self.batch
        if self.index >= self.max_batches or start >= len(self.images):
            return None
        self.index += 1
        chunk = self.images[start : start + self.batch].float() / 255.0
        return {"image": chunk.numpy()}


def score(session: ort.InferenceSession, images: torch.Tensor, batch: int = 64) -> np.ndarray:
    out = []
    for start in range(0, len(images), batch):
        x = (images[start : start + batch].float() / 255.0).numpy()
        out.append(np.asarray(session.run(None, {"image": x})[0]).reshape(-1))
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--onnx", default="models/model_fp32.onnx")
    parser.add_argument("--out-dir", default="models/quant_sweep")
    parser.add_argument("--max-eer-increase", type=float, default=0.01)
    parser.add_argument("--calibration-batches", type=int, default=16)
    args = parser.parse_args()

    cfg = load_config(args.config)
    fp32 = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = int(ort.InferenceSession(str(fp32)).get_inputs()[0].shape[2])
    cfg["data"]["img_size"] = img_size

    print("Caching the dev split in memory (uint8)...")
    images, labels = [], []
    for x, y in make_loader(cfg, "dev", train=False):
        images.append((x * 255).round().to(torch.uint8))
        labels.append(y.int())
    images, labels = torch.cat(images), torch.cat(labels).numpy()
    print(f"dev images: {len(labels)}")

    rows = []

    def evaluate(name: str, path: Path, options: dict) -> dict:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        scores = score(session, images)
        eer, threshold = eer_and_threshold(labels, scores)
        row = {
            "name": name,
            "options": options,
            "size_mb": round(path.stat().st_size / 1e6, 2),
            "p50_ms": benchmark_onnx(path, img_size)["p50_ms"],
            "dev_eer": round(eer, 4),
            "dev_auc": round(roc_auc(labels, scores), 4),
            "threshold": round(threshold, 4),
        }
        rows.append(row)
        print(
            f"{name:20s} size={row['size_mb']:5.2f}MB p50={row['p50_ms']:5.2f}ms "
            f"dev_EER={row['dev_eer']:.4f} dev_AUC={row['dev_auc']:.4f}"
        )
        return row

    baseline = evaluate("fp32", fp32, {})
    for name, options in RECIPES:
        target = out_dir / f"{name}.onnx"
        try:
            reader = CachedReader(images, 128, args.calibration_batches)
            quantize_int8(fp32, target, reader=reader, **options)
            evaluate(name, target, options)
        except Exception as error:  # noqa: BLE001 - one broken recipe must not stop the sweep
            print(f"{name:20s} FAILED: {type(error).__name__}: {str(error)[:100]}")
            rows.append({"name": name, "options": options, "error": str(error)[:200]})

    ok = [
        r
        for r in rows[1:]
        if "error" not in r
        and r["dev_eer"] <= baseline["dev_eer"] + args.max_eer_increase
        and r["dev_auc"] >= baseline["dev_auc"] - 0.005
    ]
    best = min(ok, key=lambda r: r["size_mb"]) if ok else None
    (out_dir / "results.json").write_text(json.dumps({"rows": rows, "recommended": best}, indent=2))
    if best:
        o = best["options"]
        flags = f"--quant-method {o['method']}" + (" --conv-only" if o["conv_only"] else "")
        flags += " --reduce-range" if o["reduce_range"] else ""
        print(f"\nRECOMMENDED: {best['name']}  ->  re-export with: {flags}")
    else:
        print("\nNo recipe met the dev criteria. Ship the FP32 ONNX model and report this result.")
    sys.exit(0)


if __name__ == "__main__":
    main()
