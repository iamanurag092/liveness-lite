"""Classify images as live or spoof with an exported ONNX model.

    python -m liveness.predict face1.jpg face2.png --model models/model_fp32.onnx

The model was trained on tightly cropped face images. By default each input is centre-cropped
to a square, then resized exactly as during training (256 px Lanczos, then the model size).
The decision threshold comes from ``model_card.json`` next to the model (or ``--threshold``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PREP_SIZE = 256  # the training images were pre-resized to this square size


def preprocess(
    path: str | Path, img_size: int, prep_size: int = PREP_SIZE, center_crop: bool = True
):
    """Returns float32 array (3, img_size, img_size) in [0, 1]."""
    with Image.open(path) as image:
        image = image.convert("RGB")
    if center_crop:
        width, height = image.size
        side = min(width, height)
        left, top = (width - side) // 2, (height - side) // 2
        image = image.crop((left, top, left + side, top + side))
    if prep_size:
        image = image.resize((prep_size, prep_size), Image.Resampling.LANCZOS)
    image = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
    return (np.asarray(image, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def load_session(model_path: str | Path):
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = session.get_inputs()[0].shape
    return session, int(shape[2])


def predict_files(
    session,
    files: list[str],
    img_size: int,
    threshold: float,
    center_crop: bool = True,
    prep_size: int = PREP_SIZE,
) -> list[dict]:
    input_name = session.get_inputs()[0].name
    results = []
    for file in files:
        batch = preprocess(file, img_size, prep_size, center_crop)[None]
        score = float(np.asarray(session.run(None, {input_name: batch})[0]).reshape(-1)[0])
        results.append(
            {
                "file": str(file),
                "score": round(score, 4),
                "verdict": "spoof" if score >= threshold else "live",
                "threshold": threshold,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model", default="models/model_fp32.onnx")
    parser.add_argument("--card", help="model_card.json (default: next to the model)")
    parser.add_argument("--threshold", type=float, help="override the stored threshold")
    parser.add_argument("--no-center-crop", action="store_true")
    parser.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args()

    card_path = Path(args.card) if args.card else Path(args.model).with_name("model_card.json")
    threshold = args.threshold
    if threshold is None:
        if card_path.is_file():
            threshold = float(json.loads(card_path.read_text())["threshold"])
        else:
            threshold = 0.5
            print(f"WARNING: {card_path} not found, using threshold 0.5", file=sys.stderr)

    session, img_size = load_session(args.model)
    results = predict_files(session, args.images, img_size, threshold, not args.no_center_crop)
    if args.json:
        print(json.dumps(results, indent=2))
        return
    for r in results:
        print(
            f"{r['file']}: {r['verdict'].upper()}  (spoof score {r['score']:.3f}, "
            f"threshold {r['threshold']:.3f})"
        )
    print("Note: the model expects a tightly cropped face; other inputs give unreliable scores.")


if __name__ == "__main__":
    main()
