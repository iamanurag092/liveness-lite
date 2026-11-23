"""Copy the exported models into demo/ so the static page can load them.

    python scripts/make_demo_assets.py --lite models/lite --accurate models/accurate

Each model directory must contain model_fp32.onnx and model_card.json (written by
`python -m liveness.export`). Optional, for LOCAL testing only:

    python scripts/make_demo_assets.py --lite ... --accurate ... --with-samples

copies a few test-split images into demo/samples/ together with the Python reference scores, so the
page can show that the browser reproduces them. demo/samples/ is git-ignored on purpose: the dataset
images are photos of real people and may not be redistributable, so never publish them.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

CARD_KEYS = [
    "backbone",
    "img_size",
    "params_m",
    "gmacs",
    "fp32_onnx_mb",
    "fp32_latency",
    "threshold",
    "dev_eer",
]


def copy_model(tier: str, source: Path, out: Path) -> dict:
    model, card_path = source / "model_fp32.onnx", source / "model_card.json"
    if not (model.is_file() and card_path.is_file()):
        sys.exit(f"{source} must contain model_fp32.onnx and model_card.json")
    card = json.loads(card_path.read_text())
    target = out / "models" / tier
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, target / "model.onnx")
    slim = {key: card[key] for key in CARD_KEYS if key in card}
    slim["tier"] = tier
    (target / "card.json").write_text(json.dumps(slim, indent=2))
    print(
        f"{tier}: {slim.get('backbone')}  {slim.get('fp32_onnx_mb')} MB  "
        f"threshold {slim.get('threshold')}"
    )
    return slim


def add_samples(models: dict[str, Path], data: Path, out: Path, per_class: int, seed: int) -> None:
    from liveness.predict import load_session, predict_files

    sessions = {tier: load_session(path / "model_fp32.onnx") for tier, path in models.items()}
    thresholds = {
        tier: json.loads((path / "model_card.json").read_text())["threshold"]
        for tier, path in models.items()
    }
    samples_dir = out / "samples"
    shutil.rmtree(samples_dir, ignore_errors=True)
    samples_dir.mkdir(parents=True)

    rng = random.Random(seed)
    records = []
    for label in ("live", "spoof"):
        files = sorted((data / "test" / label).glob("*.png"))
        if not files:
            sys.exit(f"No test images in {data / 'test' / label}")
        for index, source in enumerate(rng.sample(files, min(per_class, len(files)))):
            name = f"{label}_{index}.png"
            shutil.copy2(source, samples_dir / name)
            scores = {}
            for tier, (session, size) in sessions.items():
                result = predict_files(
                    session, [str(source)], size, thresholds[tier], center_crop=False
                )
                scores[tier] = result[0]["score"]
            records.append({"file": name, "label": label, "scores": scores})
    (samples_dir / "samples.json").write_text(json.dumps(records, indent=2))
    print(f"{len(records)} sample images written to {samples_dir} (git-ignored; do not publish)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--lite", help="directory of the Lite model export")
    parser.add_argument("--accurate", help="directory of the Accurate model export")
    parser.add_argument("--out", default="demo")
    parser.add_argument("--with-samples", action="store_true")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    models = {t: Path(p) for t, p in (("lite", args.lite), ("accurate", args.accurate)) if p}
    if not models:
        parser.error("Give at least one of --lite / --accurate")
    out = Path(args.out)
    for tier, source in models.items():
        copy_model(tier, source, out)
    if args.with_samples:
        add_samples(models, Path(args.data), out, args.per_class, args.seed)


if __name__ == "__main__":
    main()
