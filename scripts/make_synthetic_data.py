"""Generate a tiny, trivially separable fake dataset to exercise the pipeline (CI / smoke test).

"live" = smooth colour gradients + mild noise; "spoof" = the same plus a fine grid pattern
(a crude stand-in for moire / print texture). This says nothing about real performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

SPLIT_SIZES = {"train": 48, "dev": 24, "test": 24}  # images per class


def make_image(rng: np.random.Generator, spoof: bool, size: int) -> Image.Image:
    xs = np.linspace(0, 1, size)
    base = np.stack(
        [
            np.outer(xs, np.ones(size)) * rng.uniform(0.3, 1.0),
            np.outer(np.ones(size), xs) * rng.uniform(0.3, 1.0),
            np.full((size, size), rng.uniform(0.2, 0.8)),
        ],
        axis=-1,
    )
    image = base + rng.normal(0, 0.03, base.shape)
    if spoof:
        grid = (np.add.outer(np.arange(size), np.arange(size)) % 4 < 2).astype(float)
        image = image + 0.15 * grid[..., None]
    return Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dst", default="data/synthetic")
    parser.add_argument("--size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    for split, count in SPLIT_SIZES.items():
        for class_name in ("live", "spoof"):
            folder = Path(args.dst) / split / class_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                make_image(rng, class_name == "spoof", args.size).save(folder / f"{i:04d}.png")
    print(f"Synthetic dataset written to {args.dst}")


if __name__ == "__main__":
    main()
