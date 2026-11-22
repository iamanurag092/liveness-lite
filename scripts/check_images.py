"""Scan a prepared dataset for unreadable images; optionally delete them.

    python scripts/check_images.py --root data/processed            # report only
    python scripts/check_images.py --root data/processed --delete   # remove the bad files

After deleting, re-run scripts/prepare_lcc_fasd.py with the same arguments: it recreates exactly the
missing files from the original download (and skips any whose source is also broken).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def check(path: Path, expected_size: int | None) -> str | None:
    """Return a problem description, or None if the file is fine."""
    try:
        size_bytes = path.stat().st_size
        if size_bytes == 0:
            return "empty file (0 bytes)"
        with Image.open(path) as image:
            image.load()  # full decode catches truncated files that verify() would miss
            if expected_size and image.size != (expected_size, expected_size):
                return f"unexpected size {image.size}"
    except Exception as error:  # noqa: BLE001 - any failure means the file is unusable
        return f"{type(error).__name__}: {error}"[:120]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--root", default="data/processed")
    parser.add_argument("--size", type=int, default=0, help="also flag images that are not NxN")
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and "_preview" not in p.parts
    )
    if not files:
        sys.exit(f"No images under {root}")

    bad: list[tuple[Path, str]] = []
    for index, path in enumerate(files, 1):
        problem = check(path, args.size or None)
        if problem:
            bad.append((path, problem))
        if index % 5000 == 0:
            print(f"  scanned {index}/{len(files)}")

    print(f"Scanned {len(files)} images, {len(bad)} unreadable")
    for path, problem in bad[:25]:
        print(f"  {path.relative_to(root)}  ->  {problem}")
    if len(bad) > 25:
        print(f"  ... and {len(bad) - 25} more")

    if bad and args.delete:
        for path, _ in bad:
            path.unlink()
        print(f"Deleted {len(bad)} files. Now re-run the prepare script to recreate them.")
    elif bad:
        print("Run again with --delete to remove them.")
        sys.exit(1)


if __name__ == "__main__":
    main()
