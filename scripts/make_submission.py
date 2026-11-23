"""Build submission.zip: source, docs, report, results and the models, small enough to email.

    python scripts/make_submission.py                      # Lite model only (about 7 MB)
    python scripts/make_submission.py --include-accurate   # adds the 16.8 MB Accurate model

Left out on purpose: checkpoints (.pt), the dataset, demo/samples (photos of real people), the
virtual environment, caches and the duplicate model copies in demo/models.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

TOP_FILES = [
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    ".gitignore",
    ".gitattributes",
    "report.pdf",
]
TREES = ["src", "scripts", "configs", "tests", "docs", "demo", ".github"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "samples", "models"}
SKIP_SUFFIXES = {".pyc", ".pt", ".egg-info"}


def collect(root: Path, include_accurate: bool) -> list[Path]:
    files: list[Path] = []
    for name in TOP_FILES:
        if (root / name).is_file():
            files.append(root / name)
    for tree in TREES:
        for path in sorted((root / tree).rglob("*")):
            rel = path.relative_to(root)
            parts = set(rel.parts)
            # demo/models is regenerated from models/, demo/samples is private (real faces)
            if not path.is_file() or parts & SKIP_PARTS or path.suffix in SKIP_SUFFIXES:
                continue
            files.append(path)
    for tier in ("lite", "accurate"):
        folder = root / "models" / tier
        for name in ("model_card.json", "model_fp32.onnx"):
            if name == "model_fp32.onnx" and tier == "accurate" and not include_accurate:
                continue
            if (folder / name).is_file():
                files.append(folder / name)
    runs = root / "runs"
    if runs.is_dir():
        for run in sorted(p for p in runs.iterdir() if p.is_dir() and p.name != "smoke"):
            for path in sorted(run.iterdir()):
                keep = path.suffix in {".json", ".png", ".yaml"}
                if path.is_file() and keep and path.suffix not in SKIP_SUFFIXES:
                    files.append(path)
    return files


def build(root: Path, out: Path, include_accurate: bool) -> tuple[int, int]:
    files = collect(root, include_accurate)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("liveness-lite") / path.relative_to(root))
    return len(files), out.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--out", default="submission.zip")
    parser.add_argument("--include-accurate", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    out = Path(args.out)
    count, size = build(root, out, args.include_accurate)
    print(f"{out}: {count} files, {size / 1e6:.1f} MB")
    if size > 10e6:
        print(
            "WARNING: over 10 MB; many mail systems reject large attachments. Check the size limit."
        )
    if not (root / "report.pdf").is_file():
        print("WARNING: report.pdf is missing (python scripts/build_report.py)")
    with zipfile.ZipFile(out) as archive:
        for name in archive.namelist():
            if name.endswith((".onnx", ".pdf")):
                print("  includes", name)


if __name__ == "__main__":
    main()
