"""Turn a raw anti-spoofing dataset into the layout the pipeline expects.

    <dst>/{train,dev,test}/{live,spoof}/*.jpg   (+ <dst>/manifest.csv for auditing)

Class is inferred from folder names (token match, deepest folder wins). Two modes:

1) Dataset ships with its own splits (preferred, e.g. train/dev/eval folders):
     python scripts/prepare_data.py --src RAW --dst data/processed \\
         --split-map training=train,development=dev,evaluation=test

2) Flat dataset -> group-aware random split. --group-regex must capture the subject/identity
   id (group 1) so no person appears in more than one split (otherwise results are inflated):
     python scripts/prepare_data.py --src RAW --dst data/processed \\
         --auto-split --group-regex "(?:^|/)(\\d+)_"

Files are copied (symlinks need admin rights on Windows); pass --link to symlink instead.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def tokens(part: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", part.lower()) if t}


def infer_class(rel: Path, live_names: set[str], spoof_names: set[str]) -> str | None:
    for part in reversed(rel.parts[:-1]):  # deepest folder first
        t = tokens(part)
        if t & spoof_names:
            return "spoof"
        if t & live_names:
            return "live"
    return None


def infer_split(rel: Path, split_map: dict[str, str]) -> str | None:
    for part in rel.parts[:-1]:  # shallowest folder first
        t = tokens(part)
        for key, split in split_map.items():
            if key in t:
                return split
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", default="data/processed")
    parser.add_argument("--live-names", default="live,real,genuine,bonafide")
    parser.add_argument("--spoof-names", default="spoof,fake,attack,imposter")
    parser.add_argument("--split-map", help="e.g. training=train,development=dev,evaluation=test")
    parser.add_argument("--auto-split", action="store_true")
    parser.add_argument(
        "--group-regex", help="regex; group 1 = subject id (used with --auto-split)"
    )
    parser.add_argument("--ratios", default="0.7,0.15,0.15", help="train,dev,test for auto-split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link", action="store_true")
    args = parser.parse_args()

    if bool(args.split_map) == args.auto_split:
        parser.error("Use exactly one of --split-map or --auto-split")
    if args.auto_split and not args.group_regex:
        parser.error("--auto-split requires --group-regex (subject-disjoint splits)")

    live_names = {t.strip() for t in args.live_names.split(",")}
    spoof_names = {t.strip() for t in args.spoof_names.split(",")}
    split_map = dict(kv.split("=") for kv in args.split_map.split(",")) if args.split_map else {}

    src = Path(args.src)
    rows, skipped = [], 0
    for path in sorted(src.rglob("*")):
        if path.suffix.lower() not in IMG_EXTS:
            continue
        rel = path.relative_to(src)
        label = infer_class(rel, live_names, spoof_names)
        if label is None:
            skipped += 1
            continue
        group = None
        if args.group_regex:
            match = re.search(args.group_regex, rel.as_posix())
            group = match.group(1) if match and match.groups() else rel.parent.as_posix()
        split = infer_split(rel, split_map) if split_map else None
        if split_map and split is None:
            skipped += 1
            continue
        rows.append(
            {"src": path, "rel": rel.as_posix(), "label": label, "group": group, "split": split}
        )

    if not rows:
        raise SystemExit("No labelled images found; check --live-names/--spoof-names/--split-map")

    if args.auto_split:
        ratios = [float(r) for r in args.ratios.split(",")]
        groups = sorted({r["group"] for r in rows})
        random.Random(args.seed).shuffle(groups)
        n_train = int(len(groups) * ratios[0])
        n_dev = int(len(groups) * ratios[1])
        assignment = {}
        for i, g in enumerate(groups):
            assignment[g] = "train" if i < n_train else "dev" if i < n_train + n_dev else "test"
        for r in rows:
            r["split"] = assignment[r["group"]]
        if len(groups) == len(rows):
            print("WARNING: every image is its own group -> likely identity leakage across splits")

    dst = Path(args.dst)
    for r in rows:
        target_dir = dst / r["split"] / r["label"]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / ("__".join(Path(r["rel"]).parts))
        if args.link:
            if not target.exists():
                target.symlink_to(r["src"].resolve())
        else:
            shutil.copy2(r["src"], target)
        r["dst"] = target.relative_to(dst).as_posix()

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split", "label", "group"])
        for r in rows:
            writer.writerow([r["dst"], r["split"], r["label"], r["group"] or ""])

    counts = defaultdict(int)
    split_groups = defaultdict(set)
    for r in rows:
        counts[(r["split"], r["label"])] += 1
        if r["group"] is not None:
            split_groups[r["split"]].add(r["group"])
    print(f"Skipped (unlabelled / unmapped): {skipped}")
    for (split, label), n in sorted(counts.items()):
        print(f"  {split:5s} {label:5s} {n}")
    if split_groups:
        seen = defaultdict(list)
        for split, gs in split_groups.items():
            for g in gs:
                seen[g].append(split)
        leaked = [g for g, s in seen.items() if len(s) > 1]
        print(f"Subjects shared across splits: {len(leaked)} (must be 0)")
        if leaked:
            raise SystemExit("Identity leakage detected; fix --group-regex or the splits")


if __name__ == "__main__":
    main()
