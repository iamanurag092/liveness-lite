"""Prepare the Kaggle copy of LCC-FASD (aleksandrpikul222/lcc-fasd) for the pipeline.

The download is one flat folder of PNGs plus four official list files:

    CLIENT_TRAIN.txt   CLIENT_TEST.txt     -> live (genuine "clients")
    IMPOSTER_TRAIN.txt IMPOSTER_TEST.txt   -> spoof (attacks, "imposters")

There is no dev split, so one is carved out of the official train list. Splitting is done by
group (the ``idNN`` token in the file name, a subject/video id) so near-identical frames of the
same person never land in both train and dev. The official test list is left untouched.

Output: <dst>/{train,dev,test}/{live,spoof}/*.png, manifest.csv, dataset_summary.json and
<dst>/_preview/*.png (contact sheets: LOOK AT THEM to confirm live/spoof are not swapped).

    python scripts/prepare_lcc_fasd.py --src <...>/LCC_FASD --dst data/processed --resize 256
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional

    def tqdm(iterable, **_kwargs):
        return iterable


LISTS = {  # list file -> (official split, class)
    "CLIENT_TRAIN.txt": ("train", "live"),
    "CLIENT_TEST.txt": ("test", "live"),
    "IMPOSTER_TRAIN.txt": ("train", "spoof"),
    "IMPOSTER_TEST.txt": ("test", "spoof"),
}
ID_PATTERN = re.compile(r"(?:^|_)(id\d+)(?:_|$)")
INDEX_PATTERN = re.compile(r"^(.*?)_?(\d+)$")


def group_key(stem: str) -> str:
    """Split group: the ``idNN`` token if present, else a block of 200 consecutive indices."""
    match = ID_PATTERN.search(stem)
    if match:
        return match.group(1)
    match = INDEX_PATTERN.match(stem)
    if match:
        return f"{match.group(1) or 'file'}:{int(match.group(2)) // 200}"
    return stem


def read_list(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            names.append(line.replace("\\", "/").rsplit("/", 1)[-1])
    return names


def carve_dev(train_rows: list[dict], fraction: float, seed: int) -> set[str]:
    """Pick whole groups for dev until live AND spoof each reach ``fraction`` of train."""
    totals = Counter(r["label"] for r in train_rows)
    per_group: dict[str, Counter] = defaultdict(Counter)
    for row in train_rows:
        per_group[row["group"]][row["label"]] += 1
    groups = sorted(per_group)
    random.Random(seed).shuffle(groups)

    taken: Counter = Counter()
    chosen: set[str] = set()
    for group in groups:
        needs = [label for label in totals if taken[label] < fraction * totals[label]]
        if not needs:
            break
        if any(per_group[group][label] for label in needs):  # skip groups that add nothing needed
            chosen.add(group)
            taken.update(per_group[group])
    return chosen


def make_preview(dst: Path, split: str, label: str, count: int, seed: int) -> Path:
    thumb, cols = 128, 4
    files = sorted((dst / split / label).glob("*.png"))
    random.Random(seed).shuffle(files)
    files = files[:count]
    rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, max(rows, 1) * thumb), "white")
    for i, path in enumerate(files):
        with Image.open(path) as image:
            tile = image.convert("RGB").resize((thumb, thumb))
        sheet.paste(tile, ((i % cols) * thumb, (i // cols) * thumb))
    out = dst / "_preview" / f"{split}_{label}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--src", required=True, help="folder with the PNGs and the four .txt lists")
    parser.add_argument("--dst", default="data/processed")
    parser.add_argument("--dev-fraction", type=float, default=0.15, help="share of train for dev")
    parser.add_argument(
        "--resize", type=int, default=0, help="resize to NxN while copying (0 = keep)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-per-class", type=int, default=50)
    parser.add_argument("--preview-count", type=int, default=12)
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    rows = []
    for list_name, (split, label) in LISTS.items():
        list_path = src / list_name
        if not list_path.is_file():
            sys.exit(f"Missing list file: {list_path}")
        for name in read_list(list_path):
            rows.append({"name": name, "split": split, "label": label, "list": list_name})

    repeated = [n for n, c in Counter(r["name"] for r in rows).items() if c > 1]
    if repeated:
        sys.exit(f"{len(repeated)} files appear in more than one list, e.g. {repeated[:3]}")
    missing = [r["name"] for r in rows if not (src / r["name"]).is_file()]
    if missing:
        sys.exit(f"{len(missing)} listed files not found in {src}, e.g. {missing[:3]}")
    for row in rows:
        row["group"] = group_key(Path(row["name"]).stem)

    # Carve dev out of train, whole groups at a time, until BOTH classes reach the target share.
    train_rows = [r for r in rows if r["split"] == "train"]
    dev_groups = carve_dev(train_rows, args.dev_fraction, args.seed)
    for row in train_rows:
        if row["group"] in dev_groups:
            row["split"] = "dev"

    counts = Counter((r["split"], r["label"]) for r in rows)
    for split in ("train", "dev", "test"):
        for label in ("live", "spoof"):
            if counts[(split, label)] < args.min_per_class:
                sys.exit(
                    f"{split}/{label} has only {counts[(split, label)]} images; "
                    "try another --seed or --dev-fraction"
                )

    def id_groups(split: str) -> set[str]:
        return {r["group"] for r in rows if r["split"] == split and ":" not in r["group"]}

    shared_with_test = (id_groups("train") | id_groups("dev")) & id_groups("test")

    def block_groups(split: str) -> set[str]:
        return {r["group"] for r in rows if r["split"] == split and ":" in r["group"]}

    shared_blocks = (block_groups("train") | block_groups("dev")) & block_groups("test")
    shared_train_dev = id_groups("train") & id_groups("dev")
    if shared_train_dev:
        sys.exit("Internal error: subject ids shared between train and dev")

    moved = copied = 0
    failed: list[str] = []
    for row in tqdm(rows, desc="writing"):
        out_dir = dst / row["split"] / row["label"]
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / row["name"]
        if target.exists():
            continue
        previous = next(
            (
                dst / other / row["label"] / row["name"]
                for other in ("train", "dev", "test")
                if other != row["split"] and (dst / other / row["label"] / row["name"]).exists()
            ),
            None,
        )
        if previous is not None:  # re-run with a different split: move, do not re-copy
            previous.replace(target)
            moved += 1
            continue
        try:
            if args.resize:
                with Image.open(src / row["name"]) as image:
                    resized = image.convert("RGB").resize(
                        (args.resize, args.resize), Image.Resampling.LANCZOS
                    )
                resized.save(target)
            else:
                shutil.copy2(src / row["name"], target)
        except OSError:  # unreadable or truncated source image: skip it, report it below
            target.unlink(missing_ok=True)
            failed.append(row["name"])
            continue
        copied += 1
    print(f"files copied: {copied}, moved between splits: {moved}")
    if failed:
        bad = set(failed)
        rows = [r for r in rows if r["name"] not in bad]
        counts = Counter((r["split"], r["label"]) for r in rows)
        (dst / "skipped_unreadable.txt").write_text("\n".join(sorted(failed)) + "\n")
        print(
            f"WARNING: skipped {len(failed)} unreadable source images (see skipped_unreadable.txt)"
        )

    with open(dst / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split", "label", "group", "official_list"])
        for row in rows:
            rel = f"{row['split']}/{row['label']}/{row['name']}"
            writer.writerow([rel, row["split"], row["label"], row["group"], row["list"]])

    summary = {
        "source": str(src),
        "resize": args.resize,
        "seed": args.seed,
        "skipped_unreadable": len(failed),
        "dev_fraction": args.dev_fraction,
        "counts": {
            f"{s}/{label}": counts[(s, label)]
            for s in ("train", "dev", "test")
            for label in ("live", "spoof")
        },
        "id_groups": {s: len(id_groups(s)) for s in ("train", "dev", "test")},
        "id_groups_shared_between_train_dev_and_test": len(shared_with_test),
        "files_without_id_in_name": sum(1 for r in rows if ":" in r["group"]),
        "index_blocks_shared_between_train_dev_and_test": sorted(shared_blocks),
    }
    (dst / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    previews = [
        make_preview(dst, split, label, args.preview_count, args.seed)
        for split in ("train", "test")
        for label in ("live", "spoof")
    ]
    print(json.dumps(summary, indent=2))
    print("\nPreview sheets (check that live looks genuine and spoof looks like an attack):")
    for path in previews:
        print(f"  {path}")
    if shared_blocks:
        print(
            f"NOTE: {len(shared_blocks)} index blocks of id-less files ({sorted(shared_blocks)}) "
            "occur in both train/dev and test: neighbouring frames may straddle the official split."
        )
    if shared_with_test:
        print(
            f"\nNOTE: {len(shared_with_test)} subject ids occur in both train/dev and the official "
            "test lists. The official test split is used as given; mention this in the report."
        )


if __name__ == "__main__":
    main()
