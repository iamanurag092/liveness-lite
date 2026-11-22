import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_lcc_fasd.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_lcc_fasd", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_group_key_variants():
    module = _load_module()
    assert module.group_key("YOUTUBE_id43_s0_105") == "id43"
    assert module.group_key("YOUTUBE_VX2363SMHL_LGL90_id47_s2_15") == "id47"
    assert module.group_key("AA5742_id154_s0_112") == "id154"
    assert module.group_key("spoof_1793") == "spoof:8"
    assert module.group_key("spoof_1999") == "spoof:9"
    assert module.group_key("spoof_2000") == "spoof:10"
    assert module.group_key("weird") == "weird"


def _make_fake_dataset(folder: Path) -> None:
    folder.mkdir(parents=True)
    lists = {
        "CLIENT_TRAIN.txt": [],
        "CLIENT_TEST.txt": [],
        "IMPOSTER_TRAIN.txt": [],
        "IMPOSTER_TEST.txt": [],
    }

    def add(list_name: str, name: str) -> None:
        Image.new("RGB", (40, 40), (len(name) * 3 % 255, 50, 90)).save(folder / name)
        lists[list_name].append(f"/kaggle/input/lcc-fasd/LCC_FASD/{name}")

    for subject in range(1, 11):  # train subjects appear as live and as spoof
        for frame in range(4):
            add("CLIENT_TRAIN.txt", f"YOUTUBE_id{subject}_s0_{frame * 15}.png")
        for frame in range(6):
            add("IMPOSTER_TRAIN.txt", f"YOUTUBE_DEV_PHONE_id{subject}_s2_{frame * 15}.png")
    for subject in (11, 12):
        for frame in range(5):
            add("CLIENT_TEST.txt", f"YOUTUBE_id{subject}_s1_{frame * 15}.png")
    for index in range(1000, 1020):
        add("IMPOSTER_TEST.txt", f"spoof_{index}.png")
    for list_name, lines in lists.items():
        (folder / list_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_prepare_end_to_end(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    _make_fake_dataset(src)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--resize",
            "32",
            "--dev-fraction",
            "0.3",
            "--min-per-class",
            "1",
            "--preview-count",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    summary = json.loads((dst / "dataset_summary.json").read_text())
    counts = summary["counts"]
    assert sum(counts.values()) == 10 * 4 + 10 * 6 + 2 * 5 + 20
    assert counts["test/live"] == 10 and counts["test/spoof"] == 20
    assert counts["train/live"] + counts["dev/live"] == 40
    assert counts["dev/live"] > 0 and counts["dev/spoof"] > 0

    manifest = (dst / "manifest.csv").read_text().splitlines()[1:]
    by_split = {"train": set(), "dev": set()}
    for line in manifest:
        _, split, _, group, _ = line.split(",")
        if split in by_split:
            by_split[split].add(group)
    assert not by_split["train"] & by_split["dev"]  # no subject in both

    with Image.open(next((dst / "train" / "live").glob("*.png"))) as image:
        assert image.size == (32, 32)
    assert (dst / "_preview" / "train_live.png").exists()
    assert (dst / "_preview" / "test_spoof.png").exists()


def test_rerun_moves_files_instead_of_duplicating(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    _make_fake_dataset(src)
    base = [
        sys.executable,
        str(SCRIPT),
        "--src",
        str(src),
        "--dst",
        str(dst),
        "--resize",
        "32",
        "--dev-fraction",
        "0.3",
        "--min-per-class",
        "1",
        "--preview-count",
        "4",
    ]
    subprocess.run([*base, "--seed", "1"], check=True, capture_output=True)
    second = subprocess.run([*base, "--seed", "2"], check=True, capture_output=True, text=True)

    files = [p for p in dst.rglob("*.png") if "_preview" not in p.parts]
    assert len(files) == 10 * 4 + 10 * 6 + 2 * 5 + 20
    assert len({p.name for p in files}) == len(files)  # no file lives in two splits
    assert "files copied: 0" in second.stdout


def test_dev_gets_both_classes_even_when_live_is_rare(tmp_path):
    src, dst = tmp_path / "raw", tmp_path / "out"
    _make_fake_dataset(src)
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--src",
            str(src),
            "--dst",
            str(dst),
            "--resize",
            "32",
            "--dev-fraction",
            "0.25",
            "--min-per-class",
            "1",
            "--preview-count",
            "4",
        ],
        check=True,
        capture_output=True,
    )
    counts = json.loads((dst / "dataset_summary.json").read_text())["counts"]
    assert counts["dev/live"] >= 0.25 * 40 - 4  # at least about a quarter of train live
