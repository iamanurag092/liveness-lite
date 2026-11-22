import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "check_images.py"
PREPARE = ROOT / "scripts" / "prepare_lcc_fasd.py"


def _run(*args, check=True):
    return subprocess.run(
        [sys.executable, *map(str, args)], capture_output=True, text=True, check=check
    )


def test_check_images_finds_and_deletes_bad_files(tmp_path):
    folder = tmp_path / "train" / "live"
    folder.mkdir(parents=True)
    Image.new("RGB", (16, 16), (10, 20, 30)).save(folder / "good.png")
    (folder / "empty.png").write_bytes(b"")
    good_bytes = (folder / "good.png").read_bytes()
    (folder / "truncated.png").write_bytes(good_bytes[: len(good_bytes) // 2])

    report = _run(CHECK, "--root", tmp_path, check=False)
    assert report.returncode == 1  # bad files found, nothing deleted yet
    assert "empty.png" in report.stdout and "truncated.png" in report.stdout
    assert "good.png" not in report.stdout

    _run(CHECK, "--root", tmp_path, "--delete")
    assert sorted(p.name for p in folder.glob("*.png")) == ["good.png"]
    assert _run(CHECK, "--root", tmp_path).returncode == 0


def test_prepare_skips_corrupt_source_images(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    lists = {
        "CLIENT_TRAIN.txt": [],
        "CLIENT_TEST.txt": [],
        "IMPOSTER_TRAIN.txt": [],
        "IMPOSTER_TEST.txt": [],
    }
    for list_name, prefix in (
        ("CLIENT_TRAIN.txt", "a"),
        ("CLIENT_TEST.txt", "b"),
        ("IMPOSTER_TRAIN.txt", "c"),
        ("IMPOSTER_TEST.txt", "d"),
    ):
        for i in range(8):
            name = f"{prefix}_id{i + (10 if prefix in 'bd' else 0)}_s0_{i}.png"
            Image.new("RGB", (30, 30), (i * 20, 50, 90)).save(raw / name)
            lists[list_name].append(f"/kaggle/input/x/{name}")
    (raw / "a_id3_s0_3.png").write_bytes(b"not a png")  # corrupt one training image
    for list_name, lines in lists.items():
        (raw / list_name).write_text("\n".join(lines) + "\n")

    result = _run(
        PREPARE,
        "--src",
        raw,
        "--dst",
        out,
        "--resize",
        24,
        "--dev-fraction",
        "0.25",
        "--min-per-class",
        "1",
        "--preview-count",
        "4",
    )
    assert "skipped 1 unreadable" in result.stdout
    assert (out / "skipped_unreadable.txt").read_text().strip() == "a_id3_s0_3.png"
    assert not list(out.rglob("a_id3_s0_3.png"))
    assert "a_id3_s0_3.png" not in (out / "manifest.csv").read_text()
