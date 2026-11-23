import importlib.util
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_submission.py"


def _module():
    spec = importlib.util.spec_from_file_location("make_submission", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _touch(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_submission_contains_only_what_belongs_in_it(tmp_path):
    module = _module()
    for name in (
        "README.md",
        "report.pdf",
        "src/liveness/model.py",
        "docs/PLAN.md",
        "demo/index.html",
        "models/lite/model_fp32.onnx",
        "models/lite/model_card.json",
        "models/accurate/model_fp32.onnx",
        "models/accurate/model_card.json",
        "runs/E04/metrics_fp32.json",
        "runs/E04/confusion_fp32.png",
        "runs/E04/best.pt",
        "runs/smoke/history.json",
        "demo/samples/live_0.png",
        "demo/models/lite/model.onnx",
        "data/processed/test/live/a.png",
        "src/liveness/__pycache__/x.pyc",
        ".venv/lib/x.py",
    ):
        _touch(tmp_path / name)

    out = tmp_path / "s.zip"
    count, _ = module.build(tmp_path, out, include_accurate=False)
    names = set(zipfile.ZipFile(out).namelist())
    assert "liveness-lite/README.md" in names and "liveness-lite/report.pdf" in names
    assert "liveness-lite/models/lite/model_fp32.onnx" in names
    assert "liveness-lite/models/accurate/model_card.json" in names
    assert "liveness-lite/runs/E04/metrics_fp32.json" in names
    assert "liveness-lite/models/accurate/model_fp32.onnx" not in names
    banned = ("best.pt", "samples", "demo/models", "data/", ".venv", "__pycache__", "runs/smoke")
    assert not [n for n in names if any(b in n for b in banned)]
    assert count == len(names)

    module.build(tmp_path, out, include_accurate=True)
    assert "liveness-lite/models/accurate/model_fp32.onnx" in zipfile.ZipFile(out).namelist()
