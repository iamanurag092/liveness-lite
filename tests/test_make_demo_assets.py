import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_demo_assets.py"


def _export_dir(folder: Path, threshold: float) -> Path:
    folder.mkdir(parents=True)
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Sigmoid", ["m"], ["spoof_prob"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 32, 32])],
        [helper.make_tensor_value_info("spoof_prob", TensorProto.FLOAT, ["batch"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(folder / "model_fp32.onnx"))
    card = {
        "backbone": "tiny",
        "img_size": 32,
        "params_m": 0.0,
        "gmacs": 0.0,
        "fp32_onnx_mb": 0.0,
        "threshold": threshold,
        "dev_eer": 0.1,
        "secret_extra_field": "must not be copied",
    }
    (folder / "model_card.json").write_text(json.dumps(card))
    return folder


def _run(*args):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )


def test_models_and_samples_are_assembled(tmp_path):
    lite = _export_dir(tmp_path / "lite", 0.6)
    accurate = _export_dir(tmp_path / "accurate", 0.7)
    data = tmp_path / "data"
    for label, value in (("live", 20), ("spoof", 240)):
        folder = data / "test" / label
        folder.mkdir(parents=True)
        for i in range(4):
            Image.new("RGB", (40, 40), (value, value, value)).save(folder / f"{i}.png")
    out = tmp_path / "demo"

    _run(
        "--lite",
        lite,
        "--accurate",
        accurate,
        "--out",
        out,
        "--with-samples",
        "--data",
        data,
        "--per-class",
        3,
    )

    assert (out / "models" / "lite" / "model.onnx").is_file()
    card = json.loads((out / "models" / "accurate" / "card.json").read_text())
    assert card["threshold"] == 0.7 and card["tier"] == "accurate"
    assert "secret_extra_field" not in card

    samples = json.loads((out / "samples" / "samples.json").read_text())
    assert len(samples) == 6
    assert {s["label"] for s in samples} == {"live", "spoof"}
    for sample in samples:
        assert (out / "samples" / sample["file"]).is_file()
        assert set(sample["scores"]) == {"lite", "accurate"}
    bright = [s for s in samples if s["label"] == "spoof"][0]["scores"]["lite"]
    dark = [s for s in samples if s["label"] == "live"][0]["scores"]["lite"]
    assert bright > dark  # reference scores come from the model, not from labels


def test_requires_a_model(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--out", str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode != 0
