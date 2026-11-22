"""predict.py needs only onnxruntime + Pillow, so it is tested with a tiny hand-made ONNX model."""

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

from liveness import predict  # noqa: E402


def _tiny_model(path: Path, size: int = 32) -> None:
    """spoof_prob = sigmoid(mean(image)): white -> 0.73, black -> 0.5."""
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["m"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Sigmoid", ["m"], ["spoof_prob"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, size, size])],
        [helper.make_tensor_value_info("spoof_prob", TensorProto.FLOAT, ["batch"])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


def test_preprocess_shape_range_and_center_crop(tmp_path):
    wide = tmp_path / "wide.png"
    image = Image.new("RGB", (200, 100), (0, 0, 0))
    image.paste((255, 255, 255), (50, 0, 150, 100))  # white square exactly in the centre
    image.save(wide)
    array = predict.preprocess(wide, img_size=32, prep_size=64, center_crop=True)
    assert array.shape == (3, 32, 32)
    assert 0.0 <= array.min() and array.max() <= 1.0
    assert array.mean() > 0.95  # centre crop keeps only the white square
    assert predict.preprocess(wide, 32, 64, center_crop=False).mean() < 0.7


def test_predict_files_applies_threshold(tmp_path):
    model_path = tmp_path / "model.onnx"
    _tiny_model(model_path)
    Image.new("RGB", (40, 40), (255, 255, 255)).save(tmp_path / "white.png")
    Image.new("RGB", (40, 40), (0, 0, 0)).save(tmp_path / "black.png")

    session, size = predict.load_session(model_path)
    assert size == 32
    results = predict.predict_files(
        session, [str(tmp_path / "white.png"), str(tmp_path / "black.png")], size, threshold=0.6
    )
    assert [r["verdict"] for r in results] == ["spoof", "live"]
    assert results[0]["score"] > results[1]["score"]


def test_cli_reads_threshold_from_card(tmp_path):
    model_path = tmp_path / "model_fp32.onnx"
    _tiny_model(model_path)
    (tmp_path / "model_card.json").write_text(json.dumps({"threshold": 0.9}))
    Image.new("RGB", (40, 40), (255, 255, 255)).save(tmp_path / "white.png")
    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "liveness.predict",
            str(tmp_path / "white.png"),
            "--model",
            str(model_path),
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    )
    result = json.loads(out.stdout)[0]
    assert result["threshold"] == 0.9 and result["verdict"] == "live"  # 0.73 < 0.9
