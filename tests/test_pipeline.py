"""Torch/ONNX tests. Skipped automatically when those packages are not installed."""

import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from liveness import evaluate, export, train  # noqa: E402
from liveness.data import SpoofDataset  # noqa: E402
from liveness.model import BACKBONES, SpoofNet, count_flops, count_params  # noqa: E402
from liveness.utils import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    dst = tmp_path_factory.mktemp("synthetic")
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_synthetic_data.py"), "--dst", str(dst)],
        check=True,
    )
    return dst


@pytest.mark.parametrize("name", BACKBONES)
def test_model_forward_shape_and_size(name):
    model = SpoofNet(name=name, pretrained=False).eval()
    out = model(torch.rand(2, 3, 96, 96))
    assert out.shape == (2,)
    assert count_params(model) < 6e6
    assert count_flops(model, 96) > 0


def test_dataset_loads_labels_and_range(synthetic):
    dataset = SpoofDataset(synthetic, "train", img_size=64, train=True)
    assert dataset.class_counts() == {0: 48, 1: 48}
    x, y = dataset[0]
    assert x.shape == (3, 64, 64)
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert float(y) in (0.0, 1.0)


def test_augment_presets(synthetic):
    for preset in ("full", "basic"):
        x, _ = SpoofDataset(synthetic, "train", img_size=64, train=True, augment=preset)[0]
        assert x.shape == (3, 64, 64)
    with pytest.raises(ValueError):
        SpoofDataset(synthetic, "train", img_size=64, train=True, augment="nope")


def test_missing_split_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        SpoofDataset(tmp_path, "train", img_size=64)


def test_train_export_evaluate_end_to_end(synthetic, tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg["data"]["root"] = str(synthetic)
    cfg["output"]["dir"] = str(tmp_path / "run")

    result = train.run(cfg)
    assert (tmp_path / "run" / "best.pt").exists()
    assert result["best"]["eer"] <= 0.5

    card = export.run(cfg, str(tmp_path / "run" / "best.pt"), str(tmp_path / "models"))
    assert card["fp32_max_abs_diff_vs_torch"] < 1e-3
    assert card["int8_onnx_mb"] < card["fp32_onnx_mb"]
    assert 0.0 < card["threshold"] < 1.0 and 0.0 <= card["dev_eer"] <= 1.0

    from liveness import predict

    session, size = predict.load_session(tmp_path / "models" / "model_fp32.onnx")
    some_image = next((synthetic / "test" / "spoof").glob("*.png"))
    verdict = predict.predict_files(session, [str(some_image)], size, card["threshold"])[0]
    assert verdict["verdict"] in ("live", "spoof") and 0.0 <= verdict["score"] <= 1.0
    assert export.check_budget(card, cfg["budget"]) == []

    for tag, path in (("fp32", "model_fp32.onnx"), ("int8", "model_int8.onnx")):
        metrics = evaluate.run(
            cfg, None, str(tmp_path / "models" / path), str(tmp_path / "run"), tag
        )
        assert 0.0 <= metrics["test_at_dev_eer_threshold"]["auc"] <= 1.0
