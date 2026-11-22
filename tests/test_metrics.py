import numpy as np
import pytest

from liveness.metrics import (
    apcer_bpcer,
    eer_and_threshold,
    error_curve,
    roc_auc,
    summarize,
)


def test_auc_matches_hand_computed_value():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert roc_auc(labels, scores) == pytest.approx(0.75)


def test_auc_perfect_inverted_and_ties():
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


def test_auc_single_class_is_nan():
    assert np.isnan(roc_auc([0, 0], [0.1, 0.2]))


def test_apcer_bpcer_definitions():
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    apcer, bpcer = apcer_bpcer(labels, scores, 0.5)
    assert apcer == pytest.approx(0.5)  # spoof 0.35 is accepted as live
    assert bpcer == pytest.approx(0.0)  # no live sample reaches 0.5
    apcer, bpcer = apcer_bpcer(labels, scores, 0.3)
    assert apcer == pytest.approx(0.0)
    assert bpcer == pytest.approx(0.5)  # live 0.4 is rejected


def test_eer_is_zero_when_separable():
    eer, threshold = eer_and_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert eer == 0.0
    assert 0.2 < threshold <= 0.8


def test_eer_matches_curve_crossing_on_random_data():
    rng = np.random.default_rng(0)
    labels = np.r_[np.zeros(2000), np.ones(2000)]
    scores = np.r_[rng.normal(0.4, 0.15, 2000), rng.normal(0.6, 0.15, 2000)]
    eer, threshold = eer_and_threshold(labels, scores)
    apcer, bpcer = apcer_bpcer(labels, scores, threshold)
    assert abs(apcer - bpcer) < 0.01
    assert eer == pytest.approx((apcer + bpcer) / 2)
    assert 0.15 < eer < 0.30  # two unit-separation gaussians, EER ~ 0.21


def test_error_curve_is_monotonic():
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 500)
    scores = rng.random(500)
    _, apcer, bpcer = error_curve(labels, scores)
    assert np.all(np.diff(apcer) >= 0)  # raising the threshold accepts more attacks
    assert np.all(np.diff(bpcer) <= 0)  # ...and rejects fewer live samples


def test_summarize_consistency():
    out = summarize([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 0.5)
    assert out["acer"] == pytest.approx((out["apcer"] + out["bpcer"]) / 2)
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["n_live"] == 2 and out["n_spoof"] == 2


def test_invalid_labels_rejected():
    with pytest.raises(ValueError):
        roc_auc([0, 2], [0.1, 0.2])
