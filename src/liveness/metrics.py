"""Presentation-attack-detection metrics (numpy only).

Conventions (see docs/02_methodology.md):
  * label 1 = spoof (attack), label 0 = live (bona fide)
  * score   = estimated probability of spoof; predict spoof when score >= threshold
  * APCER   = share of attacks wrongly accepted as live   (score <  threshold)
  * BPCER   = share of live samples wrongly rejected      (score >= threshold)
  * ACER    = (APCER + BPCER) / 2
"""

from __future__ import annotations

import numpy as np


def _prepare(labels, scores) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if labels.shape != scores.shape:
        raise ValueError("labels and scores must have the same length")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError("labels must be 0 (live) or 1 (spoof)")
    return labels, scores


def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties given their average rank."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    _, inverse, counts = np.unique(sorted_a, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    cumulative = np.cumsum(counts)
    average_rank = cumulative - (counts - 1) / 2.0
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = average_rank[inverse]
    return ranks


def roc_auc(labels, scores) -> float:
    labels, scores = _prepare(labels, scores)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def apcer_bpcer(labels, scores, threshold: float) -> tuple[float, float]:
    labels, scores = _prepare(labels, scores)
    spoof, live = scores[labels == 1], scores[labels == 0]
    apcer = float(np.mean(spoof < threshold)) if len(spoof) else float("nan")
    bpcer = float(np.mean(live >= threshold)) if len(live) else float("nan")
    return apcer, bpcer


def error_curve(labels, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """APCER and BPCER for every candidate threshold. Returns (thresholds, apcer, bpcer)."""
    labels, scores = _prepare(labels, scores)
    spoof = np.sort(scores[labels == 1])
    live = np.sort(scores[labels == 0])
    if len(spoof) == 0 or len(live) == 0:
        raise ValueError("Need both live and spoof samples")
    unique = np.unique(scores)
    thresholds = np.concatenate([unique, [unique[-1] + 1e-6]])
    apcer = np.searchsorted(spoof, thresholds, side="left") / len(spoof)
    bpcer = (len(live) - np.searchsorted(live, thresholds, side="left")) / len(live)
    return thresholds, apcer, bpcer


def eer_and_threshold(labels, scores) -> tuple[float, float]:
    """Equal error rate and the threshold at which APCER and BPCER are closest."""
    thresholds, apcer, bpcer = error_curve(labels, scores)
    idx = int(np.argmin(np.abs(apcer - bpcer)))
    return float((apcer[idx] + bpcer[idx]) / 2), float(thresholds[idx])


def summarize(labels, scores, threshold: float) -> dict:
    """All headline metrics at a fixed operating threshold."""
    labels, scores = _prepare(labels, scores)
    apcer, bpcer = apcer_bpcer(labels, scores, threshold)
    predictions = (scores >= threshold).astype(int)
    eer, eer_threshold = eer_and_threshold(labels, scores)
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2,
        "auc": roc_auc(labels, scores),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "n_live": int((labels == 0).sum()),
        "n_spoof": int((labels == 1).sum()),
    }
