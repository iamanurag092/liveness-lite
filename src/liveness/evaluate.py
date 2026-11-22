"""Evaluate a checkpoint (.pt) or an exported model (.onnx) on the dev and test splits.

Protocol: the operating threshold is chosen on the *dev* split (its EER point) and then
applied unchanged to the *test* split, which is the only number to report.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

from liveness.data import make_loader
from liveness.metrics import eer_and_threshold, error_curve, summarize
from liveness.model import load_model
from liveness.utils import get_device, load_config, save_json


@torch.no_grad()
def torch_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels, scores = [], []
    for x, y in tqdm(loader, leave=False):
        probs = torch.sigmoid(model(x.to(device)))
        scores.append(probs.float().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(scores)


def onnx_scores(session, loader) -> tuple[np.ndarray, np.ndarray]:
    input_name = session.get_inputs()[0].name
    labels, scores = [], []
    for x, y in tqdm(loader, leave=False):
        out = session.run(None, {input_name: x.numpy().astype(np.float32)})[0]
        scores.append(np.asarray(out).reshape(-1))
        labels.append(y.numpy())
    return np.concatenate(labels).astype(int), np.concatenate(scores)


def _plots(out_dir: Path, tag: str, labels, scores, threshold: float) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    _, apcer, bpcer = error_curve(labels, scores)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(bpcer, 1 - apcer)
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("BPCER (live rejected)")
    ax.set_ylabel("1 - APCER (attacks caught)")
    ax.set_title(f"ROC ({tag})")
    fig.tight_layout()
    fig.savefig(out_dir / f"roc_{tag}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.hist(scores[labels == 0], bins=40, alpha=0.6, label="live")
    ax.hist(scores[labels == 1], bins=40, alpha=0.6, label="spoof")
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold {threshold:.3f}")
    ax.set_xlabel("spoof probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"scores_{tag}.png", dpi=150)
    plt.close(fig)

    predictions = (scores >= threshold).astype(int)
    matrix = np.array(
        [[np.sum((labels == t) & (predictions == p)) for p in (0, 1)] for t in (0, 1)]
    )
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, int(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["pred live", "pred spoof"])
    ax.set_yticks([0, 1], ["true live", "true spoof"])
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_{tag}.png", dpi=150)
    plt.close(fig)


def run(cfg: dict, checkpoint: str | None, onnx_path: str | None, out_dir: str, tag: str) -> dict:
    cfg = copy.deepcopy(cfg)
    if onnx_path:
        import onnxruntime as ort

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        height = session.get_inputs()[0].shape[2]
        if isinstance(height, int):  # always evaluate at the size the model was built for
            cfg["data"]["img_size"] = height
    else:
        device = get_device(cfg.get("device", "auto"))
        model, ckpt = load_model(checkpoint, device)
        cfg["data"]["img_size"] = int(ckpt["img_size"])

    dev_loader = make_loader(cfg, "dev", train=False)
    test_loader = make_loader(cfg, "test", train=False)
    if onnx_path:
        dev_labels, dev_scores = onnx_scores(session, dev_loader)
        test_labels, test_scores = onnx_scores(session, test_loader)
    else:
        dev_labels, dev_scores = torch_scores(model, dev_loader, device)
        test_labels, test_scores = torch_scores(model, test_loader, device)

    _, dev_threshold = eer_and_threshold(dev_labels, dev_scores)
    results = {
        "source": onnx_path or checkpoint,
        "img_size": cfg["data"]["img_size"],
        "dev": summarize(dev_labels, dev_scores, dev_threshold),
        "test_at_dev_eer_threshold": summarize(test_labels, test_scores, dev_threshold),
        "test_at_0.5": summarize(test_labels, test_scores, 0.5),
    }
    save_json(results, Path(out_dir) / f"metrics_{tag}.json")
    _plots(Path(out_dir), tag, test_labels, test_scores, dev_threshold)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", help="Path to best.pt")
    parser.add_argument("--onnx", help="Path to an exported .onnx model")
    parser.add_argument("--run-dir", help="Where to write metrics/plots (default: output.dir)")
    parser.add_argument("--tag", default=None, help="Suffix for output files")
    parser.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    args = parser.parse_args()
    if bool(args.checkpoint) == bool(args.onnx):
        parser.error("Provide exactly one of --checkpoint or --onnx")

    cfg = load_config(args.config, args.override)
    out_dir = args.run_dir or cfg["output"]["dir"]
    tag = args.tag or ("onnx" if args.onnx else "torch")
    results = run(cfg, args.checkpoint, args.onnx, out_dir, tag)
    test = results["test_at_dev_eer_threshold"]
    print(
        f"[{tag}] test @ dev-EER thr={test['threshold']:.3f}  "
        f"acc={test['accuracy']:.4f}  APCER={test['apcer']:.4f}  BPCER={test['bpcer']:.4f}  "
        f"ACER={test['acer']:.4f}  AUC={test['auc']:.4f}"
    )


if __name__ == "__main__":
    main()
