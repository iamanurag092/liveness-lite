"""Fine-tune the backbone on the train split; select the best epoch by dev EER."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from tqdm import tqdm

from liveness.data import make_loader
from liveness.evaluate import torch_scores
from liveness.metrics import eer_and_threshold, roc_auc
from liveness.model import SpoofNet, count_params
from liveness.utils import get_device, load_config, save_json, seed_everything


def run(cfg: dict) -> dict:
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    train_loader = make_loader(cfg, "train", train=True)
    dev_loader = make_loader(cfg, "dev", train=False)

    model = SpoofNet(**cfg["model"]).to(device)
    print(f"device={device}  params={count_params(model) / 1e6:.2f}M")

    train_cfg = cfg["train"]
    counts = train_loader.dataset.class_counts()
    pos_weight = torch.tensor([counts[0] / max(counts[1], 1)], device=device)  # live / spoof
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    smoothing = float(train_cfg.get("label_smoothing", 0.0))

    history, best, bad_epochs = [], {"eer": float("inf")}, 0
    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        started, running, seen = time.time(), 0.0, 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            targets = y * (1 - smoothing) + 0.5 * smoothing
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(x), targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        labels, scores = torch_scores(model, dev_loader, device)
        eer, threshold = eer_and_threshold(labels, scores)
        if len(np.unique(scores)) < 3:
            print(
                "WARNING: dev scores are (nearly) constant; the model output has collapsed. "
                "Check BatchNorm statistics, learning rate and data before trusting any metric."
            )
        record = {
            "epoch": epoch,
            "train_loss": running / seen,
            "dev_eer": eer,
            "dev_auc": roc_auc(labels, scores),
            "dev_threshold": threshold,
            "seconds": round(time.time() - started, 1),
        }
        history.append(record)
        print(
            f"epoch {epoch:02d} loss={record['train_loss']:.4f} "
            f"dev_EER={eer:.4f} dev_AUC={record['dev_auc']:.4f} ({record['seconds']}s)"
        )

        if eer < best["eer"]:
            best, bad_epochs = {"eer": eer, "epoch": epoch}, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_cfg": cfg["model"],
                    "img_size": cfg["data"]["img_size"],
                    "epoch": epoch,
                    "dev_eer": eer,
                },
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break

    save_json({"best": best, "history": history}, out_dir / "history.json")
    print(f"Best dev EER {best['eer']:.4f} at epoch {best['epoch']} -> {out_dir / 'best.pt'}")
    return {"best": best, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*", default=[], help="key=value config overrides")
    args = parser.parse_args()
    run(load_config(args.config, args.override))


if __name__ == "__main__":
    main()
