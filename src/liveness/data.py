"""Dataset and dataloaders.

Expected on-disk layout (built by ``scripts/prepare_data.py``)::

    <root>/{train,dev,test}/{live,spoof}/*.jpg

Images are returned as float tensors in [0, 1] (RGB). Mean/std normalisation lives
inside the model so that training, ONNX export and any browser/app client share
exactly one preprocessing contract: resize -> scale to [0, 1].
"""

from __future__ import annotations

import io
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_TO_LABEL = {"live": 0, "spoof": 1}


class RandomJPEG:
    """Re-encode the image as JPEG with random quality (simulates capture/transmit artefacts)."""

    def __init__(self, quality: tuple[int, int] = (40, 95), p: float = 0.3):
        self.quality = quality
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


AUGMENT_PRESETS = ("full", "basic")


def build_transforms(img_size: int, train: bool, augment: str = "full") -> T.Compose:
    """``full``: crop, flip, colour jitter, blur, random JPEG. ``basic``: no blur / JPEG."""
    if augment not in AUGMENT_PRESETS:
        raise ValueError(f"Unknown augment preset {augment!r}; choose from {AUGMENT_PRESETS}")
    if train:
        steps = [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        ]
        if augment == "full":
            steps += [T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2), RandomJPEG()]
        return T.Compose([*steps, T.ToTensor()])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


class SpoofDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int,
        train: bool = False,
        augment: str = "full",
    ):
        self.samples: list[tuple[Path, int]] = []
        for class_name, label in CLASS_TO_LABEL.items():
            folder = Path(root) / split / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing folder: {folder}")
            for path in sorted(folder.rglob("*")):
                if path.suffix.lower() in IMG_EXTS:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {Path(root) / split}")
        self.transform = build_transforms(img_size, train, augment)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict[int, int]:
        counts = {0: 0, 1: 0}
        for _, label in self.samples:
            counts[label] += 1
        return counts


def make_loader(cfg: dict, split: str, train: bool) -> DataLoader:
    data_cfg = cfg["data"]
    workers = int(data_cfg.get("num_workers", 0))
    batch_size = int(cfg["train"]["batch_size"]) * (1 if train else 2)
    dataset = SpoofDataset(
        data_cfg["root"],
        split,
        int(data_cfg["img_size"]),
        train=train,
        augment=data_cfg.get("augment", "full"),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train and len(dataset) > batch_size,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
    )
