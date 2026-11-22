"""Lightweight classifier: ImageNet-pretrained mobile backbone with a single-logit head."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode
from torchvision import models

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

BACKBONES = ("mobilenet_v3_small", "mobilenet_v3_large", "mobilenet_v2")


def _build_backbone(name: str, pretrained: bool, dropout: float) -> nn.Module:
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_small(weights=weights)
        net.classifier[2] = nn.Dropout(dropout)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    elif name == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v3_large(weights=weights)
        net.classifier[2] = nn.Dropout(dropout)
        net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    elif name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        net = models.mobilenet_v2(weights=weights)
        net.classifier[0] = nn.Dropout(dropout)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, 1)
    else:
        raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}")
    return net


class SpoofNet(nn.Module):
    """Input: float RGB in [0, 1], shape (N, 3, H, W). Output: spoof logit, shape (N,)."""

    def __init__(
        self,
        name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.2,
        bn_momentum: float = 0.1,
    ):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.net = _build_backbone(name, pretrained, dropout)
        # torchvision's MobileNetV3 uses BN momentum 0.01 (tuned for long ImageNet training).
        # Fine-tuning runs are short, so running statistics barely move from their start values
        # and eval-mode outputs can be badly off. PyTorch's default of 0.1 adapts in tens of steps.
        for module in self.net.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.momentum = bn_momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.net(x).squeeze(1)


def load_model(checkpoint: str | Path, device: torch.device | str = "cpu") -> tuple[SpoofNet, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    model_cfg = {**ckpt["model_cfg"], "pretrained": False}
    model = SpoofNet(**model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_flops(model: nn.Module, img_size: int) -> int:
    """Forward FLOPs for one image (1 multiply-add = 2 FLOPs)."""
    model = copy.deepcopy(model).cpu().eval()
    x = torch.rand(1, 3, img_size, img_size)
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model(x)
    return int(counter.get_total_flops())
