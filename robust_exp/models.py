from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class Normalize(nn.Module):
    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (images - self.mean) / self.std


class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(inputs))
        shortcut = self.shortcut(out) if self.shortcut is not None else inputs
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class PreActResNet(nn.Module):
    def __init__(self, blocks: list[int], num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(64, blocks[0], 1)
        self.layer2 = self._make_layer(128, blocks[1], 2)
        self.layer3 = self._make_layer(256, blocks[2], 2)
        self.layer4 = self._make_layer(512, blocks[3], 2)
        self.bn = nn.BatchNorm2d(512)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, count: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (count - 1)
        layers = []
        for current_stride in strides:
            layers.append(PreActBlock(self.in_planes, planes, current_stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.conv1(inputs)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        return self.linear(out.view(out.size(0), -1))


def build_model(
    num_classes: int = 10, architecture: str = "resnet18", dataset: str = "cifar10"
) -> nn.Module:
    normalization = {
        "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
        "cifar100": (CIFAR100_MEAN, CIFAR100_STD),
    }
    if dataset not in normalization:
        raise ValueError(f"Unknown dataset: {dataset}")
    mean, std = normalization[dataset]
    if architecture == "preact_resnet18":
        backbone = PreActResNet([2, 2, 2, 2], num_classes=num_classes)
        return nn.Sequential(Normalize(mean, std), backbone)
    if architecture != "resnet18":
        raise ValueError(f"Unknown architecture: {architecture}")
    backbone = resnet18(weights=None, num_classes=num_classes)
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    return nn.Sequential(Normalize(mean, std), backbone)
