"""ResNet1D backbone for OPS-Bench, used as a second backbone alongside TCN.

Standard ResNet1D for time series: initial conv with large kernel, several
basic residual blocks (Conv1d -> BN -> ReLU -> Conv1d -> BN, skip connection),
global average pooling, linear head. Sized to ~290K params at the default
configuration so it is directly comparable to the TCN.

Input:  (B, C, T) float32
Output: (B, num_classes) logits
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _norm(num_channels: int, kind: str) -> nn.Module:
    if kind == "bn":
        return nn.BatchNorm1d(num_channels)
    if kind == "gn":
        # GroupNorm with 8 groups (gcd of our channel widths 32/64/64/128).
        # GN is independent of batch statistics, so weak and strong views
        # do not share normalization scale -- relevant for two-view training.
        return nn.GroupNorm(num_groups=8, num_channels=num_channels)
    raise ValueError(f"Unknown norm kind: {kind!r}")


class _BasicBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 dropout: float = 0.0, norm: str = "bn"):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = _norm(out_ch, norm)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = _norm(out_ch, norm)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                _norm(out_ch, norm),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """1D ResNet for time-series classification.

    num_channels controls the per-stage channel count and number of stages.
    Default (32, 64, 64, 128) two-block stages -> ~290K params for C_in=2.
    """

    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        num_channels: Sequence[int] = (32, 64, 64, 128),
        blocks_per_stage: int = 2,
        kernel_size_init: int = 7,
        dropout: float = 0.1,
        norm: str = "bn",
    ):
        super().__init__()
        c0 = num_channels[0]
        self.stem = nn.Sequential(
            nn.Conv1d(num_inputs, c0, kernel_size=kernel_size_init,
                      stride=2, padding=kernel_size_init // 2, bias=False),
            _norm(c0, norm),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        layers = []
        in_ch = c0
        for stage_idx, out_ch in enumerate(num_channels):
            stride = 1 if stage_idx == 0 else 2
            for b in range(blocks_per_stage):
                layers.append(_BasicBlock1D(
                    in_ch, out_ch,
                    stride=stride if b == 0 else 1,
                    dropout=dropout,
                    norm=norm,
                ))
                in_ch = out_ch
        self.body = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
