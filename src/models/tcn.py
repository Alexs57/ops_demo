"""Temporal Convolutional Network (Bai, Kolter & Koltun 2018).

Dilated causal 1-D convolutions with residual blocks, mean-pooled to a
classification head. Kept small and conventional so it serves as a transparent
backbone for the OPS-Bench experiments rather than a competing contribution.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class _Chomp1d(nn.Module):
    """Trim the right end of a tensor to undo causal padding."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class _TemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        )
        self.net = nn.Sequential(
            self.conv1, _Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
            self.conv2, _Chomp1d(padding), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.activation(out + res)


class TCN(nn.Module):
    """Classification TCN. Input: (B, C, T). Output: (B, num_classes) logits."""

    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        num_channels: Sequence[int] = (64, 64, 64, 64),
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        blocks = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            blocks.append(
                _TemporalBlock(
                    in_ch, out_ch, kernel_size=kernel_size, dilation=2 ** i, dropout=dropout
                )
            )
            in_ch = out_ch
        self.network = nn.Sequential(*blocks)
        self.classifier = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.network(x)         # (B, C_last, T)
        pooled = feats.mean(dim=-1)     # (B, C_last)
        return self.classifier(pooled)  # (B, num_classes)
