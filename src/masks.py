"""Observation degradation mask generators (OPS-Bench § 4).

Each generator returns a float32 array of shape (C, T) where 1 = observed, 0 = missing.
Callers multiply the input signal X by M to get the degraded view, and may concatenate
M as additional channels for mask-aware models.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def random_point_mask(
    shape: Sequence[int],
    missing_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Independent Bernoulli mask per (channel, time) entry.

    Models packet loss / instantaneous sampling glitches.
    """
    rng = rng if rng is not None else np.random.default_rng()
    return (rng.random(shape) > missing_rate).astype(np.float32)


def temporal_block_mask(
    shape: Sequence[int],
    block_ratio: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Zero a single contiguous temporal segment across all channels.

    Models bus stall, short-time acquisition halt, or buffer write failure.
    block_ratio is the fraction of T to drop. block_ratio >= 1 returns an all-zero mask.
    """
    rng = rng if rng is not None else np.random.default_rng()
    C, T = shape
    if block_ratio >= 1.0:
        return np.zeros((C, T), dtype=np.float32)
    M = np.ones((C, T), dtype=np.float32)
    block_len = max(1, int(round(T * block_ratio)))
    block_len = min(block_len, T)
    start = int(rng.integers(0, T - block_len + 1))
    M[:, start : start + block_len] = 0.0
    return M


def channel_mask(
    shape: Sequence[int],
    channel_missing_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Drop each channel independently with probability ``channel_missing_rate``.

    Models accelerometer death / sensor offline / power failure on a channel.
    Independent per-channel Bernoulli is consistent with sensors failing
    independently in the field. Expected number of dropped channels is C*rate.

    Edge cases:
      rate=0 -> no channel dropped (M is all ones).
      rate=1 -> all channels dropped (M is all zeros, degenerate input).
    With small C (e.g. CWRU's 2-channel DE+FE), low rates leave most samples
    unaffected; this is the intended sensor-failure semantics.
    """
    rng = rng if rng is not None else np.random.default_rng()
    C, T = shape
    drop = rng.random(C) < channel_missing_rate
    M = np.ones((C, T), dtype=np.float32)
    M[drop, :] = 0.0
    return M


def mixed_mask(
    shape: Sequence[int],
    p_point: float = 0.2,
    r_block: float = 0.2,
    r_channel: float = 0.2,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Element-wise product of point + block + channel masks.

    Approximates a worst-case combined degradation typical in long-term industrial deployment.
    """
    M1 = random_point_mask(shape, p_point, rng=rng)
    M2 = temporal_block_mask(shape, r_block, rng=rng)
    M3 = channel_mask(shape, r_channel, rng=rng)
    return (M1 * M2 * M3).astype(np.float32)


def sample_mask(pattern: str, shape: Sequence[int], **kwargs) -> np.ndarray:
    """Dispatch to the right generator by string name.

    pattern: one of {"random", "block", "channel", "mixed"}.
    Per-pattern kwargs:
      random  -> missing_rate
      block   -> block_ratio
      channel -> channel_missing_rate
      mixed   -> p_point, r_block, r_channel
    All accept an optional `rng=np.random.Generator`.
    """
    rng = kwargs.get("rng")
    if pattern == "random":
        return random_point_mask(shape, kwargs.get("missing_rate", 0.3), rng=rng)
    if pattern == "block":
        return temporal_block_mask(shape, kwargs.get("block_ratio", 0.2), rng=rng)
    if pattern == "channel":
        return channel_mask(shape, kwargs.get("channel_missing_rate", 0.25), rng=rng)
    if pattern == "mixed":
        return mixed_mask(
            shape,
            p_point=kwargs.get("p_point", 0.2),
            r_block=kwargs.get("r_block", 0.2),
            r_channel=kwargs.get("r_channel", 0.2),
            rng=rng,
        )
    raise ValueError(f"Unknown mask pattern: {pattern!r}")
