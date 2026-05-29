"""CWRU bearing dataset loader.

Each CWRU ``.mat`` file holds one recording (one fault condition at one motor
load) as variable-named arrays ``X<id>_DE_time``, ``X<id>_FE_time``,
``X<id>_BA_time``. The numeric ``<id>`` identifies the recording and is mapped
to a fault class by ``DEFAULT_CLASS_MAP``. Files may be named either by that
number (``105.mat``, the official CWRU layout) or by condition
(``IR007_0.mat``, the cathysiyu/Mechanical-datasets layout). The label is always
read from the embedded ``X<id>`` key, so both on-disk layouts work and the
directory is scanned recursively.

Split protocol (OPS-Bench). To avoid leakage between overlapping windows, each
recording's *raw signal* is cut into contiguous train / validation / test
segments BEFORE windowing. Windows are then generated independently within each
segment, so no window can straddle a split boundary and no train window shares
samples with any test window. The split is deterministic (by time position
within each recording), so every seed sees the same partition; seeds vary only
the stochastic parts of training. Checkpoint selection uses the validation
split; the test split is touched once, for the final evaluation.

If real CWRU files are not available, ``make_synthetic_cwru`` produces a
dataset with the same shape so the rest of the pipeline can be tested.

Download: https://engineering.case.edu/bearingdatacenter
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# Canonical CWRU 10-class benchmark (12 kHz drive-end, 1797 rpm baseline).
# Keys: file-number string. Values: class index.
# 0=Normal, 1-3=IR 007/014/021, 4-6=Ball 007/014/021, 7-9=OR 007/014/021 (centered).
DEFAULT_CLASS_MAP: Dict[str, int] = {
    "97":  0, "98":  0, "99":  0, "100": 0,   # Normal
    "105": 1, "106": 1, "107": 1, "108": 1,   # IR 0.007"
    "169": 2, "170": 2, "171": 2, "172": 2,   # IR 0.014"
    "209": 3, "210": 3, "211": 3, "212": 3,   # IR 0.021"
    "118": 4, "119": 4, "120": 4, "121": 4,   # Ball 0.007"
    "185": 5, "186": 5, "187": 5, "188": 5,   # Ball 0.014"
    "222": 6, "223": 6, "224": 6, "225": 6,   # Ball 0.021"
    "130": 7, "131": 7, "132": 7, "133": 7,   # OR 0.007" @6
    "197": 8, "198": 8, "199": 8, "200": 8,   # OR 0.014" @6
    "234": 9, "235": 9, "236": 9, "237": 9,   # OR 0.021" @6
}
DEFAULT_NUM_CLASSES = 10


def _window_signal(
    signal: np.ndarray,
    window_size: int,
    stride: int,
) -> np.ndarray:
    """Slice a 1-D signal into overlapping windows. Returns (N, window_size)."""
    signal = np.ascontiguousarray(signal, dtype=np.float32)
    n = (len(signal) - window_size) // stride + 1
    if n <= 0:
        return np.zeros((0, window_size), dtype=np.float32)
    out = np.lib.stride_tricks.as_strided(
        signal,
        shape=(n, window_size),
        strides=(signal.strides[0] * stride, signal.strides[0]),
        writeable=False,
    )
    return np.ascontiguousarray(out, dtype=np.float32)


def _find_key(mat: dict, suffix: str) -> Optional[str]:
    """Find first key in a .mat dict whose name ends with ``suffix`` (e.g. '_DE_time')."""
    for k in mat.keys():
        if k.endswith(suffix):
            return k
    return None


def _recording_id(mat: dict) -> Optional[str]:
    """Return the CWRU file number embedded in a .mat's variable names.

    e.g. a file with key ``X118_DE_time`` returns ``"118"``. Leading zeros are
    stripped so ``X097_DE_time`` returns ``"97"``, matching DEFAULT_CLASS_MAP.
    """
    for k in mat.keys():
        m = re.match(r"X(\d+)_(DE|FE|BA)_time$", k)
        if m:
            return str(int(m.group(1)))
    return None


def _segment_bounds(length: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Contiguous train/val/test sizes for a recording of ``length`` samples.

    Layout along the time axis: ``[ train | val | test ]``. Train is the
    remainder so the three sizes always sum to ``length``.
    """
    n_test = int(round(length * test_ratio))
    n_val = int(round(length * val_ratio))
    n_train = length - n_val - n_test
    return n_train, n_val, n_test


def load_cwru(
    data_root: str | Path,
    window_size: int = 1024,
    overlap: float = 0.5,
    channels: Sequence[str] = ("DE",),
    class_map: Optional[Dict[str, int]] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load CWRU .mat files and build a leak-free train/val/test split.

    The directory is scanned recursively for .mat files; each file is one
    recording whose class is read from its embedded ``X<id>`` key. Every
    recording's raw signal is cut into contiguous train/val/test segments and
    each segment is windowed independently, so overlapping windows never cross a
    split boundary.

    Args:
        data_root: directory containing .mat files (scanned recursively).
        window_size: samples per window.
        overlap: fraction overlap in [0, 1).
        channels: which signal channels to include, subset of ("DE", "FE", "BA").
        class_map: {"<filenumber>": class_idx}; defaults to DEFAULT_CLASS_MAP.
        val_ratio, test_ratio: per-recording time fractions for validation/test.
        seed: accepted for API compatibility; the split is deterministic and
            does not depend on it.

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes
        x_*: float32 (N, C, T);  y_*: int64 (N,)

    Raises:
        FileNotFoundError if data_root is missing or contains no usable files.
    """
    from scipy.io import loadmat  # local import: avoid scipy cost for synthetic users

    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(
            f"CWRU data root not found: {root}\n"
            "Download .mat files from https://engineering.case.edu/bearingdatacenter "
            "or use the cathysiyu/Mechanical-datasets 'dataset' folder."
        )

    if class_map is None:
        class_map = DEFAULT_CLASS_MAP

    stride = max(1, int(window_size * (1.0 - overlap)))

    mat_paths = sorted(root.rglob("*.mat"))
    if not mat_paths:
        raise FileNotFoundError(f"No .mat files found under {root} (searched recursively).")

    # split -> ([x chunks], [y chunks])
    splits: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "train": ([], []), "val": ([], []), "test": ([], []),
    }
    n_recordings = 0

    for mat_path in mat_paths:
        mat = loadmat(str(mat_path))
        rec_id = _recording_id(mat)
        if rec_id is None or rec_id not in class_map:
            continue  # not a CWRU recording we have a label for
        label = class_map[rec_id]

        per_channel = []
        for ch_name in channels:
            key = _find_key(mat, f"_{ch_name}_time")
            if key is None:
                break  # this file lacks this channel; skip whole file
            sig = np.asarray(mat[key]).squeeze().astype(np.float32)
            per_channel.append(sig)
        if len(per_channel) != len(channels):
            continue

        # Align channel lengths to the minimum, then split along time.
        min_len = min(len(s) for s in per_channel)
        per_channel = [s[:min_len] for s in per_channel]
        n_train, n_val, n_test = _segment_bounds(min_len, val_ratio, test_ratio)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, min_len),
        }
        for split, (lo, hi) in bounds.items():
            windowed = [_window_signal(s[lo:hi], window_size, stride) for s in per_channel]
            stacked = np.stack(windowed, axis=1)  # (N, C, T)
            if len(stacked) == 0:
                raise ValueError(
                    f"{mat_path.name}: {split} segment ({hi - lo} samples) is too "
                    f"short for one window of {window_size}."
                )
            splits[split][0].append(stacked)
            splits[split][1].append(np.full(len(stacked), label, dtype=np.int64))
        n_recordings += 1

    if n_recordings == 0:
        raise FileNotFoundError(
            f"No labelled CWRU recordings found under {root}. Expected .mat files "
            "whose variable names embed a file number listed in DEFAULT_CLASS_MAP."
        )

    x_train = np.concatenate(splits["train"][0], axis=0)
    y_train = np.concatenate(splits["train"][1], axis=0)
    x_val = np.concatenate(splits["val"][0], axis=0)
    y_val = np.concatenate(splits["val"][1], axis=0)
    x_test = np.concatenate(splits["test"][0], axis=0)
    y_test = np.concatenate(splits["test"][1], axis=0)

    num_classes = max(class_map.values()) + 1
    found = len(np.unique(y_train))
    if found < num_classes:
        print(f"[cwru] warning: only {found}/{num_classes} classes found under {root}")

    # Per-channel z-score normalization using training statistics only.
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_val, y_val, x_test, y_test, num_classes


def make_synthetic_cwru(
    num_classes: int = 10,
    samples_per_class: int = 200,
    num_channels: int = 2,
    window_size: int = 1024,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """CWRU-shaped synthetic dataset for end-to-end smoke testing.

    Each class is a mixture of sinusoids at class-specific frequencies plus noise.
    Samples are independent, so a plain random train/val/test split is leak-free.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1.0, window_size, endpoint=False, dtype=np.float32)
    xs, ys = [], []
    for c in range(num_classes):
        base_freq = 20.0 + 10.0 * c  # 20, 30, 40, ... Hz
        for _ in range(samples_per_class):
            chans = []
            for ch in range(num_channels):
                phase = rng.uniform(0, 2 * np.pi)
                amp = 1.0 + 0.2 * rng.standard_normal()
                sig = amp * np.sin(2 * np.pi * base_freq * (1 + 0.05 * ch) * t + phase)
                sig = sig + 0.3 * rng.standard_normal(window_size).astype(np.float32)
                chans.append(sig.astype(np.float32))
            xs.append(np.stack(chans, axis=0))
            ys.append(c)
    x = np.stack(xs, axis=0)
    y = np.array(ys, dtype=np.int64)

    # shuffle and split into train/val/test
    idx = rng.permutation(len(x))
    x, y = x[idx], y[idx]
    n_test = int(round(len(x) * test_ratio))
    n_val = int(round(len(x) * val_ratio))
    x_test, y_test = x[:n_test], y[:n_test]
    x_val, y_val = x[n_test:n_test + n_val], y[n_test:n_test + n_val]
    x_train, y_train = x[n_test + n_val:], y[n_test + n_val:]

    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    return x_train, y_train, x_val, y_val, x_test, y_test, num_classes
