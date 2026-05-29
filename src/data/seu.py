"""Southeast University (SEU) gearbox dataset loader.

The SEU dataset (Shao et al., 2018) was collected on a Drivetrain Dynamic
Simulator. It ships as the "Mechanical-datasets" repository, with two
sub-datasets under ``gearbox/``:

  bearingset/  -- 5 bearing conditions: health, ball, comb(ination), inner, outer
  gearset/     -- 5 gear conditions:    Health, Chipped, Miss, Root, Surface

Each condition has two operating points (rotating speed 20 Hz - load 0 V, and
30 Hz - 2 V), stored as tab-separated .csv files with a short text header
followed by 8 channels of synchronously sampled vibration signal. Each
(condition, operating-point) .csv is treated as one recording.

Following common practice in the SEU literature, the bearing and gear
sub-datasets are treated as two *separate* 5-class problems; ``subset`` selects
which. Both operating points are pooled per class.

Split protocol (OPS-Bench). As in the CWRU loader, each recording's raw signal
is cut into contiguous train / validation / test segments BEFORE windowing, so
overlapping windows never straddle a split boundary. The split is deterministic.

Returned arrays match the CWRU loader's contract so the pipeline is
dataset-agnostic:
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes
    x_*: float32 (N, C, T);  y_*: int64 (N,)

Data: https://github.com/cathysiyu/Mechanical-datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SEU_NUM_CHANNELS = 8
SEU_CONDITIONS = ("20_0", "30_2")
# File-name stem -> class index. Bearing stems are lowercase, gear stems are
# capitalized, matching the on-disk file names.
SEU_BEARING_CLASSES = {"health": 0, "ball": 1, "comb": 2, "inner": 3, "outer": 4}
SEU_GEAR_CLASSES = {"Health": 0, "Chipped": 1, "Miss": 2, "Root": 3, "Surface": 4}


def _read_seu_csv(path: Path, n_channels: int = 8, max_rows: int = 300000) -> np.ndarray:
    """Parse one SEU .csv into an (n_channels, T) float32 array.

    The header is skipped implicitly: any line that does not yield at least
    ``n_channels`` parseable floats is dropped. Commas are normalised to tabs so
    the occasional comma-delimited file in this dataset is handled transparently.
    Reading stops after ``max_rows`` data rows.
    """
    rows = []
    with open(path, "r") as f:
        for ln in f:
            parts = [p for p in ln.replace(",", "\t").split("\t") if p.strip() != ""]
            if len(parts) < n_channels:
                continue
            try:
                vals = [float(p) for p in parts[:n_channels]]
            except ValueError:
                continue  # header / non-numeric line
            rows.append(vals)
            if len(rows) >= max_rows:
                break
    if not rows:
        raise ValueError(f"No numeric {n_channels}-channel data found in {path}")
    return np.asarray(rows, dtype=np.float32).T  # (n_channels, T)


def _window_multichannel(sig: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """Slice a (C, T_full) signal into (N, C, window_size) overlapping windows."""
    c, t_full = sig.shape
    n = (t_full - window_size) // stride + 1
    if n <= 0:
        return np.zeros((0, c, window_size), dtype=np.float32)
    out = np.empty((n, c, window_size), dtype=np.float32)
    for i in range(n):
        s = i * stride
        out[i] = sig[:, s:s + window_size]
    return out


def _segment_bounds(length: int, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    """Contiguous train/val/test sizes for a recording of ``length`` samples."""
    n_test = int(round(length * test_ratio))
    n_val = int(round(length * val_ratio))
    n_train = length - n_val - n_test
    return n_train, n_val, n_test


def load_seu(
    data_root: str | Path,
    subset: str = "bearing",
    window_size: int = 1024,
    overlap: float = 0.5,
    channels: Optional[Sequence[int]] = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 0,
    max_samples_per_file: int = 300000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load an SEU sub-dataset and build a leak-free train/val/test split.

    Each (condition, operating-point) .csv is one recording. Its raw signal is
    cut into contiguous train/val/test segments and each segment is windowed
    independently, so overlapping windows never cross a split boundary.

    Args:
        data_root: directory containing the ``gearbox/`` folder.
        subset: "bearing" or "gear" -- which 5-class sub-dataset to load.
        window_size: samples per window.
        overlap: fraction overlap in [0, 1).
        channels: 0-based channel indices to keep (default: all 8).
        val_ratio, test_ratio: per-recording time fractions for validation/test.
        seed: accepted for API compatibility; the split is deterministic.
        max_samples_per_file: cap on raw rows read per .csv (the files are huge).

    Returns:
        x_train, y_train, x_val, y_val, x_test, y_test, num_classes.

    Raw per-recording signals are cached as an .npz next to ``gearbox/`` so
    repeated calls (e.g. different seeds) skip the slow csv parse.
    """
    if subset not in ("bearing", "gear"):
        raise ValueError(f"subset must be 'bearing' or 'gear', got {subset!r}")

    root = Path(data_root)
    sub_dir = root / "gearbox" / f"{subset}set"
    if not sub_dir.exists():
        raise FileNotFoundError(
            f"SEU {subset} directory not found: {sub_dir}\n"
            "Expected <data_root>/gearbox/bearingset and .../gearset "
            "(the Mechanical-datasets layout)."
        )

    class_map = SEU_BEARING_CLASSES if subset == "bearing" else SEU_GEAR_CLASSES
    stride = max(1, int(window_size * (1.0 - overlap)))

    # Cache raw per-recording signals (windowing happens after the split, so the
    # cache must store raw signal, not windows).
    cache = sub_dir.parent / f"_seu_raw_{subset}_m{max_samples_per_file}.npz"
    recordings: Dict[str, np.ndarray] = {}
    if cache.exists():
        d = np.load(cache)
        recordings = {k: d[k] for k in d.files}
    else:
        for name in class_map:
            for cond in SEU_CONDITIONS:
                fp = sub_dir / f"{name}_{cond}.csv"
                if not fp.exists():
                    print(f"[seu] warning: missing {fp.name}")
                    continue
                sig = _read_seu_csv(fp, n_channels=SEU_NUM_CHANNELS,
                                    max_rows=max_samples_per_file)
                recordings[f"{name}_{cond}"] = sig
                print(f"[seu] {fp.name}: signal {sig.shape}")
        if not recordings:
            raise FileNotFoundError(f"No SEU {subset} csv files parsed under {sub_dir}")
        np.savez(cache, **recordings)
        print(f"[seu] cached {len(recordings)} raw recordings -> {cache.name}")

    # Split each recording by time, then window each segment.
    splits: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {
        "train": ([], []), "val": ([], []), "test": ([], []),
    }
    for rec_name, sig in sorted(recordings.items()):
        cls_name = rec_name.rsplit("_", 2)[0]  # "ball_20_0" -> "ball"
        if cls_name not in class_map:
            continue
        label = class_map[cls_name]
        if channels is not None:
            sig = sig[list(channels), :]
        length = sig.shape[1]
        n_train, n_val, n_test = _segment_bounds(length, val_ratio, test_ratio)
        bounds = {
            "train": (0, n_train),
            "val": (n_train, n_train + n_val),
            "test": (n_train + n_val, length),
        }
        for split, (lo, hi) in bounds.items():
            w = _window_multichannel(sig[:, lo:hi], window_size, stride)
            if len(w) == 0:
                raise ValueError(
                    f"{rec_name}: {split} segment ({hi - lo} samples) too short "
                    f"for one window of {window_size}."
                )
            splits[split][0].append(w)
            splits[split][1].append(np.full(len(w), label, dtype=np.int64))

    x_train = np.concatenate(splits["train"][0], axis=0)
    y_train = np.concatenate(splits["train"][1], axis=0)
    x_val = np.concatenate(splits["val"][0], axis=0)
    y_val = np.concatenate(splits["val"][1], axis=0)
    x_test = np.concatenate(splits["test"][0], axis=0)
    y_test = np.concatenate(splits["test"][1], axis=0)

    num_classes = max(class_map.values()) + 1

    # Per-channel z-score normalization using training statistics only.
    mean = x_train.mean(axis=(0, 2), keepdims=True)
    std = x_train.std(axis=(0, 2), keepdims=True) + 1e-8
    x_train = ((x_train - mean) / std).astype(np.float32)
    x_val = ((x_val - mean) / std).astype(np.float32)
    x_test = ((x_test - mean) / std).astype(np.float32)

    return (x_train, y_train.astype(np.int64), x_val, y_val.astype(np.int64),
            x_test, y_test.astype(np.int64), num_classes)
