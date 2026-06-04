"""OPS-Bench: a minimal, self-contained example.

Demonstrates the evaluation protocol and the training recipes from the paper
"Robust Industrial Fault Diagnosis under Observation-Pattern Shift". It

  1. trains a fixed-severity baseline and the three cross-severity coverage
     recipes from the paper, all on the same data and TCN backbone:
       * MixedMask   -- the natural defense (baseline): one mixed-pattern
         masked view per sample at a single severity (--train-rate);
       * weak/strong -- paired two-view coverage: each sample is shown at two
         severities per step, a heavy "strong" view (mixed mask at
         --train-rate) and a near-clean "weak" view (mixed mask at
         --weak-rate), both supervised by cross-entropy. This is the canonical
         lambda=0 form, so there is no consistency/KL term and no mask-channel
         input -- the paired weak/strong CE is the whole recipe;
       * two-point   -- single-view coverage: each step shows one view whose
         severity is --train-rate or --weak-rate with probability 1/2 each;
       * random-rate -- single-view coverage: each step shows one view whose
         severity is drawn from Uniform(0, --train-rate);
  2. evaluates all four under OPS-Bench -- clean test data plus the four
     observation-degradation patterns (random point / temporal block / channel
     drop / mixed) at --test-rate, with per-sample independent masks;
  3. reports per-pattern accuracy, WOPA (worst case over all patterns) and AOPD
     (average drop from clean), plus each coverage recipe's WOPA gain over the
     fixed-severity baseline.

Quick smoke test (no data needed, CPU, ~1 minute):

    python example.py --dataset synthetic --epochs 3 --device cpu

Real run on CWRU (after `python scripts/download_cwru.py`):

    python example.py --dataset cwru --data-root data/cwru --channels DE FE \
        --epochs 30 --train-rate 0.5 --test-rate 0.5 --device cuda

To benchmark your own backbone, see "Use OPS-Bench on your own model" in
README.md -- the only contract is an nn.Module mapping (B, C, T) -> (B, K).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.cwru import load_cwru, make_synthetic_cwru  # noqa: E402
from src.masks import (  # noqa: E402
    channel_mask, mixed_mask, random_point_mask, temporal_block_mask,
)
from src.metrics import average_pattern_drop, worst_pattern_accuracy  # noqa: E402
from src.models.tcn import TCN  # noqa: E402
from src.training.trainer import (  # noqa: E402
    MaskAugDataset, MaskAugTwoViewDataset, apply_mask_per_sample, evaluate,
    set_seed, train_classifier, train_classifier_two_view,
)

PATTERNS = ("random", "block", "channel", "mixed")


def build_mask_fn(pattern: str, rate: float):
    """Return a sampler mask_fn(shape, rng) -> (C, T) for one pattern at one severity.

    The single ``rate`` knob drives each family naturally: Bernoulli drop
    probability (random), dropped fraction of the time axis (block), per-channel
    drop probability (channel), and all three components together (mixed).
    """
    if pattern == "random":
        return lambda shape, rng: random_point_mask(shape, missing_rate=rate, rng=rng)
    if pattern == "block":
        return lambda shape, rng: temporal_block_mask(shape, block_ratio=rate, rng=rng)
    if pattern == "channel":
        return lambda shape, rng: channel_mask(shape, channel_missing_rate=rate, rng=rng)
    if pattern == "mixed":
        return lambda shape, rng: mixed_mask(shape, p_point=rate, r_block=rate,
                                             r_channel=rate, rng=rng)
    raise ValueError(f"unknown pattern: {pattern}")


def build_twopoint_mask_fn(r_strong: float, r_weak: float):
    """Single-view two-point coverage: each draw picks severity ``r_strong`` or
    ``r_weak`` with probability 1/2, then a mixed mask at that severity."""
    def fn(shape, rng):
        rate = r_strong if rng.random() < 0.5 else r_weak
        return mixed_mask(shape, p_point=rate, r_block=rate, r_channel=rate, rng=rng)
    return fn


def build_randomrate_mask_fn(r_strong: float):
    """Single-view random-rate coverage: each draw samples severity from
    Uniform(0, ``r_strong``), then a mixed mask at that severity."""
    def fn(shape, rng):
        rate = float(rng.uniform(0.0, r_strong))
        return mixed_mask(shape, p_point=rate, r_block=rate, r_channel=rate, rng=rng)
    return fn


def new_tcn(in_channels: int, num_classes: int) -> TCN:
    return TCN(num_inputs=in_channels, num_classes=num_classes,
               num_channels=(64, 64, 64, 64), kernel_size=7, dropout=0.2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OPS-Bench minimal example")
    p.add_argument("--dataset", default="synthetic", choices=["synthetic", "cwru"],
                   help="synthetic needs no data; cwru reads .mat files from --data-root")
    p.add_argument("--data-root", default="data/cwru")
    p.add_argument("--channels", nargs="+", default=["DE", "FE"],
                   help="CWRU channels, subset of {DE, FE, BA}")
    p.add_argument("--window-size", type=int, default=1024)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-rate", type=float, default=0.5,
                   help="strong (heavy) training severity; also the baseline severity")
    p.add_argument("--weak-rate", type=float, default=0.1,
                   help="severity of the near-clean weak training view")
    p.add_argument("--test-rate", type=float, default=0.5,
                   help="severity applied to every test pattern")
    p.add_argument("--device", default="cuda",
                   help="cuda or cpu (auto-falls back to cpu if cuda is unavailable)")
    return p.parse_args()


def evaluate_ops_bench(model, x_test, y_test, test_deg, patterns, *, batch_size, device):
    """Clean + per-pattern accuracy, then WOPA and AOPD. Returns a result dict."""
    clean = evaluate(model, x_test, y_test, batch_size=batch_size, device=device)
    per_pattern = {p: evaluate(model, test_deg[p], y_test,
                               batch_size=batch_size, device=device)
                   for p in patterns}
    return {
        "clean": clean,
        "per_pattern": per_pattern,
        "WOPA": worst_pattern_accuracy(per_pattern),
        "AOPD": average_pattern_drop(clean, per_pattern),
    }


def train_single_view(x_train, y_train, x_val, y_val, in_channels, num_classes,
                      mask_fn, args):
    """One masked view per sample from ``mask_fn``, plain cross-entropy.

    Shared by the fixed-severity MixedMask baseline and the single-view coverage
    recipes (two-point, random-rate); they differ only in ``mask_fn``.
    """
    ds = MaskAugDataset(x_train, y_train, mask_fn, concat_mask=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = new_tcn(in_channels, num_classes)
    model, _ = train_classifier(
        model, x_train, y_train, x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed, train_loader=loader,
    )
    return model


def train_weak_strong(x_train, y_train, x_val, y_val, in_channels, num_classes,
                      strong_mask_fn, weak_mask_fn, args):
    """Weak/strong recipe: paired strong + weak views, both under CE (canonical lambda=0)."""
    ds = MaskAugTwoViewDataset(x_train, y_train, strong_mask_fn,
                               concat_mask=False, weak_mask_fn=weak_mask_fn)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = new_tcn(in_channels, num_classes)
    model, _ = train_classifier_two_view(
        model, loader, x_val, y_val,
        consistency_weight=0.0, consistency_mode="asymmetric",
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, seed=args.seed,
    )
    return model


def print_row(name, res, patterns):
    cells = "  ".join(f"{res['per_pattern'][p]:>8.4f}" for p in patterns)
    print(f"{name:<11}  {cells}  {res['clean']:>8.4f}  "
          f"{res['WOPA']:>8.4f}  {res['AOPD']:>8.4f}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    # 1. data ---------------------------------------------------------------
    if args.dataset == "synthetic":
        data = make_synthetic_cwru(num_channels=len(args.channels),
                                   window_size=args.window_size, seed=args.seed)
    else:
        data = load_cwru(data_root=args.data_root, window_size=args.window_size,
                         overlap=args.overlap, channels=tuple(args.channels),
                         seed=args.seed)
    x_train, y_train, x_val, y_val, x_test, y_test, num_classes = data
    in_channels = x_train.shape[1]
    print(f"[data] {args.dataset}: train={x_train.shape} val={x_val.shape} "
          f"test={x_test.shape} classes={num_classes}")

    patterns = list(PATTERNS)
    if in_channels < 2 and "channel" in patterns:
        print("[warn] channel drop needs C >= 2; dropping the 'channel' pattern.")
        patterns.remove("channel")

    # 2. pre-generate the degraded test sets ONCE, so every model is scored
    #    on identical masked inputs (a fair, paired comparison) ------------
    test_rng = np.random.default_rng(args.seed + 1000)
    test_deg = {p: apply_mask_per_sample(x_test, build_mask_fn(p, args.test_rate), test_rng)
                for p in patterns}

    strong_mask_fn = build_mask_fn("mixed", args.train_rate)
    weak_mask_fn = build_mask_fn("mixed", args.weak_rate)
    twopoint_mask_fn = build_twopoint_mask_fn(args.train_rate, args.weak_rate)
    randomrate_mask_fn = build_randomrate_mask_fn(args.train_rate)

    # 3. train the baseline and the three coverage recipes ------------------
    print(f"\n[train] MixedMask baseline  (mixed mask @ rate {args.train_rate})")
    mixed_model = train_single_view(x_train, y_train, x_val, y_val, in_channels,
                                    num_classes, strong_mask_fn, args)
    print(f"\n[train] weak/strong  (strong @ {args.train_rate} + weak @ {args.weak_rate}, "
          f"canonical lambda=0)")
    ws_model = train_weak_strong(x_train, y_train, x_val, y_val, in_channels,
                                 num_classes, strong_mask_fn, weak_mask_fn, args)
    print(f"\n[train] two-point  (severity in {{{args.weak_rate}, {args.train_rate}}}, single view)")
    tp_model = train_single_view(x_train, y_train, x_val, y_val, in_channels,
                                 num_classes, twopoint_mask_fn, args)
    print(f"\n[train] random-rate  (severity ~ U(0, {args.train_rate}), single view)")
    rr_model = train_single_view(x_train, y_train, x_val, y_val, in_channels,
                                 num_classes, randomrate_mask_fn, args)

    # 4. evaluate every model on OPS-Bench ---------------------------------
    def score(model):
        return evaluate_ops_bench(model, x_test, y_test, test_deg, patterns,
                                  batch_size=args.batch_size, device=args.device)

    mixed_res = score(mixed_model)
    ws_res = score(ws_model)
    tp_res = score(tp_model)
    rr_res = score(rr_model)

    # 5. report -------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"OPS-Bench results   (test severity = {args.test_rate})")
    print("=" * 78)
    header = f"{'method':<11}  " + "  ".join(f"{p:>8}" for p in patterns) + \
             f"  {'clean':>8}  {'WOPA':>8}  {'AOPD':>8}"
    print(header)
    print_row("MixedMask", mixed_res, patterns)
    print_row("weak/strong", ws_res, patterns)
    print_row("two-point", tp_res, patterns)
    print_row("random-rate", rr_res, patterns)
    print("-" * 78)
    print("delta-WOPA vs MixedMask baseline (fixed severity):")
    for name, res in (("weak/strong", ws_res), ("two-point", tp_res),
                      ("random-rate", rr_res)):
        d = res["WOPA"] - mixed_res["WOPA"]
        print(f"  {name:<12} {d:+.4f}   (clean {res['clean']:.4f})")
    print("WOPA = worst accuracy over all patterns (higher is better); "
          "AOPD = mean drop from clean (lower is better).")


if __name__ == "__main__":
    main()
