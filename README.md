# OPS-Bench & OPACT

Code for the paper "Cross-Severity Training-Distribution Coverage for Robust
Industrial Fault Diagnosis under Observation-Pattern Shift".

**OPS-Bench** is a benchmark for robustness to test-time sensor missingness. It
defines four mask families (random point, temporal block, channel drop, mixed)
and scores a model by worst-case accuracy over the patterns (WOPA) and average
drop from clean (AOPD). **OPACT** is the training recipe. Each sample is shown
at two severities per step, a heavy view and a near-clean view, both supervised
by cross-entropy. The canonical form adds no consistency term and no
mask-channel input.

## Install

Install torch for your platform (CPU or CUDA build) from pytorch.org, then:

    pip install -r requirements.txt

## Run

Synthetic smoke test (no data, CPU, about a minute):

    python example.py --dataset synthetic --epochs 3 --device cpu

CWRU:

    python scripts/download_cwru.py
    python example.py --dataset cwru --data-root data/cwru --channels DE FE \
        --epochs 30 --train-rate 0.5 --test-rate 0.5 --device cuda

`example.py` trains a MixedMask baseline and OPACT and prints per-pattern
accuracy with WOPA and AOPD for each.

## Use OPS-Bench on your own model

Any model mapping `(B, C, T)` to `(B, num_classes)` logits works:

```python
import numpy as np
from src.masks import random_point_mask, temporal_block_mask, channel_mask, mixed_mask
from src.training.trainer import apply_mask_per_sample, evaluate
from src.metrics import worst_pattern_accuracy, average_pattern_drop

rng = np.random.default_rng(0)
r = 0.5
patterns = {
    "random":  lambda s, g: random_point_mask(s, missing_rate=r, rng=g),
    "block":   lambda s, g: temporal_block_mask(s, block_ratio=r, rng=g),
    "channel": lambda s, g: channel_mask(s, channel_missing_rate=r, rng=g),
    "mixed":   lambda s, g: mixed_mask(s, p_point=r, r_block=r, r_channel=r, rng=g),
}
clean = evaluate(model, x_test, y_test, device="cuda")
acc = {k: evaluate(model, apply_mask_per_sample(x_test, fn, rng), y_test, device="cuda")
       for k, fn in patterns.items()}
print(worst_pattern_accuracy(acc), average_pattern_drop(clean, acc))
```

To train with OPACT, build a `MaskAugTwoViewDataset` and call
`train_classifier_two_view(..., consistency_mode="asymmetric", consistency_weight=0)`,
as in `example.py`.

## Layout

```
example.py                 demo (MixedMask vs OPACT on OPS-Bench)
scripts/download_cwru.py   CWRU downloader
src/masks.py               the four mask families
src/metrics.py             WOPA and AOPD
src/data/                  CWRU (leak-free split), synthetic, SEU loaders
src/models/                TCN, ResNet1D
src/training/trainer.py    training (including OPACT) and masked evaluation
```

CWRU data and run outputs are git-ignored. MIT licensed (see LICENSE).
