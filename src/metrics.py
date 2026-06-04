"""Cross-pattern robustness metrics for OPS-Bench.

WOPA (Worst-case Observation-Pattern Accuracy): the minimum classification
accuracy over every observation degradation pattern a model is evaluated on.
The matched / mixed pattern is NOT excluded, so WOPA is a genuine worst case
across the full set of test patterns -- a lower bound on the accuracy a
deployed model can show under any of the modelled degradations.

(Earlier versions excluded the pattern matching the training distribution and
called the result "worst-case off-pattern accuracy". That exclusion hid the
mixed pattern, which is usually the hardest, from the worst-case figure; it has
been removed so the metric reports the true worst case.)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def worst_pattern_accuracy(acc_per_test_pattern: Dict[str, float]) -> float:
    """WOPA: minimum accuracy over every test pattern (nothing excluded)."""
    if not acc_per_test_pattern:
        raise ValueError("acc_per_test_pattern is empty")
    return float(min(acc_per_test_pattern.values()))


def average_pattern_drop(
    acc_clean: float,
    acc_per_test_pattern: Dict[str, float],
) -> float:
    """Mean accuracy drop from clean accuracy, averaged over every test pattern."""
    if not acc_per_test_pattern:
        raise ValueError("acc_per_test_pattern is empty")
    drops = [acc_clean - acc for acc in acc_per_test_pattern.values()]
    return float(np.mean(drops))


def cross_pattern_summary(
    results: Dict[tuple, float],
    train_patterns: List[str],
    test_patterns: List[str],
    acc_clean_per_train: Dict[str, float],
) -> List[Dict]:
    """Build a per-row WOPA / drop summary for a cross-pattern matrix.

    results: dict keyed by (train_pattern, test_pattern) -> accuracy.
    train_patterns: row labels, e.g. ["clean", "random", "block", "channel", "mixed"].
    test_patterns: column labels, e.g. ["random", "block", "channel", "mixed"].
    acc_clean_per_train: each train setting's accuracy on undegraded test data.

    Each output row carries the per-pattern accuracies (``test_<p>``), ``WOPA``
    (worst over ALL test patterns), and ``AOPD`` (mean drop from clean over all
    test patterns). No pattern is treated as in-distribution or excluded.
    """
    rows: List[Dict] = []
    for tp in train_patterns:
        row: Dict = {"train": tp}
        per_test: Dict[str, float] = {}
        for ep in test_patterns:
            acc = results.get((tp, ep))
            row[f"test_{ep}"] = acc
            if acc is not None:
                per_test[ep] = acc
        clean_acc = acc_clean_per_train.get(tp)

        row["WOPA"] = worst_pattern_accuracy(per_test) if per_test else None
        row["AOPD"] = (
            average_pattern_drop(clean_acc, per_test)
            if (clean_acc is not None and per_test)
            else None
        )
        rows.append(row)
    return rows
