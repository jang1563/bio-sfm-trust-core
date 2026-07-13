"""Pure-stdlib ranking statistics (AUROC, Pearson).

Provenance: extracted from bio-sfm-trust-audit
(experiments/trust_cue_attribution/phase1c_specialist_metric_check.py). No numpy.
"""

from __future__ import annotations

import math
from typing import Optional


def _finite_values(values: list[float], name: str) -> list[float]:
    checked: list[float] = []
    for index, value in enumerate(values):
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        checked.append(number)
    return checked


def _binary_labels(labels: list[int]) -> list[int]:
    checked: list[int] = []
    for index, label in enumerate(labels):
        if not isinstance(label, (bool, int)) or label not in (0, 1):
            raise ValueError(f"labels[{index}] must be binary (0 or 1)")
        checked.append(int(label))
    return checked


def _avg_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def auroc(scores: list[float], labels: list[int]) -> Optional[float]:
    """Mann-Whitney AUROC of `scores` against binary `labels`. None if degenerate."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must have equal length")
    checked_scores = _finite_values(scores, "scores")
    checked_labels = _binary_labels(labels)
    pos = sum(checked_labels)
    neg = len(checked_labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _avg_ranks(checked_scores)
    sum_pos = sum(r for r, label in zip(ranks, checked_labels) if label == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def pearson(x: list[float], y: list[float]) -> Optional[float]:
    """Pearson correlation. None if n<2 or a variable is constant."""
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    checked_x = _finite_values(x, "x")
    checked_y = _finite_values(y, "y")
    n = len(checked_x)
    if n < 2:
        return None
    mx = sum(checked_x) / n
    my = sum(checked_y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(checked_x, checked_y))
    sxx = sum((a - mx) ** 2 for a in checked_x)
    syy = sum((b - my) ** 2 for b in checked_y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)
