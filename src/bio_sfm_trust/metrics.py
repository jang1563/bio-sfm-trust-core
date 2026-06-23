"""Pure-stdlib ranking statistics (AUROC, Pearson).

Provenance: extracted from bio-sfm-trust-audit
(experiments/trust_cue_attribution/phase1c_specialist_metric_check.py). No numpy.
"""

from __future__ import annotations

from typing import Optional


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
    pos = sum(1 for l in labels if l == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = _avg_ranks(scores)
    sum_pos = sum(r for r, l in zip(ranks, labels) if l == 1)
    return (sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def pearson(x: list[float], y: list[float]) -> Optional[float]:
    """Pearson correlation. None if n<2 or a variable is constant."""
    n = len(x)
    if n < 2:
        return None
    mx = sum(x) / n
    my = sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx ** 0.5 * syy ** 0.5)
