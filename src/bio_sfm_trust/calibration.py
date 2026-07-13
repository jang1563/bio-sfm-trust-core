"""Isotonic calibration of a raw risk signal to P(wrong).

Provenance: extracted from bio-sfm-trust-audit
(experiments/trust_cue_attribution/phase2_calibrated_gate.py). The raw gate uses
risk = 1 - confidence, which is compressed near 0, so the principled
"verify iff risk > lambda" rule never fires and the policy degenerates to
trust-all. A monotonic isotonic fit risk -> P(wrong) (leave-one-out, no in-sample
overfit) lands the calibrated risk on the P(wrong) scale so the verify rule
triggers where the specialist is actually likely wrong.

A single isotonic map does not reverse ordering, but pooled ties can change a
tie-sensitive ranking metric such as AUROC. Leave-one-out predictions also come
from different fitted maps, so their ranking must not be assumed invariant.
Pure standard library.
"""

from __future__ import annotations

import bisect
import math
from itertools import groupby
from numbers import Real
from typing import Callable, Sequence


def _weighted_pava(values: list[float], weights: list[int]) -> list[float]:
    """Weighted PAVA, returning one fitted value per supplied value/weight pair."""
    blocks: list[list[float]] = []  # [mean, total weight, number of input groups]
    for value, weight in zip(values, weights):
        blocks.append([value, float(weight), 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            mean1, weight1, groups1 = blocks.pop()
            mean0, weight0, groups0 = blocks.pop()
            total_weight = weight0 + weight1
            pooled = math.fsum((mean0 * weight0, mean1 * weight1)) / total_weight
            blocks.append([pooled, total_weight, groups0 + groups1])
    out: list[float] = []
    for mean, _, groups in blocks:
        out.extend([mean] * int(groups))
    return out


def _pava(values: list[float]) -> list[float]:
    """Pool-adjacent-violators isotonic fit (non-decreasing), values pre-sorted by x."""
    return _weighted_pava(values, [1] * len(values))


def _validated_calibration_data(
    x: Sequence[Real],
    y: Sequence[Real],
) -> tuple[list[float], list[float]]:
    if len(x) != len(y):
        raise ValueError("x and y must have equal length")
    if not x:
        raise ValueError("x and y must be non-empty")

    xs: list[float] = []
    ys: list[float] = []
    for index, value in enumerate(x):
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"x[{index}] must be a finite number")
        xs.append(float(value))
    for index, value in enumerate(y):
        if not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"y[{index}] must be a finite probability")
        probability = float(value)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"y[{index}] must be between 0 and 1")
        ys.append(probability)
    return xs, ys


def isotonic_calibrator(x: list[float], y: list[float]) -> Callable[[float], float]:
    """Fit monotonic x -> y (isotonic); return a step predictor for new x."""
    x_values, y_values = _validated_calibration_data(x, y)
    pairs = sorted(zip(x_values, y_values), key=lambda pair: pair[0])

    # Isotonic regression assigns one value to one predictor location.  Collapse
    # duplicate x values before PAVA, retaining their sample counts as weights;
    # otherwise a stable sort makes the fit depend on the input order of tied rows.
    xs: list[float] = []
    tie_means: list[float] = []
    tie_weights: list[int] = []
    for x_value, group in groupby(pairs, key=lambda pair: pair[0]):
        targets = [target for _, target in group]
        xs.append(x_value)
        tie_means.append(math.fsum(targets) / len(targets))
        tie_weights.append(len(targets))
    fitted = _weighted_pava(tie_means, tie_weights)

    def predict(xq: float) -> float:
        if not isinstance(xq, Real) or not math.isfinite(float(xq)):
            raise ValueError("xq must be a finite number")
        idx = bisect.bisect_right(xs, float(xq)) - 1
        if idx < 0:
            return fitted[0]
        if idx >= len(fitted):
            return fitted[-1]
        return fitted[idx]

    return predict


def loo_calibrated_risks(raw_risks: list[float], wrong: list[int]) -> list[float]:
    """Leave-one-out isotonic calibration of raw risk -> P(wrong)."""
    if any(target not in (0, 1) for target in wrong):
        raise ValueError("wrong values must be binary (0 or 1)")
    risks, targets = _validated_calibration_data(raw_risks, wrong)
    n = len(risks)
    out: list[float] = []
    for i in range(n):
        x = [risks[j] for j in range(n) if j != i]
        y = [targets[j] for j in range(n) if j != i]
        out.append(isotonic_calibrator(x, y)(risks[i]) if x else 0.0)
    return out
