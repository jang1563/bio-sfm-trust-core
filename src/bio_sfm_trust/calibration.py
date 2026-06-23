"""Isotonic calibration of a raw risk signal to P(wrong).

Provenance: extracted from bio-sfm-trust-audit
(experiments/trust_cue_attribution/phase2_calibrated_gate.py). The raw gate uses
risk = 1 - confidence, which is compressed near 0, so the principled
"verify iff risk > lambda" rule never fires and the policy degenerates to
trust-all. A monotonic isotonic fit risk -> P(wrong) (leave-one-out, no in-sample
overfit) lands the calibrated risk on the P(wrong) scale so the verify rule
triggers where the specialist is actually likely wrong.

Isotonic is monotonic, so it does NOT change a ranking metric like AUROC; it only
fixes the decision threshold/policy. Pure standard library.
"""

from __future__ import annotations

import bisect
from typing import Callable


def _pava(values: list[float]) -> list[float]:
    """Pool-adjacent-violators isotonic fit (non-decreasing), values pre-sorted by x."""
    blocks: list[list[float]] = []  # [mean, size]
    for v in values:
        blocks.append([v, 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            v1, s1 = blocks.pop()
            v0, s0 = blocks.pop()
            blocks.append([(v0 * s0 + v1 * s1) / (s0 + s1), s0 + s1])
    out: list[float] = []
    for mean, size in blocks:
        out.extend([mean] * int(size))
    return out


def isotonic_calibrator(x: list[float], y: list[float]) -> Callable[[float], float]:
    """Fit monotonic x -> y (isotonic); return a step predictor for new x."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    xs = [x[i] for i in order]
    fitted = _pava([y[i] for i in order])

    def predict(xq: float) -> float:
        idx = bisect.bisect_right(xs, xq) - 1
        if idx < 0:
            return fitted[0]
        if idx >= len(fitted):
            return fitted[-1]
        return fitted[idx]

    return predict


def loo_calibrated_risks(raw_risks: list[float], wrong: list[int]) -> list[float]:
    """Leave-one-out isotonic calibration of raw risk -> P(wrong)."""
    n = len(raw_risks)
    out: list[float] = []
    for i in range(n):
        x = [raw_risks[j] for j in range(n) if j != i]
        y = [float(wrong[j]) for j in range(n) if j != i]
        out.append(isotonic_calibrator(x, y)(raw_risks[i]) if x else 0.0)
    return out
