"""Conformal / RCPS risk control for the trust gate's accept ("trust") route.

The lambda-threshold (verify iff risk > lambda) trades verification cost against benefit, but gives
NO guarantee on how often a TRUSTED design is actually wrong. `rcps_threshold` chooses, from
held-out calibration data, the most permissive calibrated-risk threshold tau whose accept set
{risk <= tau} has a false-accept rate controlled at <= alpha -- the RCPS idea (Bates et al. 2021,
"Distribution-Free, Risk-Controlling Prediction Sets"): each candidate threshold's false-accept
rate is a mean of Bernoulli outcomes, and we keep the largest threshold whose one-sided Hoeffding
upper confidence bound (at level delta) is <= alpha.

HONESTY NOTE: this uses a POINTWISE Hoeffding UCB (ln(1/delta)), the standard selective-risk
construction, NOT a Bonferroni/Bentkus correction over the whole threshold grid -- so the
finite-sample guarantee is the usual RCPS one rather than a fully grid-corrected bound. The
accompanying test verifies EMPIRICALLY that the held-out false-accept rate is controlled at alpha.

Trusting only `risk <= tau` then carries a controlled false-accept rate, instead of an uncalibrated
cost heuristic. Returns None when no threshold can be certified (the signal is too weak or alpha
too strict) -- the caller should then trust nothing in that regime. Pure standard library.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence


def false_accept_rate(risks: Sequence[float], wrong: Sequence[int], tau: float) -> Optional[float]:
    """Empirical P(wrong | risk <= tau). None if the accept set {risk <= tau} is empty."""
    accepted = [w for r, w in zip(risks, wrong) if r <= tau]
    return (sum(accepted) / len(accepted)) if accepted else None


def rcps_threshold(
    risks: List[float],
    wrong: List[int],
    alpha: float,
    delta: float = 0.1,
) -> Optional[float]:
    """Largest calibrated-risk threshold tau whose accept set {risk <= tau} has false-accept rate
    <= alpha with probability >= 1 - delta.

    For each candidate threshold tau (the distinct risk values), the false-accept rate of the
    accept set is a mean of n Bernoulli(<=1) outcomes; its one-sided Hoeffding upper confidence
    bound at level delta is rhat + sqrt(ln(1/delta) / (2n)). We return the LARGEST tau whose UCB
    <= alpha (maximal coverage subject to the risk being controlled), or None if none qualifies.
    """
    pairs = sorted(zip([float(r) for r in risks], [int(w) for w in wrong]))
    if not pairs:
        return None
    slack_num = math.log(1.0 / delta)
    best: Optional[float] = None
    for tau in sorted({r for r, _ in pairs}):
        accepted = [w for r, w in pairs if r <= tau]
        n = len(accepted)
        if n == 0:
            continue
        rhat = sum(accepted) / n
        ucb = rhat + math.sqrt(slack_num / (2 * n))
        if ucb <= alpha:
            best = tau  # keep climbing; the largest certified tau gives the most coverage
    return best
