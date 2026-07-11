"""Selective-risk threshold learning and independent certification.

The lambda threshold trades verification cost against benefit but does not bound how often a
trusted design is wrong. A valid finite-sample certificate needs to separate threshold selection
from evaluation. This module therefore provides a learn-then-test path:

1. select a candidate threshold on a fit split;
2. freeze that threshold;
3. validate its accepted set on an independent certification split with a predeclared one-sided bound.

`rcps_threshold` remains as a backward-compatible, exploratory selector. Because it searches a
threshold grid and evaluates candidates on the same observations, its pointwise UCB must not be
reported as a distribution-free certificate. Use `split_ltt_threshold` for certified decisions.
Pure standard library.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence


def _validate_inputs(risks: Sequence[float], wrong: Sequence[int]) -> None:
    if len(risks) != len(wrong):
        raise ValueError("risks and wrong must have equal length")
    if any(int(w) not in (0, 1) for w in wrong):
        raise ValueError("wrong values must be binary (0 or 1)")


def _validate_probability(name: str, value: float) -> None:
    if not 0.0 < float(value) < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")


def false_accept_rate(risks: Sequence[float], wrong: Sequence[int], tau: float) -> Optional[float]:
    """Empirical P(wrong | risk <= tau). None if the accept set {risk <= tau} is empty."""
    _validate_inputs(risks, wrong)
    accepted = [w for r, w in zip(risks, wrong) if r <= tau]
    return (sum(accepted) / len(accepted)) if accepted else None


def hoeffding_upper_bound(empirical_rate: float, n: int, delta: float = 0.1) -> float:
    """One-sided Hoeffding upper bound for a fixed Bernoulli selection rule."""
    _validate_probability("delta", delta)
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 <= float(empirical_rate) <= 1.0:
        raise ValueError("empirical_rate must be between 0 and 1")
    return min(1.0, float(empirical_rate) + math.sqrt(math.log(1.0 / delta) / (2 * n)))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """P(X <= k) for X ~ Binomial(n, probability), using the standard library."""
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    q = 1.0 - probability
    if k <= n // 2:
        term = q ** n
        total = term
        odds = probability / q
        for value in range(1, k + 1):
            term *= ((n - value + 1) / value) * odds
            total += term
        return min(1.0, total)

    # Sum the shorter upper tail from X=n downward, then complement. This avoids
    # underflow in q**n when both p and k are close to one and n is moderate.
    term = probability ** n
    upper_tail = term
    reverse_odds = q / probability
    for value in range(n, k + 1, -1):
        term *= (value / (n - value + 1)) * reverse_odds
        upper_tail += term
    return max(0.0, 1.0 - min(1.0, upper_tail))


def clopper_pearson_upper_bound(
    false_accepts: int,
    n: int,
    delta: float = 0.1,
) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial error rate."""
    _validate_probability("delta", delta)
    if n <= 0:
        raise ValueError("n must be positive")
    if isinstance(false_accepts, bool) or int(false_accepts) != false_accepts:
        raise ValueError("false_accepts must be an integer")
    false_accepts = int(false_accepts)
    if not 0 <= false_accepts <= n:
        raise ValueError("false_accepts must be between 0 and n")
    if false_accepts == n:
        return 1.0

    # Invert P_p(X <= false_accepts) = delta. The CDF is monotone decreasing in p.
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if _binomial_cdf(false_accepts, n, midpoint) > delta:
            low = midpoint
        else:
            high = midpoint
    return high


def _upper_bound(false_accepts: int, n: int, delta: float, bound: str) -> float:
    if bound == "hoeffding":
        return hoeffding_upper_bound(false_accepts / n, n, delta)
    if bound == "clopper_pearson":
        return clopper_pearson_upper_bound(false_accepts, n, delta)
    raise ValueError("bound must be 'hoeffding' or 'clopper_pearson'")


def validate_fixed_threshold(
    risks: Sequence[float],
    wrong: Sequence[int],
    tau: float,
    alpha: float,
    delta: float = 0.1,
    bound: str = "hoeffding",
) -> Dict[str, Any]:
    """Validate one pre-specified threshold on an independent certification sample."""
    _validate_inputs(risks, wrong)
    _validate_probability("alpha", alpha)
    _validate_probability("delta", delta)
    accepted = [int(w) for r, w in zip(risks, wrong) if float(r) <= float(tau)]
    n_accepted = len(accepted)
    if not accepted:
        return {
            "tau": float(tau),
            "n": len(risks),
            "n_accepted": 0,
            "false_accepts": 0,
            "empirical_false_accept_rate": None,
            "ucb": None,
            "alpha": float(alpha),
            "delta": float(delta),
            "certified": False,
            "reason": "empty_certification_accept_set",
        }
    false_accepts = sum(accepted)
    empirical = false_accepts / n_accepted
    ucb = _upper_bound(false_accepts, n_accepted, delta, bound)
    certified = ucb <= alpha
    return {
        "tau": float(tau),
        "n": len(risks),
        "n_accepted": n_accepted,
        "false_accepts": false_accepts,
        "empirical_false_accept_rate": empirical,
        "ucb": ucb,
        "alpha": float(alpha),
        "delta": float(delta),
        "certified": certified,
        "bound": bound,
        "reason": "certified" if certified else f"{bound}_ucb_exceeds_alpha",
    }


def select_candidate_threshold(
    risks: Sequence[float],
    wrong: Sequence[int],
    alpha: float,
    delta: float = 0.1,
) -> Optional[float]:
    """Select a conservative candidate tau on fit data; this step is not a certificate."""
    _validate_inputs(risks, wrong)
    _validate_probability("alpha", alpha)
    _validate_probability("delta", delta)
    pairs = sorted(zip([float(r) for r in risks], [int(w) for w in wrong]))
    best: Optional[float] = None
    for tau in sorted({r for r, _ in pairs}):
        accepted = [w for r, w in pairs if r <= tau]
        empirical = sum(accepted) / len(accepted)
        if hoeffding_upper_bound(empirical, len(accepted), delta) <= alpha:
            best = tau
    return best


def select_empirical_threshold(
    risks: Sequence[float],
    wrong: Sequence[int],
    alpha: float,
) -> Optional[float]:
    """Largest fit-split threshold with empirical selective risk at most alpha.

    This is a learning rule only. Its output becomes certifiable only after independent validation.
    """
    _validate_inputs(risks, wrong)
    _validate_probability("alpha", alpha)
    pairs = sorted(zip([float(r) for r in risks], [int(w) for w in wrong]))
    best: Optional[float] = None
    for tau in sorted({r for r, _ in pairs}):
        accepted = [w for r, w in pairs if r <= tau]
        if sum(accepted) / len(accepted) <= alpha:
            best = tau
    return best


def split_ltt_threshold(
    fit_risks: Sequence[float],
    fit_wrong: Sequence[int],
    certification_risks: Sequence[float],
    certification_wrong: Sequence[int],
    alpha: float,
    delta: float = 0.1,
    bound: str = "hoeffding",
) -> Dict[str, Any]:
    """Learn a threshold on fit data and certify the frozen rule on independent data."""
    if bound not in ("hoeffding", "clopper_pearson"):
        raise ValueError("bound must be 'hoeffding' or 'clopper_pearson'")
    method = f"split_learn_then_test_{bound}"
    candidate = select_empirical_threshold(fit_risks, fit_wrong, alpha)
    if candidate is None:
        return {
            "method": method,
            "tau_candidate": None,
            "tau": None,
            "alpha": float(alpha),
            "delta": float(delta),
            "n_fit": len(fit_risks),
            "n_certification": len(certification_risks),
            "certified": False,
            "reason": "no_candidate_threshold_on_fit_split",
            "certification": None,
        }
    certification = validate_fixed_threshold(
        certification_risks,
        certification_wrong,
        candidate,
        alpha,
        delta,
        bound,
    )
    certified = bool(certification["certified"])
    return {
        "method": method,
        "tau_candidate": candidate,
        "tau": candidate if certified else None,
        "alpha": float(alpha),
        "delta": float(delta),
        "n_fit": len(fit_risks),
        "n_certification": len(certification_risks),
        "certified": certified,
        "reason": certification["reason"],
        "certification": certification,
    }


def rcps_threshold(
    risks: List[float],
    wrong: List[int],
    alpha: float,
    delta: float = 0.1,
) -> Optional[float]:
    """Backward-compatible exploratory threshold selector.

    The returned threshold has a pointwise Hoeffding screen on the same data used for grid search.
    It is useful for planning but is not, by itself, a distribution-free certificate.
    """
    return select_candidate_threshold(risks, wrong, alpha, delta)
