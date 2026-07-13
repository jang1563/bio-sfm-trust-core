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
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Sequence


def _validate_inputs(risks: Sequence[float], wrong: Sequence[int]) -> None:
    if len(risks) != len(wrong):
        raise ValueError("risks and wrong must have equal length")
    for index, risk in enumerate(risks):
        if not isinstance(risk, Real) or not math.isfinite(float(risk)):
            raise ValueError(f"risks[{index}] must be a finite number")
        if not 0.0 <= float(risk) <= 1.0:
            raise ValueError(f"risks[{index}] must be between 0 and 1")
    # Check the supplied values before converting them.  Casting first would
    # silently turn fractional labels such as 0.9 into 0 and could therefore
    # produce a false certificate.
    if any(w not in (0, 1) for w in wrong):
        raise ValueError("wrong values must be binary (0 or 1)")


def _validate_probability(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number strictly between 0 and 1")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise ValueError(f"{name} must be strictly between 0 and 1")
    return number


def _validate_bound(bound: str) -> None:
    if bound not in ("hoeffding", "clopper_pearson"):
        raise ValueError("bound must be 'hoeffding' or 'clopper_pearson'")


def _validate_threshold(tau: float) -> float:
    if isinstance(tau, bool) or not isinstance(tau, Real) or not math.isfinite(float(tau)):
        raise ValueError("tau must be a finite number")
    threshold = float(tau)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("tau must be between 0 and 1")
    return threshold


def _validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def false_accept_rate(risks: Sequence[float], wrong: Sequence[int], tau: float) -> Optional[float]:
    """Empirical P(wrong | risk <= tau). None if the accept set {risk <= tau} is empty."""
    _validate_inputs(risks, wrong)
    tau = _validate_threshold(tau)
    accepted = [w for r, w in zip(risks, wrong) if r <= tau]
    return (sum(accepted) / len(accepted)) if accepted else None


def hoeffding_upper_bound(empirical_rate: float, n: int, delta: float = 0.1) -> float:
    """One-sided Hoeffding upper bound for a fixed Bernoulli selection rule."""
    delta = _validate_probability("delta", delta)
    n = _validate_positive_integer("n", n)
    if (
        isinstance(empirical_rate, bool)
        or not isinstance(empirical_rate, Real)
        or not math.isfinite(float(empirical_rate))
        or not 0.0 <= float(empirical_rate) <= 1.0
    ):
        raise ValueError("empirical_rate must be between 0 and 1")
    return min(1.0, float(empirical_rate) + math.sqrt(math.log(1.0 / delta) / (2 * n)))


@lru_cache(maxsize=256)
def _log_binomial_coefficient(n: int, value: int) -> float:
    """Accurate log(n choose value), cached across bound inversion steps."""
    return math.log(math.comb(n, value))


def _binomial_boundary_pmf(value: int, n: int, probability: float) -> float:
    """Binomial PMF at one tail boundary, evaluated in log space."""
    return math.exp(
        _log_binomial_coefficient(n, value)
        + value * math.log(probability)
        + (n - value) * math.log1p(-probability)
    )


def _binomial_lower_tail(k: int, n: int, probability: float) -> float:
    """Directly sum P(X <= k) while moving away from the binomial mode."""
    q = 1.0 - probability
    term = _binomial_boundary_pmf(k, n, probability)
    terms = [term]
    reverse_odds = q / probability
    for value in range(k, 0, -1):
        term *= (value / (n - value + 1)) * reverse_odds
        terms.append(term)
    return min(1.0, math.fsum(terms))


def _binomial_upper_tail(k: int, n: int, probability: float) -> float:
    """Directly sum P(X > k) while moving away from the binomial mode."""
    q = 1.0 - probability
    first = k + 1
    term = _binomial_boundary_pmf(first, n, probability)
    terms = [term]
    odds = probability / q
    for value in range(first, n):
        term *= ((n - value) / (value + 1)) * odds
        terms.append(term)
    return min(1.0, math.fsum(terms))


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    """P(X <= k) for X ~ Binomial(n, probability), using the standard library.

    The shorter tail is seeded near the distribution's mode with a log-PMF,
    then accumulated away from the mode with a decreasing recurrence. Seeding
    at an endpoint with ``p**n`` or ``(1-p)**n`` can underflow for ordinary
    large samples and make an exact certificate anti-conservative.
    """
    if k >= n:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0
    if k < (n + 1) * probability:
        return _binomial_lower_tail(k, n, probability)
    return max(0.0, 1.0 - _binomial_upper_tail(k, n, probability))


def _binomial_survival(k: int, n: int, probability: float) -> float:
    """P(X > k), using a direct upper-tail sum when that tail is small."""
    if k >= n:
        return 0.0
    if probability <= 0.0:
        return 0.0
    if probability >= 1.0:
        return 1.0
    if k >= (n + 1) * probability:
        return _binomial_upper_tail(k, n, probability)
    return max(0.0, 1.0 - _binomial_lower_tail(k, n, probability))


def clopper_pearson_upper_bound(
    false_accepts: int,
    n: int,
    delta: float = 0.1,
) -> float:
    """Exact one-sided Clopper-Pearson upper bound for a binomial error rate."""
    delta = _validate_probability("delta", delta)
    n = _validate_positive_integer("n", n)
    if isinstance(false_accepts, bool) or not isinstance(false_accepts, Integral):
        raise ValueError("false_accepts must be an integer")
    false_accepts = int(false_accepts)
    if not 0 <= false_accepts <= n:
        raise ValueError("false_accepts must be between 0 and n")
    if false_accepts == n:
        return 1.0

    # Invert P_p(X <= false_accepts) = delta. The CDF is monotone decreasing in
    # p. For delta > 0.5, compare the directly summed survival tail against
    # 1-delta to avoid cancellation near one.
    low, high = 0.0, 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if delta <= 0.5:
            cdf_exceeds_delta = _binomial_cdf(false_accepts, n, midpoint) > delta
        else:
            cdf_exceeds_delta = (
                _binomial_survival(false_accepts, n, midpoint) < 1.0 - delta
            )
        if cdf_exceeds_delta:
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
    alpha = _validate_probability("alpha", alpha)
    delta = _validate_probability("delta", delta)
    _validate_bound(bound)
    tau = _validate_threshold(tau)
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
            "bound": bound,
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
    alpha = _validate_probability("alpha", alpha)
    delta = _validate_probability("delta", delta)
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
    alpha = _validate_probability("alpha", alpha)
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
    # Validate every predeclared component before threshold selection so an
    # early no-candidate return cannot conceal malformed certification data.
    _validate_inputs(fit_risks, fit_wrong)
    _validate_inputs(certification_risks, certification_wrong)
    alpha = _validate_probability("alpha", alpha)
    delta = _validate_probability("delta", delta)
    _validate_bound(bound)
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
