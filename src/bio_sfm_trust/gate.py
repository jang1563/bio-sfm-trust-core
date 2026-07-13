"""Deterministic offline trust gate over a calibrated confidence signal.

Provenance: merged from bio-sfm-trust-audit's two complementary modules
(experiments/trust_cue_attribution/phase2_calibration_gate.py and
phase2_calibrated_gate.py), with intra-package imports instead of sys.path shims.

Two gate variants:
- `phase2_calibration_gate`: deterministic gate on the RAW confidence-derived risk
  (uses `truth.correct`). The cost-optimal rule "verify iff risk > lambda" must
  beat trust-all, default-template, and shuffled/inverted controls before any LLM
  spend.
- `calibrated_gate`: same idea but with leave-one-out isotonic calibration of the
  raw risk to P(wrong), re-deriving the wrong-label from continuous `truth.quality`
  at a substrate-appropriate lDDT cutoff.

No LLM calls; `truth.*` fields are used only by these scorers, never model-visible.

Input records (one target per line / list element):
    {
      "target_id": "T1",
      "regime": "monomer" | "complex",
      "mean_plddt": 0..100,
      "iptm": 0..1 | null,
      "pae_interaction": non-negative float | null,
      "template_baseline_correct": bool,
      "truth": {"correct": bool, "quality": 0..1}   # HIDDEN
    }
"""

from __future__ import annotations

import math
from typing import Any, Optional

from .calibration import loo_calibrated_risks
from .metrics import auroc, pearson

VALIDITY_AUROC_MIN = 0.70
SHUFFLED_GAP_MIN = 0.05          # real-vs-shuffled net/target; must beat near-noise gaps
MIN_PER_REGIME = 30              # pre-specified power floor per regime
DEFAULT_LAMBDA = 0.5
DEFAULT_CORRECT_LDDT = 0.9


def _finite_float(value: Any, name: str) -> float:
    """Return a finite float or raise a field-specific validation error."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _binary_labels(values: list[Any], name: str) -> list[int]:
    """Validate binary labels without accepting truthy non-binary values."""
    labels: list[int] = []
    for index, value in enumerate(values):
        if not isinstance(value, (bool, int)) or value not in (0, 1):
            raise ValueError(f"{name}[{index}] must be binary (0 or 1)")
        labels.append(int(value))
    return labels


def _probability_values(values: list[Any], name: str) -> list[float]:
    """Validate finite risk/probability values in the closed unit interval."""
    checked = [_finite_float(value, f"{name}[{index}]") for index, value in enumerate(values)]
    for index, value in enumerate(checked):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{index}] must be in [0, 1]")
    return checked


def _verification_cost(value: Any) -> float:
    """Validate a finite, non-negative verification cost."""
    checked = _finite_float(value, "lam")
    if checked < 0.0:
        raise ValueError("lam must be non-negative")
    return checked


def _template_baseline_labels(records: list[dict[str, Any]]) -> list[int]:
    """Require an explicit boolean baseline outcome for every gate record."""
    labels: list[int] = []
    for index, record in enumerate(records):
        try:
            value = record["template_baseline_correct"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"records[{index}].template_baseline_correct is required"
            ) from exc
        if not isinstance(value, bool):
            raise ValueError(
                f"records[{index}].template_baseline_correct must be a boolean"
            )
        labels.append(int(value))
    return labels


def _truth_quality_values(records: list[dict[str, Any]]) -> list[float]:
    """Require finite normalized structure-quality truth for every gate record."""
    qualities: list[float] = []
    for index, record in enumerate(records):
        try:
            raw_quality = record["truth"]["quality"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"records[{index}].truth.quality is required") from exc
        quality = _finite_float(raw_quality, f"records[{index}].truth.quality")
        if not 0.0 <= quality <= 1.0:
            raise ValueError(f"records[{index}].truth.quality must be in [0, 1]")
        qualities.append(quality)
    return qualities


def _correct_lddt_cutoff(value: Any) -> float:
    """Validate the normalized quality cutoff used to derive wrong labels."""
    cutoff = _finite_float(value, "correct_lddt")
    if not 0.0 <= cutoff <= 1.0:
        raise ValueError("correct_lddt must be in [0, 1]")
    return cutoff


def confidence_to_risk(record: dict[str, Any]) -> float:
    """Map model-emitted confidence to an estimated wrong-risk in [0, 1].

    Monomer: risk = 1 - pLDDT/100. Complex: prefer interface predicted-aligned-error
    (pae_interaction, A, lower=better) when present -- it is the validated interface signal
    (pAE can discriminate interface success even among well-folded binders where ipTM is
    uninformative); risk = pae/30 clamped. Falls back to the pLDDT+ipTM blend, then pLDDT.
    The downstream gate re-calibrates this raw risk per regime (isotonic), so only its MONOTONICITY
    with failure matters here.
    """
    regime = record.get("regime")
    if regime not in ("monomer", "complex"):
        raise ValueError("regime must be explicitly set to 'monomer' or 'complex'")
    if record.get("mean_plddt") is None:
        raise ValueError("mean_plddt is required")

    mean_plddt = _finite_float(record["mean_plddt"], "mean_plddt")
    if not 0.0 <= mean_plddt <= 100.0:
        raise ValueError("mean_plddt must be in [0, 100]")

    iptm = None
    if record.get("iptm") is not None:
        iptm = _finite_float(record["iptm"], "iptm")
        if not 0.0 <= iptm <= 1.0:
            raise ValueError("iptm must be in [0, 1]")

    pae_interaction = None
    if record.get("pae_interaction") is not None:
        pae_interaction = _finite_float(record["pae_interaction"], "pae_interaction")
        if pae_interaction < 0.0:
            raise ValueError("pae_interaction must be non-negative")

    plddt = mean_plddt / 100.0
    if regime == "complex" and pae_interaction is not None:
        risk = pae_interaction / 30.0
    elif regime == "complex" and iptm is not None:
        risk = 1.0 - 0.5 * plddt - 0.5 * iptm
    else:
        risk = 1.0 - plddt
    return max(0.0, min(1.0, risk))


def _decorrelated_permutation(values: list[float]) -> list[float]:
    """Deterministic shuffle that breaks the value<->target association regardless
    of input ordering (no RNG): rotate by n//2 in value-sorted space."""
    n = len(values)
    if n < 2:
        return list(values)
    order = sorted(range(n), key=lambda i: values[i])
    k = n // 2
    shuffled = [0.0] * n
    for rank, idx in enumerate(order):
        shuffled[idx] = values[order[(rank + k) % n]]
    return shuffled


def risk_threshold_policy_net(records: list[dict[str, Any]], risks: list[float], lam: float) -> float:
    """Net reward per target for 'verify iff risk > lambda', else trust the specialist."""
    if len(records) != len(risks):
        raise ValueError("records and risks must have equal length")
    checked_risks = _probability_values(list(risks), "risks")
    checked_lam = _verification_cost(lam)
    truth_values: list[Any] = []
    for index, record in enumerate(records):
        try:
            truth_values.append(record["truth"]["correct"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"records[{index}].truth.correct is required") from exc
    truth_correct = _binary_labels(truth_values, "truth.correct")
    if not records:
        return 0.0
    correct = 0.0
    assays = 0.0
    for is_correct, risk in zip(truth_correct, checked_risks):
        if risk > checked_lam:
            correct += 1.0          # verify -> guaranteed correct
            assays += 1.0
        elif is_correct:
            correct += 1.0          # trust a correct specialist call
    n = len(records)
    return correct / n - checked_lam * (assays / n)


def _regime_calibration(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pearson(pLDDT, structure quality) within each regime; the gap is the stakes."""
    out: dict[str, Any] = {}
    for regime in ("monomer", "complex"):
        rows = [r for r in records if r.get("regime") == regime]
        plddt = [float(r["mean_plddt"]) for r in rows]
        quality = [float(r["truth"]["quality"]) for r in rows]
        out[regime] = {
            "n": len(rows),
            "pearson_plddt_vs_quality": _round(pearson(plddt, quality)) if len(rows) >= 2 else None,
        }
    mono = out["monomer"]["pearson_plddt_vs_quality"]
    comp = out["complex"]["pearson_plddt_vs_quality"]
    out["monomer_minus_complex"] = _round(mono - comp) if (mono is not None and comp is not None) else None
    return out


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(float(value), 6)


def _decide(checks: dict[str, bool]) -> str:
    core_ok = (
        checks["signal_validity_auroc_ok"]
        and checks["policy_beats_trust_all"]
        and checks.get("policy_beats_default_template", True)
        and checks["policy_beats_best_control"]
        and checks["real_vs_shuffled_gap_large"]
    )
    if core_ok and checks["power_sufficient"]:
        return "eligible_for_phase2_interface_pilot"
    if core_ok:
        return "eligible_pending_more_targets"
    if checks["signal_validity_auroc_ok"]:
        return "redesign_policy_before_pilot"
    return "do_not_run_signal_not_calibrated"


def phase2_calibration_gate(
    records: list[dict[str, Any]],
    *,
    lam: float = DEFAULT_LAMBDA,
    validity_auroc_min: float = VALIDITY_AUROC_MIN,
    shuffled_gap_min: float = SHUFFLED_GAP_MIN,
    min_per_regime: int = MIN_PER_REGIME,
) -> dict[str, Any]:
    """Deterministic gate on the RAW confidence-derived risk (uses truth.correct)."""
    template_baseline_labels = _template_baseline_labels(records)
    real_risks = [confidence_to_risk(r) for r in records]
    wrong_labels = [0 if r["truth"]["correct"] else 1 for r in records]
    shuffled_risks = _decorrelated_permutation(real_risks)
    inverted_risks = [1.0 - r for r in real_risks]

    validity_auroc = auroc(real_risks, wrong_labels)

    real_net = risk_threshold_policy_net(records, real_risks, lam)
    shuffled_net = risk_threshold_policy_net(records, shuffled_risks, lam)
    inverted_net = risk_threshold_policy_net(records, inverted_risks, lam)
    best_control = max(shuffled_net, inverted_net)

    n = len(records)
    trust_all = sum(1 for r in records if r["truth"]["correct"]) / n if n else 0.0
    default_template = sum(template_baseline_labels) / n if n else 0.0
    verify_all = 1.0 - lam
    oracle = 1.0 - lam * (sum(wrong_labels) / n) if n else 0.0

    regime = _regime_calibration(records)
    n_mono = regime["monomer"]["n"]
    n_comp = regime["complex"]["n"]

    checks = {
        "signal_validity_auroc_ok": bool(validity_auroc is not None and validity_auroc >= validity_auroc_min),
        "policy_beats_trust_all": real_net > trust_all,
        "policy_beats_default_template": real_net > default_template,
        "policy_beats_best_control": real_net > best_control,
        "real_vs_shuffled_gap_large": (real_net - shuffled_net) >= shuffled_gap_min,
        "power_sufficient": n_mono >= min_per_regime and n_comp >= min_per_regime,
    }
    return {
        "phase": "phase2",
        "status": "calibration_gate_ready",
        "claim_boundary": (
            "Offline deterministic gate over a calibrated structure-confidence "
            "signal; no LLM calls, no leakage, no claim of faithful internal "
            "interpretation or general SFM generalization."
        ),
        "scope": {"n_targets": n, "n_monomer": n_mono, "n_complex": n_comp},
        "lambda": lam,
        "signal_validity": {"wrong_risk_auroc": _round(validity_auroc), "threshold": validity_auroc_min},
        "regime_calibration": regime,
        "net_reward_per_target": {
            "real_risk_policy": _round(real_net),
            "shuffled_risk_control": _round(shuffled_net),
            "inverted_risk_control": _round(inverted_net),
            "trust_all": _round(trust_all),
            "default_template": _round(default_template),
            "verify_all": _round(verify_all),
            "oracle": _round(oracle),
        },
        "margins": {
            "vs_trust_all": _round(real_net - trust_all),
            "vs_default_template": _round(real_net - default_template),
            "vs_best_control": _round(real_net - best_control),
            "real_minus_shuffled": _round(real_net - shuffled_net),
        },
        "checks": checks,
        "decision": _decide(checks),
    }


def _policy_net_loo(wrong: list[int], risks: list[float], lam: float) -> float:
    """verify iff risk>lam (correct, cost lam), else trust (correct iff not wrong)."""
    if len(wrong) != len(risks):
        raise ValueError("wrong and risks must have equal length")
    checked_wrong = _binary_labels(list(wrong), "wrong")
    checked_risks = _probability_values(list(risks), "risks")
    checked_lam = _verification_cost(lam)
    if not wrong:
        return 0.0
    correct = assays = 0.0
    for w, r in zip(checked_wrong, checked_risks):
        if r > checked_lam:
            correct += 1.0
            assays += 1.0
        elif not w:
            correct += 1.0
    n = len(wrong)
    return correct / n - checked_lam * (assays / n)


def calibrated_gate(
    records: list[dict[str, Any]],
    *,
    lam: float = DEFAULT_LAMBDA,
    correct_lddt: float = DEFAULT_CORRECT_LDDT,
    validity_auroc_min: float = VALIDITY_AUROC_MIN,
    shuffled_gap_min: float = SHUFFLED_GAP_MIN,
    min_per_regime: int = MIN_PER_REGIME,
) -> dict[str, Any]:
    """Offline gate with leave-one-out isotonic calibration of raw risk -> P(wrong)."""
    checked_correct_lddt = _correct_lddt_cutoff(correct_lddt)
    truth_quality = _truth_quality_values(records)
    raw = [confidence_to_risk(r) for r in records]
    wrong = [0 if quality >= checked_correct_lddt else 1 for quality in truth_quality]
    cal = loo_calibrated_risks(raw, wrong)

    validity_auroc = auroc(raw, wrong)
    real = _policy_net_loo(wrong, cal, lam)
    shuffled = _policy_net_loo(wrong, _decorrelated_permutation(cal), lam)
    inverted = _policy_net_loo(wrong, [1.0 - c for c in cal], lam)
    best_control = max(shuffled, inverted)
    n = len(records)
    trust_all = sum(1 for w in wrong if not w) / n if n else 0.0
    oracle = 1.0 - lam * (sum(wrong) / n) if n else 0.0
    n_mono = sum(1 for r in records if r.get("regime") == "monomer")
    n_cplx = sum(1 for r in records if r.get("regime") == "complex")

    checks = {
        "signal_validity_auroc_ok": bool(validity_auroc is not None and validity_auroc >= validity_auroc_min),
        "policy_beats_trust_all": real > trust_all,
        "policy_beats_best_control": real > best_control,
        "real_vs_shuffled_gap_large": (real - shuffled) >= shuffled_gap_min,
        "power_sufficient": n_mono >= min_per_regime and n_cplx >= min_per_regime,
    }
    return {
        "phase": "phase2",
        "status": "calibrated_gate_ready",
        "claim_boundary": "Offline gate with LOO isotonic risk->P(wrong) calibration; no LLM calls.",
        "calibration": "leave_one_out_isotonic_raw_risk_to_p_wrong",
        "correct_lddt_cutoff": checked_correct_lddt,
        "lambda": lam,
        "scope": {"n_targets": n, "n_monomer": n_mono, "n_complex": n_cplx, "n_wrong": sum(wrong)},
        "signal_validity": {"wrong_risk_auroc": _round(validity_auroc), "threshold": validity_auroc_min},
        "net_reward_per_target": {
            "calibrated_real_policy": _round(real),
            "calibrated_shuffled_control": _round(shuffled),
            "calibrated_inverted_control": _round(inverted),
            "trust_all": _round(trust_all),
            "oracle": _round(oracle),
        },
        "margins": {
            "vs_trust_all": _round(real - trust_all),
            "vs_best_control": _round(real - best_control),
            "real_minus_shuffled": _round(real - shuffled),
            "oracle_headroom": _round(oracle - trust_all),
        },
        "checks": checks,
        "decision": _decide(checks),
    }
