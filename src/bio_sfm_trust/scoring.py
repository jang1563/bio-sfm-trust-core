"""Cost-aware, modality-agnostic action scoring: net = correct - lambda * assays.

Provenance: generalized from bio-sfm-trust-audit's per-item action->(correct, assays)
mapping (experiments/trust_cue_attribution/phase2_score_episodes.py `outcome()` and
`_summary()`). Unlike the audit's panel/gene-keyed `score_actions`, this operates on
a flat list of per-candidate decisions, which is what a generative designer needs.
"""

from __future__ import annotations

from typing import Any, Iterable

from .actions import normalize_action


def action_outcome(
    action: str,
    *,
    sfm_correct: bool,
    baseline_correct: bool,
) -> tuple[int, int]:
    """Return (correct, assays) for one decision.

    - verify_assay:      guaranteed correct, costs 1 assay
    - trust_sfm:         correct iff the specialist call was correct, no assay
    - default_baseline:  correct iff the cheap baseline was correct, no assay
    - defer:             not correct, no assay
    """
    a = normalize_action(action)
    if a == "verify_assay":
        return 1, 1
    if a == "trust_sfm":
        return (1 if sfm_correct else 0), 0
    if a == "default_baseline":
        return (1 if baseline_correct else 0), 0
    return 0, 0  # defer


def net_reward(outcomes: Iterable[tuple[int, int]], lam: float = 0.5) -> float:
    """Per-item net reward for a sequence of (correct, assays) outcomes."""
    rows = list(outcomes)
    n = len(rows)
    if n == 0:
        return 0.0
    correct = sum(c for c, _ in rows)
    assays = sum(a for _, a in rows)
    return (correct - lam * assays) / n


def summarize_actions(rows: list[dict[str, Any]], lam: float = 0.5) -> dict[str, Any]:
    """Summarize a list of decision rows.

    Each row must carry `action` and the ground-truth fields `sfm_correct` and
    `baseline_correct` (booleans). Truth fields are scorer-side only.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0}
    outs = [
        action_outcome(
            r["action"],
            sfm_correct=bool(r.get("sfm_correct", False)),
            baseline_correct=bool(r.get("baseline_correct", False)),
        )
        for r in rows
    ]
    correct = sum(c for c, _ in outs)
    assays = sum(a for _, a in outs)
    acts = [normalize_action(r["action"]) for r in rows]
    trust_err = sum(
        1 for r, a in zip(rows, acts) if a == "trust_sfm" and not bool(r.get("sfm_correct", False))
    )
    return {
        "n": n,
        "accuracy": round(correct / n, 6),
        "assays_per_item": round(assays / n, 6),
        "net_reward_per_item": round((correct - lam * assays) / n, 6),
        "trust_error_rate": round(trust_err / n, 6),
        "trust_rate": round(acts.count("trust_sfm") / n, 6),
        "verify_rate": round(acts.count("verify_assay") / n, 6),
        "default_rate": round(acts.count("default_baseline") / n, 6),
        "defer_rate": round(acts.count("defer") / n, 6),
    }
