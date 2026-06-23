"""bio_sfm_trust — calibrated trust-routing engine.

The reusable core of the LLM-over-SFM trust-routing decision, extracted (with
provenance) from `bio-sfm-trust-audit` (experiments/trust_cue_attribution/). It
provides:

- the action set (`trust_sfm | verify_assay | default_baseline | defer`),
- isotonic calibration of a raw confidence-derived risk to P(wrong),
- a deterministic offline gate (verify iff calibrated-risk > lambda) with
  shuffled/inverted controls,
- cost-aware scoring (net = correct - lambda * assays),
- the modality-agnostic AdapterContract schema and a mock LLM provider.

Pure standard library; no GPU/weights/network required.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .actions import ACTIONS, normalize_action, parse_action_record
from .adapters import AdapterContract, adapter_contracts, get_adapter_contract
from .calibration import isotonic_calibrator, loo_calibrated_risks
from .gate import (
    calibrated_gate,
    confidence_to_risk,
    phase2_calibration_gate,
    risk_threshold_policy_net,
)
from .metrics import auroc, pearson
from .providers import get_provider, mock_defer_response
from .scoring import action_outcome, net_reward, summarize_actions

__all__ = [
    "ACTIONS",
    "normalize_action",
    "parse_action_record",
    "isotonic_calibrator",
    "loo_calibrated_risks",
    "confidence_to_risk",
    "risk_threshold_policy_net",
    "phase2_calibration_gate",
    "calibrated_gate",
    "auroc",
    "pearson",
    "action_outcome",
    "net_reward",
    "summarize_actions",
    "AdapterContract",
    "get_adapter_contract",
    "adapter_contracts",
    "get_provider",
    "mock_defer_response",
]
