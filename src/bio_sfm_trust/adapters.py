"""AdapterContract: a modality-agnostic interface so any specialist SFM plugs into
the same action / reward / scoring stack.

Provenance: schema mirrors bio-sfm-trust-audit
(experiments/trust_cue_attribution/adapters.py). An adapter is *metadata only* — it
declares what inputs/outputs/evidence a specialist must provide and what preflight
checks it must pass. Actual model wrappers live downstream (in the designer's
generate/ and predict/ modules), behind this contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class AdapterContract:
    name: str
    phase: str
    status: str
    specialist_model: str
    task_family: str
    required_inputs: list[str] = field(default_factory=list)
    standardized_outputs: list[str] = field(default_factory=list)
    evidence_fields: list[str] = field(default_factory=list)
    internal_signal_fields: list[str] = field(default_factory=list)
    hidden_fields: list[str] = field(default_factory=list)
    preflight_checks: list[str] = field(default_factory=list)
    claim_boundary: str = ""

    def describe(self) -> dict[str, Any]:
        return asdict(self)


BoltzStructureAdapter = AdapterContract(
    name="BoltzStructureAdapter",
    phase="phase2",
    status="preflight_contract",
    specialist_model="Boltz-1/Boltz-2 (AlphaFold3-class structure predictor)",
    task_family="protein_structure_prediction_with_calibrated_confidence",
    required_inputs=["sequence(s) per chain", "regime (monomer|complex)"],
    standardized_outputs=["predicted_structure", "mean_plddt", "iptm", "ptm"],
    evidence_fields=["mean_plddt", "iptm", "regime"],
    internal_signal_fields=["per_residue_plddt", "pae"],
    hidden_fields=["truth.correct", "truth.quality"],
    preflight_checks=[
        "targets released after specialist training cutoff (leakage-safe)",
        "calibrated confidence validated against held-out lDDT",
    ],
    claim_boundary="Calibrated structure-confidence only; no functional guarantee.",
)

ESMSequenceAdapter = AdapterContract(
    name="ESMSequenceAdapter",
    phase="future",
    status="contract_only",
    specialist_model="ESM-2 / ESMFold (protein language model)",
    task_family="sequence_property_or_embedding",
    required_inputs=["amino-acid sequence"],
    standardized_outputs=["scalar_property", "calibrated_confidence"],
    evidence_fields=["scalar_property", "calibrated_confidence"],
    internal_signal_fields=["embedding (route via trained read-out head, never raw to LLM)"],
    hidden_fields=["truth.correct", "truth.quality"],
    preflight_checks=["confidence calibrated vs held-out truth", "no raw-latent prompt-pasting"],
    claim_boundary="Scalar property via trained read-out head; raw embeddings are not LLM-readable.",
)

ProteinMPNNGeneratorAdapter = AdapterContract(
    name="ProteinMPNNGeneratorAdapter",
    phase="future",
    status="contract_only",
    specialist_model="ProteinMPNN / RFdiffusion (generative design)",
    task_family="sequence_or_backbone_generation",
    required_inputs=["design objective", "scaffold/constraints"],
    standardized_outputs=["candidate designs", "generator_score"],
    evidence_fields=["generator_score"],
    internal_signal_fields=[],
    hidden_fields=[],
    preflight_checks=["candidates screened before they leave the generator"],
    claim_boundary="Generative proposals only; advancement gated by trust + safety.",
)

ADAPTERS: dict[str, AdapterContract] = {
    a.name: a
    for a in (BoltzStructureAdapter, ESMSequenceAdapter, ProteinMPNNGeneratorAdapter)
}


def get_adapter_contract(name: str) -> AdapterContract:
    if name not in ADAPTERS:
        raise ValueError(f"unknown adapter {name!r}; expected one of {sorted(ADAPTERS)}")
    return ADAPTERS[name]


def adapter_contracts() -> dict[str, dict[str, Any]]:
    return {name: a.describe() for name, a in ADAPTERS.items()}
