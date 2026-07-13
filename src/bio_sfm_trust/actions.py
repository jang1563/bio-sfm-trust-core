"""Action parsing and normalization.

Provenance: lifted verbatim from bio-sfm-trust-audit
(experiments/trust_cue_attribution/actions.py). The action set is the scientific
object of that audit; it is reused here unchanged so routing decisions are
comparable across the audit and the designer.
"""

from __future__ import annotations

import json
import re
from typing import Any

ACTIONS = ("trust_sfm", "verify_assay", "default_baseline", "defer")

_ALIASES = {
    "trust": "trust_sfm",
    "trust_fm": "trust_sfm",
    "trust_model": "trust_sfm",
    "use_sfm": "trust_sfm",
    "verify": "verify_assay",
    "assay": "verify_assay",
    "run_assay": "verify_assay",
    "default": "default_baseline",
    "baseline": "default_baseline",
    "additive": "default_baseline",
    "abstain": "defer",
    "untested": "defer",
}

_ACTION_TOKEN_PATTERN = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(action)
        for action in sorted(set(ACTIONS) | set(_ALIASES), key=lambda action: (-len(action), action))
    )
    + r")(?!\w)",
    re.I,
)


def normalize_action(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key not in ACTIONS:
        raise ValueError(f"unknown action {value!r}; expected one of {ACTIONS}")
    return key


def parse_action_record(value: str | dict[str, Any]) -> dict[str, Any]:
    """Parse an LLM action record from JSON, a dict, or a plain action label."""
    if isinstance(value, dict):
        rec = dict(value)
    else:
        text = value.strip()
        try:
            rec = json.loads(text)
        except json.JSONDecodeError:
            matches = {
                normalize_action(match.group(0)) for match in _ACTION_TOKEN_PATTERN.finditer(text)
            }
            if not matches:
                raise ValueError(f"could not parse action from {value!r}") from None
            if len(matches) > 1:
                raise ValueError(
                    f"ambiguous actions in {value!r}: {sorted(matches)}"
                ) from None
            rec = {"action": matches.pop()}
        if not isinstance(rec, dict):
            raise ValueError("action record JSON must be an object")
    rec["action"] = normalize_action(str(rec.get("action", "")))
    if "confidence" in rec:
        rec["confidence"] = float(rec["confidence"])
    return rec
