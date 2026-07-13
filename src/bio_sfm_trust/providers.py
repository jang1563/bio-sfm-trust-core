"""LLM provider abstraction with a dependency-free mock.

Provenance: distilled from bio-sfm-trust-audit
(experiments/trust_cue_attribution/llm_runner.py) provider dispatch. A provider is
a callable `(prompt: str) -> str` returning the model's raw text. The `mock`
providers let the whole loop run with no API key or network. The `anthropic`
provider is lazy: it only imports the SDK when actually requested.
"""

from __future__ import annotations

import json
from typing import Callable

Provider = Callable[[str], str]


def mock_defer_response(prompt: str) -> str:
    """Return a valid all-defer action record without calling any API."""
    return json.dumps({"action": "defer", "confidence": 0.0, "rationale": "mock_defer provider"})


def _mock_trust_response(prompt: str) -> str:
    return json.dumps({"action": "trust_sfm", "confidence": 0.5, "rationale": "mock_trust provider"})


def _anthropic_provider(model: str) -> Provider:
    """Lazily construct an Anthropic-backed provider; import the SDK on demand."""
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only when SDK absent
        raise RuntimeError(
            "anthropic provider requested but the 'anthropic' SDK is not installed. "
            "Install the 'anthropic' extra and set ANTHROPIC_API_KEY."
        ) from exc

    from anthropic import Anthropic

    client = Anthropic()

    def call(prompt: str) -> str:
        # Adaptive thinking cannot be combined with temperature or top_p. Low effort
        # and a constrained prompt reduce variance without claiming determinism.
        msg = client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text")

    return call


def get_provider(name: str, *, model: str = "claude-opus-4-8") -> Provider:
    """Return a provider callable by name.

    - "mock_defer": always defers (default, dependency-free)
    - "mock_trust": always trusts the specialist
    - "anthropic":  real Claude calls (lazy import; needs SDK + key)
    """
    if name in ("mock", "mock_defer"):
        return mock_defer_response
    if name == "mock_trust":
        return _mock_trust_response
    if name in ("anthropic", "anthropic_messages"):
        return _anthropic_provider(model)
    raise ValueError(f"unknown provider {name!r}")
