# bio-sfm-trust

Calibrated **trust-routing engine** for LLM orchestration over specialist scientific
foundation models (SFMs). Given a specialist's output and a model-emitted confidence,
it decides — under a verification cost — whether to

`trust_sfm | verify_assay | default_baseline | defer`,  scored by `net = correct − λ·assays`.

This is the reusable core extracted (with provenance) from
[**bio-sfm-trust-audit**](https://github.com/jang1563/bio-sfm-trust-audit), so the
audit and downstream designers share one calibration engine. Pure standard library —
no GPU, weights, or network required.

## Install

```bash
pip install -e .
python -m unittest discover -s tests -v
```

## What's inside

| Module | Purpose |
|---|---|
| `actions` | the action set + `normalize_action`, `parse_action_record` |
| `calibration` | `isotonic_calibrator`, `loo_calibrated_risks` (PAVA, leave-one-out) |
| `gate` | `confidence_to_risk`, deterministic offline gates (`phase2_calibration_gate`, `calibrated_gate`) with shuffled/inverted controls |
| `scoring` | `action_outcome`, `net_reward`, `summarize_actions` (modality-agnostic, per-item) |
| `metrics` | pure-stdlib `auroc`, `pearson` |
| `adapters` | `AdapterContract` schema — how any SFM plugs into the action/reward stack |
| `providers` | `mock_defer` (dependency-free) + lazy `anthropic` provider |

## Design principles (carried from the audit)

- The trust decision is **external and engineered**, never delegated to the LLM's own
  sense of confidence (measured ≈ chance allocation; stronger models over-verify).
- A raw confidence (e.g. `1 − pLDDT`) is **compressed**; isotonic calibration lands it
  on the P(wrong) scale so `verify iff risk > λ` actually fires.
- An offline deterministic gate must beat trust-all + shuffled/inverted controls
  **before any LLM spend**.

## License

MIT (see `LICENSE`). Provenance headers in each module cite the upstream audit file.
