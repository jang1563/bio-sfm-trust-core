# bio-sfm-trust

Calibrated **trust-routing engine** for LLM orchestration over specialist scientific
foundation models (SFMs). Given a specialist's output and a model-emitted confidence,
it decides — under a verification cost — whether to

`trust_sfm | verify_assay | default_baseline | defer`,  scored by `net = correct − λ·assays`.

This is the reusable core extracted (with provenance) from
[**bio-sfm-trust-audit**](https://github.com/jang1563/bio-sfm-trust-audit), so the
audit and downstream designers share one calibration engine. Pure standard library —
no GPU, weights, or network required.

**Used by [bio_sfm_designer](https://github.com/jang1563/bio_sfm_designer)** — the biology DBTL
application built on this engine. The dependency is one-way: `designer → trust-core` (this repo has no
protein/HPC/application code).

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
| `conformal` | split learn-then-test thresholding: fit-split selection plus independent fixed-threshold Hoeffding certification |
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

## Status

Stable engine — 36 tests, pure stdlib. The certified path uses `split_ltt_threshold`: threshold selection
and isotonic fitting happen on a fit split, then a frozen rule is checked on an independent certification
split. `rcps_threshold` remains only as a backward-compatible exploratory selector; its same-data grid
search is not a formal certificate. Consumed by
[bio_sfm_designer](https://github.com/jang1563/bio_sfm_designer); see that repo's README for the
application-level status and honest findings.

## License

MIT (see `LICENSE`). Provenance headers in each module cite the upstream audit file.
