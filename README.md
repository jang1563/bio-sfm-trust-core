# bio-sfm-trust-core

[![CI](https://github.com/jang1563/bio-sfm-trust-core/actions/workflows/ci.yml/badge.svg)](https://github.com/jang1563/bio-sfm-trust-core/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

`bio-sfm-trust` is a pure-Python trust-routing engine for LLM orchestration over
specialist scientific foundation models (SFMs). Given a specialist output and a
model-visible confidence signal, it supports cost-aware decisions among:

`trust_sfm | verify_assay | default_baseline | defer`

The package scores decisions with `net = correct - lambda * assays` and provides
calibration plus split learn-then-test certification for selective trust. The
repository is named `bio-sfm-trust-core`; the installable package is
`bio-sfm-trust`, imported as `bio_sfm_trust`.

## Scope

This repository contains the reusable routing and certification code. It does not
ship specialist-model weights, biological datasets, or benchmark results, so a
Hugging Face mirror is not part of this release.

The empirical evidence and result artifacts live in
[bio-sfm-trust-audit](https://github.com/jang1563/bio-sfm-trust-audit). The
downstream DBTL application is
[bio_sfm_designer](https://github.com/jang1563/bio_sfm_designer). Dependencies are
one-way: `designer -> trust-core`; this repository has no protein-generation or
HPC application code.

## Install and verify

The package is currently distributed from source and GitHub release artifacts;
no PyPI release is claimed.

```bash
git clone https://github.com/jang1563/bio-sfm-trust-core.git
cd bio-sfm-trust-core
python -m pip install .
python -m unittest discover -s tests -v
```

The core has no runtime dependencies and needs no GPU, model weights, credentials,
or network access. The optional Anthropic provider is installed explicitly:

```bash
python -m pip install ".[anthropic]"
export ANTHROPIC_API_KEY="..."
```

## Certification example

`split_ltt_threshold` learns a candidate risk threshold on a fit split and tests
the frozen threshold on an independent certification split:

```python
from bio_sfm_trust import split_ltt_threshold

report = split_ltt_threshold(
    fit_risks=[0.05] * 50 + [0.90] * 20,
    fit_wrong=[0] * 50 + [1] * 20,
    certification_risks=[0.05] * 50 + [0.90] * 20,
    certification_wrong=[0] * 50 + [1] * 20,
    alpha=0.25,
    delta=0.10,
    bound="clopper_pearson",
)
print(report["certified"], report["tau"])
```

The function consumes already-computed risks; it does not fit an isotonic model.
If calibration is needed, fit it using only the fit split before calling the
certification function. Never use the certification split to choose the threshold
or fit the calibrator.

## Package map

| Module | Purpose |
|---|---|
| `actions` | Canonical action set plus fail-closed parsing and normalization |
| `calibration` | Isotonic calibration and leave-one-out calibrated risks |
| `conformal` | Split learn-then-test threshold selection and independent certification |
| `gate` | Confidence-to-risk mapping and deterministic offline gates |
| `scoring` | Cost-aware action outcomes and summaries |
| `metrics` | Dependency-free AUROC and Pearson implementations |
| `adapters` | `AdapterContract` schemas for specialist-model integration |
| `providers` | Dependency-free mock providers plus a lazy Anthropic provider |

## Statistical contract and limitations

- The trust decision is external and engineered; it is not delegated to an LLM's
  self-reported confidence.
- `split_ltt_threshold` separates threshold selection from evaluation. A result is
  called certified only when the frozen rule passes the predeclared one-sided bound
  on the independent certification split.
- `rcps_threshold` remains a backward-compatible exploratory selector. Because it
  searches and evaluates on the same observations, its pointwise bound is not a
  distribution-free certificate.
- Certification assumes the certification observations are independent of fitting
  and representative of the intended deployment distribution. It does not guarantee
  performance under distribution shift or validate the upstream scientific model.
- Adapter entries marked `contract_only` describe interfaces, not implemented or
  validated model integrations.

## Status

Version `0.2.0` is research-stage software. The public CI matrix installs the built
package and exercises the test suite across supported Python versions, then validates
the source distribution and wheel. Application-level findings and readiness claims
belong to the upstream audit or downstream designer, not to this package alone.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests
python -m unittest discover -s tests -v
python -m build
twine check dist/*
cffconvert --validate
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md), and
[CHANGELOG.md](CHANGELOG.md). Citation metadata is available in
[CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE). Provenance headers in the modules identify the
upstream audit implementations from which the core was extracted.
