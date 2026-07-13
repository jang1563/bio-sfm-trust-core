# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-07-13

### Added

- Split learn-then-test threshold selection with independent Hoeffding or exact
  Clopper-Pearson certification.
- Strict validation for calibration, certification, risk, and metric inputs.
- Public CI across supported Python versions, package build checks, citation
  metadata, and release documentation.
- An explicit optional dependency group for the lazy Anthropic provider.

### Changed

- Duplicate calibration inputs are aggregated before weighted isotonic fitting,
  making tied-risk predictions independent of row order.
- Public documentation now separates this code package from the empirical audit
  and downstream application.

### Fixed

- Clopper-Pearson tail evaluation now avoids endpoint underflow and near-one
  complement cancellation that could underestimate a large-sample upper bound.
- Gate, metric, parser, and policy inputs now fail closed on missing, ambiguous,
  nonfinite, out-of-range, or non-binary values.
- Trust-error summaries retain the historical per-item rate while reporting the
  conditional error rate among trust actions separately.

## [0.1.0] - 2026-07-04

### Added

- Initial dependency-free trust-routing core.
- Canonical actions, cost-aware scoring, calibration, routing gates, adapter
  contracts, and mock/Anthropic provider dispatch.
- Complex-regime preference for the validated `pae_interaction` signal when present.

[Unreleased]: https://github.com/jang1563/bio-sfm-trust-core/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jang1563/bio-sfm-trust-core/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jang1563/bio-sfm-trust-core/tree/v0.1.0
