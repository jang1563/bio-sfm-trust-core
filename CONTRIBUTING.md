# Contributing

Contributions should preserve the package's small, dependency-free core and its
fail-closed routing semantics.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Before opening a pull request, run:

```bash
ruff check src tests
python -m unittest discover -s tests -v
python -m build
twine check dist/*
cffconvert --validate
```

## Change requirements

- Add regression tests for every behavior change or bug fix.
- Validate malformed inputs before early returns; certification and routing paths
  must fail closed.
- Keep threshold fitting and certification data independent.
- Document public API or statistical-contract changes in `CHANGELOG.md`.
- Do not commit credentials, local paths, generated build products, model weights,
  or private research data.

Scientific results and application-specific integrations belong in the upstream
audit or downstream application unless they change this reusable core directly.
