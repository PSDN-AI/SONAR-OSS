# SONAR-OSS

[![CI](https://github.com/PSDN-AI/SONAR-OSS/actions/workflows/ci.yml/badge.svg)](https://github.com/PSDN-AI/SONAR-OSS/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/psdn-sonar.svg)](https://pypi.org/project/psdn-sonar/)
[![Python versions](https://img.shields.io/pypi/pyversions/psdn-sonar.svg)](https://pypi.org/project/psdn-sonar/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Multi-language ASR (automatic speech recognition) evaluation toolkit — metrics,
reporting, and benchmarks. Distributed as the `psdn-sonar` Python package.

![Architecture](docs/architecture.png)

> **Status: pre-release.** The library (metrics, language processors, dataset
> loaders, evaluators, reporting, CLI) is in place and heading toward a first
> `0.1.0` release; see [`CHANGELOG.md`](CHANGELOG.md). Content imported from
> the upstream codebase passes the checklist in
> [`docs/import-gate.md`](docs/import-gate.md).

## Requirements

- Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

**Supported environments.** CI validates Linux x86_64 with CPython 3.10, 3.11,
and 3.12. macOS and Windows are expected to work for the core package but are
not CI-validated; some optional extras have platform-sensitive dependencies
(`[korean]` needs a Java runtime at runtime, `[ml]`/`[pyannote]` pull large
PyTorch trees). Open an issue if an install fails on a supported Python.

## Installation

Contributors install the frozen, locked environment (exactly what CI runs):

```bash
git clone https://github.com/PSDN-AI/SONAR-OSS.git
cd SONAR-OSS
make setup            # uv sync --frozen with dev extras
source .venv/bin/activate
```

Or with plain pip (editable, freshly resolved):

```bash
pip install -e ".[dev]"
```

Note that `pip` does not read `uv.lock`: pip installs resolve dependency
versions fresh from PyPI within the ranges in `pyproject.toml`, so they are
not byte-for-byte reproducible the way `uv sync --frozen` is. This is the
normal contract for downstream package installs; use `uv` when you need the
locked contributor environment.

Copy `.env.example` to `.env` and fill in only the values you need (API keys
are required only for optional hosted-model backends).

## Usage

See [`docs/USAGE.md`](docs/USAGE.md) for a short quickstart with runnable
examples (scoring a pair, evaluating a model over a dataset, listing models),
and [`docs/FAQ.md`](docs/FAQ.md) for common workflows: CLI commands, required
input files, output layout, and how success is measured.

## Development

```bash
make lint              # ruff lint + format check
make typecheck         # ty type checker
make test              # pytest
make pre-commit-install
make check-internal-refs
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor guide,
including PR title conventions and the import gate.

## License

MIT — see [`LICENSE`](LICENSE).
