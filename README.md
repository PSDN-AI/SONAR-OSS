# SONAR-OSS

Multi-language ASR (automatic speech recognition) evaluation toolkit — metrics,
reporting, and benchmarks. Distributed as the `psdn-sonar` Python package.

> **Status: pre-release.** The package skeleton, tooling, and CI are in place;
> the library (metrics, language processors, dataset loaders, evaluators,
> reporting, CLI) is being imported incrementally. See
> [`docs/import-gate.md`](docs/import-gate.md) for the checklist every import
> must pass.

## Requirements

- Python 3.10–3.12
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Installation

```bash
git clone https://github.com/PSDN-AI/SONAR-OSS.git
cd SONAR-OSS
make setup            # creates .venv and installs with dev extras via uv
source .venv/bin/activate
```

Or with plain pip:

```bash
pip install -e ".[dev]"
```

Copy `.env.example` to `.env` and fill in only the values you need (API keys
are required only for optional hosted-model backends).

## Usage

See [`docs/USAGE.md`](docs/USAGE.md) for a short quickstart with runnable
examples (scoring a pair, evaluating a model over a dataset, listing models).

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
