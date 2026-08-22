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
- `ffmpeg` **required** by the pipeline-based ASR adapters — including the
  English defaults `whisper_base_en` / `whisper_small_en` and any `--hf-model`
  that falls back to the generic pipeline — for **all** audio input, WAV
  included (the `transformers` pipeline shells out to `ffmpeg` to decode file
  paths). These adapters refuse to load without it and name the missing
  binary. Adapters that decode audio themselves (the `wav2vec2_*` models and
  the non-pipeline Whisper fine-tunes) evaluate WAV without `ffmpeg`; MP3 and
  some `pydub` paths need it regardless. Install: `sudo apt-get install
  ffmpeg` (Debian/Ubuntu) or `brew install ffmpeg` (macOS). The `[pyannote]`
  extra needs `ffmpeg` too: pyannote.audio 4.x decodes audio through
  torchcodec, which loads the system ffmpeg libraries at runtime

**Supported environments.** CI validates Linux x86_64 with CPython 3.10, 3.11,
and 3.12, plus macOS arm64 with CPython 3.12 and the `[ml]` extra installed
(so the HuggingFace model adapters are exercised on every PR, not just
importable). Windows is expected to work for the core package but is not
CI-validated; some optional extras have platform-sensitive dependencies
(`[korean]` needs a Java runtime at runtime, `[ml]`/`[pyannote]` pull large
PyTorch trees). Open an issue if an install fails on a supported Python.
The checked-in `.python-version` pins `uv` to CPython 3.12, so `make setup`
on a machine with no Python selected builds a supported interpreter instead
of whatever newest version `uv` manages (CI passes `--python` explicitly and
is unaffected).

**What a first run costs.** On a fresh machine, budget several gigabytes of
downloads and tens of minutes before the first number appears; everything is
cached, so later runs against the same data and models start in seconds.
Measured example (Bengali FLEURS + one CTC model): ~3.4 GB / ~27 min for the
dataset, 1.26 GB for the model checkpoint, and ~1.5 GB on disk for the `[ml]`
extra (torch alone ~0.5 GB). `psdn-sonar discover --max-samples` bounds how
many samples are *prepared*, not the download — each requested split is
fetched into the HuggingFace cache in full on first run. Two more downloads
happen lazily: the first POSEIDON / semantic-similarity call fetches the
~64 MB sentence-transformers scorer, and audio-quality analysis fetches a
~390 MB UTMOS checkpoint via `torch.hub`. Checkpoint size can also hide
behind an adapter: `khushids_bengali` is a 62 MB PEFT adapter whose
`openai/whisper-large-v3` base adds ~2.9 GB.

**Compute device and runtime.** Local HuggingFace adapters auto-select the
best available device — CUDA, then MPS (Apple Silicon), then CPU — and the
device used is recorded in each run's `scores_<model>.json` (`config.device`),
since a GPU run and a CPU run can produce different transcripts for the same
audio. On CPU, architecture dominates runtime: measured on the same
200-utterance FLEURS Bengali set, a Wav2Vec2 CTC model ran at ~1.2 s/sample
while a Whisper-medium fine-tune ran at ~19 s/sample. A full multi-model
language default (Bengali has 9 models, mostly Whisper-class) is a
multi-hour job without a GPU — trim `--models` and `--max-samples` to size
your run first.

## Installation

### Install the pre-release from TestPyPI (for testing)

This repository is version **`0.1.0`**. A TestPyPI wheel may still be
published under an older dev tag (historically `0.1.0.dev2`). Check
[TestPyPI](https://test.pypi.org/project/psdn-sonar/) for the latest
filename before copying the commands below. The package is not yet on
PyPI. To install a TestPyPI wheel exactly as released:

1. Create and activate a fresh Python 3.10–3.12 virtual environment. These
   examples use Python 3.12; substitute 3.10 or 3.11 if needed.

   macOS or Linux (bash/zsh):

   ```bash
   python3.12 --version
   python3.12 -m venv sonar-env
   source sonar-env/bin/activate
   ```

   Windows PowerShell:

   ```powershell
   py -3.12 --version
   py -3.12 -m venv sonar-env
   .\sonar-env\Scripts\Activate.ps1
   ```

2. Download only the released wheel from TestPyPI, without resolving its
   dependencies:

   ```text
   python -m pip download --index-url https://test.pypi.org/simple/ --no-deps --only-binary=:all: --no-cache-dir --dest testpypi-dist "psdn-sonar==0.1.0.dev2"
   # If 0.1.0 is on TestPyPI, substitute psdn-sonar==0.1.0 and the matching wheel name below.
   ```

3. Install that wheel, resolving dependencies from PyPI only:

   ```text
   python -m pip install --index-url https://pypi.org/simple/ --no-cache-dir "testpypi-dist/psdn_sonar-0.1.0.dev2-py3-none-any.whl"
   ```

4. Verify the install:

   ```bash
   psdn-sonar --version                                           # matches the wheel you installed
   python -c "import psdn_sonar; print(psdn_sonar.__version__)"
   ```

Then follow [`docs/USAGE.md`](docs/USAGE.md) for runnable examples. To install
an optional extra in step 3, append it to the wheel path, for example
`"testpypi-dist/psdn_sonar-0.1.0.dev2-py3-none-any.whl[ml]"`. Once `0.1.0` is
released, this section becomes a plain `pip install psdn-sonar`.

### Contributor install (from source)

Contributors install the frozen, locked environment (exactly what CI runs):

```bash
git clone https://github.com/PSDN-AI/SONAR-OSS.git
cd SONAR-OSS
make setup            # uv sync --frozen with dev extras — does NOT include [ml]
# or, to run the docs/USAGE.md examples (local models, semantic similarity):
make setup-ml         # dev + [ml] extras, ~1.5 GB on disk
source .venv/bin/activate
```

Or with plain pip (editable, freshly resolved). **Create and activate a
virtual environment first** — exactly as in step 1 of the TestPyPI section
above. On "externally managed" interpreters (PEP 668: Homebrew and
Debian/Ubuntu system Pythons) pip otherwise refuses with
`error: externally-managed-environment` and installs nothing:

```bash
pip install -e ".[dev]"       # contributor tooling only — no [ml]
pip install -e ".[dev,ml]"    # what the docs/USAGE.md examples need
```

The `[ml]` extra is what runs local HuggingFace models and POSEIDON's
semantic similarity; without it the USAGE examples fail with a `TypeError`
naming the extra. Add it to an existing `make setup` environment with
`uv pip install -e ".[ml]"`.

One model in the Bengali defaults needs more than `[ml]`: `khushids_bengali`
is a PEFT/LoRA adapter and requires `peft` from the `[bengali]` extra
(`pip install "psdn-sonar[bengali]"`). Without it, a `--language bn` run
skips that model with a message naming the extra and evaluates the rest of
the defaults; one unavailable model never aborts a multi-model run.

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

Before comparing published numbers, read
[`docs/SCORE_INTERPRETATION.md`](docs/SCORE_INTERPRETATION.md): what the
scores measure — multi-speaker results include preprocessing error, scores
are comparable within a dataset only, and cells where a model is evaluated
on its declared training corpus carry an in-domain marker.

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
