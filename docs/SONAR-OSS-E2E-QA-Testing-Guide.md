# SONAR OSS End-to-End QA Testing Guide

**Package:** `psdn-sonar`
**Repository:** [PSDN-AI/SONAR-OSS](https://github.com/PSDN-AI/SONAR-OSS)
**Source version in this checkout:** `0.1.0` (`psdn_sonar/__init__.py`, `pyproject.toml`)
**Audience:** a QA engineer installing SONAR as a new external user
**Verified against:** CLI, model registry, dataset registry, language processors, and docs in this repository (not invented from older write-ups)

How to use this document:

1. Work top to bottom. Most commands are copy-paste.
2. Fill the **Status** column in the matrix as you go (`PASS` / `FAIL` / `BLOCKED`).
3. If documentation and the live CLI disagree, treat the **live CLI / source** as truth and record the mismatch in section 12.
4. Prefer an explicit `--models` or `--hf-model`. Omitting `--models` on `single` now runs default **local** models only; hosted API defaults are skipped unless their keys are set. Always pass `--language`.

---

## Compact test matrix

| Test | Language | Dataset | Model | Single/Multi | Expected Result | Status |
| ---- | -------- | ------- | ----- | ------------ | --------------- | ------ |
| INST-01 Fresh venv + core install | — | — | — | — | `psdn-sonar --version` prints a version; `import psdn_sonar` works | |
| INST-02 `[ml]` extra | — | — | — | — | `import torch`, `import transformers` succeed | |
| INST-03 Editable/source install | — | — | — | — | Same CLI as INST-01 from a repo checkout | |
| INST-04 Optional extras present | — | — | — | — | `[hindi]`, `[bengali]`, `[korean]`, `[pyannote]` install or fail with a clear extra name | |
| SMOKE-01 CLI help/version | — | — | — | — | Four subcommands: `single`, `multi`, `custom`, `discover` | |
| SMOKE-02 Discover dry-run (all 4 langs) | en/hi/bn/ko | registry | — | — | en: fleurs+voxpopuli; hi/bn: fleurs; ko: fleurs+zeroth | |
| DISC-EN-FLEURS | en | `google/fleurs` `en_us` | — | — | `data/qa/en/fleurs/{train,val,test}.tsv` + `metadata.json` | |
| DISC-EN-VOXPOPULI | en | `facebook/voxpopuli` `en` | — | — | `data/qa/en/voxpopuli/{train,val,test}.tsv` | |
| DISC-HI-FLEURS | hi | `google/fleurs` `hi_in` | — | — | `data/qa/hi/fleurs/{train,val,test}.tsv` | |
| DISC-BN-FLEURS | bn | `google/fleurs` `bn_in` | — | — | `data/qa/bn/fleurs/{train,val,test}.tsv` | |
| DISC-KO-FLEURS | ko | `google/fleurs` `ko_kr` | — | — | `data/qa/ko/fleurs/{train,val,test}.tsv` | |
| DISC-KO-ZEROTH | ko | `Bingsu/zeroth-korean` | — | — | `data/qa/ko/zeroth/{train,test}.tsv` (no val split upstream) | |
| E2E-EN | en | FLEURS test (≤10) | `whisper_base_en` | Single | CSV + `scores_*.json`; WER/CER/sem/POSEIDON present | |
| E2E-EN-HF | en | FLEURS test (≤5) | `--hf-model openai/whisper-tiny` | Single | Results saved as `custom_openai_whisper_tiny` | |
| E2E-HI | hi | FLEURS test (≤10) | `whisper_small_hi` | Single | Hindi processor used; metrics + scores.json | |
| E2E-BN | bn | FLEURS test (≤10) | `wav2vec2_bengali` | Single | Registered Bengali CTC model; Bengali-script predictions; metrics + scores.json | |
| E2E-BN-HF | bn | FLEURS test (≤5) | `--hf-model openai/whisper-small` | Single | BYO mechanics only; WER ≈ 1.0 expected (whisper-small hallucinates on bn) | |
| E2E-KO | ko | FLEURS test (≤10) | `whisper_small_ko` | Single | Korean processor used; metrics + scores.json | |
| E2E-KO-Z | ko | Zeroth test (≤5) | `wav2vec2_base_korean` | Single | Second Korean public corpus evaluates | |
| MULTI-01 Fixture + `no_trim` | en | Constructed 2-speaker FLEURS mix | `whisper_base_en` | Multi | Per-speaker CSV rows; `best_method=no_trim` | |
| MULTI-02 Auto method | en | Same fixture | `whisper_base_en` | Multi | Completes; method logged (`energy_trim` / `no_trim` / …) | |
| MULTI-03 Missing pyannote | en | Same fixture | `whisper_base_en` + `--method pyannote_vad` | Multi | Clear skip/error; no silent hang | |
| MULTI-04 Missing HF token | en | Same fixture | `whisper_base_en` + pyannote | Multi | Actionable auth error, not a stack dump only | |
| CLI-SINGLE | en | FLEURS | `whisper_base_en` | Single | Happy path (covered by E2E-EN) | |
| CLI-MULTI | en | Fixture | `whisper_base_en` | Multi | Happy path (covered by MULTI-01) | |
| CLI-CUSTOM | pt | YAML + FLEURS `pt_br` or local TSV | `openai/whisper-small` | Custom | HF model evaluated; API models skipped if keys absent | |
| CLI-DISCOVER | — | — | — | — | `--help` + dry-run + small prepare | |
| NORM-EN/HI/BN/KO | all | Controlled strings | Python API | — | Exact normalize + WER=0 on equivalent pairs | |
| NEG-* | various | Synthetic | various | — | Clear error or explicit per-row `error` | |
| REPRO-01 | en | Same TSV | `whisper_base_en` | Single | Second run writes `scores_*.json` with seed/git/model + `lineage` (HF revision, normalization contract) | |

---

## What SONAR actually supports today

Recorded from the running CLI and source. Use this as the contract.

### Public CLI

```text
psdn-sonar {single,custom,multi,discover}
```

There is no other public console script. `psdn-sonar --version` in this checkout prints `psdn-sonar 0.1.0`.

### Discoverable datasets (`psdn_sonar/data/registry.py`)

| Registry name | Hugging Face ID | Languages among en/hi/bn/ko | Configs |
| --- | --- | --- | --- |
| `fleurs` | `google/fleurs` @ `70bb2e84b976b7e960aa89f1c648e09c59f894dd` | en, hi, bn, ko | `en_us`, `hi_in`, `bn_in`, `ko_kr` |
| `voxpopuli` | `facebook/voxpopuli` @ `42f01879c780b4a2e90ec0b4f616c2ece526e4f1` | **en only** | `en` |
| `zeroth` | `Bingsu/zeroth-korean` @ `bd173fe2c8ed0dccd47acb4eda77542593651622` | **ko only** | none (default config) |

**Not available via `psdn-sonar discover`:**

- Mozilla Common Voice — present in `benchmark_catalog.yaml` with `enabled: false`. Not in `DATASET_REGISTRY`.
- OpenSLR 37 / 53 — catalogued, and `psdn_sonar.core` has loaders, but they are **not** wired into `discover`.

Verified dry-run output:

```text
en → fleurs (en_us), voxpopuli (en)
hi → fleurs (hi_in)
bn → fleurs (bn_in)
ko → fleurs (ko_kr), zeroth (Bingsu/zeroth-korean)
```

### Registered local ASR models (`psdn_sonar/models/registry.py`)

Use these IDs with `--models`. Do not invent names such as `whisper-ko` or `whisper-hindi`.

| SONAR ID | Hugging Face repo | Language | Approx. size / notes |
| --- | --- | --- | --- |
| `whisper_base_en` | `openai/whisper-base` | en | ~74M params, ~150–300 MB. **Preferred English smoke model.** |
| `whisper_small_en` | `openai/whisper-small` | en | ~244M params, ~500 MB |
| `whisper_small_hi` | `openai/whisper-small` (`language=hi`) | hi | Same weights as whisper-small; forced Hindi decode. **Preferred Hindi smoke model.** |
| `whisper_hindi_large_v2` | `vasista22/whisper-hindi-large-v2` | hi | Whisper large-v2 class, ~1.5 GB. Skip for smoke. |
| `wav2vec2_bengali` | `arijitx/wav2vec2-xls-r-300m-bengali` | bn | ~300M CTC, ~1.2 GB. **Preferred Bengali smoke model** — do not use generic `openai/whisper-small` for Bengali (cross-script hallucinations, WER ≈ 1.04). |
| `banglaspeech2text` | `anuragshas/whisper-large-v2-bn` | bn | large-v2. Skip for smoke. |
| `khushids_bengali` | `KhushiDS/whisper-large-v3-Bengali` | bn | PEFT on whisper-large-v3; needs `[bengali]` (`peft`). Skip for smoke. |
| `tugstugi_bengali` | `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` | bn | medium. Skip for smoke. |
| `tugstugi_bengali_regional` | `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium` | bn | medium. Skip for smoke. |
| `banglaasr` | `bangla-speech-processing/BanglaASR` | bn | Whisper-class. Skip for smoke. |
| `banglaasr_v5` | `arif11/bangla-ASR-v5` | bn | Not in language defaults. |
| `whisper_small_ko` | `SungBeom/whisper-small-ko` | ko | ~244M. **Preferred Korean smoke model.** |
| `wav2vec2_base_korean` | `Kkonjeong/wav2vec2-base-korean` | ko | ~360 MB. Good second Korean model. |
| `kresnik_wav2vec2_large_xlsr_korean` | `kresnik/wav2vec2-large-xlsr-korean` | ko | large XLS-R, ~1.2 GB |
| `wav2vec2_xlsr_korean` | same as kresnik | ko | Alias of the large Korean XLS-R model |
| `elevenlabs_api` | hosted | all defaults | Needs `ELEVENLABS_API_KEY`. **Not for OSS smoke.** |
| `whisper_api` | hosted OpenAI | all defaults | Needs `OPENAI_API_KEY`. **Not for OSS smoke.** |
| `assemblyai_api` | hosted | all defaults | Needs `ASSEMBLYAI_API_KEY`. **Not for OSS smoke.** |

Bring-your-own Hugging Face model (single-speaker only, fully implemented):

```bash
psdn-sonar single --input DATA.tsv --hf-model openai/whisper-tiny --language en
```

Results are written under the sanitized name `custom_<org>_<model>` with `/` and `-` replaced by `_`. Example: `openai/whisper-tiny` → `custom_openai_whisper_tiny`.

`--hf-model` on `multi` is supported (same sanitized results name as `single`). Always pass `--language` so WER/CER use the correct normalizer.

### Language processors

| Code | Processor | Extra for full fidelity | WER path |
| --- | --- | --- | --- |
| `en` | `EnglishProcessor` | none | processor `normalize()` |
| `hi` | `HindiProcessor` | `[hindi]` (`indic-nlp-library`) | processor `normalize()`; falls back to NFC |
| `bn` | `BengaliProcessor` | `[bengali]` (`bnlp_toolkit`) | **eval uses `normalize_bengali_for_wer()`, not `BengaliProcessor.normalize()`** |
| `ko` | `KoreanProcessor` | `[korean]` (needs a **Java** runtime for konlpy/MeCab) | processor `normalize()`; G2P/spacing extras optional |

Supported CLI language aliases for default-model lookup: `en`/`english`, `hi`/`hindi`, `bn`/`bengali`, `ko`/`korean`.

**Default `--language` is `bn`.** Forgetting `--language en` on English audio will score with the Bengali WER normalizer.

---

## 1. Environment setup

### 1.1 Purpose

Prove a new user can install SONAR in a clean environment and reach a working CLI.

### 1.2 System prerequisites

| Requirement | Why | How to check |
| --- | --- | --- |
| Python **3.10, 3.11, or 3.12** | README supported range. `requires-python = ">=3.10"` has no upper bound, but CI only validates 3.10–3.12. | `python3.12 --version` |
| ~10 GB free disk | torch + one small Whisper + FLEURS subset + sentence-transformers | `df -h .` |
| 8 GB RAM (16 GB safer) | Whisper-small + MOS scorers | — |
| Network | Hugging Face Hub for datasets and models | — |
| `ffmpeg` (**required** for pipeline adapters) | The `transformers` ASR pipeline shells out to ffmpeg to decode file paths — **WAV included**. This covers `whisper_base_en`, `whisper_small_en`, `khushids_bengali`, and any `--hf-model` on the generic-pipeline path; those adapters refuse to load without it (see 10.28). Adapters that decode audio themselves (`wav2vec2_*`, non-pipeline Whisper fine-tunes) evaluate WAV without it. MP3 and some `pydub` paths need it regardless. | `ffmpeg -version` |
| Java runtime (only if installing `[korean]`) | konlpy/MeCab. Core Korean number/loanword normalization works without it. | `java -version` |

Install ffmpeg if missing (Linux):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

Optional but recommended Hugging Face login (higher rate limits; **required** later for gated pyannote models):

```bash
# After INST-02, from the activated venv
huggingface-cli login
# or
export HF_TOKEN=hf_...
```

Pyannote models used by SONAR are gated:

- `pyannote/segmentation-3.0` (VAD)
- `pyannote/speaker-diarization-3.1` (diarization)

Accept the terms on those Hugging Face model pages while logged in, then set `HF_TOKEN` in `.env`. Hugging Face ASR checkpoints listed above are public; they do not require a token.

### 1.3 INST-01 — Fresh environment + documented package install

**Purpose:** Install the published package the way a new user would.

**Prerequisites:** Python 3.10–3.12, internet.

**Commands:**

```bash
python3.12 --version
python3.12 -m venv sonar-qa-env
source sonar-qa-env/bin/activate   # Windows: .\sonar-qa-env\Scripts\Activate.ps1
python -m pip install -U pip

# Documented TestPyPI path from README.md (see mismatch INST-DOC below)
python -m pip download --index-url https://test.pypi.org/simple/ \
  --no-deps --only-binary=:all: --no-cache-dir --dest testpypi-dist \
  "psdn-sonar==0.1.0.dev2"

python -m pip install --index-url https://pypi.org/simple/ --no-cache-dir \
  "testpypi-dist/psdn_sonar-0.1.0.dev2-py3-none-any.whl"
```

**Expected result:**

```bash
psdn-sonar --version
# README promises: psdn-sonar 0.1.0.dev2
python -c "import psdn_sonar; print(psdn_sonar.__version__)"
```

**Pass/fail:**

- PASS if the wheel installs and the CLI runs.
- FAIL if the wheel is missing, dependencies fail, or the CLI is not on `PATH`.
- **Record INST-DOC:** this checkout is version `0.1.0`, not `0.1.0.dev2`. If TestPyPI no longer has `0.1.0.dev2`, install from source (INST-03) and file the README mismatch.

### 1.4 INST-02 — `[ml]` extra

**Purpose:** Local Hugging Face models, semantic similarity, and POSEIDON need `[ml]` (torch, torchaudio, transformers, sentence-transformers, speechmos, onnxruntime).

**Prerequisites:** INST-01 succeeded, or INST-03.

**Commands (TestPyPI wheel):**

```bash
python -m pip install --index-url https://pypi.org/simple/ --no-cache-dir \
  "testpypi-dist/psdn_sonar-0.1.0.dev2-py3-none-any.whl[ml]"
```

**Commands (source / editable — use this if TestPyPI is stale):**

```bash
python -m pip install "psdn-sonar[ml]"          # if a wheel/index is available
# or from a clone:
python -m pip install -e ".[ml]"
```

**Expected result:**

```bash
python -c "import torch, transformers, sentence_transformers; print(torch.__version__)"
```

**Pass/fail:** PASS if those imports succeed. FAIL if torch/transformers are missing and `psdn-sonar single --models whisper_base_en` later dies with an import error.

Core WER/CER (`jiwer`, language processors) work without `[ml]`. Semantic similarity and HF ASR do not.

### 1.5 INST-03 — Editable / contributor install from the repository

**Purpose:** Confirm the source-checkout path a contributor (and this QA workspace) uses.

**Prerequisites:** git, Python 3.10–3.12. `uv` is optional.

**Commands:**

```bash
git clone https://github.com/PSDN-AI/SONAR-OSS.git
cd SONAR-OSS

# Path A — documented contributor path (locked, what CI runs)
# requires: https://docs.astral.sh/uv/
make setup
source .venv/bin/activate

# Path B — documented pip editable path (fresh resolve, not uv.lock)
python3.12 -m venv .venv-pip
source .venv-pip/bin/activate
python -m pip install -U pip
python -m pip install -e ".[ml,dev]"
```

USAGE.md currently shows only `pip install -e ".[ml]"` — that is a source-tree command, not a PyPI one-liner.

**Expected result:**

```bash
psdn-sonar --version          # this checkout: psdn-sonar 0.1.0
python -c "from psdn_sonar.models.registry import list_models; print(list_models())"
```

Expected `list_models()` set:

```text
assemblyai_api, banglaasr, banglaasr_v5, banglaspeech2text, elevenlabs_api,
khushids_bengali, kresnik_wav2vec2_large_xlsr_korean, tugstugi_bengali,
tugstugi_bengali_regional, wav2vec2_base_korean, wav2vec2_bengali,
wav2vec2_xlsr_korean, whisper_api, whisper_base_en, whisper_hindi_large_v2,
whisper_small_en, whisper_small_hi, whisper_small_ko
```

**Pass/fail:** PASS if version is `0.1.0` and the model list matches. FAIL if the CLI is missing or the registry is empty.

### 1.6 INST-04 — Optional extras and auth files

**Purpose:** Confirm extras exist and fail clearly when skipped.

**Commands:**

```bash
# From the same venv as INST-03
python -m pip install -e ".[hindi]"      # indic-nlp-library; Hindi unicode/tokenize
python -m pip install -e ".[bengali]"    # bnlp, peft, …; Bengali tokenize + KhushiDS PEFT
python -m pip install -e ".[korean]"     # konlpy/g2pk/jamo; needs Java at runtime
python -m pip install -e ".[pyannote]"   # pyannote.audio>=3.1,<4; needs [ml]/torch
# Full kitchen sink (large):
# python -m pip install -e ".[all]"
```

Copy env template (API keys are **not** required for the local-model tests in this guide):

```bash
cp .env.example .env
# Uncomment and set only what you need:
# HF_TOKEN=hf_...          # required for pyannote VAD/diarization
# OPENAI_API_KEY=...       # only for whisper_api
# ELEVENLABS_API_KEY=...   # only for elevenlabs_api
# ASSEMBLYAI_API_KEY=...   # only for assemblyai_api
```

**Expected result:** each extra installs, or pip names the extra and the missing package. Korean extra may succeed at pip-time and fail at tokenize-time without Java — that is acceptable if the error mentions konlpy/Java.

**Pass/fail:** PASS if extras are selectable and missing-extra errors name `psdn-sonar[…]`. FAIL if extras are undeclared or install silently does nothing.

### 1.7 Working directory for the rest of this guide

```bash
export QA_ROOT="$PWD/qa-work"
mkdir -p "$QA_ROOT"
cd "$QA_ROOT"
# If you installed from a clone, keep the clone path:
# export SONAR_SRC=/path/to/SONAR-OSS
```

Commands below assume the venv is still active.

---

## 2. Smoke test

### 2.1 SMOKE-01 — CLI surface

**Purpose:** Every public entry point exists and `--help` matches the implementation.

**Prerequisites:** INST-01 or INST-03.

**Commands:**

```bash
psdn-sonar --help
psdn-sonar --version
psdn-sonar single --help
psdn-sonar multi --help
psdn-sonar custom --help
psdn-sonar discover --help
```

**Expected result (captured from this checkout):**

Top-level modes: `single`, `custom`, `multi`, `discover`.

`single` flags: `--input` (required), `--models`, `--hf-model`, `--language` (default `bn`), `--output` (default `results/single-speaker-eval`), `--max-samples` (default `0` = all), `--significant-wer-threshold`, `--report`.

`multi` flags: `--input` (required, manifest.jsonl), `--models`, `--hf-model`, `--output` (default `results/multispeaker-eval`), `--max-samples`, `--language` (default `bn`), `--method` (`energy_trim`, `timestamp_trim`, `no_trim`, `pyannote_vad`), `--sweep`, `--demographics`, `--dataset-dir`, `--report`.

`custom` flags: `--config` (required), `--output`, `--max-samples`, `--report`.

`discover` flags: `--language` (required), `--output`, `--datasets`, `--max-samples`, `--split-ratio`, `--skip-audio-validation`, `--validate`, `--dry-run`.

No mode:

```text
psdn-sonar: error: the following arguments are required: mode
```

Exit code `2`.

**Pass/fail:** PASS if all four subcommands exist and the flags above appear. FAIL if a documented extra subcommand appears or one of these four is missing.

### 2.2 SMOKE-02 — Discover dry-run for all four languages

**Purpose:** Confirm the live dataset registry before downloading anything.

**Commands:**

```bash
psdn-sonar discover --language en --dry-run
psdn-sonar discover --language hi --dry-run
psdn-sonar discover --language bn --dry-run
psdn-sonar discover --language ko --dry-run
```

**Expected result:** tables matching the discoverable-dataset section above. `xx` is checked later under negative tests.

**Pass/fail:** PASS if the four tables match. FAIL if Common Voice or OpenSLR appear (they must not) or if FLEURS is missing for any of the four languages.

### 2.3 Prepare small public subsets

**Purpose:** Download only enough audio for the rest of the guide.

**Prerequisites:** SMOKE-02, Hugging Face Hub reachable. FLEURS is CC-BY-4.0.

`--max-samples` limits **per split**. FLEURS has train/validation/test, so `10` yields about 30 clips per language. `--skip-audio-validation` skips SNR during prepare (evaluation still computes SNR).

**Commands:**

```bash
# English — FLEURS
psdn-sonar discover --language en --datasets fleurs \
  --max-samples 10 --skip-audio-validation --output data/qa/en

# English — VoxPopuli (second supported public corpus)
psdn-sonar discover --language en --datasets voxpopuli \
  --max-samples 10 --skip-audio-validation --output data/qa/en

# Hindi — FLEURS only (no second discoverable public corpus)
psdn-sonar discover --language hi --datasets fleurs \
  --max-samples 10 --skip-audio-validation --output data/qa/hi

# Bengali — FLEURS only
psdn-sonar discover --language bn --datasets fleurs \
  --max-samples 10 --skip-audio-validation --output data/qa/bn

# Korean — FLEURS
psdn-sonar discover --language ko --datasets fleurs \
  --max-samples 10 --skip-audio-validation --output data/qa/ko

# Korean — Zeroth (second supported public corpus; splits are train, test)
psdn-sonar discover --language ko --datasets zeroth \
  --max-samples 10 --skip-audio-validation --output data/qa/ko
```

**Expected result per dataset directory:**

```text
data/qa/<lang>/<dataset>/
  train.tsv          # fleurs; zeroth also has train
  val.tsv            # fleurs only (HF name "validation" is rewritten to val)
  test.tsv
  metadata.json
  audio/*.wav
```

`test.tsv` columns written by `DatasetPreparer`:

```text
audio_path	transcription	transcription_norm	duration_s	snr_db
```

`psdn-sonar single` only **requires** `audio_path` and `transcription`. Extra columns are ignored.

`metadata.json` must include `source`, `source_revision`, `config`, `language`, `split_sizes`, `download_date`.

**Workspace shortcut:** this checkout already has untracked FLEURS 10-clip subsets at `data/e2e/{en,hi,bn,ko}/fleurs/` from a prior prepare. Those TSVs contain **absolute** paths and only work on the machine that created them. Prefer `data/qa/...` from the commands above.

**Pass/fail:** PASS if `test.tsv` has a header plus up to 10 data rows and every `audio_path` exists. FAIL if prepare exits 1 or audio files are missing.

**Do not run** `psdn-sonar discover --language en` without `--datasets`. That downloads **both** FLEURS and VoxPopuli at `--max-samples 0` (all).

---

## 3. English E2E

### 3.1 E2E-EN — Registered model, full metric path

**Purpose:** public dataset → model → transcription → English normalization → WER/CER → semantic similarity → POSEIDON → files.

**Prerequisites:** INST-02, DISC-EN-FLEURS. First run downloads `openai/whisper-base` and `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~500 MB).

**Command:**

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 10 \
  --output results/e2e-en \
  --report
```

**Expected result:**

Log lines include `Single-speaker evaluation`, `Loaded N samples`, `Evaluating model: whisper_base_en`, then an aggregate table with CER, WER, Semantic Similarity, POSEIDON.

Files:

```text
results/e2e-en/asr_detailed_whisper_base_en.csv
results/e2e-en/scores_whisper_base_en.json
results/e2e-en/analysis/whisper_base_en/EVAL_REPORT.md
```

`--report` also writes plot directories under `analysis/whisper_base_en/` (diversity, cross-dataset, hard-negatives, audio-quality, latency). Plot steps may warn and continue; missing `EVAL_REPORT.md` is a fail.

**`asr_detailed_*.csv` required columns** (from `SingleSpeakerEvaluator._result_row`):

| Column | Must |
| --- | --- |
| `audio_path` | existing wav path |
| `ground_truth` | non-empty reference |
| `prediction` | model text on success |
| `wer`, `cer` | floats on success |
| `semantic_similarity` | float in `[0, 1]` when `[ml]` is installed |
| `poseidon_score` | float in `[0, 1]` when sem + WER/CER exist |
| `significant_wer` | boolean vs threshold (default `0.30`) |
| `inference_latency_s` | float |
| `snr_db`, `clipping_ratio`, `silence_ratio`, `snr_tier` | audio quality |
| `error` | empty/`None` on success; `"Empty prediction"` or exception text on failure |

MOS columns (`dnsmos_*`, UTMOS, SQUIM) may be null if those optional scorers fail to load. That is not a fail by itself.

**`scores_whisper_base_en.json` schema** (from `psdn_sonar/benchmark/scores.py`):

```json
{
  "submission": {
    "provider": "local",
    "model_snapshot": "whisper_base_en",
    "region": "local",
    "protocol": "batch",
    "inference_params": {"language_code": "en"},
    "seed": 42,
    "git_sha": "<40-char sha or unknown>",
    "package_version": "0.1.0",
    "timestamp_utc": "<ISO-8601 Z>"
  },
  "model_name": "whisper_base_en",
  "lineage": {
    "hf_model_id": "openai/whisper-base",
    "hf_revision": "<40-char checkpoint sha>",
    "normalization": "en:v2"
  },
  "aggregate": {
    "cer_mean": 0.0,
    "wer_mean": 0.0,
    "semantic_similarity_mean": 0.0,
    "poseidon_score_mean": 0.0,
    "significant_wer_rate": 0.0,
    "significant_wer_threshold": 0.3,
    "total_samples": 10,
    "successful": 10,
    "failed": 0,
    "elapsed_time_s": 0.0
  },
  "utterances": [
    {
      "audio_path": "...",
      "wer": 0.0,
      "cer": 0.0,
      "semantic_similarity": 0.0,
      "poseidon_score": 0.0,
      "significant_wer": false,
      "inference_latency_s": 0.0,
      "error": null
    }
  ]
}
```

Numeric values above are placeholders. **Success means the keys exist**, `successful > 0`, and aggregate WER is a finite float. FLEURS English with whisper-base is typically well below 0.30 mean WER; do not fail the release on a slightly high WER if transcriptions are clearly English.

POSEIDON formula (defaults): `0.35*(1-WER) + 0.20*(1-CER) + 0.45*similarity`.

**Pass/fail:**

- PASS: audio loads, model loads, predictions are English-like, all four metrics populate, both files exist, `--language en` is recorded in `submission.inference_params.language_code`.
- FAIL: process exits 1, all rows have `error`, sem/POSEIDON are all null with `[ml]` installed, or `language_code` is `bn`.

### 3.2 E2E-EN-HF — Bring-your-own `--hf-model`

**Purpose:** `--hf-model` path (`CustomHuggingFaceModel`).

**Command:**

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --hf-model openai/whisper-tiny \
  --language en \
  --max-samples 5 \
  --output results/e2e-en-hf
```

**Expected result:** files `asr_detailed_custom_openai_whisper_tiny.csv` and `scores_custom_openai_whisper_tiny.json`. Log: `Using custom HuggingFace model: openai/whisper-tiny`.

**Pass/fail:** PASS if those filenames exist and `successful > 0`. FAIL if the CLI rejects `--hf-model` or writes the wrong model name.

### 3.3 E2E-EN-VP — Second English public dataset

**Purpose:** VoxPopuli English is the only second discoverable English corpus.

**Command:**

```bash
psdn-sonar single \
  --input data/qa/en/voxpopuli/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 5 \
  --output results/e2e-en-voxpopuli
```

**Expected result:** same schema as E2E-EN. Parliamentary speech may yield higher WER than FLEURS.

**Pass/fail:** PASS if evaluation completes with `successful > 0`. FAIL if the TSV is empty or the model cannot read the audio.

---

## 4. Hindi E2E

### 4.1 E2E-HI — Registered Hindi model

**Purpose:** FLEURS Hindi → `whisper_small_hi` → Devanagari normalization → metrics.

**Prerequisites:** DISC-HI-FLEURS. Downloads `openai/whisper-small` if not cached (~500 MB).

**Command:**

```bash
psdn-sonar single \
  --input data/qa/hi/fleurs/test.tsv \
  --models whisper_small_hi \
  --language hi \
  --max-samples 10 \
  --output results/e2e-hi \
  --report
```

**Expected result:**

- `results/e2e-hi/asr_detailed_whisper_small_hi.csv`
- `results/e2e-hi/scores_whisper_small_hi.json` with `"language_code": "hi"`
- Predictions contain Devanagari (`validate_text` range U+0900–U+097F) for most rows
- WER/CER/sem/POSEIDON populated

There is **no second Hindi dataset** in `discover`. Do not invent a Common Voice or OpenSLR Hindi discover command.

**Pass/fail:** PASS if files exist, `successful > 0`, and references in the TSV are Devanagari. FAIL if the run used default `bn` (forgot `--language hi`) or the model failed to load.

### 4.2 E2E-HI-HF — BYO model forced to Hindi

**Command:**

```bash
psdn-sonar single \
  --input data/qa/hi/fleurs/test.tsv \
  --hf-model openai/whisper-small \
  --language hi \
  --max-samples 5 \
  --output results/e2e-hi-hf
```

**Expected result:** `CustomHuggingFaceModel` maps `--language hi` to Whisper’s long name `hindi` for generate kwargs. Files under `custom_openai_whisper_small`.

**Pass/fail:** PASS if predictions are mostly Hindi script. FAIL if the model ignores language and emits only Latin (note this as a product issue, not a QA procedure error).

---

## 5. Bengali E2E

### 5.1 E2E-BN — Registered Bengali model (recommended smoke)

**Purpose:** FLEURS Bengali → `wav2vec2_bengali` (`arijitx/wav2vec2-xls-r-300m-bengali`, ~1.2 GB) → Bengali normalization → metrics.

Do **not** smoke-test Bengali with generic `openai/whisper-small`: on Bengali audio it hallucinates Latin/wrong-script output against Bengali references (measured WER ≈ 1.04). A registered Bengali checkpoint is required for meaningful Bengali numbers.

**Prerequisites:** DISC-BN-FLEURS. First load may take several minutes.

**Command:**

```bash
psdn-sonar single \
  --input data/qa/bn/fleurs/test.tsv \
  --models wav2vec2_bengali \
  --language bn \
  --max-samples 10 \
  --output results/e2e-bn \
  --report
```

**Expected result:**

- `asr_detailed_wav2vec2_bengali.csv`
- `scores_wav2vec2_bengali.json` with `"language_code": "bn"`
- Ground truth is Bengali script (U+0980–U+09FF)
- Metrics populated; predictions are mostly Bengali script

Bengali **evaluation** normalization is `normalize_bengali_for_wer()` (suffix splitting, nasals, number variants). That is **not** identical to `BengaliProcessor.normalize()`. Example: `এটি একটি পরীক্ষা` becomes `এ টি এক টি পরীক্ষা` on the WER path. See section 9.

**Pass/fail:** PASS if files exist, `successful > 0`, and predictions are Bengali script. FAIL if `Unknown model` or import/`peft` errors (this adapter does not need peft; `khushids_bengali` does), or if language_code is not `bn`.

### 5.2 E2E-BN-HF — BYO mechanics only (not a quality test)

**Purpose:** Exercise the `--hf-model` bring-your-own path on Bengali input. This validates the **mechanism** (custom model load, file naming, Bengali WER path), not transcription quality — generic multilingual Whisper hallucinates on Bengali, so expect WER near 1.0.

**Command:**

```bash
psdn-sonar single \
  --input data/qa/bn/fleurs/test.tsv \
  --hf-model openai/whisper-small \
  --language bn \
  --max-samples 5 \
  --output results/e2e-bn-hf
```

**Expected result:** `asr_detailed_custom_openai_whisper_small.csv` and `scores_custom_openai_whisper_small.json` written; run completes. WER ≈ 1.0 and non-Bengali predictions are **expected** here and are not a failure of the harness.

**Pass/fail:** PASS if files exist and the run exits 0. FAIL only on crash, `Unknown model`, or missing output files. Do not file the high WER as a bug; it is the documented behavior of this pairing.

There is **no second Bengali dataset** in `discover`. OpenSLR 37/53 exist only as catalog + `core.py` loaders, not as CLI discover targets.

---

## 6. Korean E2E

### 6.1 E2E-KO — Registered Korean Whisper

**Purpose:** FLEURS Korean → `whisper_small_ko` → Hangul normalization → metrics.

**Prerequisites:** DISC-KO-FLEURS. Downloads `SungBeom/whisper-small-ko`.

**Command:**

```bash
psdn-sonar single \
  --input data/qa/ko/fleurs/test.tsv \
  --models whisper_small_ko \
  --language ko \
  --max-samples 10 \
  --output results/e2e-ko \
  --report
```

**Expected result:**

- `asr_detailed_whisper_small_ko.csv` / `scores_whisper_small_ko.json`
- `"language_code": "ko"`
- Predictions mostly Hangul (U+AC00–U+D7AF)
- Four metrics populated

**Pass/fail:** PASS if `successful > 0` and files exist. FAIL if default language `bn` was used.

### 6.2 E2E-KO-Z — Second Korean public dataset

**Purpose:** Zeroth Korean via discover.

**Command:**

```bash
psdn-sonar single \
  --input data/qa/ko/zeroth/test.tsv \
  --models wav2vec2_base_korean \
  --language ko \
  --max-samples 5 \
  --output results/e2e-ko-zeroth
```

**Expected result:** `asr_detailed_wav2vec2_base_korean.csv`. Zeroth text is Korean; WER may differ from FLEURS.

**Pass/fail:** PASS if evaluation completes. FAIL if `test.tsv` is missing (Zeroth has `train` and `test` only — there is no `val.tsv` unless the preparer also wrote leftover split names).

---

## 7. Multi-speaker E2E

### 7.0 Read this before running anything

The live loader (`psdn_sonar/loaders/manifest.py`) requires:

```json
{
  "audio_id": "TEST001",
  "audio_filepaths": {"speaker_a": "sample_audio/TEST001/speaker_a.wav", "speaker_b": "sample_audio/TEST001/speaker_b.wav"},
  "transcript_filepath": "sample_audio/TEST001/transcript.json",
  "num_speakers": 2
}
```

Transcripts must be **JSON**, not `.txt`. Combined audio, if used, must be named:

```text
{directory_of_speaker_a}/{audio_id}_Combined_Audio.wav
```

`examples/test_manifest.jsonl` plus `examples/sample_audio/TEST001/` is a working format fixture (short tones, not real speech). You can smoke the CLI with it; use the FLEURS-mix fixture below for a real-speech E2E.

`--models` or `--hf-model` is required. Language auto-select is single-speaker only. Always pass `--language`.

Default packaged `multi_speaker_config.yaml` enables only `no_trim`. Omitting `--method` still auto-selects among whatever methods the config lists (here: `no_trim` only), unless you pass `--method` or `--sweep`.

Registered Hugging Face models have `supports_diarization = False`. `pyannote_diarize` / scribe-style per-clip diarization is skipped for them. Isolated-channel VAD (`pyannote_vad`) does not need ASR diarization support.

### 7.1 Build a legal two-speaker fixture from FLEURS

**Purpose:** Public, CC-BY-4.0 audio QA can download themselves.

**Prerequisites:** DISC-EN-FLEURS (or any two English FLEURS wavs + transcripts).

**Command:**

```bash
python << 'PY'
from pathlib import Path
import json
import csv
import numpy as np
import soundfile as sf

root = Path("data/qa/multi/conv_001")
root.mkdir(parents=True, exist_ok=True)

tsv = Path("data/qa/en/fleurs/test.tsv")
rows = list(csv.DictReader(tsv.open(encoding="utf-8"), delimiter="\t"))
assert len(rows) >= 2, "need at least two FLEURS clips"

def load(row):
    audio, sr = sf.read(row["audio_path"])
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), sr, row["transcription"]

a, sr_a, text_a = load(rows[0])
b, sr_b, text_b = load(rows[1])
assert sr_a == sr_b, "sample rates must match"
sr = sr_a

sf.write(root / "speaker_a.wav", a, sr)
sf.write(root / "speaker_b.wav", b, sr)

# Sequential "conversation": A then short gap then B (still two speakers).
gap = np.zeros(int(0.4 * sr), dtype=np.float32)
combined = np.concatenate([a, gap, b])
peak = np.max(np.abs(combined)) or 1.0
combined = combined / peak * 0.9
sf.write(root / "conv_001_Combined_Audio.wav", combined, sr)

dur_a = len(a) / sr
dur_b = len(b) / sr
transcript = {
    "segments": [
        {"speaker": "speaker_a", "text": text_a, "start": 0.0, "end": dur_a},
        {"speaker": "speaker_b", "text": text_b, "start": dur_a + 0.4, "end": dur_a + 0.4 + dur_b},
    ]
}
(root / "transcript.json").write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

manifest = {
    "audio_id": "conv_001",
    "audio_filepaths": {
        "speaker_a": "conv_001/speaker_a.wav",
        "speaker_b": "conv_001/speaker_b.wav",
    },
    "transcript_filepath": "conv_001/transcript.json",
    "num_speakers": 2,
}
Path("data/qa/multi/manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
print("Wrote data/qa/multi/manifest.jsonl")
PY
```

Attribution: clips originate from Google FLEURS (CC-BY-4.0). Keep FLEURS attribution in the QA report.

### 7.2 MULTI-01 — Happy path (`no_trim`)

**Purpose:** Isolated speaker files → preprocess → transcribe → per-speaker metrics.

**Command:**

```bash
psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method no_trim \
  --output results/e2e-multi
```

**Expected result:**

```text
results/e2e-multi/asr_eval_results_whisper_base_en_manifest.csv
results/e2e-multi/asr_eval_results_whisper_base_en_manifest.txt
```

CSV columns (`get_output_fieldnames()`):

```text
audio_id, speaker, best_method, path, transcription, transcription_norm,
asr_transcription, asr_transcription_norm_non, asr_transcription_norm_conv,
cer_non, wer_non, semantic_similarity_non, poseidon_score_non,
cer_conv, wer_conv, semantic_similarity_conv, poseidon_score_conv,
original_duration_s, trimmed_duration_s, snr_db, clipping_ratio,
silence_ratio, snr_tier, quality_warnings, inference_latency_s,
all_method_scores
```

Expect **two rows** (`speaker` A and B), `best_method=no_trim`, non-empty `asr_transcription` on success, and numeric `wer_non` / `cer_non`. Semantic/POSEIDON columns fill when sentence-transformers is installed.

`display_aggregate_stats` prefers `cer_conv`/`wer_conv` when those columns exist (they always do here).

**Pass/fail:** PASS if two speaker rows exist and at least one has WER. FAIL if both clips are skipped (`could not load transcript` / `both audio files missing`).

### 7.3 MULTI-02 — Automatic preprocessing selection

**Command:**

```bash
psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --output results/e2e-multi-auto
```

**Expected result:** Completes. With the packaged config, the only configured method is `no_trim`, so auto-select should pick `no_trim`. If you override methods (see `psdn_sonar/multi_speaker_config.yaml`) to include `energy_trim`, auto-select scores silence/duration and picks one method per clip **without** using ground truth (unlike `--sweep`).

**Pass/fail:** PASS if a CSV is written and `best_method` is populated. FAIL if `No valid preprocessing methods available`.

### 7.4 MULTI-03 — Missing pyannote extra

**Purpose:** Graceful behavior without `psdn-sonar[pyannote]`.

**Command (venv without pyannote):**

```bash
psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method pyannote_vad \
  --output results/e2e-multi-nopyannote
```

**Expected result:** A warning of the form:

```text
Skipping pyannote_vad: pyannote.audio not installed. Install with: pip install 'psdn-sonar[pyannote]'
```

Then either a fallback to remaining methods or:

```text
No valid preprocessing methods available
```

and exit code 1. Either is acceptable if the message is actionable. A hang or an unrelated traceback is a fail.

### 7.5 MULTI-04 — pyannote installed, credentials missing

**Prerequisites:** `pip install "psdn-sonar[pyannote]"` (and torch from `[ml]`). Unset token:

```bash
env -u HF_TOKEN psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method pyannote_vad \
  --output results/e2e-multi-notoken
```

**Expected result:** Load of `pyannote/segmentation-3.0` fails with an auth / gated-model / 401-style error. Evaluation should exit 1 or write failure rows, not hang.

To run the **success** path after accepting model terms:

```bash
# HF_TOKEN in .env
psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method pyannote_vad \
  --output results/e2e-multi-vad
```

Expect `best_method=pyannote_vad` and `trimmed_duration_s` ≤ `original_duration_s` when speech is detected.

### 7.6 MULTI-05 — No-speech / silent channels

Create a silent speaker file and point `speaker_b` at it (same manifest schema). Expected: VAD returns no segments and the pipeline either passes through the file or writes a failure row with empty `asr_transcription`. Must not crash.

---

## 8. CLI tests

Every public command gets a happy path. Reuse artifacts from earlier sections.

### 8.1 CLI-HELP — already SMOKE-01

Record PASS/FAIL there.

### 8.2 CLI-SINGLE — happy path

Covered by E2E-EN. Additional flag check:

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 2 \
  --significant-wer-threshold 0.5 \
  --output results/cli-single-threshold
```

**Expected:** `scores_*.json` has `"significant_wer_threshold": 0.5`.

### 8.3 CLI-MULTI — happy path

Covered by MULTI-01.

`--demographics` and `--report` both **require** `--dataset-dir` or argparse exits 2:

```text
--demographics requires --dataset-dir
--report requires --dataset-dir for multi-speaker mode
```

Skip demographic plots unless you have a dataset directory with the demographic metadata the analyzer expects. That is not required for OSS smoke.

### 8.4 CLI-CUSTOM — happy path

**Purpose:** YAML-driven BYO language.

`examples/custom_eval_portuguese.yaml` now has `api_models.enabled: false`. It still downloads FLEURS `pt_br` unless you pass `--max-samples` or point `dataset.tsv_path` at a local file.

Safer config:

```bash
cat > data/qa/custom_pt.yaml << 'EOF'
language:
  code: pt
  name: Portuguese

models:
  - hf_model_id: openai/whisper-tiny

dataset:
  tsv_path: data/qa/en/fleurs/test.tsv

api_models:
  enabled: false
EOF

psdn-sonar custom \
  --config data/qa/custom_pt.yaml \
  --output results/cli-custom \
  --max-samples 3 \
  --report
```

Using an English TSV with `language: pt` is intentional: it proves the custom runner loads a YAML, instantiates `CustomHuggingFaceModel`, and writes results. For a true Portuguese run, set `dataset.hf_dataset_id: google/fleurs`, `hf_subset: pt_br`, `hf_split: test`, `text_column: transcription`, and keep `--max-samples 3`.

**Expected result:** log `Custom evaluation: language=Portuguese (pt)`, results CSV `asr_detailed_custom_openai_whisper_tiny.csv`, API models skipped when `enabled: false`. With `enabled: true` and no keys:

```text
Skipping whisper_api: OPENAI_API_KEY not set
```

**Pass/fail:** PASS if at least one HF model CSV is written. FAIL if missing keys abort the whole run (they must not).

### 8.5 CLI-DISCOVER — happy path

Covered by SMOKE-02 + section 2.3.

Additional:

```bash
psdn-sonar discover --language ur --dry-run
```

Urdu is in the FLEURS map (`ur_pk`). Expect `fleurs` to appear. This is listed in the CLI epilog; it is a valid discover language even though SONAR has no `ur` processor.

```bash
psdn-sonar discover --language en --datasets fleurs --validate --dry-run
```

`--validate` hits the Hub. Expect the same fleurs row, optionally with split sizes.

---

## 9. Language normalization tests

These are **controlled** tests. No ASR. They prove the scoring path QA will rely on.

**Prerequisites:** INST-03 (or any install that includes the package). `[ml]` not required for WER/CER.

**Command:**

```bash
python << 'PY'
from psdn_sonar.config_loader import load_config
from psdn_sonar.registry import get_language_processor
from psdn_sonar.evaluators.utterance import UtteranceEvaluator
import psdn_sonar.language  # noqa: F401 — register processors

def proc(lang):
    return get_language_processor(lang)(load_config(language=lang, backend="huggingface"))

failed = 0

def check(name, got, expected):
    global failed
    ok = got == expected
    print(("PASS" if ok else "FAIL"), name)
    if not ok:
        print("  expected:", repr(expected))
        print("  got:     ", repr(got))
        failed += 1

# --- processor.normalize() (config-driven) ---
en, hi, bn, ko = proc("en"), proc("hi"), proc("bn"), proc("ko")

check("en lowercase/ws", en.normalize("  Hello   WORLD  "), "hello world")
check("en punct", en.normalize("hello, world!"), "hello world")
check("en 50%", en.normalize("50%"), "fifty percent")
check("en 3.14", en.normalize("3.14"), "threefourteen")
check("en v2 preserved", en.normalize("release v2"), "release v2")
check("en 25 beds", en.normalize("The hospital has 25 beds."), "the hospital has twenty five beds")
check("en 1,000", en.normalize("1,000 dollars"), "one thousand dollars")

check("hi digits ५००", hi.normalize("५००"), "पाँच सौ")
check("hi 50%", hi.normalize("50%"), "पचास प्रतिशत")
check("hi punct", hi.normalize("नमस्ते, दुनिया!"), "नमस्ते दुनिया")
check("hi loanword customer", hi.normalize("customer सेवा"), "कस्टमर सेवा")
check("hi 10", hi.normalize("यह 10 रुपये है।"), "यह दस रुपये है")
check("hi 1,000", hi.normalize("1,000 रुपये") == hi.normalize("1000 रुपये"), True)

check("bn punct", bn.normalize("এটি, একটি! পরীক্ষা?"), "এটি একটি পরীক্ষা")
check("bn digit ২", bn.normalize("২"), "দুই")
check("bn 123", bn.normalize("123"), "একশত তেইশ")
check("bn ৫০%", bn.normalize("৫০%"), "পঞ্চাশ শতাংশ")
check("bn ws", bn.normalize("এটি   একটি  পরীক্ষা"), "এটি একটি পরীক্ষা")

check("ko punct", ko.normalize("안녕하세요!"), "안녕하세요")
check("ko 123", ko.normalize("123"), "백이십삼")
check("ko 100원", ko.normalize("100원"), "백원")
check("ko 50%", ko.normalize("50%"), "오십 퍼센트")
check("ko 5,500원", ko.normalize("5,500원"), ko.normalize("5500원"))
check("ko loanword phone", ko.normalize("phone 번호"), "폰 번호")

# --- eval WER path (what single-speaker scoring uses) ---
def score(lang, ref, hyp):
    return UtteranceEvaluator.score_single_variant(ref, hyp, language=lang)

for name, lang, ref, hyp, exp_wer in [
    ("en punct pair", "en", "hello, world!", "hello world", 0.0),
    ("en verbalize pair", "en", "The hospital has 25 beds.", "the hospital has twenty five beds", 0.0),
    ("en real error", "en", "hello world", "hello there", 0.5),
    ("en 1,000 pair", "en", "1,000 dollars", "one thousand dollars", 0.0),
    ("hi punct pair", "hi", "नमस्ते, दुनिया!", "नमस्ते दुनिया", 0.0),
    ("hi 50% pair", "hi", "50%", "पचास प्रतिशत", 0.0),
    ("bn punct pair", "bn", "এটি, একটি! পরীক্ষা?", "এটি একটি পরীক্ষা", 0.0),
    ("bn ৫০% pair", "bn", "৫০%", "৫০ শতাংশ", 0.0),
    ("ko punct pair", "ko", "안녕하세요!", "안녕하세요", 0.0),
    ("ko 100원 pair", "ko", "100원", "백원", 0.0),
]:
    cer, wer, rn, hn = score(lang, ref, hyp)
    check(name + f" WER={wer}", wer, exp_wer)

# Bengali eval path splits suffixes — processor.normalize does not.
# এটি stays whole (single-cluster stem guard); একটি splits (এক is a word).
cer, wer, rn, hn = score("bn", "এটি একটি পরীক্ষা", "এটি একটি পরীক্ষা")
check("bn wer-path suffix split", rn, "এটি এক টি পরীক্ষা")
check("bn wer-path WER0", wer, 0.0)

# Whole words must not be cut by suffix-lookalike endings (issue #142)
from psdn_sonar.utils.text_processing import normalize_bengali_for_wer as bn_wer
check("bn whole word মাটি", bn_wer("মাটি"), "মাটি")
check("bn whole word ছেলে", bn_wer("ছেলে"), "ছেলে")
check("bn conjunct ঘণ্টা", bn_wer("ঘণ্টা"), "ঘণ্টা")
check("bn real suffix প্যাকেটটা", bn_wer("প্যাকেটটা"), "প্যাকেট টা")

raise SystemExit(failed)
PY
```

**Expected result:** every line prints `PASS`. Exit code 0.

These values were generated from the processors in this checkout. If `[hindi]` / `[korean]` extras change unicode/tokenize behavior, loanword and digit tests should still match (they use core deps + shipped caches).

**Pass/fail:** PASS if the script exits 0. FAIL if any assertion prints `FAIL` — that is a scoring-regression, not a flaky ASR issue.

---

## 10. Negative and edge-case tests

For each case: **Purpose → Prerequisites → Command → Expected Result → Pass/Fail**.

SONAR should either finish with an explicit per-row `error` or exit with a short, actionable message. A raw traceback without a one-line `ERROR:` / argparse message is a product defect.

### 10.1 NEG-MISSING-AUDIO-FILE (path in TSV, file absent)

**Purpose:** Missing wav referenced by TSV.

```bash
mkdir -p data/qa/neg
printf 'audio_path\ttranscription\n/tmp/does-not-exist-sonar.wav\thello world\n' > data/qa/neg/missing-file.tsv
psdn-sonar single --input data/qa/neg/missing-file.tsv --models whisper_base_en --language en --output results/neg-missing-file
```

**Expected:** CLI does **not** reject the TSV (the file exists). Evaluation logs `Audio file not found`, writes a row with `prediction=""` and `error="Empty prediction"`, `failed >= 1`. Exit 0 is current behavior.

**Pass:** clear row-level error. **Fail:** uncaught exception / no output files.

### 10.2 NEG-MISSING-INPUT (TSV path itself missing)

```bash
psdn-sonar single --input /nonexistent/data.tsv --models whisper_base_en --language en
```

**Expected (exact):**

```text
psdn-sonar: error: Input file not found: /nonexistent/data.tsv
```

Exit code `2`.

### 10.3 NEG-CORRUPT-AUDIO

```bash
printf 'audio_path\ttranscription\n' > data/qa/neg/corrupt.tsv
printf 'not-a-wav' > data/qa/neg/corrupt.wav
printf 'data/qa/neg/corrupt.wav\thello world\n' >> data/qa/neg/corrupt.tsv
# audio_path is resolved relative to the TSV directory; use a name-only path:
printf 'audio_path\ttranscription\ncorrupt.wav\thello world\n' > data/qa/neg/corrupt.tsv
psdn-sonar single --input data/qa/neg/corrupt.tsv --models whisper_base_en --language en --output results/neg-corrupt
```

**Expected:** `transcribe()` returns `None` on failure (HF adapters catch exceptions). Row gets `error="Empty prediction"` or the exception string. Run continues.

**Pass:** no crash. **Fail:** process abort without a results CSV.

### 10.4 NEG-UNSUPPORTED-FORMAT

```bash
printf 'this is not audio' > data/qa/neg/clip.txt
printf 'audio_path\ttranscription\nclip.txt\thello world\n' > data/qa/neg/badfmt.tsv
psdn-sonar single --input data/qa/neg/badfmt.tsv --models whisper_base_en --language en --output results/neg-badfmt
```

**Expected:** same as corrupt audio — failed row, not a hang.

### 10.5 NEG-EMPTY-TRANSCRIPTION (empty hypothesis)

Already covered when the model returns `""`: `error="Empty prediction"`, WER/CER left empty, not counted in `successful`.

Also test empty **reference** (row skipped at load time):

```bash
printf 'audio_path\ttranscription\nclip.wav\t\n' > data/qa/neg/empty-ref.tsv
# copy a real wav next to it
cp data/qa/en/fleurs/audio/test_000000.wav data/qa/neg/clip.wav
psdn-sonar single --input data/qa/neg/empty-ref.tsv --models whisper_base_en --language en --output results/neg-empty-ref
```

**Expected:** the empty-reference row is kept and counted as **failed** with a per-row `error` in the CSV (`Row missing/empty 'transcription' field`); since no samples succeed, the run logs `No samples were successfully evaluated` and exits **1**. `avg_wer`/`avg_cer` in the summary and `wer_mean`/`cer_mean` in `scores_*.json` are `null`, and the results CSV still has its header.

**Pass:** exit 1 with the per-row error recorded and null means. **Fail:** exit 0, or the row silently dropped with no error recorded.

Also test a reference that is non-empty but **normalizes to empty** (punctuation-only). It passes the load check but cannot be scored; per D39 the row must be counted as failed and excluded from aggregates — it used to be scored as a perfect WER/CER of 0.0:

```bash
printf 'audio_path\ttranscription\nclip.wav\t...!?\n' > data/qa/neg/punct-ref.tsv
psdn-sonar single --input data/qa/neg/punct-ref.tsv --models whisper_base_en --language en --output results/neg-punct-ref
echo "exit=$?"
```

**Expected:** exit 1 (zero successful samples). The row is failed with `error` starting `CER/WER uncomputable`, its `wer`/`cer` cells empty, and the model's transcription preserved in the `prediction` column. **Fail:** the row counted successful with WER/CER `0.0`.

### 10.6 NEG-WRONG-LANGUAGE-CODE

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language hi \
  --max-samples 1 \
  --output results/neg-wrong-lang
```

**Expected:** Run **succeeds**. English text is scored with the Hindi processor (loanwords/digits). WER will look odd. This is not rejected.

**Pass:** completes. File as a usability issue if there is no warning that references fail `validate_text`.

### 10.7 NEG-UNSUPPORTED-LANGUAGE

Unknown codes are rejected before any model is loaded or any score is written, with or without `--models`:

```bash
psdn-sonar single --input data/qa/en/fleurs/test.tsv --language xx
psdn-sonar single --input data/qa/en/fleurs/test.tsv --models whisper_base_en --language xx --max-samples 1 --output results/neg-lang-xx
```

**Expected:** both commands exit **1** with:

```text
Unknown --language 'xx': not a recognized ISO 639-1 code, so no scores were written. Languages with dedicated normalizers: bn, en, hi, ko. ...
```

No `results/neg-lang-xx` scores are written.

A **recognized** ISO code without a dedicated normalizer (e.g. `pt`) is a separate case — it proceeds but warns:

```bash
psdn-sonar single --input data/qa/en/fleurs/test.tsv --models whisper_base_en --language pt --max-samples 1 --output results/neg-lang-pt
```

**Expected:** run completes; warning `Language 'pt' (portuguese) has no dedicated normalizer; WER/CER will use the generic fallback normalization ...` appears before evaluation starts.

**Pass:** `xx` → exit 1 with no scores; `pt` → completes with the fallback warning. **Fail:** any unknown code producing a `scores_*.json`.

### 10.8 NEG-INVALID-HF-MODEL

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --hf-model this-org/definitely-not-a-real-asr-model-xyz \
  --language en \
  --max-samples 1 \
  --output results/neg-bad-hf
```

**Expected:** `Failed to load custom HuggingFace model '…'` (or Hub 404) and

```text
Evaluation failed: ...
```

Exit code `1`.

### 10.9 NEG-UNKNOWN-REGISTRY-MODEL

```bash
psdn-sonar single --input data/qa/en/fleurs/test.tsv --models whisper-hindi --language hi --output results/neg-bad-id
```

**Expected:** `Model whisper-hindi not found` and the run skips that model (`create_model` raises `Unknown model 'whisper-hindi'. Available: …`). If it is the only model, you get no CSV. Current CLI still exits 0 after a skip — record that.

### 10.10 NEG-MISSING-ML-DEPS

In a venv with **only** the core package (no `[ml]`):

```bash
psdn-sonar single --input data/qa/en/fleurs/test.tsv --models whisper_base_en --language en --max-samples 1 --output results/neg-no-ml
```

**Expected:** import error for `torch` / `transformers` mentioning `pip install "psdn-sonar[ml]"` (the huggingface adapter module docstring states this). Exit 1.

### 10.11 NEG-MISSING-HF-CREDENTIALS

Public Whisper models should **work without** `HF_TOKEN`. If Hub rate-limits you, the error should mention login/token.

Gated pyannote without token: see MULTI-04.

### 10.12 NEG-MISSING-PYANNOTE-CREDENTIALS

See MULTI-03 and MULTI-04.

### 10.13 NEG-DATASET-DOWNLOAD-FAILURE

```bash
psdn-sonar discover --language en --datasets fleurs --output /proc/cannot-write-here
```

**Expected:** prepare fails, `Failed to prepare fleurs`, and if it is the only dataset, exit 1:

```text
All 1 dataset(s) failed to prepare: fleurs
```

Also, `--datasets` entries are validated per name with distinct reasons and exit **1**:

```bash
psdn-sonar discover --language en --datasets common_voice --dry-run              # known but disabled
psdn-sonar discover --language en --datasets definitely-not-a-dataset --dry-run  # unknown name
psdn-sonar discover --language bn --datasets openslr37_bd --dry-run              # catalogued non-HF source
```

**Expected:** each exits **1** before any download. The error names the entry and the reason: `catalogued but disabled (review decision: pending)` for `common_voice`; `unknown dataset name` plus the discoverable list (`fleurs, voxpopuli, zeroth`) for the typo; `catalogued as an openslr source; \`discover\` covers HuggingFace-hosted sources only` for `openslr37_bd`.

A **valid** filter that matches nothing for the language also exits **1**, blaming the filter rather than the language:

```bash
psdn-sonar discover --language en --datasets zeroth --dry-run
```

**Expected:** exit **1** with `The --datasets filter, not the language, excluded everything (zeroth supports: ko)`.

Every `discover` summary additionally prints a scope note naming the catalogued entries the command cannot reach (the three OpenSLR Bengali corpora, `multilingual_librispeech`, and disabled `common_voice`), so the table is no longer presented as the complete catalog.

### 10.14 NEG-MALFORMED-TSV

```bash
printf 'not a tsv at all\x00\xff' > data/qa/neg/garbage.tsv
psdn-sonar single --input data/qa/neg/garbage.tsv --models whisper_base_en --language en --output results/neg-garbage
```

**Expected:** `Evaluation failed:` with a decode/parse error. Exit 1.

### 10.15 NEG-MISSING-COLUMNS

```bash
printf 'path\tsentence\na.wav\thello\n' > data/qa/neg/wrong-cols.tsv
psdn-sonar single --input data/qa/neg/wrong-cols.tsv --models whisper_base_en --language en --output results/neg-cols
```

**Expected:**

```text
Evaluation failed: TSV missing required columns: audio_path, transcription
```

Exit 1. (Exact missing-column list depends on which names are absent.)

### 10.15b NEG-SURPLUS-FIELD

A literal tab inside the transcription makes a row carry more fields than the header. The surplus used to be discarded silently, scoring against a truncated reference (exit 0).

```bash
W=data/qa/en/fleurs/audio/test_000000.wav
printf 'audio_path\ttranscription\n' > data/qa/neg/extra-tab.tsv
printf '%s\thello\tworld extra\n' "$W" >> data/qa/neg/extra-tab.tsv
psdn-sonar single --input data/qa/neg/extra-tab.tsv --models whisper_base_en --language en --output results/neg-extra-tab
```

**Expected:** a warning naming the line — `TSV line 2: 3 fields for 2 header columns (a literal tab inside a field?) — refusing to truncate the transcription ... row will be counted as failed, not dropped`. The row appears in results with an `error`; as the only sample, the run exits 1 under the zero-successful-samples rule. Never exit 0 with a silently truncated reference.

### 10.15c NEG-BOM-TSV

Excel prepends a UTF-8 BOM to exported TSVs. It used to corrupt the first column name and produce `TSV missing required columns: audio_path` for a present column.

```bash
W=data/qa/en/fleurs/audio/test_000000.wav
printf '\xEF\xBB\xBFaudio_path\ttranscription\n' > data/qa/neg/bom.tsv
printf '%s\thello world\n' "$W" >> data/qa/neg/bom.tsv
psdn-sonar single --input data/qa/neg/bom.tsv --models whisper_base_en --language en --output results/neg-bom
```

**Expected:** the BOM is stripped and the run behaves exactly like the same file without a BOM (normal evaluation, exit 0). No missing-column error.

### 10.16 NEG-VERY-SHORT-AUDIO

Use `ffmpeg` or Python to write a 50 ms wav and a one-word transcript. Evaluation must complete. WER may be 1.0. `conf/config.yaml` validation `min_duration_seconds: 0.5` is **not** enforced by `psdn-sonar single` (that config tree is for the recipe/validation layer). Record if a warning appears.

### 10.17 NEG-LONG-AUDIO

Concatenate several FLEURS clips to >60 s (Whisper adapters use 30 s chunking on the pipeline class). Must complete; latency will be higher. Skip if disk/time is tight; mark BLOCKED.

### 10.18 NEG-SILENT-AUDIO

```bash
python -c "import numpy as np, soundfile as sf; sf.write('data/qa/neg/silent.wav', np.zeros(16000, np.float32), 16000)"
printf 'audio_path\ttranscription\nsilent.wav\thello world\n' > data/qa/neg/silent.tsv
psdn-sonar single --input data/qa/neg/silent.tsv --models whisper_base_en --language en --output results/neg-silent
```

**Expected:** `silence_ratio` = `1.0` (an all-zero file is fully silent — it no longer scores `0.0` against its own maximum), `snr_db` and `snr_tier` empty (SNR undefined without signal; no more `inf`/`High`), and `quality_warnings` contains `high_silence:`. Prediction is often empty → `Empty prediction`; if it is the only sample and the model returns nothing, the run exits 1 under the zero-successful-samples rule.

### 10.19 NEG-NOISY-AUDIO

Mix a FLEURS wav with loud white noise. Expected: completes; `snr_db` lower; WER usually worse. Not a fail unless the process crashes.

### 10.20 NEG-CODE-SWITCH

Add a TSV row whose reference mixes English and Hindi (or use a FLEURS sentence that already contains a Latin name). Expected: completes. Hindi loanword cache may rewrite some Latin tokens. Record actual `prediction` and normalized reference.

### 10.21 NEG-UNICODE

Hindi/Bengali/Korean E2E already exercise non-Latin text. Additional:

```bash
python -c "from psdn_sonar.language.hindi import HindiProcessor; from psdn_sonar.config_loader import load_config
p=HindiProcessor(load_config(language='hi', backend='huggingface'))
print(p.normalize('नमस्ते 🙂 १२३'))"
```

**Expected:** digits verbalized; emoji stripped or left without crashing.

### 10.22 NEG-MULTIPLE-SPEAKERS-ON-SINGLE

Point `psdn-sonar single` at the **combined** multi-speaker wav with a concatenated transcript. Expected: completes as one utterance (no diarization). High WER is OK. This proves single-speaker mode does not crash on overlapping speech.

### 10.23 NEG-NO-SPEECH / MULTI SKIP

Manifest with missing `transcript.json`: clip is skipped with `Skipping {id}: could not load transcript`. Other clips still run.

### 10.24 NEG-MODELS-AND-HF-MODEL

```bash
psdn-sonar single --input data/qa/en/fleurs/test.tsv --models whisper_base_en --hf-model openai/whisper-tiny
```

**Expected (exact):**

```text
psdn-sonar: error: Cannot use both --models and --hf-model. Choose one.
```

Exit 2.

### 10.25 NEG-MULTI-NO-MODEL

```bash
psdn-sonar multi --input data/qa/multi/manifest.jsonl
```

**Expected (exact):**

```text
Either --models or --hf-model must be specified for multi-speaker mode. Language-based auto-selection is only supported for single-speaker mode.
```

Exit 1.

### 10.26 NEG-CUSTOM-MISSING-CONFIG

```bash
psdn-sonar custom --config /nonexistent.yaml
```

**Expected (exact):**

```text
psdn-sonar: error: Config file not found: /nonexistent.yaml
```

Exit 2.

### 10.27 NEG-PATH-TRAVERSAL (audio_path escaping the TSV directory)

**Purpose:** A relative `audio_path` with `../` must not be followed out of the dataset directory (a TSV received from someone else could otherwise read audio from anywhere on disk).

```bash
printf 'audio_path\ttranscription\n../../../../etc/hosts\tignored\n' > /tmp/traversal.tsv
psdn-sonar single --input /tmp/traversal.tsv --models whisper_base_en --language en --output results/neg-traversal
echo "exit=$?"
```

**Expected:** exit 1 with `ERROR ... audio_path escapes dataset root: ../../../../etc/hosts` before any model loads. The file is never opened — no audio-decoder error mentioning `/etc/hosts` appears.

Absolute paths in the TSV remain allowed by default (SONAR's own `discover` output writes them, and they are explicit in the file). To also reject absolute paths and require every path to be an existing regular file inside the TSV's directory, pass `--strict-audio-paths`:

```bash
printf 'audio_path\ttranscription\n/etc/hosts\tignored\n' > /tmp/absolute.tsv
psdn-sonar single --input /tmp/absolute.tsv --models whisper_base_en --language en --strict-audio-paths --output results/neg-absolute
echo "exit=$?"
```

**Expected:** exit 1 with `ERROR ... audio_path must be relative inside bundle: /etc/hosts`.

### 10.28 NEG-NO-FFMPEG (pipeline adapter without ffmpeg on PATH)

**Purpose:** The pipeline-based adapters (`whisper_base_en`, `whisper_small_en`, `khushids_bengali`, generic-pipeline `--hf-model`) need the `ffmpeg` binary to decode **any** file path, WAV included. Without it the run must fail **once at model load** with an error naming the binary — not one `Transcription failed` per utterance followed by a normal-looking summary (issue #109).

```bash
# Run with ffmpeg hidden from PATH (venv bin + minimal system dirs only)
VENVBIN=$(dirname "$(which python)")
env PATH="$VENVBIN:/usr/bin:/bin" sh -c 'command -v ffmpeg' || echo "ffmpeg hidden OK"
env PATH="$VENVBIN:/usr/bin:/bin" psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en --language en --max-samples 2 \
  --output results/neg-no-ffmpeg
echo "exit=$?"
```

If `/usr/bin/ffmpeg` exists on your machine, use a directory set that excludes it (e.g. copy `python`/`psdn-sonar` symlink targets into a scratch bin) — the point is that `ffmpeg` must not be resolvable.

**Expected:** exit 1. A single `ERROR ... Evaluation failed: StandardHuggingFaceASR (openai/whisper-base) hands audio file paths to the transformers ASR pipeline, which requires the ffmpeg binary to decode them — including WAV.` The message names install commands (`apt-get install ffmpeg` / `brew install ffmpeg`) and the ffmpeg-free adapter alternatives. No per-utterance `Transcription failed` lines, no checkpoint download, no `scores_*.json` claiming success, no traceback.

**Control:** the same command with normal `PATH` (ffmpeg present) must complete as in section 3. `wav2vec2_bengali` and other self-decoding adapters must still evaluate WAV with ffmpeg hidden.

---

## 11. Reproducibility tests

### 11.1 REPRO-01 — Run twice, compare artifacts

**Purpose:** Same command, two output dirs; configuration is recorded.

```bash
psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 5 \
  --output results/repro-a

psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 5 \
  --output results/repro-b
```

**Expected in both `scores_whisper_base_en.json` files:**

| Field | Must match across runs |
| --- | --- |
| `submission.package_version` | yes |
| `submission.model_snapshot` | `whisper_base_en` |
| `submission.inference_params.language_code` | `en` |
| `submission.seed` | `42` (from `psdn_sonar/conf/config.yaml` `run.seed`) |
| `submission.protocol` | `batch` |
| `submission.git_sha` | same if run from the same git checkout |
| `lineage.hf_model_id` | `openai/whisper-base` (the repo id actually loaded) |
| `lineage.hf_revision` | same 40-char checkpoint SHA across both runs |
| `lineage.normalization` | `en:v2` (Bengali runs also carry `+bnlp`/`-bnlp`) |
| `aggregate.total_samples` | `5` |

`timestamp_utc` **must differ** (or may differ). `elapsed_time_s` will differ.

WER/CER: Whisper on CPU is usually stable for the same checkpoint; GPU kernels can differ slightly. **Require** identical `package_version`, model id, seed, language, and dataset path. Treat WER deltas > 0.02 mean as a note, not an automatic fail, unless you pinned CPU and `CUBLAS` determinism.

Dataset traceability: `data/qa/en/fleurs/metadata.json` must still show `source: google/fleurs` and the pinned `source_revision`.

**Pass/fail:** PASS if both scores files exist and the table matches. FAIL if seed/model/language are missing (the run cannot be audited).

### 11.2 REPRO-02 — Normalization determinism

Re-run the section 9 script twice. Output must be identical. These paths do not use GPU sampling.

---

## 12. Documentation / implementation mismatches

Items marked **Fixed** were corrected in this repository before this guide was sent to QA. Re-verify them; do not treat the old broken behavior as expected.

| ID | Where | Status | Notes |
| --- | --- | --- | --- |
| D1 | README Installation | **Fixed (docs)** | README now states source version `0.1.0` and that TestPyPI may still list `0.1.0.dev2`. Confirm the live TestPyPI filename. |
| D2 | README vs `pyproject.toml` | Open | Python 3.10–3.12 documented; `requires-python = ">=3.10"` has no upper cap. 3.13 is untested. |
| D3 | `docs/USAGE.md` | **Fixed (docs)** | USAGE now points at README for package install; `-e ".[ml]"` is labeled as a clone/contributor command. |
| D4 | `docs/FAQ.md` Q1 | **Fixed (docs)** | FAQ discover examples use `fleurs` / `voxpopuli` / `zeroth`. Common Voice is documented as not discoverable. |
| D5 | `discover --help` | **Fixed** | Example filter is `fleurs,voxpopuli`. |
| D6 | Common Voice + OpenSLR | **Mitigated** | Still not preparable via `discover`, but the CLI now prints a scope note naming them and `--datasets` errors explain why each is unreachable (see 10.13). Preparing them remains a product gap. |
| D7 | FAQ + examples manifest | **Fixed** | Docs and `examples/test_manifest.jsonl` use `audio_filepaths` + JSON `transcript_filepath`. |
| D8 | Sample audio | **Fixed** | `examples/sample_audio/TEST001/` ships with the repo (synthetic tones). |
| D9 | `examples/test_data.tsv` | **Fixed** | Points at `sample_audio/single/sample.wav`. |
| D10 | `tests/unit/test_examples.py` | **Fixed** | Asserts the runtime schema and that fixture files exist. |
| D11 | `multi --hf-model` | **Fixed** | CLI forwards `custom_hf_model` into `create_model`. |
| D12 | Default models include APIs | **Fixed** | Language-default `single` skips hosted APIs unless their keys are set. |
| D13 | `--language` default `bn` | **Mitigated** | Still defaults to `bn` (compatibility) but logs a warning. Always pass `--language` in this guide. |
| D14 | FAQ Q2 output layout | **Fixed (docs)** | Tree shows `asr_detailed_*.csv` + `scores_*.json`. |
| D15 | FAQ latency plots | **Fixed (docs)** | Latency plots listed under `latency-analysis/`. |
| D16 | FAQ “PSDN Score” | **Fixed (docs)** | Table now says POSEIDON. |
| D17 | Portuguese custom YAML | **Fixed** | `api_models.enabled: false`. Still pass `--max-samples`. |
| D18 | CLI `--method` vs packaged YAML | Open | Help lists four methods; packaged YAML enables `no_trim` only. |
| D19 | `[korean]` Java | Open | Called out in README/CONTRIBUTING, not repeated in FAQ. |
| D20 | ffmpeg | **Fixed (docs)** | README Requirements now mention ffmpeg for MP3 / pydub. |
| D21 | Bengali WER path | Open | Eval uses `normalize_bengali_for_wer()` (suffix split), not `BengaliProcessor.normalize()`. |
| D22 | `benchmark/README.md` scores example | Open | Example JSON omits `significant_wer*`. |
| D23 | FAQ audio formats | Open | No hard allow-list in `single`; MP3 needs ffmpeg. |
| D24 | `core.py` docstring | Open | Library loaders for CV/OpenSLR are not `discover`. |
| D25 | PyPI badge vs TestPyPI status | Open | Badge vs “not yet on PyPI” paragraph. |
| D26 | `--language` accepted any string | **Fixed** | `single`/`multi` now exit 1 on unknown codes before scoring; recognized codes without a dedicated normalizer (e.g. `pt`) warn about generic fallback. See 10.7. |
| D27 | Bengali paired with `openai/whisper-small` | **Fixed (docs)** | CLI epilogue, examples, and this guide now use `wav2vec2_bengali` for Bengali; the BYO whisper-small run is kept as a mechanics-only test (WER ≈ 1.0 expected). |
| D28 | `--datasets` accepted any string, exit 0 | **Fixed** | Entries are validated per name (unknown / disabled / non-HF source / not wired), zero matches with a filter exit 1, and the error blames the filter rather than the language. Summary prints a catalog scope note. See 10.13. |
| D29 | Silent audio scored `silence_ratio=0.0`, `snr_db=inf` | **Fixed** | Uniformly quiet files (loudest RMS frame below ~-60 dBFS) now score `1.0` and trip the `high_silence` gate; SNR is `None` (blank) for signal-less audio and capped at 100 dB for noise-free audio. See 10.18. |
| D30 | Multi-speaker assignment scored perfect CER/WER (0.0) as worst case | **Fixed** | The `dual_assignment_score` heuristic used `or` fallbacks, so a perfect transcription could lose to a swapped pairing with higher similarity, corrupting both speakers' WER/CER. Missing metrics now default via explicit `None` checks. |
| D31 | TSV `audio_path` with `../` escaped the dataset directory and was opened | **Fixed** | The boundary guard existed but defaulted off and the CLI could not enable it. Relative paths escaping the TSV directory are now always rejected; `--strict-audio-paths` additionally rejects absolute paths and requires existing regular files. See 10.27. |
| D32 | No stated position on what scores measure (preprocessing confound, cross-dataset length gap, unmarked in-domain cells) | **Documented / data emitted** | `docs/SCORE_INTERPRETATION.md` states the position on all three. Multi-speaker runs log their pipeline scope; `precompute_benchmarks.py` emits `public_length_stats_<language>.json` and per-cell `domain_markers.json` (from `psdn_sonar.models.provenance`). Leaderboard rendering is owned by `PSDN-AI/psdn-portals`. |
| D33 | Bengali had no symbol map — `৫০%` normalized to `50 %` while en/hi/ko verbalize `%` | **Fixed** | `BENGALI_SYMBOL_MAP` added and wired into both the canonical WER pipeline and `BengaliProcessor`; `৫০%` and `৫০ শতাংশ` now normalize identically, and no symbol survives whose spacing could differ by `bnlp` availability. Bengali normalization contract bumped to `bn:v2` — v1 and v2 Bengali scores are not like-for-like. |
| D34 | Thousands separators split number verbalization — `1,000 dollars` normalized to `one000 dollars` (en/hi/ko), inflating WER on corpora with separated numerals | **Fixed** | `verbalize_digits` now strips separators inside well-formed grouped numbers (Western `1,234,567`, Indian `12,34,567`, space/thin-space grouping) before digit-run extraction, matching what the canonical Bengali pipeline always did; `BengaliProcessor`'s own digit path got the same comma strip. en/hi/ko normalization contracts bumped to `en:v2`/`hi:v2`/`ko:v2` — v1 and v2 scores are not like-for-like. |
| D35 | Bengali suffix splitting had no stem check — whole words were cut in two (`মাটি` → `মা টি`, `ছেলে` → `ছেল এ`) and `ঘণ্টা` split inside a conjunct leaving a virama fragment, inflating the WER denominator | **Fixed** | `_split_suffixes` now requires a splittable stem (≥2 grapheme clusters, never virama-terminated) and consults a small protected whole-word lexicon for cases structure cannot decide (`ছেলে` vs `দেশে`). Real suffixes still split (`প্যাকেটটা` → `প্যাকেট টা`); `হাতে` now splits at the true morpheme boundary (`হাত এ`, matching the `দেশে` → `দেশ এ` precedent). Bengali contract bumped to `bn:v3`. |
| D36 | A surplus TSV field (literal tab in the transcription) silently truncated the reference and scored it (exit 0); a UTF-8 BOM produced `TSV missing required columns: audio_path` for a column that is present | **Fixed** | `load_data` reads TSVs as `utf-8-sig` (BOM stripped, no-op otherwise) and marks surplus-field rows as failed with a warning naming the line and field counts, following the issue-#102 failed-not-dropped pattern. See 10.15b/10.15c. |
| D37 | README claimed "WAV evaluation works without ffmpeg", but the pipeline adapters (both English defaults, `khushids_bengali`, generic-pipeline `--hf-model`) shell out to ffmpeg for **all** file-path input including WAV; without it every utterance failed individually and the run still printed its normal summary | **Fixed** | Those adapters now preflight for ffmpeg at model load (before the checkpoint download) and raise `MissingFfmpegError` naming the binary, install commands, and ffmpeg-free alternatives; the CLI exits 1 cleanly. README Requirements corrected. See 10.28. |
| D38 | A run with zero successful samples intermittently exited 134 (SIGABRT: a native extension aborting in interpreter teardown with `recursive_mutex lock failed`) instead of the reported exit 1, so automation could not tell a failed evaluation from a crashed process | **Fixed** | The `psdn-sonar` console script now runs through `entrypoint()`, which flushes logging and the std streams and leaves via `os._exit` with the code the run decided on, skipping interpreter teardown entirely. All artifacts are written and closed before that point. Exit codes are now stable across repeat runs: 0 success, 1 evaluation/data failure, 2 argparse usage error, 130 Ctrl-C. |
| D39 | Three scoring paths handled a missing metric three contradictory ways (single-speaker rollups: best case 0.0; `PoseidonScorer`: worst case 1.0/0.0; `significant_wer_rate`: excluded), and `semantic_similarity` was clamped to `[0,1]` inside POSEIDON but stored/averaged raw, so `semantic_similarity_mean` could go negative while `poseidon_score_mean` could not | **Fixed** | One convention everywhere: an uncomputable metric is `null`, the row is failed with the reason in `error` (prediction preserved), and aggregates cover only present values; `ensure_poseidon_score` leaves `NaN` instead of fabricating scores. Similarity is cosine clamped to `[0,1]` at the point of computation, so every artifact reports the same range. Documented in `psdn_sonar/benchmark/README.md` ("Missing values and metric ranges"). |

---

## 13. QA sign-off checklist

Copy this into the test report.

### Install

- [ ] Clean venv created (3.10, 3.11, or 3.12)
- [ ] Documented TestPyPI path tried; result recorded (including if `0.1.0.dev2` is gone)
- [ ] Source editable + `[ml]` works
- [ ] `psdn-sonar --version` recorded
- [ ] ffmpeg present (required by the pipeline adapters even for WAV — see 10.28); without it MP3 cases marked BLOCKED
- [ ] `HF_TOKEN` documented as optional except pyannote

### Data

- [ ] Discover dry-run matches FLEURS / VoxPopuli / Zeroth only
- [ ] Common Voice via discover confirmed **unavailable**
- [ ] Small FLEURS subsets prepared for en, hi, bn, ko
- [ ] English VoxPopuli subset prepared
- [ ] Korean Zeroth subset prepared
- [ ] `metadata.json` records source + revision

### Models

- [ ] Used only IDs from the registry table (or `--hf-model`)
- [ ] Default `--language en` run (no `--models`) did **not** call hosted APIs without keys
- [ ] `--hf-model` single-speaker and multi-speaker paths verified
- [ ] Approximate download sizes recorded

### Single-speaker E2E (en, hi, bn, ko)

- [ ] Audio loaded
- [ ] Model loaded
- [ ] Transcription completed (`successful > 0`)
- [ ] Correct `--language` in `scores_*.json`
- [ ] WER column present
- [ ] CER column present
- [ ] Semantic similarity present (`[ml]`)
- [ ] POSEIDON present
- [ ] Per-utterance CSV present
- [ ] Aggregate table logged and `aggregate` in JSON
- [ ] `--report` produced `EVAL_REPORT.md` (or plot failures only warned)

### Multi-speaker

- [ ] Fixture used the **runtime** manifest schema (not FAQ)
- [ ] `no_trim` happy path produced 2 speaker rows
- [ ] Auto-select completed
- [ ] Missing pyannote extra: actionable message
- [ ] Missing `HF_TOKEN` with pyannote: actionable message
- [ ] `--hf-model` on multi loads the custom checkpoint (D11 fixed — verify)

### CLI

- [ ] `single` happy path
- [ ] `multi` happy path
- [ ] `custom` happy path (APIs skipped without keys)
- [ ] `discover` happy path
- [ ] `--help` / `--version` / missing mode

### Normalization

- [ ] Section 9 script exit 0 for en, hi, bn, ko
- [ ] Bengali WER-path suffix split recorded

### Negatives

- [ ] Missing input file → exit 2, exact argparse text
- [ ] Missing TSV columns → `TSV missing required columns`
- [ ] Surplus TSV field (tab inside transcription) → row failed with line number, never a silently truncated reference
- [ ] TSV with UTF-8 BOM → accepted like the BOM-free file, no missing-column error
- [ ] Invalid HF model → exit 1
- [ ] Unknown language code (`xx`) → exit 1 before scoring, with or without `--models`
- [ ] Recognized code without normalizer (`pt`) → runs with explicit fallback warning
- [ ] Invalid `--datasets` entry (typo / disabled / non-HF source) → exit 1 with per-entry reason
- [ ] Valid `--datasets` matching nothing for the language → exit 1 blaming the filter
- [ ] `--models` + `--hf-model` → exit 2
- [ ] Silent / corrupt / short audio did not crash
- [ ] `audio_path` with `../` escaping the TSV directory → exit 1, file never opened
- [ ] `--strict-audio-paths` rejects absolute `audio_path` values → exit 1
- [ ] Pipeline adapter without ffmpeg on PATH → exit 1 at model load naming ffmpeg, no per-utterance errors
- [ ] Any failing case re-run several times returns the same exit code every time (never 134/SIGABRT)
- [ ] Punctuation-only reference → row failed as `CER/WER uncomputable`, never scored as WER 0.0

### Reproducibility

- [ ] Two runs share seed `42`, model id, language, package version
- [ ] `lineage` block present: HF repo id, checkpoint SHA, normalization contract
- [ ] Dataset `metadata.json` retained

### Release recommendation

- [ ] Re-verify D7/D8/D11/D12 on the branch under test (previously blockers; now claimed fixed)
- [ ] Confirm the TestPyPI wheel name matches README (D1)
- [ ] Hindi/Bengali “second public dataset” story is still a product gap (D6) — waive or enable loaders before promising Common Voice/OpenSLR

**Sign-off**

| Role | Name | Date | Verdict |
| --- | --- | --- | --- |
| QA | | | Ready / Ready with waivers / Not ready |
| Notes | | | |

---

## Appendix A — Quick command index

```bash
psdn-sonar --version
psdn-sonar --help
psdn-sonar single --help
psdn-sonar multi --help
psdn-sonar custom --help
psdn-sonar discover --help

psdn-sonar discover --language en --dry-run
psdn-sonar discover --language en --datasets fleurs --max-samples 10 --skip-audio-validation --output data/qa/en

psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --models whisper_base_en \
  --language en \
  --max-samples 10 \
  --output results/e2e-en \
  --report

psdn-sonar single \
  --input data/qa/en/fleurs/test.tsv \
  --hf-model openai/whisper-tiny \
  --language en \
  --max-samples 5 \
  --output results/e2e-en-hf

psdn-sonar multi \
  --input data/qa/multi/manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method no_trim \
  --output results/e2e-multi

psdn-sonar custom \
  --config data/qa/custom_pt.yaml \
  --output results/cli-custom \
  --max-samples 3 \
  --report
```

## Appendix B — Licenses for public data used in this guide

| Dataset | Hub ID | License (from catalog) |
| --- | --- | --- |
| Google FLEURS | `google/fleurs` | CC-BY-4.0 |
| Meta VoxPopuli | `facebook/voxpopuli` | CC0-1.0 AND LicenseRef-European-Parliament |
| Zeroth Korean | `Bingsu/zeroth-korean` | CC-BY-4.0 |

Retain attribution in the QA report. Catalog `review.decision` is still `pending` for all three — they are usable for evaluation, not for redistribution by SONAR.

## Appendix C — How this guide was verified

Commands, model IDs, dataset names, error strings, and normalization outputs were taken from:

- `psdn-sonar --help` / subcommand `--help` on this checkout (`0.1.0`)
- `psdn_sonar/cli.py`, `psdn_sonar/models/registry.py`, `psdn_sonar/data/registry.py`
- Live `psdn-sonar discover --language {en,hi,bn,ko,xx} --dry-run`
- Live `EnglishProcessor` / `HindiProcessor` / `BengaliProcessor` / `KoreanProcessor` and `UtteranceEvaluator.score_single_variant`
- `docs/FAQ.md`, `README.md`, `docs/USAGE.md`, `examples/` compared against those sources
