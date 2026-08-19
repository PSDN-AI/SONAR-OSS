# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- `psdn-sonar single`/`multi` now validate `--language` before scoring (#103):
  unrecognized codes (e.g. `xx`) exit 1 with an actionable error instead of
  silently scoring with the fallback normalizer, recognized ISO codes without
  a dedicated normalizer (e.g. `pt`) log an explicit fallback warning, and
  long names / mixed case (`Bengali`, `KO`) are canonicalized to the ISO code
  so they select the intended normalizer. The CLI epilogue, examples, and QA
  guide no longer pair Bengali with generic `openai/whisper-small`, which
  hallucinates non-Bengali script (WER ≈ 1.04); they recommend the registered
  `wav2vec2_bengali` instead.
- Runs that evaluate zero samples no longer exit 0 looking like clean runs
  (#102): unknown model names raise a `ValueError` listing the registered ids,
  a run with zero successful samples raises `NoSamplesEvaluatedError` after
  writing artifacts, `wer_mean`/`cer_mean` are `null` (not `0.0`) when
  `successful == 0`, TSV rows with missing/blank fields are counted in
  `failed` with per-row error entries instead of being silently dropped,
  per-utterance CSVs always carry a header, a multi run that processes zero
  clips raises instead of logging completion, and `--method pyannote_vad`
  without the `[pyannote]` extra fails fast with the install hint.

- `calculate_poseidon_score` now raises an actionable `TypeError` naming the
  missing `psdn-sonar[ml]` extra when given the `None` that
  `compute_semantic_similarity` returns without sentence-transformers, instead
  of an opaque comparison error (#101). README and USAGE now flag that the
  `[dev]` install does not include `[ml]`.
- `psdn-sonar multi --hf-model` now loads the HuggingFace checkpoint instead of
  calling `create_model(None)`. `--language` is passed through to scoring.
- Language-default `single` runs skip hosted API models unless their keys are
  set, so `psdn-sonar single --language en` no longer tries ElevenLabs/OpenAI/
  AssemblyAI on a fresh install.
- Omitting `--language` logs a warning that scoring defaults to Bengali.
- Multi-speaker FAQ/examples now use the runtime manifest schema
  (`audio_filepaths`, JSON `transcript_filepath`) and ship a tiny fixture.
- `docs/USAGE.md` section 1 now scores through
  `UtteranceEvaluator.score_single_variant` — the evaluation path, which
  normalizes before computing CER/WER — and labels `calculate_cer_wer` as a
  raw-text primitive, so the first documented number matches what an
  evaluation run reports (#100).

## [0.1.0] - 2026-08-17

### Added

- Multi-language text normalization pipeline (Bengali, Hindi, English, Korean)
  with loanword replacement, symbol/number verbalization, and script checks.
- Config-driven ASR backends: HuggingFace models, hosted APIs (OpenAI Whisper,
  AssemblyAI, ElevenLabs), and a central model registry.
- Evaluation metrics: CER, WER, semantic similarity, and the composite
  PSDN score with configurable weights.
- Audio-quality metrics: SNR, clipping ratio, silence ratio, and optional
  reference-free MOS scorers (DNSMOS, UTMOS, SQUIM).
- Single-speaker, multi-speaker (manifest-driven), and custom (YAML-configured)
  evaluation pipelines, exposed via the `psdn-sonar` CLI.
- Dataset discovery and preparation for public corpora (`psdn-sonar discover`).
- Reporting suite: markdown report generator plus lexical-diversity,
  cross-dataset, demographic, audio-quality, and latency plots.
- Precomputed public benchmark data and aggregation scripts
  (`precompute_benchmarks.py`, `extract_benchmarks.py`, `build_macro_summary.py`).
- Data utilities: YAML-driven cloud sync, Common Voice and Zeroth-Korean
  TSV converters, transcript-JSON data prep, SNR-vs-WER analysis.
- Runnable examples for every pipeline under `examples/`.
