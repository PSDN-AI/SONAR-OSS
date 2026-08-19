# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

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
