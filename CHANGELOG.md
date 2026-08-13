# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-08-14

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
