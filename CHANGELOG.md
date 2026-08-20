# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Documented what the published scores measure and made the caveats
  machine-readable (#119): `docs/SCORE_INTERPRETATION.md` states the
  project's position on the three reader traps — multi-speaker WER/CER
  measures the preprocessing + ASR pipeline end to end (the multi CLI now
  logs this scope at run start), scores are comparable within a dataset
  only (`precompute_benchmarks.py` now publishes per-dataset
  utterance-length stats as `public_length_stats_<language>.json`), and
  cells where a model is evaluated on a corpus its card declares as
  training data are marked (`psdn_sonar.models.provenance` records the
  audited card declarations; the precompute script writes per-cell
  `domain_markers.json` and warns when running an in-domain pair).
  Leaderboard rendering of the markers is owned by `PSDN-AI/psdn-portals`.
- `scores.json` now carries a `lineage` block recording the facts needed to
  compare two runs like-for-like (#120): the HuggingFace repo id and
  checkpoint commit SHA actually loaded (`hf_model_id`/`hf_revision` — the
  model registry pins no revisions, so this is the only record of which
  weights produced the numbers) and the WER normalization contract in force
  (`normalization`, e.g. `bn:v1+bnlp`). The Bengali contract marks whether
  the optional `bnlp` tokenizer was active, because its silent
  whitespace-split fallback changes tokenization — and therefore absolute
  WER/POSEIDON — between otherwise identical environments.

### Fixed

- A TSV `audio_path` with `../` can no longer escape the dataset
  directory (#127): the boundary guard in the single-speaker evaluator
  existed but was gated behind a parameter that defaulted off and had no
  CLI exposure, so a manifest received from someone else could read
  audio from anywhere on disk. Relative paths that resolve outside the
  TSV's directory are now always rejected with a clear error before any
  model loads. Absolute paths remain allowed by default (they are
  explicit in the TSV and `discover` output writes them); the new
  `psdn-sonar single --strict-audio-paths` flag additionally rejects
  absolute paths and requires every path to be an existing regular file.
- Multi-speaker assignment no longer treats a perfect score as missing
  (#106): the `dual_assignment_score` selection heuristic defaulted
  missing metrics with `or`, which also fires on a legitimate `0.0` —
  a perfect transcription (CER/WER 0.0) was scored as worst case, so a
  swapped speaker-to-reference pairing with real errors but slightly
  higher similarity could win and corrupt both speakers' reported
  WER/CER. Missing metrics now default via explicit `None` checks,
  matching the equivalent computation in `core.py`.
- Fully silent audio no longer passes the quality gate (#105):
  `calculate_silence_ratio` measures frames relative to the file's own
  loudest frame, so uniformly quiet files scored `0.0` — identical to
  all-speech. Files whose loudest RMS frame is below an absolute silence
  floor (`SONAR_SILENCE_FLOOR_AMP`, default 1e-3 ≈ -60 dBFS) now score
  `1.0` and trip the `high_silence` warning; mixed-content behavior is
  unchanged. `calculate_snr` returns `None` instead of `inf` for
  signal-less audio (SNR is undefined; the old `inf` leaked into CSVs,
  gave silent files an SNR tier of "High", and poisoned plot columns) and
  caps noise-free audio at 100 dB.
- `psdn-sonar discover` no longer reports failure as success (#104):
  `--datasets` entries are validated per name with distinct reasons — unknown
  name (with the discoverable list), catalogued-but-disabled
  (`common_voice`), catalogued non-HuggingFace source (the OpenSLR Bengali
  corpora), or catalogued-but-not-wired (`multilingual_librispeech`) — and
  exit 1 instead of warning and exiting 0. A valid filter that matches
  nothing for the language also exits 1, and the error blames the filter
  (naming each dataset's supported languages) rather than the language. The
  summary now prints a catalog scope note so a one-row listing is no longer
  presented as the complete catalog.
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
