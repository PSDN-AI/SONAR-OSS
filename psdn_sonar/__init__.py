"""psdn-sonar: multi-language ASR evaluation toolkit.

Currently available layers:

- :mod:`psdn_sonar.utils` — metrics (WER/CER/semantic/POSEIDON), text
  normalization, number/symbol verbalization, loanword normalization,
  plotting helpers, and the S3 data downloader.
- :mod:`psdn_sonar.config` — environment-based POSEIDON weight configuration.
- :mod:`psdn_sonar.config_loader` — YAML/OmegaConf run configuration
  (``conf/`` tree: language, backend, and validation profiles).
- :mod:`psdn_sonar.registry` — ASR-backend and language-processor registries.
- :mod:`psdn_sonar.language_codes` — ISO 639-1 language code mappings.
- :mod:`psdn_sonar.language` — per-language text processors (en, bn, hi, ko).
- :mod:`psdn_sonar.models` — ASR model registry, adapter implementations
  (HuggingFace and hosted vendor APIs), and protocol-aware latency types
  (``LatencyMetrics``).
- :mod:`psdn_sonar.backends` — config-driven ASR backends.
- :mod:`psdn_sonar.analysis` — demographic performance analysis
  (``DemographicAnalyzer``).
- :mod:`psdn_sonar.core` — dataset/manifest evaluation loops.
- :mod:`psdn_sonar.recipe` — per-language evaluation recipes
  (models, datasets, normalization).
- :mod:`psdn_sonar.multispeaker_pipeline` — manifest-driven multi-speaker
  evaluation pipeline.
- :mod:`psdn_sonar.orchestrators` — multi-dataset / multi-model workflows
  (``CrossDatasetComparison``).
- :mod:`psdn_sonar.custom_eval` — bring-your-own HuggingFace model + dataset
  evaluation from a YAML config.
- :mod:`psdn_sonar.cli` — the ``psdn-sonar`` command-line interface
  (``single`` / ``multi`` / ``discover`` / ``custom``).

Examples and public scripts are imported incrementally in subsequent PRs —
see docs/import-gate.md for the checklist every import must pass.
"""

__version__ = "0.1.1"
