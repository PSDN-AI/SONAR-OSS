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

Language processors, model backends, dataset loaders, evaluators, reporting,
and the CLI are imported incrementally in subsequent PRs — see
docs/import-gate.md for the checklist every import must pass.
"""

__version__ = "0.1.0"
