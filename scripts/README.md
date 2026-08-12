# Scripts

Utility scripts for repository maintenance and dataset acquisition. They are not
part of the installable `psdn_sonar` package; run them from the repo root with
`uv run python scripts/<name>.py`.

## Repository maintenance

| Script | Role |
|--------|------|
| `check_internal_refs.sh` | CI gate: fails on references to private infrastructure |
| `check_installed_package.py` | Smoke-checks an installed wheel (used by the package workflow) |
| `dependency_audit.py` | Validates `pip-audit` findings against reviewed exceptions |

## Data acquisition

| Script | Role |
|--------|------|
| `download_dataset.py` | Fetch a manifest + audio dataset from S3/R2 and convert to TSV (requires the `cloud` extra) |
| `download_data.py` | Sync remote directories/files to local disk from a YAML config (see `download_config.example.yaml`) |
| `download_commonvoice_english.sh` | Download Common Voice Spontaneous Speech (English) from Mozilla Data Collective |

`download_data.py` is a thin wrapper over the packaged
`psdn_sonar.utils.data_downloader.sync_from_config`, so the same sync can be
driven programmatically from an installed wheel. Cloud scripts never embed
credentials — supply them via environment variables (`AWS_*`, `R2_*`,
`MOZILLA_API_KEY`), a `.env` file, or CLI flags.
