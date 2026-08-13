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

## Benchmark precomputation

| Script | Role |
|--------|------|
| `precompute_benchmarks.py` | Evaluate public datasets with a language's registered models and compute lexical statistics, producing the cached benchmark data under `psdn_sonar/benchmarks/` |
| `extract_benchmarks.py` | Aggregate `raw-evaluations/<model>/<dataset>.csv` files into `public_benchmarks.csv` (per-model per-dataset mean/std) |
| `build_macro_summary.py` | Unweighted macro-mean leaderboard across locales from the benchmarks CSV (one vote per locale) |
| `snr_vs_wer.py` | Enrich evaluation results with audio-quality metrics (SNR, clipping, silence) and render SNR-vs-WER plots |

```bash
# Korean: prepare FLEURS + Zeroth from HuggingFace, evaluate, compute stats
python scripts/precompute_benchmarks.py --language korean --prepare fleurs zeroth

# English: use a locally prepared Common Voice TSV
python scripts/precompute_benchmarks.py --language english --tsv commonvoice=path/to/test.tsv
```

## Data acquisition

| Script | Role |
|--------|------|
| `download_dataset.py` | Fetch a manifest + audio dataset from S3/R2 and convert to TSV (requires the `cloud` extra) |
| `download_data.py` | Sync remote directories/files to local disk from a YAML config (see `download_config.example.yaml`) |
| `download_commonvoice_english.sh` | Download Common Voice Spontaneous Speech (English) from Mozilla Data Collective |

## Data preparation

| Script | Role |
|--------|------|
| `prepare_data.py` | Convert a transcript-JSON delivery (audio + per-recording JSON) to eval + metadata TSVs |
| `convert_commonvoice_to_tsv.py` | Convert a Common Voice archive or extracted dir (classic or Spontaneous Speech layout) to the toolkit TSV |
| `convert_zeroth_korean_to_tsv.py` | Convert an extracted Zeroth-Korean (OpenSLR SLR40) directory to the toolkit TSV |

`download_data.py` is a thin wrapper over the packaged
`psdn_sonar.utils.data_downloader.sync_from_config`, so the same sync can be
driven programmatically from an installed wheel. Cloud scripts never embed
credentials — supply them via environment variables (`AWS_*`, `R2_*`,
`MOZILLA_API_KEY`), a `.env` file, or CLI flags.
