# Examples

Runnable examples for the main psdn-sonar workflows. Each script has a
`--help` flag with full options.

## Evaluation

| Example | What it shows |
| --- | --- |
| `single_speaker_audio_dataset.py` | Evaluate registered models on a TSV dataset (`audio_path` + `transcription` columns). |
| `multispeaker_audio_dataset.py` | Evaluate a model on a multi-speaker `manifest.jsonl` dataset. |
| `custom_dataset.py` | Convert a dataset with arbitrary column names into psdn-sonar format and evaluate on it. |
| `custom_eval_portuguese.yaml` | Config for `psdn-sonar custom`: bring-your-own HuggingFace model + dataset for a new language. |

## HuggingFace workflows

| Example | What it shows |
| --- | --- |
| `huggingface_dataset_loader.py` | Download any HF audio dataset and convert it to psdn-sonar TSV format. |
| `huggingface_complete_workflow.py` | End-to-end: HF dataset + HF model to evaluation results and a full report. |
| `korean_language_smoke.py` | Config-driven registry: Korean normalization/tokenization and backend setup. |

```bash
# Convert FLEURS Korean to psdn-sonar format
python examples/huggingface_dataset_loader.py --dataset google/fleurs --config ko_kr --output data/fleurs-ko/test.tsv

# Full pipeline: dataset + model to report
python examples/huggingface_complete_workflow.py --dataset google/fleurs --config ko_kr \
    --hf-model openai/whisper-small --language ko --max-samples 50 --output-dir results/korean-eval
```

```bash
# Single-speaker evaluation on the bundled sample data
python examples/single_speaker_audio_dataset.py --tsv-path examples/test_data.tsv --models whisper_api

# Multi-speaker evaluation
python examples/multispeaker_audio_dataset.py --manifest examples/test_manifest.jsonl --model elevenlabs_api

# Custom-language evaluation from a YAML config
psdn-sonar custom --config examples/custom_eval_portuguese.yaml --output results/custom-eval --report
```

## Analysis and data utilities

| Example | What it shows |
| --- | --- |
| `demographic_analysis.py` | ASR performance across demographics (age, gender, region) from multi-speaker results. |
| `visualization.py` | Summary WER/CER plots from an evaluation summary CSV. |
| `download_from_cloud.py` | Fetching datasets from S3/Cloudflare R2 (requires the `cloud` extra). |

```bash
# Demographic breakdown of multi-speaker results
python examples/demographic_analysis.py --results-csv results/asr_eval_results_whisper_api_manifest.csv \
    --dataset-dir /path/to/dataset

# Summary plots from a results summary CSV
python examples/visualization.py --summary-csv results/summary.csv --output-dir results/plots
```

## Sample data

- `test_data.tsv` — minimal single-speaker TSV (Bengali).
- `test_manifest.jsonl` — minimal multi-speaker manifest; each entry points
  to per-speaker audio, combined audio, and a transcript.
