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
# Single-speaker evaluation on the bundled sample clip (needs [ml])
psdn-sonar single --input examples/test_data.tsv --models whisper_base_en --language en --output results/example-single

# Multi-speaker evaluation on the bundled two-speaker fixture (needs [ml])
psdn-sonar multi --input examples/test_manifest.jsonl --models whisper_base_en --language en --method no_trim --output results/example-multi

# Custom-language evaluation from a YAML config (downloads FLEURS pt_br unless you change the dataset)
psdn-sonar custom --config examples/custom_eval_portuguese.yaml --max-samples 10 --output results/custom-eval --report
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

- `test_data.tsv` — one-row single-speaker TSV pointing at `sample_audio/single/sample.wav`.
- `test_manifest.jsonl` — one two-speaker clip. Each line must have `audio_id`,
  `audio_filepaths` (`speaker_a` / `speaker_b`), `transcript_filepath` (JSON),
  and `num_speakers`. Combined audio, when present, is
  `{audio_id}_Combined_Audio.wav` next to the speaker files.
- `sample_audio/TEST001/` — short synthetic tones plus `transcript.json`
  (format fixture, not real speech).
