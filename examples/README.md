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

```bash
# Single-speaker evaluation on the bundled sample data
python examples/single_speaker_audio_dataset.py --tsv-path examples/test_data.tsv --models whisper_api

# Multi-speaker evaluation
python examples/multispeaker_audio_dataset.py --manifest examples/test_manifest.jsonl --model elevenlabs_api

# Custom-language evaluation from a YAML config
psdn-sonar custom --config examples/custom_eval_portuguese.yaml --output results/custom-eval --report
```

## Sample data

- `test_data.tsv` — minimal single-speaker TSV (Bengali).
- `test_manifest.jsonl` — minimal multi-speaker manifest; each entry points
  to per-speaker audio, combined audio, and a transcript.
