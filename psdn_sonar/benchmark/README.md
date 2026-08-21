# Benchmark run artifacts

## `scores.json`

Each `SingleSpeakerEvaluator` model run writes a machine-readable artifact next to
`asr_detailed_<model>.csv`:

```
results/my-run/
  asr_detailed_whisper_api.csv
  scores_whisper_api.json
```

Multi-model runs use one `scores_<model>.json` per model to avoid clobbering.

### Schema

| Field | Description |
|-------|-------------|
| `submission` | `SubmissionConfig` — provider, model snapshot, region, protocol, env |
| `model_name` | Registry model id for this run |
| `aggregate` | Mean CER/WER, optional sem/POSEIDON, latency avg/median/p95, counts |
| `lineage` | `RunLineage` — resolved HF checkpoint (`hf_model_id`, `hf_revision`) and the WER normalization contract in force (`normalization`, e.g. `"bn:v3+bnlp"`). Best-effort; fields are `null` for hosted API models. Recorded because the registry pins no model revisions, so without it the exact weights and rule set behind a number are unrecoverable. |
| `utterances` | Slim per-row metrics (paths, WER/CER, sem, POSEIDON, latency, errors) |

### Example

```json
{
  "submission": {
    "provider": "openai",
    "model_snapshot": "whisper-1@2024-06-01",
    "region": "us-east-1",
    "protocol": "batch",
    "inference_params": {
      "language_code": "bn",
      "temperature": 0.0
    },
    "sample_rate_hz": 16000,
    "seed": 42,
    "judge_model": null,
    "prompt_version": null,
    "git_sha": "3a4c7b106ff16bd3f689c9d88c24b0134dfc883e",
    "package_version": "0.1.0",
    "timestamp_utc": "2026-05-22T12:00:00Z"
  },
  "model_name": "whisper_api",
  "lineage": {
    "hf_model_id": null,
    "hf_revision": null,
    "normalization": "bn:v3+bnlp"
  },
  "aggregate": {
    "cer_mean": 0.12,
    "wer_mean": 0.18,
    "semantic_similarity_mean": 0.91,
    "poseidon_score_mean": 0.85,
    "latency_avg_s": 1.2,
    "latency_median_s": 1.1,
    "latency_p95_s": 2.4,
    "total_samples": 100,
    "successful": 98,
    "failed": 2,
    "elapsed_time_s": 120.5
  },
  "utterances": [
    {
      "audio_path": "/data/clip001.wav",
      "wer": 0.1,
      "cer": 0.05,
      "semantic_similarity": 0.95,
      "poseidon_score": 0.88,
      "inference_latency_s": 1.05,
      "error": null
    }
  ]
}
```

## `SubmissionConfig`

Use `SubmissionConfig.from_env()` to fill `git_sha`, `package_version`, `timestamp_utc`,
and `seed` (from `conf/config.yaml` via `config_loader.get_run_seed()`). Callers supply
run-specific fields (`provider`, `model_snapshot`, `region`, `protocol`, etc.).

```python
from psdn_sonar.benchmark import SubmissionConfig

cfg = SubmissionConfig.from_env(
    provider="assemblyai",
    model_snapshot="assemblyai_api",
    region="us-east-1",
    protocol="batch",
    inference_params={"language_code": "bn"},
    sample_rate_hz=16000,
)
```

`protocol` must be `"batch"` or `"streaming"`. Unknown keys in `inference_params` are rejected.
