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
| `warnings` | Run-level configuration warnings that make the numbers suspect, preserved verbatim (issue #148). Currently: the dominant Unicode script of the reference transcriptions contradicting `--language` (e.g. Latin-script references scored as `ko`), which means WER/CER were normalized with the wrong rules. Empty for a clean run; absent in artifacts written before the field existed. |
| `utterances` | Slim per-row metrics (paths, WER/CER, sem, POSEIDON, latency, errors) |

### Missing values and metric ranges

One convention applies to every scoring path in the package (issue #107):

- **A metric that cannot be computed is `null` — never substituted with a
  best- or worst-case value.** CER/WER are `null` when the normalized
  reference is empty or jiwer is unavailable; semantic similarity is `null`
  when the reference is empty or sentence-transformers (`[ml]` extra) is
  unavailable; `poseidon_score` is `null` whenever any of its three
  components is. A row whose metrics could not be computed is counted in
  `failed` (with the reason in its `error` field), even when transcription
  itself succeeded — the prediction is preserved on the row.
- **Aggregates are computed only over present values.** `cer_mean` /
  `wer_mean` average the successfully scored rows; `semantic_similarity_mean`
  / `poseidon_score_mean` average the rows where that metric exists;
  `significant_wer_rate` excludes missing WERs from both numerator and
  denominator. When no row has a value, the aggregate is `null` (not `0.0`).
  Use `total_samples` / `successful` / `failed` to see how many rows
  contributed.
- **An empty hypothesis against a non-empty reference is measurable, not
  missing**: WER/CER are genuinely 1.0 (every word wrong).
- **`semantic_similarity` is cosine similarity clamped to `[0, 1]`** at the
  point of computation. Raw cosine ranges `[-1, 1]`, but every artifact —
  the per-utterance CSV, `semantic_similarity_mean`, the POSEIDON input, and
  the public leaderboard — reports the clamped value, so
  `semantic_similarity_mean` and `poseidon_score_mean` are computed over the
  same range and neither can go negative. A negative raw cosine (unrelated
  texts) reads as `0.0`.

The same convention governs derived artifacts: `ensure_poseidon_score`
(used by the reporting plots to backfill POSEIDON on legacy CSVs) leaves
`NaN` for rows with any missing metric instead of fabricating a score, and
the public `PoseidonScorer` API returns `None` metrics for unmeasurable
pairs.

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
    "timestamp_utc": "2026-05-22T12:00:00Z",
    "poseidon_weights": {"wer": 0.35, "cer": 0.20, "semantic": 0.45},
    "similarity_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "os_platform": "Linux-6.8.0-57-generic-x86_64-with-glibc2.39",
    "python_version": "3.12.3",
    "device": "cpu"
  },
  "model_name": "whisper_api",
  "lineage": {
    "hf_model_id": null,
    "hf_revision": null,
    "normalization": "bn:v3+bnlp"
  },
  "warnings": [],
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
`seed` (from `conf/config.yaml` via `config_loader.get_run_seed()`), and the
score-changing inputs: `poseidon_weights` and `similarity_model` as actually
in effect (including `POSEIDON_*_WEIGHT` / `SIMILARITY_MODEL` env overrides),
plus `os_platform`, `python_version`, and `device`. Callers supply
run-specific fields (`provider`, `model_snapshot`, `region`, `protocol`, etc.).

`git_sha` identifies the psdn-sonar checkout the package ran from: it is
resolved against the package's own directory (never the caller's working
directory) and only when the package files are tracked by that repository.
Installs without a checkout (wheel/pip) record `"unknown"` — use
`package_version` there, or set the `SONAR_GIT_SHA` environment variable to
stamp a known commit (e.g. in CI when evaluating a built wheel). `device` is
`null` when torch is not installed (API-only environments).

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

When the single-speaker evaluator builds this block itself (no `submission`
passed), `provider` is derived from the adapter that actually served
inference (`openai`/`elevenlabs`/`assemblyai` for hosted APIs, `local` for
in-process models), `model_snapshot` records the provider-side model id the
adapter requested (falling back to the registry alias, which is separately
recorded as `model_name`), and `region` is `null` unless the `SONAR_REGION`
environment variable supplies one — hosted providers do not disclose a
region, so the toolkit never invents one (issue #184). `judge_model` and
`prompt_version` stay `null` on this path because it never runs the LLM
judge; callers that do run it supply their own `SubmissionConfig`.
