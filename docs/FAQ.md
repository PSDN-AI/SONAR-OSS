# SONAR — FAQ

---

## Q1: What CLI commands to issue, and what files are needed?

**Dataset discovery** — find and download public datasets for a language:

```bash
psdn-sonar discover --language bn --dry-run          # list what's available
psdn-sonar discover --language bn --output data/bn   # download + prepare
psdn-sonar discover --language ko --max-samples 500  # limit sample count
```

**Single-speaker evaluation** — the main workhorse:

```bash
psdn-sonar single \
  --input data/test.tsv \
  --models wav2vec2_bengali whisper_api elevenlabs_api \
  --output results/single-speaker-eval \
  --report
```

**Multi-speaker evaluation** — for conversational audio:

```bash
psdn-sonar multi \
  --input data/manifest.jsonl \
  --models elevenlabs_api \
  --method energy_trim \
  --demographics --dataset-dir /path/to/dataset \
  --output results/multi-eval
```

**Custom / BYOL evaluation** — bring your own language:

```bash
psdn-sonar custom --config examples/custom_eval_portuguese.yaml --report
```

### Files needed before running

| File | Format | Where it comes from |
|------|--------|-------------------|
| `data/*.tsv` | Tab-separated: `audio_path \t transcription` | `psdn-sonar discover`, manual prep, or `scripts/prepare_data.py` |
| `data/manifest.jsonl` | JSON lines, one recording per line (see manifest format below) | Written by hand or by your recording tooling |
| `.env` | API keys: `OPENAI_API_KEY`, `ASSEMBLYAI_API_KEY`, `ELEVENLABS_API_KEY`, `HF_TOKEN` | Manually created from `.env.example` |
| `examples/custom_eval_*.yaml` | BYOL config: language, HF model IDs, dataset path, text processing flags | Written by hand per new language |
| Audio files (`.wav`/`.flac`) | Referenced by `audio_path` column in the TSV | HuggingFace download or your own recordings |

### Manifest format (multi-speaker)

Each line of `manifest.jsonl` is a JSON object describing one recording.
Paths are relative to the manifest's directory:

```json
{
  "audio_id": "REC001",
  "audio_a_path": "REC001/speaker_a.wav",
  "audio_b_path": "REC001/speaker_b.wav",
  "combined_audio_path": "REC001/combined.wav",
  "transcript_path": "REC001/transcript.txt"
}
```

See `examples/test_manifest.jsonl` for a working sample.

---

## Q2: Where would the output show up: the JSON & MD files?

All outputs land under the `results/` directory (or wherever `--output` points).

### After a single-speaker eval (`psdn-sonar single`)

```
results/single-speaker-eval/
├── asr_detailed_wav2vec2_bengali.csv    # per-utterance results
├── asr_detailed_whisper_api.csv         # one CSV per model
├── asr_detailed_elevenlabs_api.csv
├── scores_wav2vec2_bengali.json         # machine-readable run artifact
└── scores_whisper_api.json              # (see psdn_sonar/benchmark/README.md)
```

### After adding `--report`

One analysis directory per model:

```
results/single-speaker-eval/analysis/<model>/
├── EVAL_REPORT.md
├── diversity-analysis/
│   ├── diversity_gt_comparative_diversity.png
│   ├── diversity_gt_vocabulary_growth_curve.png
│   └── diversity_gt_zipf_law.png
├── cross-dataset-analysis/
│   └── <metric>_by_dataset_model.png
├── hard-negatives-analysis/
│   └── wer/cer hard-negatives comparisons
└── audio-quality-analysis/
    ├── snr_vs_wer_scatter.png
    ├── snr_distribution.png
    ├── latency_boxplot.png
    └── quality_summary.json
```

### After multi-speaker eval

```
results/multi-eval/
├── asr_eval_results_elevenlabs_api_manifest.csv
├── asr_eval_results_elevenlabs_api_manifest_stats.txt
└── demographic-analysis/       # if --demographics was used
    ├── cer_by_gender.png
    ├── wer_by_age_group.png
    └── poseidon_by_region.png
```

---

## Q3: How do we transfer / transform data from one place / form to another?

### The data transformation chain (e.g. HF, S3/R2, or local)

```
HuggingFace Hub                    Local recordings
     │                                   │
     ▼                                   ▼
psdn-sonar discover              scripts/prepare_data.py
     │                                   │
     ▼                                   ▼
 TSV files in data/              TSV files in data/
 (audio_path \t transcription)   (audio_path \t transcription)
     │                                   │
     └──────────────┬────────────────────┘
                    ▼
          psdn-sonar single / multi / custom
                    │
                    ▼
         asr_detailed_{model}.csv
         (per-utterance metrics)
                    │
                    ▼
              --report flag
                    │
                    ▼
         EVAL_REPORT.md + plots/
```

Datasets stored in S3-compatible object storage (AWS S3, Cloudflare R2) can be
synced locally with `scripts/download_data.py --config your_sync.yaml` — see
`scripts/download_config.example.yaml` for the schema.

---

## Q4: Pre-run checklist before another evaluation run

**Environment:**

- [ ] `.env` file exists with valid, non-expired API keys (especially if using `whisper_api`, `assemblyai_api`, or `elevenlabs_api`)
- [ ] `HF_TOKEN` is set if using pyannote VAD/diarization

**Input data:**

- [ ] TSV file has correct tab-separated columns: `audio_path` and `transcription`
- [ ] Audio files actually exist at the paths referenced in the TSV
- [ ] Audio files are in a supported format (`.wav`, `.flac`, `.mp3`)
- [ ] For multi-speaker: `manifest.jsonl` has valid audio and transcript paths

**Previous results (if re-running):**

- [ ] Open previous `asr_detailed_*.csv` — scan for rows with very high CER/WER (> 0.9), which often indicate bad audio, missing files, or API errors
- [ ] Check for empty `hypothesis` fields — means transcription failed entirely
- [ ] Look at `EVAL_REPORT.md` summary — are mean CER/WER in expected ranges? Compare against the baselines in `psdn_sonar/benchmarks/`

---

## Q5: How do we measure / benchmark success?

### Primary metrics (per utterance)

| Metric | What it measures | Good value |
|--------|-----------------|------------|
| **CER** | Character-level transcription accuracy | < 0.15 |
| **WER** | Word-level transcription accuracy | < 0.25 |
| **Semantic Similarity** | Meaning preservation (cosine similarity of sentence embeddings) | > 0.85 |
| **PSDN Score** | Weighted composite: `w_wer×(1−WER) + w_cer×(1−CER) + w_sem×Similarity` (defaults: 0.35 / 0.20 / 0.45, configurable per-call or via env vars) | > 0.75 |

### Secondary metrics

| Metric | What it measures |
|--------|-----------------|
| **SNR (dB)** | Audio signal-to-noise ratio — flags bad recordings |
| **Clipping ratio** | Fraction of samples at max amplitude — flags distortion |
| **Silence ratio** | Fraction of audio that is silence |
| **Latency (s)** | Transcription time per utterance — for model speed comparison |
| **DNSMOS / UTMOS / SQUIM** | Reference-free MOS scores — predicted audio quality |
