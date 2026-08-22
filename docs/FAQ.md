# SONAR — FAQ

---

## Q1: What CLI commands to issue, and what files are needed?

**Dataset discovery** — find and download public datasets for a language.
`--datasets` accepts registry names currently wired into discover: `fleurs`,
`voxpopuli` (English among the four supported eval languages), and `zeroth`
(Korean). Common Voice is not discoverable.

```bash
psdn-sonar discover --language en --dry-run                        # list what's available
psdn-sonar discover --language en --datasets fleurs --output data/en
psdn-sonar discover --language ko --datasets fleurs --max-samples 10
```

**Single-speaker evaluation** — the main workhorse. Always pass `--language`
and `--models` (or `--hf-model`). Hosted IDs such as `whisper_api` need API
keys; local IDs such as `whisper_base_en` need `psdn-sonar[ml]`.

```bash
psdn-sonar single \
  --input data/test.tsv \
  --models whisper_base_en \
  --language en \
  --output results/single-speaker-eval \
  --report
```

Omitting `--models` runs the language's default *local* models. Hosted API
defaults are skipped unless their keys are set. Omitting `--language`
defaults to Bengali (`bn`) and logs a warning — pass the code explicitly.

**Multi-speaker evaluation** — for conversational audio. The input is a
`manifest.jsonl` in the schema below (not a TSV). `--models` or `--hf-model`
is required.

```bash
psdn-sonar multi \
  --input examples/test_manifest.jsonl \
  --models whisper_base_en \
  --language en \
  --method no_trim \
  --output results/multi-eval
```

**Custom / BYOL evaluation** — bring your own language. Pass `--max-samples`
on first run; the example config downloads FLEURS Portuguese unless you point
it at a local TSV.

```bash
psdn-sonar custom --config examples/custom_eval_portuguese.yaml --max-samples 10 --report
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

Each line of `manifest.jsonl` is a JSON object. Paths are relative to the
manifest's directory. The transcript must be JSON (not `.txt`).

```json
{
  "audio_id": "TEST001",
  "audio_filepaths": {
    "speaker_a": "sample_audio/TEST001/speaker_a.wav",
    "speaker_b": "sample_audio/TEST001/speaker_b.wav"
  },
  "transcript_filepath": "sample_audio/TEST001/transcript.json",
  "num_speakers": 2
}
```

Transcript JSON:

```json
{
  "segments": [
    {"speaker": "speaker_a", "text": "hello from speaker a", "start": 0.0, "end": 0.4},
    {"speaker": "speaker_b", "text": "hello from speaker b", "start": 0.65, "end": 1.05}
  ]
}
```

Optional combined / mixed audio, when used for VAD or diarization, must be
named `{audio_id}_Combined_Audio.wav` in the same directory as `speaker_a`
(for example `sample_audio/TEST001/TEST001_Combined_Audio.wav`).

See `examples/test_manifest.jsonl` for a working sample.

---

## Q2: Where would the output show up: the JSON & MD files?

All outputs land under the `results/` directory (or wherever `--output` points).

### After a single-speaker eval (`psdn-sonar single`)

```
results/single-speaker-eval/
├── asr_detailed_whisper_base_en.csv     # per-utterance results
├── scores_whisper_base_en.json          # machine-readable run artifact
└── analysis/whisper_base_en/            # only when --report is set
```

The per-utterance CSV carries `normalized_reference` / `normalized_hypothesis`
next to the raw `ground_truth` / `prediction` — the exact strings WER/CER were
computed over, which is what you need when two visually identical references
score differently (e.g. an invisible zero-width character).

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
├── audio-quality-analysis/
│   ├── snr_vs_wer_scatter.png
│   ├── snr_distribution.png
│   └── quality_summary.json
└── latency-analysis/
```

### After multi-speaker eval

```
results/multi-eval/
├── asr_eval_results_whisper_base_en_manifest.csv
├── asr_eval_results_whisper_base_en_manifest.txt
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
- [ ] Disk, bandwidth, and time are budgeted for a **first** run: datasets and model checkpoints download in full (several GB — see "What a first run costs" in the README), and `--max-samples` bounds processing, not the download
- [ ] You know which device the run will use: local models pick CUDA, then MPS, then CPU (recorded in `scores_<model>.json` under `config.device`); on CPU, Whisper-class models run at roughly 20 s/sample vs ~1 s/sample for CTC models

**Input data:**

- [ ] TSV file has correct tab-separated columns: `audio_path` and `transcription`
- [ ] Audio files actually exist at the paths referenced in the TSV
- [ ] Audio files are in a supported format (`.wav`, `.flac`, `.mp3`)
- [ ] For multi-speaker: `manifest.jsonl` uses `audio_filepaths` + `transcript_filepath` (JSON), and those files exist

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
| **POSEIDON** | Weighted composite: `w_wer×(1−WER) + w_cer×(1−CER) + w_sem×Similarity` (defaults: 0.35 / 0.20 / 0.45, configurable per-call or via env vars) | > 0.75 |

### Secondary metrics

| Metric | What it measures |
|--------|-----------------|
| **SNR (dB)** | Audio signal-to-noise ratio — flags bad recordings |
| **Clipping ratio** | Fraction of samples at max amplitude — flags distortion |
| **Silence ratio** | Fraction of audio that is silence |
| **Latency (s)** | Transcription time per utterance — for model speed comparison |
| **DNSMOS / UTMOS / SQUIM** | Reference-free MOS scores — predicted audio quality |
