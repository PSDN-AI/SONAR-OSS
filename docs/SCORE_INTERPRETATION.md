# What the SONAR scores measure

SONAR publishes WER, CER, semantic similarity, and POSEIDON scores per
(model, dataset) cell. Three properties of those numbers are easy to misread.
This page states the project's position on each and where the supporting
data lives. It resolves the questions raised in
[issue #119](https://github.com/PSDN-AI/SONAR-OSS/issues/119).

## 1. Multi-speaker scores measure the pipeline, not the model alone

On the multi-speaker path, audio passes through SONAR's own preprocessing
(energy trimming, VAD, or diarization) before it reaches the model under
test. Each stage can fail in ways the model cannot recover from: quiet
speech trimmed as silence becomes deletion errors, missed VAD segments never
reach the model, and diarization can assign speech to the wrong speaker.
Method selection is signal-based (`(1 - silence_ratio) x (1 - duration_change)`)
with no content awareness.

**Position: multi-speaker WER/CER is an end-to-end measurement of
preprocessing + ASR, by design.**

- Every model in a run passes through the same preprocessing configuration,
  so *within-run* comparisons share the confound — but preprocessing error
  interacts with model behavior, so it is not provably uniform across models.
- Absolute multi-speaker numbers include preprocessing error and must not be
  quoted as isolated model capability or mixed into single-speaker
  comparisons.
- The multi-speaker CLI logs this scope statement at the start of every run.

Models without native diarization support are evaluated on multi-speaker
data through per-channel preprocessing rather than excluded; that is a
deliberate trade-off to keep the model roster comparable on the same clips,
at the cost of the confound documented here.

## 2. Scores are comparable within a dataset only

Datasets shown side by side can have utterance-length profiles that differ
by ~6x (e.g. OpenSLR53 median 3 words vs FLEURS-bn median 17). WER on a
3-word utterance moves in steps of 1/3, so short-utterance corpora produce
quasi-binary per-utterance scores — a median or q3 of exactly 1.0 on such a
corpus is a length artifact, not evidence of a perfect model.

**Position: a score is meaningful relative to other models on the same
dataset, never across datasets.**

- `scripts/precompute_benchmarks.py` publishes
  `public_length_stats_<language>.json` next to the evaluation results:
  per-dataset word-count quartiles, means, and the fraction of utterances
  with 5 words or fewer, so the length profile is visible wherever the
  scores are shown.
- Leaderboard rendering (dataset selector separation or length badges) is
  owned by `PSDN-AI/psdn-portals` and should consume these stats.

## 3. In-domain cells are marked

Some registered models are evaluated on corpora their own HuggingFace model
cards declare as training data:

| Registry name | Checkpoint | Card declares training on | Overlapping benchmark |
| --- | --- | --- | --- |
| `khushids_bengali` | `KhushiDS/whisper-large-v3-Bengali` | `google/fleurs` | FLEURS |
| `wav2vec2_bengali` | `arijitx/wav2vec2-xls-r-300m-bengali` | `openslr`, `SLR53`, `AI4Bharat/IndicCorp` | OpenSLR53 |
| `kresnik_wav2vec2_large_xlsr_korean` (and alias `wav2vec2_xlsr_korean`) | `kresnik/wav2vec2-large-xlsr-korean` | `kresnik/zeroth_korean` | Zeroth-Korean |

(Card declarations read 2026-08-18/19; recorded in
`psdn_sonar/models/provenance.py`.)

**Position: in-domain evaluation is legitimate when the authors held out the
evaluated split, but the reader must be able to tell — so every cell carries
a marker.**

- `psdn_sonar.models.provenance.evaluation_domain(model, dataset)` classifies
  each cell as `in-domain` (card declares the dataset), `not-declared`
  (audited card declares other corpora — not proof of held-out), or
  `unknown` (unaudited card, custom model, or hosted API with undisclosed
  training data).
- `scripts/precompute_benchmarks.py` writes `domain_markers.json` next to the
  raw evaluations and logs a warning when it evaluates an in-domain pair.
- None of the cards states which split was used for training, so split
  hygiene cannot be verified from public metadata. The marker states the
  declared relationship; it does not adjudicate it. The measured effect is
  real either way: `kresnik_wav2vec2_large_xlsr_korean` scores a POSEIDON
  median of 1.0000 on Zeroth-Korean vs 0.7389 on FLEURS-KO in an identical
  environment.
- Rendering the marker on the leaderboard page is owned by
  `PSDN-AI/psdn-portals`, consuming `domain_markers.json`.
