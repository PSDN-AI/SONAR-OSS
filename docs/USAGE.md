# Using SONAR

SONAR (`psdn-sonar`) scores ASR (speech-to-text) output with reproducible
metrics: **WER**, **CER**, semantic similarity, and the composite **POSEIDON**
score.

> These examples use the Python API. The same workflows are available from the
> `psdn-sonar` command-line interface — see the CLI examples in
> [`docs/FAQ.md`](FAQ.md) and `psdn-sonar --help`.

## Install

New users: follow the package install in [`README.md`](../README.md)
(`pip install "psdn-sonar[ml]"`, or the source steps listed there).

From a clone, the contributor install that includes local-model extras is:

```bash
make setup-ml            # uv: frozen dev + [ml] extras
# or, inside an activated virtual environment:
pip install -e ".[ml]"   # core + ML models/backends
```

Plain `make setup` and `pip install -e ".[dev]"` do **not** include `[ml]`,
and the examples below need it.

Core WER/CER work on the base install; POSEIDON's semantic similarity and
running HuggingFace models need the `[ml]` extra.

## 1. Score a single reference vs. hypothesis

Score the way an evaluation run scores: normalize both sides for the target
language, then compute CER/WER. `UtteranceEvaluator.score_single_variant` is
the same per-utterance code path `SingleSpeakerEvaluator` uses:

```python
from psdn_sonar.evaluators.utterance import UtteranceEvaluator

cer, wer, ref_norm, hyp_norm = UtteranceEvaluator.score_single_variant(
    "Hello, World!", "hello world", language="en"
)
print(f"WER={wer:.2f}  CER={cer:.2f}")   # WER=0.00  CER=0.00
```

Case and punctuation differences normalize away (`ref_norm` and `hyp_norm`
both come back as `hello world`), so a perfect transcription scores WER 0.0
here exactly as it does in an evaluation report.

The lower-level primitive `calculate_cer_wer(reference, hypothesis)` scores
**raw text with no normalization** — on the pair above it reports WER 1.0,
because both raw tokens differ in case or punctuation. Use it only for inputs
you have already normalized (or deliberately want compared raw); evaluation
runs always normalize both sides first.

Add the composite POSEIDON score (needs `[ml]`):

```python
from psdn_sonar.evaluators.utterance import UtteranceEvaluator
from psdn_sonar.utils.metrics import calculate_poseidon_score, compute_semantic_similarity

ref, hyp = "the quick brown fox", "the quick brown box"
cer, wer, ref_norm, hyp_norm = UtteranceEvaluator.score_single_variant(ref, hyp, language="en")
sim = compute_semantic_similarity(ref_norm, hyp_norm)
print(f"POSEIDON={calculate_poseidon_score(cer, wer, sim):.3f}")
```

Without `[ml]`, `compute_semantic_similarity()` returns `None` and
`calculate_poseidon_score()` rejects it with a `TypeError` telling you to
install `psdn-sonar[ml]`. Handle the `None` explicitly if semantic similarity
is optional in your pipeline.

Note that the first `compute_semantic_similarity()` call in an environment
touches the network: it downloads the ~64 MB sentence-transformers scorer
into the HuggingFace cache. Later calls are offline.

## 2. Evaluate a model over a dataset

Create a tab-separated `eval.tsv` with `audio_path` and `transcription` columns:

```
audio_path	transcription
clips/0001.wav	the quick brown fox
clips/0002.wav	she sells sea shells
```

Then run one or more models over it:

```python
from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

SingleSpeakerEvaluator.run_evaluation(
    tsv_path="eval.tsv",
    models=["whisper_small_en"],
    language="en",
    output_dir="results/demo",
)
```

This writes per-utterance metrics to `results/demo/asr_detailed_<model>.csv`
and a machine-readable `results/demo/scores_<model>.json` — WER/CER/POSEIDON
aggregates plus a reproducibility record (git SHA, seed, model, timestamp).
The CSV includes `normalized_reference` / `normalized_hypothesis`, the exact
strings WER/CER were computed over after language normalization.

Local models auto-select the best available device (CUDA, then MPS, then
CPU), and the device used is recorded in `scores_<model>.json` under
`submission.device`. Before sizing a run, read "What a first run costs" in the
[README](../README.md#requirements): first runs download models and datasets
in full (several GB), and Whisper-class models on CPU run at roughly
20 s/sample versus ~1 s/sample for CTC models.

## 3. Discover available models

```python
from psdn_sonar.models.registry import list_models, get_language_defaults

print(list_models())                 # every registered model id
print(get_language_defaults("bn"))   # default models for a language: bn/hi/ko/en
```

Most local models need only the `[ml]` extra, but `khushids_bengali` (in the
Bengali defaults) is a PEFT/LoRA adapter that additionally requires the
`[bengali]` extra: `pip install "psdn-sonar[bengali]"`. Without it the model
is skipped with a message naming the extra, and the rest of a multi-model
run continues. Hosted API models (`*_api`) need their respective API keys
and are skipped from language defaults when the keys are unset.

## 4. Compare completed runs: the leaderboard

Every single-speaker evaluation run (`psdn-sonar single` or `run_evaluation`)
writes a `scores_<model>.json` artifact into its output directory; `multi`
writes per-clip CSV/TXT results but no scores artifact, so its runs do not
appear here. `psdn-sonar leaderboard` scans one or more directories for those
artifacts and renders a comparison table:

```bash
psdn-sonar leaderboard --runs results/ --language bn
psdn-sonar leaderboard --runs results/ --sort wer
psdn-sonar leaderboard --runs results/ --json   # machine-readable
```

The table shows only measured numbers. A metric no contributing run computed
(for example POSEIDON without the `[ml]` extra) is rendered as `—` — it is
never derived or back-solved from the other columns. Models evaluated under
multiple runs show the mean of their per-run aggregates and the run count,
and rows whose runs recorded configuration warnings in `scores.json` (such as
a reference-script/`--language` mismatch) are marked with `!`.

---

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) to contribute, and the module
docstrings for the full API.
