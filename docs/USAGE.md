# Using SONAR

SONAR (`psdn-sonar`) scores ASR (speech-to-text) output with reproducible
metrics: **WER**, **CER**, semantic similarity, and the composite **POSEIDON**
score.

> These examples use the Python API. The same workflows are available from the
> `psdn-sonar` command-line interface — see the CLI examples in
> [`docs/FAQ.md`](FAQ.md) and `psdn-sonar --help`.

## Install

New users: follow the package install in [`README.md`](../README.md)
(`pip install "psdn-sonar[ml]"` once published, or the TestPyPI / source
steps listed there).

From a clone, the contributor install that includes local-model extras is:

```bash
pip install -e ".[ml]"   # core + ML models/backends
```

Core WER/CER work on the base install; POSEIDON's semantic similarity and
running HuggingFace models need the `[ml]` extra.

## 1. Score a single reference vs. hypothesis

```python
from psdn_sonar.utils.metrics import calculate_cer_wer

cer, wer = calculate_cer_wer("the quick brown fox", "the quick brown box")
print(f"WER={wer:.2f}  CER={cer:.2f}")
```

Add the composite POSEIDON score (needs `[ml]`):

```python
from psdn_sonar.utils.metrics import (
    calculate_cer_wer,
    calculate_poseidon_score,
    compute_semantic_similarity,
)

ref, hyp = "the quick brown fox", "the quick brown box"
cer, wer = calculate_cer_wer(ref, hyp)
sim = compute_semantic_similarity(ref, hyp)
print(f"POSEIDON={calculate_poseidon_score(cer, wer, sim):.3f}")
```

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

## 3. Discover available models

```python
from psdn_sonar.models.registry import list_models, get_language_defaults

print(list_models())                 # every registered model id
print(get_language_defaults("bn"))   # default models for a language: bn/hi/ko/en
```

---

See [`CONTRIBUTING.md`](../CONTRIBUTING.md) to contribute, and the module
docstrings for the full API.
