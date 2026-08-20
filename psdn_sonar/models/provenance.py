"""Training-corpus provenance for registered models (issue #119).

Several registered models are evaluated on the very corpora their
HuggingFace model cards declare as training data. That is legitimate when
the model authors held out the evaluated split, but a reader comparing
leaderboard cells cannot tell an in-domain score from a held-out one —
e.g. ``kresnik_wav2vec2_large_xlsr_korean`` scores a POSEIDON median of
1.0 on Zeroth-Korean (declared training corpus) vs 0.74 on FLEURS-KO.

This module records what the model cards *declare*, so evaluation
tooling can mark each (model, dataset) cell. It makes no claim about
split hygiene: none of the cards state which split was used for
training, so whether the evaluated utterances were seen in training
cannot be settled from public metadata.

Declarations verified against the HuggingFace model cards on 2026-08-18/19:

- ``KhushiDS/whisper-large-v3-Bengali``: "fine-tuned ... on the fleurs
  dataset", card metadata ``datasets: [google/fleurs]``.
- ``arijitx/wav2vec2-xls-r-300m-bengali``: card metadata
  ``datasets: [openslr, SLR53, AI4Bharat/IndicCorp]``, model-index
  result on "Open SLR".
- ``kresnik/wav2vec2-large-xlsr-korean``: card metadata
  ``datasets: [kresnik/zeroth_korean]``, own model-index reports
  Test WER 4.74 on Zeroth Korean.

Dataset names are benchmark-catalog keys (see
``psdn_sonar/data/benchmark_catalog.yaml``).
"""

from typing import Dict, FrozenSet, Optional

# Registry model name -> catalog dataset names its card declares as training
# data. Only card-audited models appear here; absence means "not audited",
# not "trained on nothing".
_DECLARED_TRAINING_DATASETS: Dict[str, FrozenSet[str]] = {
    "khushids_bengali": frozenset({"fleurs"}),
    # Card also lists a generic "openslr" tag; only the specifically named
    # SLR53 is recorded to avoid overstating the declaration.
    "wav2vec2_bengali": frozenset({"openslr53"}),
    # Two registry names, same checkpoint (kresnik/wav2vec2-large-xlsr-korean).
    "kresnik_wav2vec2_large_xlsr_korean": frozenset({"zeroth"}),
    "wav2vec2_xlsr_korean": frozenset({"zeroth"}),
}

# Hosted APIs never disclose training data; their domain status is always
# "unknown" rather than "not-declared".
_UNDISCLOSED_TRAINING_DATA = frozenset({"whisper_api", "elevenlabs_api", "assemblyai_api"})

IN_DOMAIN = "in-domain"
NOT_DECLARED = "not-declared"
UNKNOWN = "unknown"


def declared_training_datasets(model_name: str) -> Optional[FrozenSet[str]]:
    """Catalog dataset names the model's card declares as training data.

    Returns ``None`` when no audited declaration exists — either the card
    was not audited or the vendor discloses nothing (hosted APIs).
    """
    return _DECLARED_TRAINING_DATASETS.get(model_name)


def evaluation_domain(model_name: str, dataset_name: str) -> str:
    """Classify one (model, dataset) evaluation cell.

    - ``"in-domain"``: the model card declares the dataset as training data.
    - ``"not-declared"``: the card was audited and declares other corpora,
      not this one. Not proof of held-out — only that no overlap is declared.
    - ``"unknown"``: no audited declaration (unaudited card, custom model,
      or hosted API with undisclosed training data).
    """
    declared = _DECLARED_TRAINING_DATASETS.get(model_name)
    if declared is None:
        return UNKNOWN
    return IN_DOMAIN if dataset_name in declared else NOT_DECLARED
