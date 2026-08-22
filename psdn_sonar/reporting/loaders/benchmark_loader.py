"""Loaders for precomputed public-benchmark statistics shipped in ``benchmarks/``.

No benchmark data ships in the repository (a deliberate import-gate rule:
see ``docs/import-gate.md``); the directory is only populated by running the
precompute scripts locally. Every loader here therefore degrades to an empty
value, and ``available_benchmark_datasets`` below is the single availability
probe the report generator uses to decide whether it may claim a
public-benchmark comparison at all (issue #113).
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"

_LANGUAGE_ALIASES = {
    "bn": "bengali",
    "ko": "korean",
    "hi": "hindi",
    "en": "english",
}

# Filename-stem fragments of known public benchmarks mapped to display names.
# Shared with the cross-dataset plots, which resolve CSVs the same way.
DATASET_DISPLAY = {
    "user_dataset": "Your dataset",
    "commonvoice": "Common Voice",
    "fleurs": "FLEURS",
    "zeroth": "Zeroth",
    "librispeech": "LibriSpeech",
    "openslr37_bd": "OpenSLR37 BD",
    "openslr37_in": "OpenSLR37 IN",
    "openslr53": "OpenSLR53",
}


def canonical_language(language: str) -> str:
    """Long-form language name used in benchmark paths (``bn`` -> ``bengali``)."""
    return _LANGUAGE_ALIASES.get(language.lower(), language.lower())


def raw_evaluations_dir(language: str) -> Path:
    """Directory holding precomputed per-model benchmark evaluation CSVs."""
    return _BENCHMARKS_DIR / canonical_language(language) / "raw-evaluations"


def available_benchmark_datasets(language: str) -> list:
    """Display names of public benchmarks with at least one precomputed
    evaluation CSV present for *language*, in the order they are found.

    Returns ``[]`` in a stock checkout or install — no benchmark data ships —
    which is the signal that a generated report must not claim any
    public-benchmark comparison (issue #113).
    """
    eval_dir = raw_evaluations_dir(language)
    if not eval_dir.is_dir():
        return []

    names = []
    for model_path in sorted(p for p in eval_dir.iterdir() if p.is_dir()):
        for csv_file in sorted(model_path.glob("*.csv")):
            key = next((k for k in DATASET_DISPLAY if k != "user_dataset" and k in csv_file.stem), None)
            if key is not None and DATASET_DISPLAY[key] not in names:
                names.append(DATASET_DISPLAY[key])
    return names


def _load_benchmark_json(prefix: str, language: str) -> dict:
    """Load ``{prefix}_{language}.json``, falling back to the legacy un-suffixed
    ``{prefix}.json`` for Bengali (the first shipped benchmark). Returns ``{}``
    when neither file exists or parsing fails."""
    language = canonical_language(language)
    try:
        benchmark_file = _BENCHMARKS_DIR / f"{prefix}_{language}.json"
        if benchmark_file.exists():
            with open(benchmark_file, "r", encoding="utf-8") as f:
                return json.load(f)

        legacy_file = _BENCHMARKS_DIR / f"{prefix}.json"
        if legacy_file.exists() and language == "bengali":
            with open(legacy_file, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Could not load %s for %s: %s", prefix, language, e)

    return {}


def load_public_benchmark_diversity(language: str = "bengali") -> dict:
    """Load public benchmark diversity stats for a specific language."""
    return _load_benchmark_json("public_diversity_stats", language)


def load_public_lexical_data(language: str = "bengali") -> dict:
    """Load public lexical data for a specific language."""
    return _load_benchmark_json("public_lexical_data", language)
