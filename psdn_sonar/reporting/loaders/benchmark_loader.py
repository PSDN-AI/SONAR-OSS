"""Loaders for precomputed public-benchmark statistics shipped in ``benchmarks/``."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_BENCHMARKS_DIR = Path(__file__).parent.parent.parent / "benchmarks"


def _load_benchmark_json(prefix: str, language: str) -> dict:
    """Load ``{prefix}_{language}.json``, falling back to the legacy un-suffixed
    ``{prefix}.json`` for Bengali (the first shipped benchmark). Returns ``{}``
    when neither file exists or parsing fails."""
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
