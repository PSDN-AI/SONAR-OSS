"""Corpus and error metrics for report generation."""

from .hard_negatives import calculate_hard_negatives
from .lexical import (
    calculate_gini_coefficient,
    calculate_lexical_diversity_metrics,
    calculate_ngram_diversity,
    calculate_ngram_diversity_chunked,
)

__all__ = [
    "calculate_ngram_diversity",
    "calculate_ngram_diversity_chunked",
    "calculate_lexical_diversity_metrics",
    "calculate_gini_coefficient",
    "calculate_hard_negatives",
]
