"""Report generation: metrics, loaders, and plots for evaluation results."""

from .loaders import load_public_benchmark_diversity, load_transcripts_from_file
from .metrics import (
    calculate_gini_coefficient,
    calculate_hard_negatives,
    calculate_lexical_diversity_metrics,
    calculate_ngram_diversity,
)

__all__ = [
    "calculate_gini_coefficient",
    "calculate_hard_negatives",
    "calculate_lexical_diversity_metrics",
    "calculate_ngram_diversity",
    "load_public_benchmark_diversity",
    "load_transcripts_from_file",
]
