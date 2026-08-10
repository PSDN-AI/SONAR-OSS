"""Plot generators for evaluation reports."""

from .hard_negatives import generate_hard_negatives_comparison
from .latency import generate_latency_plots
from .lexical_diversity import plot_ngram_diversity_comparison, plot_vocabulary_growth, plot_zipf_law

__all__ = [
    "generate_hard_negatives_comparison",
    "generate_latency_plots",
    "plot_ngram_diversity_comparison",
    "plot_vocabulary_growth",
    "plot_zipf_law",
]
