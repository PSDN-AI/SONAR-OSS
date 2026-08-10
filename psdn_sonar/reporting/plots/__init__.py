"""Plot generators for evaluation reports."""

from .audio_quality import generate_audio_quality_plots
from .demographic import generate_demographic_plots
from .hard_negatives import generate_hard_negatives_comparison
from .latency import generate_latency_plots
from .lexical_diversity import plot_ngram_diversity_comparison, plot_vocabulary_growth, plot_zipf_law

__all__ = [
    "generate_audio_quality_plots",
    "generate_demographic_plots",
    "generate_hard_negatives_comparison",
    "generate_latency_plots",
    "plot_ngram_diversity_comparison",
    "plot_vocabulary_growth",
    "plot_zipf_law",
]
