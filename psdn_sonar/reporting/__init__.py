"""Report generation: metrics, loaders, and plots for evaluation results."""

from .loaders import load_public_benchmark_diversity, load_transcripts_from_file
from .metrics import (
    calculate_gini_coefficient,
    calculate_hard_negatives,
    calculate_lexical_diversity_metrics,
    calculate_ngram_diversity,
)
from .plots import (
    generate_audio_quality_plots,
    generate_cross_dataset_plots,
    generate_demographic_plots,
    generate_hard_negatives_comparison,
    generate_latency_plots,
    generate_model_comparison_plots,
    plot_ngram_diversity_comparison,
    plot_vocabulary_growth,
    plot_zipf_law,
)

__all__ = [
    "calculate_gini_coefficient",
    "calculate_hard_negatives",
    "calculate_lexical_diversity_metrics",
    "calculate_ngram_diversity",
    "generate_audio_quality_plots",
    "generate_cross_dataset_plots",
    "generate_demographic_plots",
    "generate_hard_negatives_comparison",
    "generate_latency_plots",
    "generate_model_comparison_plots",
    "load_public_benchmark_diversity",
    "load_transcripts_from_file",
    "plot_ngram_diversity_comparison",
    "plot_vocabulary_growth",
    "plot_zipf_law",
]
