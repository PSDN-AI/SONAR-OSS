"""Lexical diversity visualisations: n-gram diversity, vocabulary growth, Zipf's law."""

import logging
from collections import Counter

import pandas as pd
import plotnine as p9
from plotnine import (
    aes,
    geom_col,
    geom_errorbar,
    geom_hline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_x_log10,
    scale_y_log10,
)

from psdn_sonar.utils.plot_theme import get_swarm_colors, save_plot, theme_swarm_lab

from ..loaders.benchmark_loader import load_public_benchmark_diversity, load_public_lexical_data

logger = logging.getLogger(__name__)

_DATASET_COLORS = get_swarm_colors(10)

BENGALI_DATASET_ORDER = ["Common Voice", "FLEURS", "OpenSLR37 BD", "OpenSLR37 IN", "OpenSLR53"]
ENGLISH_DATASET_ORDER = ["Common Voice", "FLEURS", "LibriSpeech"]
KOREAN_DATASET_ORDER = ["Common Voice", "FLEURS"]
HINDI_DATASET_ORDER = ["Common Voice", "FLEURS"]

_DATASET_ORDERS = {
    "english": ENGLISH_DATASET_ORDER,
    "en": ENGLISH_DATASET_ORDER,
    "korean": KOREAN_DATASET_ORDER,
    "ko": KOREAN_DATASET_ORDER,
    "hindi": HINDI_DATASET_ORDER,
    "hi": HINDI_DATASET_ORDER,
}


def get_dataset_order(language: str = "bengali"):
    """Return the preferred dataset display order for *language*."""
    return _DATASET_ORDERS.get(language.lower(), BENGALI_DATASET_ORDER)


def sorted_dataset_keys(keys, language="bengali"):
    """Sort dataset names by the language's display order, unknown names last."""
    dataset_order = get_dataset_order(language)
    order = {k: i for i, k in enumerate(dataset_order)}
    return sorted(keys, key=lambda k: (order.get(k, 999), k))


def plot_ngram_diversity_comparison(results, output_path, include_benchmarks=True, language="bengali"):
    """Grouped bar chart of unigram/bigram/trigram diversity per dataset.

    *results* maps dataset name to the dict returned by
    :func:`psdn_sonar.reporting.metrics.calculate_lexical_diversity_metrics`.
    Public-benchmark stats are overlaid when available.
    """
    if include_benchmarks:
        public_benchmarks = load_public_benchmark_diversity(language=language)
        combined_results = {**public_benchmarks, **results}
        has_benchmarks = bool(public_benchmarks)
    else:
        combined_results = results
        has_benchmarks = False

    datasets = sorted_dataset_keys(list(combined_results.keys()), language=language)

    plot_data = []
    for dataset in datasets:
        for metric in ["unigram", "bigram", "trigram"]:
            diversity_val = combined_results[dataset].get(f"{metric}_diversity", 0)
            diversity_std = combined_results[dataset].get(f"{metric}_diversity_std", 0)
            diversity_pct = diversity_val * 100
            diversity_std_pct = diversity_std * 100
            plot_data.append(
                {
                    "Dataset": dataset,
                    "Metric": metric.capitalize(),
                    "Diversity": diversity_pct,
                    "ymin": max(0, diversity_pct - diversity_std_pct),
                    "ymax": min(100, diversity_pct + diversity_std_pct),
                }
            )

    df = pd.DataFrame(plot_data)

    dataset_colors = {ds: _DATASET_COLORS[i % len(_DATASET_COLORS)] for i, ds in enumerate(datasets)}
    has_errorbars = (df["ymax"] - df["ymin"]).abs().sum() > 0

    plot = (
        ggplot(df, aes(x="Metric", y="Diversity", fill="Dataset"))
        + geom_col(position=position_dodge(0.9), alpha=0.85, color="black", size=0.5)
        + geom_hline(yintercept=30, linetype="dashed", color="#FEB308", size=1.5, alpha=0.7)
        + geom_hline(yintercept=70, linetype="dashed", color="#39AD48", size=1.5, alpha=0.7)
        + scale_fill_manual(values=dataset_colors)
        + labs(
            title="N-gram Diversity Comparison (Chunked Avg)" + (" vs Public Benchmarks" if has_benchmarks else ""),
            x="N-gram Type",
            y="Diversity (%)",
            fill="Dataset",
        )
        + theme_swarm_lab(figure_size=(14, 8))
        + p9.theme(
            plot_title=p9.element_text(size=14, weight="bold", ha="center"),
            axis_title_x=p9.element_text(size=12, weight="bold"),
            axis_title_y=p9.element_text(size=12, weight="bold"),
            axis_text_x=p9.element_text(size=11),
            axis_text_y=p9.element_text(size=11),
            legend_text=p9.element_text(size=10),
            legend_title=p9.element_text(size=11, face="bold"),
        )
    )

    if has_errorbars:
        plot = plot + geom_errorbar(
            aes(ymin="ymin", ymax="ymax"),
            position=position_dodge(0.9),
            width=0.25,
            size=0.6,
        )

    save_plot(plot, output_path, dpi=600, width=14, height=8)
    logger.info("  ✓ Saved plot: %s", output_path)


def plot_vocabulary_growth(
    all_transcripts, output_path, title_prefix="", include_public_benchmarks=True, language="bengali"
):
    """Vocabulary growth curve (unique words vs. total tokens) per dataset.

    *all_transcripts* maps dataset name to a list of transcript strings.
    Precomputed public-benchmark curves are overlaid when available.
    """
    datasets = sorted_dataset_keys(list(all_transcripts.keys()), language=language)
    if not datasets:
        return

    growth_data = []
    for idx, dataset_name in enumerate(datasets):
        texts = all_transcripts[dataset_name]
        all_words = []
        vocab_sizes = []
        seen_vocab = set()
        for t in texts:
            for w in t.split():
                all_words.append(w)
                seen_vocab.add(w)
                vocab_sizes.append(len(seen_vocab))
        if len(all_words) == 0:
            continue
        token_counts = list(range(1, len(all_words) + 1))

        sample_rate = max(1, len(token_counts) // 1000)
        for tc, vs in zip(token_counts[::sample_rate], vocab_sizes[::sample_rate]):
            if tc <= 1e6:
                growth_data.append({"Dataset": dataset_name, "Total_Tokens": tc, "Vocab_Size": vs})

    has_benchmarks = False
    if include_public_benchmarks:
        public_data = load_public_lexical_data(language=language)
        if public_data:
            has_benchmarks = True
            for dataset_name, data in public_data.items():
                for point in data["vocabulary_growth"]:
                    token_val = point.get("tokens", point.get("token_count"))
                    growth_data.append(
                        {"Dataset": dataset_name, "Total_Tokens": token_val, "Vocab_Size": point["vocab_size"]}
                    )

    df_growth = pd.DataFrame(growth_data)

    all_datasets = df_growth["Dataset"].unique()
    dataset_colors = {ds: _DATASET_COLORS[i % len(_DATASET_COLORS)] for i, ds in enumerate(sorted(all_datasets))}

    title = (
        "Vocabulary Growth Curve: User Dataset vs Public Benchmarks" if has_benchmarks else "Vocabulary Growth Curve"
    )
    if title_prefix and not include_public_benchmarks:
        title = f"{title} ({title_prefix})"

    plot = (
        ggplot(df_growth, aes(x="Total_Tokens", y="Vocab_Size", color="Dataset"))
        + geom_line(size=2, alpha=0.8)
        + scale_color_manual(values=dataset_colors)
        + labs(title=title, x="Total Tokens", y="Unique Words (Vocabulary Size)")
        + theme_swarm_lab(figure_size=(14, 8))
        + p9.theme(
            plot_title=p9.element_text(size=14, weight="bold", ha="center"),
            legend_text=p9.element_text(size=10),
        )
    )

    save_plot(plot, output_path, dpi=600, width=14, height=8)
    logger.info("Saved: %s", output_path)


def plot_zipf_law(all_transcripts, output_path, title_prefix="", include_public_benchmarks=True, language="bengali"):
    """Log-log word frequency vs. rank scatter per dataset.

    *all_transcripts* maps dataset name to a list of transcript strings.
    Precomputed public-benchmark points are overlaid when available.
    """
    datasets = sorted_dataset_keys(list(all_transcripts.keys()), language=language)
    if not datasets:
        return

    zipf_data = []
    for idx, dataset_name in enumerate(datasets):
        texts = all_transcripts[dataset_name]
        all_words = []
        for t in texts:
            all_words.extend(t.split())
        if len(all_words) == 0:
            continue
        word_counts = Counter(all_words)
        frequencies = sorted(word_counts.values(), reverse=True)
        ranks = list(range(1, len(frequencies) + 1))

        sample_rate = max(1, len(ranks) // 1000)
        for rank, freq in zip(ranks[::sample_rate], frequencies[::sample_rate]):
            zipf_data.append({"Dataset": dataset_name, "Rank": rank, "Frequency": freq, "Type": "Observed"})

    has_benchmarks = False
    if include_public_benchmarks:
        public_data = load_public_lexical_data(language=language)
        if public_data:
            has_benchmarks = True
            for dataset_name, data in public_data.items():
                for point in data["zipf_law"]:
                    zipf_data.append(
                        {
                            "Dataset": dataset_name,
                            "Rank": point["rank"],
                            "Frequency": point["frequency"],
                            "Type": "Observed",
                        }
                    )

    df_zipf = pd.DataFrame(zipf_data)

    all_datasets = df_zipf["Dataset"].unique()
    dataset_colors = {ds: _DATASET_COLORS[i % len(_DATASET_COLORS)] for i, ds in enumerate(sorted(all_datasets))}

    title = (
        "Zipf's Law: User Dataset vs Public Benchmarks" if has_benchmarks else "Zipf's Law: Word Frequency Distribution"
    )

    plot_zipf = (
        ggplot(df_zipf, aes(x="Rank", y="Frequency", color="Dataset"))
        + geom_point(size=1.5, alpha=0.6)
        + scale_x_log10()
        + scale_y_log10()
        + scale_color_manual(values=dataset_colors)
        + labs(title=title, x="Rank (log scale)", y="Frequency (log scale)")
        + theme_swarm_lab(figure_size=(14, 8))
        + p9.theme(
            plot_title=p9.element_text(size=14, weight="bold", ha="center"),
            legend_text=p9.element_text(size=10),
        )
    )

    save_plot(plot_zipf, output_path, dpi=600, width=14, height=8)
    logger.info("Saved: %s", output_path)
