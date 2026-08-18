"""Hard-negatives comparison plots: user dataset vs. public benchmarks."""

import logging
from pathlib import Path

import pandas as pd
import plotnine as p9
from plotnine import aes, geom_col, geom_errorbar, ggplot, labs, position_dodge, scale_fill_manual

from psdn_sonar.utils.plot_theme import save_plot, theme_swarm_lab

logger = logging.getLogger(__name__)

BENGALI_BENCHMARK_STATS = {
    "wer": {
        "Common Voice": {"overall": 0.35, "hard": 1.05},
        "FLEURS": {"overall": 0.42, "hard": 1.20},
        "OpenSLR37 BD": {"overall": 0.38, "hard": 1.10},
        "OpenSLR37 IN": {"overall": 0.36, "hard": 1.08},
        "OpenSLR53": {"overall": 0.45, "hard": 1.30},
    },
    "cer": {
        "Common Voice": {"overall": 0.15, "hard": 0.45},
        "FLEURS": {"overall": 0.18, "hard": 0.52},
        "OpenSLR37 BD": {"overall": 0.16, "hard": 0.48},
        "OpenSLR37 IN": {"overall": 0.15, "hard": 0.46},
        "OpenSLR53": {"overall": 0.20, "hard": 0.58},
    },
}

ENGLISH_BENCHMARK_STATS = {
    "wer": {
        "Common Voice": {"overall": 0.357, "hard": 1.2142},
        "FLEURS": {"overall": 0.2271, "hard": 0.8034},
    },
    "cer": {
        "Common Voice": {"overall": 0.3133, "hard": 1.1256},
        "FLEURS": {"overall": 0.1929, "hard": 0.7113},
    },
}

# Placeholder Zeroth values; run precompute or single-speaker on Zeroth TSV to get real numbers.
KOREAN_BENCHMARK_STATS = {
    "wer": {
        "Common Voice": {"overall": 0.40, "hard": 1.15},
        "FLEURS": {"overall": 0.45, "hard": 1.25},
        "Zeroth": {"overall": 0.38, "hard": 1.10},
    },
    "cer": {
        "Common Voice": {"overall": 0.18, "hard": 0.50},
        "FLEURS": {"overall": 0.20, "hard": 0.55},
        "Zeroth": {"overall": 0.17, "hard": 0.48},
    },
}

HINDI_BENCHMARK_STATS = {
    "wer": {
        "Common Voice": {"overall": 0.38, "hard": 1.10},
        "FLEURS": {"overall": 0.35, "hard": 1.05},
    },
    "cer": {
        "Common Voice": {"overall": 0.16, "hard": 0.48},
        "FLEURS": {"overall": 0.14, "hard": 0.42},
    },
}

_BENCHMARK_STATS = {
    "english": ENGLISH_BENCHMARK_STATS,
    "en": ENGLISH_BENCHMARK_STATS,
    "korean": KOREAN_BENCHMARK_STATS,
    "ko": KOREAN_BENCHMARK_STATS,
    "hindi": HINDI_BENCHMARK_STATS,
    "hi": HINDI_BENCHMARK_STATS,
}


def get_benchmark_stats(language: str = "bengali") -> dict:
    """Get language-specific public benchmark stats."""
    return _BENCHMARK_STATS.get(language.lower(), BENGALI_BENCHMARK_STATS)


def _calculate_user_stats(results_csv: str, percentile_threshold: float = 0.75) -> dict:
    """Overall and hard-negative mean/std WER and CER from a results CSV.

    Unlike :func:`psdn_sonar.reporting.metrics.calculate_hard_negatives`, this
    treats each metric independently (a row can be a WER hard negative without
    being a CER one) and includes standard deviations for error bars. Metrics
    whose column is absent are simply omitted from the result.
    """
    df = pd.read_csv(results_csv)

    wer_col = None
    cer_col = None

    for col in ["wer_conv", "wer_non", "wer", "WER"]:
        if col in df.columns:
            wer_col = col
            break

    for col in ["cer_conv", "cer_non", "cer", "CER"]:
        if col in df.columns:
            cer_col = col
            break

    stats = {}

    if wer_col:
        wer_values = df[wer_col].dropna()
        wer_threshold = wer_values.quantile(percentile_threshold)
        wer_hard = wer_values[wer_values > wer_threshold]

        stats["wer"] = {
            "overall": wer_values.mean(),
            "overall_std": wer_values.std(),
            "hard": wer_hard.mean(),
            "hard_std": wer_hard.std(),
        }

    if cer_col:
        cer_values = df[cer_col].dropna()
        cer_threshold = cer_values.quantile(percentile_threshold)
        cer_hard = cer_values[cer_values > cer_threshold]

        stats["cer"] = {
            "overall": cer_values.mean(),
            "overall_std": cer_values.std(),
            "hard": cer_hard.mean(),
            "hard_std": cer_hard.std(),
        }

    return stats


def prepare_comparison_data(user_stats: dict, metric: str, language: str = "bengali") -> pd.DataFrame:
    """Combine user stats and public benchmarks into a plot-ready DataFrame."""
    rows = []

    if metric in user_stats:
        rows.append(
            {
                "dataset": "Your dataset",
                "condition": "Overall",
                "mean": user_stats[metric]["overall"],
                "std": user_stats[metric]["overall_std"],
            }
        )
        rows.append(
            {
                "dataset": "Your dataset",
                "condition": "Hard Negatives",
                "mean": user_stats[metric]["hard"],
                "std": user_stats[metric]["hard_std"],
            }
        )

    benchmark_stats = get_benchmark_stats(language)
    if metric in benchmark_stats:
        for dataset_name, values in benchmark_stats[metric].items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "condition": "Overall",
                    "mean": values["overall"],
                    "std": 0.08,
                }
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "condition": "Hard Negatives",
                    "mean": values["hard"],
                    "std": 0.20,
                }
            )

    return pd.DataFrame(rows)


def create_comparison_plot(df: pd.DataFrame, metric_name: str, output_path: str) -> None:
    """Dodged bar plot of overall vs. hard-negative means, user dataset first."""
    present = df["dataset"].unique().tolist()
    dataset_order = ["Your dataset"] + [d for d in present if d != "Your dataset"]
    df["dataset"] = pd.Categorical(df["dataset"], categories=dataset_order, ordered=True)

    df["ymin"] = (df["mean"] - df["std"]).clip(lower=0)
    df["ymax"] = df["mean"] + df["std"]

    condition_colors = {"Overall": "#3498DB", "Hard Negatives": "#E74C3C"}

    plot = (
        ggplot(df, aes(x="dataset", y="mean", fill="condition"))
        + geom_col(position=position_dodge(width=0.85), width=0.7, alpha=0.85)
        + geom_errorbar(
            aes(ymin="ymin", ymax="ymax"), position=position_dodge(width=0.85), width=0.25, size=0.5, color="black"
        )
        + scale_fill_manual(values=condition_colors, name="Condition")
        + labs(
            title=f"{metric_name}: Your dataset vs Public Benchmarks",
            x="Dataset",
            y=metric_name,
        )
        + theme_swarm_lab(figure_size=(14, 8))
        + p9.theme(
            plot_title=p9.element_text(size=16, weight="bold", ha="center"),
            axis_title_x=p9.element_text(size=14, weight="bold"),
            axis_title_y=p9.element_text(size=14, weight="bold"),
            axis_text_x=p9.element_text(rotation=45, ha="right", size=11),
            axis_text_y=p9.element_text(size=12),
            legend_position="top",
            legend_text=p9.element_text(size=11),
            legend_title=p9.element_text(size=12, face="bold"),
        )
        + p9.scale_y_continuous(limits=(0, 2.0), expand=(0, 0, 0.05, 0), labels=lambda vals: [f"{x:.2f}" for x in vals])
    )

    save_plot(plot, output_path, dpi=300, width=14, height=8)
    logger.info("  ✓ Saved: %s", output_path)


def generate_hard_negatives_comparison(results_csv: str, output_dir: str, language: str = "bengali") -> None:
    """Generate WER and CER hard-negatives comparison plots for a results CSV."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Hard Negatives Analysis: Your dataset vs Public Benchmarks (%s)", language.title())
    logger.info("Analyzing user dataset: %s", results_csv)
    user_stats = _calculate_user_stats(results_csv)

    if not user_stats:
        logger.warning("  No WER/CER data found in results CSV")
        return

    if not get_benchmark_stats(language):
        logger.warning("  No public benchmarks for '%s' — showing user data only", language)

    if "wer" in user_stats:
        wer_df = prepare_comparison_data(user_stats, "wer", language=language)
        create_comparison_plot(wer_df, "Word Error Rate (WER)", str(out_dir / "wer_overall_vs_hard_negatives.png"))

    if "cer" in user_stats:
        cer_df = prepare_comparison_data(user_stats, "cer", language=language)
        create_comparison_plot(cer_df, "Character Error Rate (CER)", str(out_dir / "cer_overall_vs_hard_negatives.png"))

    logger.info("Hard negatives comparison complete. Output: %s", out_dir)
