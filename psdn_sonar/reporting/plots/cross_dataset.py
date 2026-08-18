"""Cross-dataset and cross-model comparison boxplots.

Combines a user's evaluation results with precomputed public-benchmark
evaluations (shipped under ``psdn_sonar/benchmarks/<language>/raw-evaluations/``
when present) and renders per-metric boxplots. Model names are discovered
dynamically; nothing is hardcoded to a specific model registry.
"""

import logging
import os
from pathlib import Path

import pandas as pd
import plotnine as p9
from plotnine import aes, geom_boxplot, ggplot, labs, position_dodge, scale_y_continuous

from psdn_sonar.utils.metrics import _POSEIDON_COLUMN_LAYOUTS, ensure_poseidon_score
from psdn_sonar.utils.plot_theme import get_swarm_colors, save_plot, theme_swarm_lab

from ._common import prettify_model_name

logger = logging.getLogger(__name__)

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

# User dataset first, then public benchmarks.
_DATASET_ORDER = [
    "user_dataset",
    "librispeech",
    "commonvoice",
    "fleurs",
    "zeroth",
    "openslr37_bd",
    "openslr37_in",
    "openslr53",
]

_LANGUAGE_ALIASES = {
    "bn": "bengali",
    "ko": "korean",
    "hi": "hindi",
    "en": "english",
}

_METRIC_CONFIG = [
    ("cer", "Character Error Rate (CER)", True),
    ("wer", "Word Error Rate (WER)", True),
    ("sem", "Semantic Similarity", False),
    ("poseidon", "Poseidon Score", False),
]


def _find_metric_columns(df: pd.DataFrame):
    """Return the first (cer, wer, sem) column triple present, or (None, None, None)."""
    for cer_col, wer_col, sem_col in _POSEIDON_COLUMN_LAYOUTS:
        if cer_col in df.columns and wer_col in df.columns:
            return cer_col, wer_col, (sem_col if sem_col in df.columns else None)
    return None, None, None


def _shape_rows(df: pd.DataFrame, dataset: str, model: str) -> list:
    """Convert a results DataFrame to plot rows, skipping rows with NaN CER/WER."""
    cer_col, wer_col, sem_col = _find_metric_columns(df)
    if cer_col is None:
        return []

    rows = []
    for _, row in df.iterrows():
        cer_val = row.get(cer_col)
        wer_val = row.get(wer_col)
        if pd.isna(cer_val) or pd.isna(wer_val):
            continue
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "cer": cer_val,
                "wer": wer_val,
                "sem": row.get(sem_col) if sem_col else None,
                "poseidon": row.get("poseidon_score"),
            }
        )
    return rows


def load_public_benchmark_data(language: str = "bengali") -> list:
    """Load precomputed public-benchmark evaluations for *language*.

    Scans ``benchmarks/<language>/raw-evaluations/<model>/<dataset>.csv``;
    every subdirectory is treated as a model and the dataset is parsed from
    the CSV filename. Returns [] when no benchmark data is shipped.
    """
    lang = _LANGUAGE_ALIASES.get(language.lower(), language.lower())
    benchmark_dir = Path(__file__).parent.parent.parent / "benchmarks" / lang / "raw-evaluations"
    if not benchmark_dir.is_dir():
        logger.info("No public benchmark evaluations for '%s' — showing user data only", language)
        return []

    all_rows = []
    for model_path in sorted(p for p in benchmark_dir.iterdir() if p.is_dir()):
        for csv_file in sorted(model_path.glob("*.csv")):
            dataset_key = next((key for key in DATASET_DISPLAY if key in csv_file.stem), None)
            if dataset_key is None:
                continue
            try:
                df = ensure_poseidon_score(pd.read_csv(csv_file))
            except Exception as e:
                logger.warning("Skipping %s: %s", csv_file, e)
                continue
            all_rows.extend(_shape_rows(df, dataset_key, model_path.name))

    return all_rows


def load_user_dataset_results(results_csv: str, model_name: str) -> list:
    """Load the user's evaluation results CSV as plot rows."""
    try:
        df = ensure_poseidon_score(pd.read_csv(results_csv))
    except Exception as e:
        logger.warning("Failed to load user results: %s", e)
        return []

    rows = _shape_rows(df, "user_dataset", model_name)
    if not rows:
        if _find_metric_columns(df)[0] is None:
            logger.warning("No CER/WER columns found in user dataset")
        else:
            logger.error("All user dataset samples have NaN values! Check if model evaluation succeeded.")
    elif len(rows) < len(df):
        logger.warning("Skipped %s/%s user dataset rows with NaN CER/WER values", len(df) - len(rows), len(df))
    return rows


def _model_labels(df: pd.DataFrame) -> tuple:
    """Ordered display labels and a label->color map for the models present."""
    labels = []
    for model in sorted(df["model"].unique()):
        label = prettify_model_name(model)
        if label not in labels:
            labels.append(label)
    colors = dict(zip(labels, get_swarm_colors(len(labels))))
    return labels, colors


def _apply_metric_y_scale(plot, values: pd.Series, lower_better: bool):
    """Y scale: IQR-capped for error rates, fixed 0-1 band for scores."""
    if lower_better:
        q75 = values.quantile(0.75)
        y_max = min(1.5, q75 + 1.5 * (q75 - values.quantile(0.25)))
        return plot + scale_y_continuous(
            limits=(-0.03, y_max), expand=(0, 0), labels=lambda vals: [f"{x:.2f}" if x >= 0 else "" for x in vals]
        )
    return plot + scale_y_continuous(
        limits=(-0.05, 1.15), expand=(0, 0), labels=lambda vals: [f"{x:.2f}" for x in vals]
    )


def create_cross_dataset_boxplots(df: pd.DataFrame, output_dir: str, language: str = "bengali") -> None:
    """One boxplot per metric: X=dataset, Y=value, fill=model (dodged)."""
    datasets_present = [d for d in _DATASET_ORDER if d in df["dataset"].unique()]
    model_labels, model_colors = _model_labels(df)
    fig_width = max(20, len(datasets_present) * 3)

    for metric_key, ylabel, lower_better in _METRIC_CONFIG:
        df_metric = df[df[metric_key].notna()].copy()
        if df_metric.empty:
            logger.warning("No data for %s", metric_key)
            continue

        df_metric["dataset_label"] = pd.Categorical(
            df_metric["dataset"].map(DATASET_DISPLAY).fillna(df_metric["dataset"]),
            categories=[DATASET_DISPLAY.get(d, d) for d in datasets_present],
            ordered=True,
        )
        df_metric["model_label"] = pd.Categorical(
            df_metric["model"].map(prettify_model_name),
            categories=model_labels,
            ordered=True,
        )
        df_metric["value"] = df_metric[metric_key]

        plot = (
            ggplot(df_metric, aes(x="dataset_label", y="value", fill="model_label"))
            + geom_boxplot(position=position_dodge(width=0.85), width=0.7, alpha=0.85, outlier_alpha=0.3)
            + p9.scale_fill_manual(values=model_colors, name="Model")
            + labs(
                title=f"{ylabel} Distribution Across Datasets and Models",
                x="Datasets",
                y=ylabel,
                fill="Model",
            )
            + theme_swarm_lab(figure_size=(fig_width, 10))
            + p9.theme(
                plot_title=p9.element_text(size=16, weight="bold", ha="center"),
                axis_title_x=p9.element_text(size=14, weight="bold"),
                axis_title_y=p9.element_text(size=14, weight="bold"),
                axis_text_x=p9.element_text(rotation=0, ha="center", size=12),
                axis_text_y=p9.element_text(size=12),
                legend_position="right",
                legend_text=p9.element_text(size=11),
                legend_title=p9.element_text(size=12, face="bold"),
            )
        )
        plot = _apply_metric_y_scale(plot, df_metric["value"], lower_better)

        out_path = os.path.join(output_dir, f"{metric_key}_by_dataset_model.png")
        save_plot(plot, out_path, dpi=300, width=fig_width, height=10)
        logger.info("  ✓ Saved: %s", out_path)


def generate_cross_dataset_plots(results_csv: str, model_name: str, output_dir: str, language: str = "bengali") -> None:
    """Generate cross-dataset comparison plots: user dataset vs public benchmarks.

    Parameters
    ----------
    results_csv : path to the user's evaluation results CSV
    model_name  : model name used for the user evaluation
    output_dir  : directory to save plots
    language    : language for selecting shipped benchmark evaluations
    """
    os.makedirs(output_dir, exist_ok=True)

    benchmark_rows = load_public_benchmark_data(language=language)
    logger.info("Loaded %s samples from public benchmarks", len(benchmark_rows))

    user_rows = load_user_dataset_results(results_csv, model_name)
    logger.info("Loaded %s valid samples from user dataset", len(user_rows))
    if not user_rows and results_csv:
        logger.warning("User dataset has no valid predictions — cross-dataset plots will show benchmarks only.")

    all_rows = benchmark_rows + user_rows
    if not all_rows:
        logger.warning("No data found — skipping cross-dataset plots.")
        return

    df = pd.DataFrame(all_rows)
    logger.info("Total samples: %s across %s datasets", len(df), df["dataset"].nunique())
    create_cross_dataset_boxplots(df, output_dir, language=language)
    logger.info("Cross-dataset plots saved to %s", output_dir)


def generate_model_comparison_plots(evaluated_models, output_dir: str, language: str = "custom") -> None:
    """Generate boxplots comparing multiple models on a single user dataset.

    Parameters
    ----------
    evaluated_models : list of (model_name, results_csv_path) tuples
    output_dir       : directory to save plots
    language         : language name (used in plot titles)
    """
    os.makedirs(output_dir, exist_ok=True)

    all_rows = []
    for model_name, csv_path in evaluated_models:
        if not Path(csv_path).exists():
            logger.warning("%s not found, skipping", csv_path)
            continue
        rows = load_user_dataset_results(csv_path, model_name)
        logger.info("Loaded %s samples for %s", len(rows), model_name)
        all_rows.extend(rows)

    if not all_rows:
        logger.warning("No data to plot.")
        return

    df = pd.DataFrame(all_rows)
    model_labels, model_colors = _model_labels(df)
    fig_width = max(12, len(model_labels) * 3)

    for metric_key, ylabel, lower_better in _METRIC_CONFIG:
        df_m = df[df[metric_key].notna()].copy()
        if df_m.empty:
            logger.warning("No data for %s", metric_key)
            continue

        df_m["model_label"] = pd.Categorical(
            df_m["model"].map(prettify_model_name),
            categories=model_labels,
            ordered=True,
        )
        df_m["value"] = df_m[metric_key]

        plot = (
            ggplot(df_m, aes(x="model_label", y="value", fill="model_label"))
            + geom_boxplot(width=0.6, alpha=0.85, outlier_alpha=0.3)
            + p9.scale_fill_manual(values=model_colors, name="Model")
            + labs(
                title=f"{ylabel} — Model Comparison ({language.title()})",
                x="Model",
                y=ylabel,
            )
            + theme_swarm_lab(figure_size=(fig_width, 8))
            + p9.theme(
                plot_title=p9.element_text(size=16, weight="bold", ha="center"),
                axis_title_x=p9.element_text(size=14, weight="bold"),
                axis_title_y=p9.element_text(size=14, weight="bold"),
                axis_text_x=p9.element_text(rotation=0, ha="center", size=11),
                axis_text_y=p9.element_text(size=12),
                legend_position="none",
            )
        )
        plot = _apply_metric_y_scale(plot, df_m["value"], lower_better)

        out_path = os.path.join(output_dir, f"{metric_key}_model_comparison.png")
        save_plot(plot, out_path, dpi=300, width=fig_width, height=8)
        logger.info("  ✓ Saved: %s", out_path)

    logger.info("Model comparison plots saved to %s", output_dir)
