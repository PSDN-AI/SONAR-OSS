"""Demographic performance plots: per-model metrics by gender, age group, and region.

Joins multi-speaker evaluation results with speaker metadata
(``<dataset_dir>/<audio_id>/metadata.json``) and renders boxplots comparing
models across demographic groups, plus a summary statistics CSV.
"""

import glob
import json
import logging
import os

import pandas as pd
import plotnine as p9
from plotnine import aes, geom_boxplot, ggplot, labs, position_dodge, scale_y_continuous

from psdn_sonar.utils.metrics import ensure_poseidon_score
from psdn_sonar.utils.plot_theme import get_swarm_colors, save_plot, theme_swarm_lab

from ._common import prettify_model_name

logger = logging.getLogger(__name__)

METRICS = {
    "cer_conv": ("Character Error Rate (CER)", True),
    "wer_conv": ("Word Error Rate (WER)", True),
    "semantic_similarity_conv": ("Semantic Similarity", False),
    "poseidon_score": ("Poseidon Score", False),
}

DEMOGRAPHICS = {
    "gender": "Gender",
    "age_group": "Age Group",
    "region": "Region",
}

AGE_BINS = [0, 20, 23, 26, 100]
AGE_LABELS = ["≤20", "21-23", "24-26", ">26"]

# Preferred category orders; groups absent from the data are dropped and
# demographics without an entry (e.g. region) fall back to sorted order.
_GROUP_ORDERS = {
    "age_group": AGE_LABELS,
    "gender": ["Female", "Male"],
}

_RESULT_PATTERNS = ("results_*_manifest.csv", "asr_eval_results_*_manifest.csv")


def _model_name_from_filename(basename: str) -> str:
    if basename.startswith("asr_eval_results_"):
        return basename.replace("asr_eval_results_", "").replace("_manifest.csv", "")
    return basename.replace("results_", "").replace("_manifest.csv", "")


def _speaker_metadata(dataset_dir: str, audio_id: str, speaker: str) -> dict:
    metadata_path = os.path.join(dataset_dir, str(audio_id), "metadata.json")
    if not os.path.isfile(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    speaker_key = "speaker_a" if speaker == "A" else "speaker_b"
    speaker_meta = metadata.get(speaker_key, {})
    return {
        "age": speaker_meta.get("age"),
        "gender": speaker_meta.get("gender"),
        "region": speaker_meta.get("region"),
    }


def load_all_multispeaker_with_metadata(results_dir: str, dataset_dir: str) -> pd.DataFrame:
    """Load every per-model results CSV and join speaker metadata (age, gender, region)."""
    csv_list = []
    for pattern in _RESULT_PATTERNS:
        csv_list.extend(sorted(glob.glob(os.path.join(results_dir, pattern))))

    all_rows = []
    for path in csv_list:
        basename = os.path.basename(path)
        model_name = _model_name_from_filename(basename)
        try:
            df = pd.read_csv(path)
            df = ensure_poseidon_score(df)

            metadata_list = [
                _speaker_metadata(dataset_dir, row["audio_id"], row["speaker"]) for _, row in df.iterrows()
            ]
            df_with_meta = pd.concat([df, pd.DataFrame(metadata_list)], axis=1)
            df_with_meta["model"] = model_name
            df_with_meta["model_label"] = prettify_model_name(model_name)
            all_rows.append(df_with_meta)
            logger.info("  Loaded %d samples for %s", len(df), model_name)
        except Exception as e:
            logger.warning("  Warning: Skipping %s: %s", basename, e)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)


def create_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bin ages into groups for clearer visualization."""
    df = df.copy()
    df["age_group"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, include_lowest=True)
    return df


def _ordered_categories(values: pd.Series, demographic: str) -> list:
    present = values.unique()
    preferred = _GROUP_ORDERS.get(demographic)
    if preferred:
        return [cat for cat in preferred if cat in present]
    return sorted(present)


def create_demographic_comparison_boxplot(
    df: pd.DataFrame, demographic: str, metric: str, metric_label: str, lower_better: bool, output_path: str
) -> None:
    """Dodged boxplot of *metric* by demographic group, one box per model.

    For lower-is-better metrics, extreme outliers beyond Q3 + 1.5*IQR
    (capped at 1.5) are dropped so boxes stay readable.
    """
    df_clean = df.dropna(subset=[demographic, metric, "model"]).copy()
    if df_clean.empty:
        logger.warning("  Warning: No data for %s vs %s", demographic, metric)
        return

    if lower_better:
        q75 = df_clean[metric].quantile(0.75)
        q25 = df_clean[metric].quantile(0.25)
        reasonable_max = min(1.5, q75 + 1.5 * (q75 - q25))
        n_clipped = len(df_clean[df_clean[metric] > reasonable_max])
        if n_clipped > 0:
            logger.info("    Note: %d extreme outliers (>%.2f) clipped for %s", n_clipped, reasonable_max, metric)
        df_clean = df_clean[df_clean[metric] <= reasonable_max]

    df_clean["demographic_str"] = df_clean[demographic].astype(str)
    df_clean["demographic_str"] = pd.Categorical(
        df_clean["demographic_str"],
        categories=_ordered_categories(df_clean["demographic_str"], demographic),
        ordered=True,
    )

    model_labels_ordered = sorted(df_clean["model_label"].unique())
    df_clean["model_label"] = pd.Categorical(df_clean["model_label"], categories=model_labels_ordered, ordered=True)
    model_colors = dict(zip(model_labels_ordered, get_swarm_colors(len(model_labels_ordered))))

    n_groups = df_clean["demographic_str"].nunique()
    n_models = len(model_labels_ordered)
    fig_width = max(14, n_groups * n_models * 0.8)

    plot = (
        ggplot(df_clean, aes(x="demographic_str", y=metric, fill="model_label"))
        + geom_boxplot(position=position_dodge(width=0.85), width=0.7, alpha=0.85, outlier_alpha=0.3)
        + p9.scale_fill_manual(values=model_colors, name="Model")
        + labs(
            title=f"{metric_label} Distribution by {DEMOGRAPHICS[demographic]} Across Models",
            x=DEMOGRAPHICS[demographic],
            y=metric_label,
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

    if lower_better:
        y_max = df_clean[metric].max() * 1.1
        plot = plot + scale_y_continuous(
            limits=(-0.03, y_max), expand=(0, 0), labels=lambda vals: [f"{x:.2f}" if x >= 0 else "" for x in vals]
        )
    else:
        plot = plot + scale_y_continuous(
            limits=(-0.05, 1.15), expand=(0, 0), labels=lambda vals: [f"{x:.2f}" for x in vals]
        )

    save_plot(plot, output_path, dpi=300, width=fig_width, height=10)
    logger.info("  ✓ Saved: %s", output_path)


def create_summary_statistics(df: pd.DataFrame, output_dir: str) -> None:
    """Write a CSV of count/mean/median/std/quartiles by model and demographic group."""
    summaries = []

    for metric_key, (metric_label, _) in METRICS.items():
        if metric_key not in df.columns:
            continue

        for demographic in DEMOGRAPHICS:
            df_clean = df.dropna(subset=[demographic, metric_key, "model"]).copy()
            if df_clean.empty:
                continue

            grouped = (
                df_clean.groupby(["model", demographic], observed=True)[metric_key]
                .agg(
                    [
                        "count",
                        "mean",
                        "median",
                        "std",
                        ("q25", lambda x: x.quantile(0.25)),
                        ("q75", lambda x: x.quantile(0.75)),
                    ]
                )
                .reset_index()
            )

            grouped["metric"] = metric_label
            grouped["demographic"] = DEMOGRAPHICS[demographic]
            grouped["model_display"] = grouped["model"].map(prettify_model_name)
            grouped = grouped.rename(columns={demographic: "demographic_value"})

            summaries.append(grouped)

    if summaries:
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_df = summary_df[
            [
                "metric",
                "demographic",
                "demographic_value",
                "model_display",
                "count",
                "mean",
                "median",
                "std",
                "q25",
                "q75",
            ]
        ]
        summary_df = summary_df.sort_values(["metric", "demographic", "demographic_value", "model_display"])

        output_path = os.path.join(output_dir, "demographic_summary_statistics.csv")
        summary_df.to_csv(output_path, index=False, float_format="%.4f")
        logger.info("✓ Saved summary statistics: %s", output_path)


def generate_demographic_plots(results_dir: str, dataset_dir: str, output_dir: str) -> None:
    """Generate demographic analysis plots and a summary statistics CSV.

    Parameters
    ----------
    results_dir : directory containing per-model ``results_*_manifest.csv`` files
    dataset_dir : directory containing ``<audio_id>/metadata.json`` speaker metadata
    output_dir  : directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading multispeaker data with metadata...")
    df = load_all_multispeaker_with_metadata(results_dir, dataset_dir)
    if df.empty:
        logger.warning("No demographic data found — skipping demographic plots.")
        return

    df = create_age_groups(df)
    logger.info("Loaded %d samples across %d models", len(df), df["model"].nunique())

    for metric_key, (metric_label, lower_better) in METRICS.items():
        if metric_key not in df.columns:
            continue
        for demographic in DEMOGRAPHICS:
            if demographic not in df.columns:
                continue
            demographic_prefix = "age" if demographic == "age_group" else demographic
            output_path = os.path.join(output_dir, f"{demographic_prefix}_{metric_key}.png")
            create_demographic_comparison_boxplot(df, demographic, metric_key, metric_label, lower_better, output_path)

    create_summary_statistics(df, output_dir)
    logger.info("Demographic analysis complete. Output: %s", output_dir)
