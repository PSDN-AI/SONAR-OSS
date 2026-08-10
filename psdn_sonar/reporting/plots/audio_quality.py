"""Audio quality visualisations: SNR/MOS scatter, distributions, and quality-tier plots."""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotnine as p9

from psdn_sonar.audio_quality import get_audio_quality_config
from psdn_sonar.utils.plot_theme import get_swarm_colors, save_plot, theme_swarm_lab

from ._common import load_and_tag_results

logger = logging.getLogger(__name__)

_TIER_ORDER = ["Low", "Medium", "High"]
_TIER_COLORS = {
    "Low": "#C84848",
    "Medium": "#D4A017",
    "High": "#2D8B57",
}


def plot_wer_by_snr_tier(df: pd.DataFrame, output_path: str) -> None:
    """Grouped boxplot of WER per SNR quality tier, one box per model."""
    subset = df.dropna(subset=["snr_tier", "wer"]).copy()
    subset = subset[subset["wer"] <= 1.0]
    if subset.empty:
        logger.warning("No data for WER-by-SNR-tier plot; skipping.")
        return

    subset["snr_tier"] = pd.Categorical(subset["snr_tier"], categories=_TIER_ORDER, ordered=True)
    models = sorted(subset["model"].unique())
    colors = get_swarm_colors(len(models))

    plot = (
        p9.ggplot(subset, p9.aes(x="snr_tier", y="wer", fill="model"))
        + p9.geom_boxplot(position=p9.position_dodge(width=0.8), width=0.7, outlier_size=1.5, outlier_alpha=0.4)
        + p9.scale_fill_manual(values=colors)
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(
            title="WER by SNR Quality Tier",
            x="SNR Quality Tier",
            y="Word Error Rate (WER)",
            fill="Model",
        )
        + theme_swarm_lab(figure_size=(12, 8))
        + p9.theme(axis_text_x=p9.element_text(rotation=0, ha="center"))
    )
    save_plot(plot, output_path, width=12, height=8)


def plot_quality_tier_composition(df: pd.DataFrame, output_path: str) -> None:
    """Stacked bar chart of each model's audio-quality tier percentages."""
    subset = df.dropna(subset=["snr_tier"]).copy()
    if subset.empty:
        logger.warning("No data for quality-tier composition plot; skipping.")
        return

    subset["snr_tier"] = pd.Categorical(subset["snr_tier"], categories=_TIER_ORDER, ordered=True)

    source_col = "model"
    counts = subset.groupby([source_col, "snr_tier"], observed=False).size().reset_index(name="count")
    totals = counts.groupby(source_col)["count"].transform("sum")
    counts["pct"] = counts["count"] / totals * 100

    plot = (
        p9.ggplot(counts, p9.aes(x=source_col, y="pct", fill="snr_tier"))
        + p9.geom_bar(stat="identity", position="stack", color="black", size=0.3, alpha=0.85)
        + p9.scale_fill_manual(values=[_TIER_COLORS[t] for t in _TIER_ORDER])
        + p9.labs(
            title="Audio Quality Tier Composition",
            x="Model / Dataset",
            y="% of Audio Files",
            fill="SNR Tier",
        )
        + theme_swarm_lab(figure_size=(12, 8))
        + p9.theme(axis_text_x=p9.element_text(rotation=0, ha="center", size=10))
    )
    save_plot(plot, output_path, width=12, height=8)


def plot_model_tier_heatmap(df: pd.DataFrame, output_path: str) -> None:
    """Model × SNR-tier heatmap of median WER."""
    subset = df.dropna(subset=["snr_tier", "wer"]).copy()
    subset = subset[subset["wer"] <= 1.0]
    if subset.empty:
        logger.warning("No data for model-tier heatmap; skipping.")
        return

    subset["snr_tier"] = pd.Categorical(subset["snr_tier"], categories=_TIER_ORDER, ordered=True)
    pivot = subset.groupby(["model", "snr_tier"], observed=False)["wer"].median().reset_index(name="median_wer")

    plot = (
        p9.ggplot(pivot, p9.aes(x="snr_tier", y="model", fill="median_wer"))
        + p9.geom_tile(color="white", size=1.5)
        + p9.geom_text(p9.aes(label="median_wer"), format_string="{:.3f}", size=12, color="black")
        + p9.scale_fill_gradient(low="#2D8B57", high="#C84848", name="Median WER")
        + p9.labs(
            title="Model × SNR Quality Tier — Median WER",
            x="SNR Quality Tier",
            y="Model",
        )
        + theme_swarm_lab(figure_size=(10, max(6, len(pivot["model"].unique()) * 1.2)))
        + p9.theme(axis_text_x=p9.element_text(rotation=0, ha="center"))
    )
    save_plot(plot, output_path, width=10, height=max(6, len(pivot["model"].unique()) * 1.2))


def plot_snr_vs_wer_scatter(df: pd.DataFrame, output_path: str, snr_cap: float = 100.0) -> None:
    """SNR vs WER scatter with per-model regression lines.

    SNR values above *snr_cap* are treated as measurement artifacts and
    excluded so the x-axis stays in a realistic range.
    """
    subset = df.dropna(subset=["snr_db", "wer"]).copy()
    subset = subset[np.isfinite(subset["snr_db"])]
    subset = subset[(subset["wer"] <= 1.0) & (subset["snr_db"] <= snr_cap)]
    if subset.empty:
        logger.warning("No data for SNR-vs-WER scatter; skipping.")
        return

    models = sorted(subset["model"].unique())
    colors = get_swarm_colors(len(models))

    plot = (
        p9.ggplot(subset, p9.aes(x="snr_db", y="wer", color="model"))
        + p9.geom_point(size=2.5, alpha=0.5)
        + p9.geom_smooth(method="lm", se=True, size=1.5, alpha=0.2)
        + p9.scale_color_manual(values=colors)
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(
            title="SNR vs. WER per Utterance",
            x="SNR (dB)",
            y="Word Error Rate (WER)",
            color="Model",
        )
        + theme_swarm_lab(figure_size=(12, 8))
    )
    save_plot(plot, output_path, width=12, height=8)


def plot_snr_distribution(df: pd.DataFrame, output_path: str, snr_cap: float = 100.0) -> None:
    """Per-model SNR histogram with tier-threshold reference lines."""
    subset = df.dropna(subset=["snr_db"]).copy()
    subset = subset[np.isfinite(subset["snr_db"]) & (subset["snr_db"] <= snr_cap)]
    if subset.empty:
        logger.warning("No data for SNR distribution histogram; skipping.")
        return

    cfg = get_audio_quality_config()
    models = sorted(subset["model"].unique())
    colors = get_swarm_colors(len(models))
    n_bins = max(10, min(40, len(subset) // 3))

    plot = (
        p9.ggplot(subset, p9.aes(x="snr_db", fill="model"))
        + p9.geom_histogram(bins=n_bins, alpha=0.65, position="dodge", color="black", size=0.3)
        + p9.geom_vline(xintercept=cfg.snr_tier_low_db, linetype="dashed", color="#C84848", size=1)
        + p9.geom_vline(xintercept=cfg.snr_tier_high_db, linetype="dashed", color="#2D8B57", size=1)
        + p9.scale_fill_manual(values=colors)
        + p9.annotate("text", x=cfg.snr_tier_low_db - 1, y=0, label="Low / Medium", ha="right", size=9, color="#C84848")
        + p9.annotate(
            "text", x=cfg.snr_tier_high_db + 1, y=0, label="Medium / High", ha="left", size=9, color="#2D8B57"
        )
        + p9.labs(
            title="SNR Distribution (Histogram)",
            x="SNR (dB)",
            y="Count",
            fill="Model",
        )
        + theme_swarm_lab(figure_size=(12, 8))
    )
    save_plot(plot, output_path, width=12, height=8)


def compute_quality_summary(df: pd.DataFrame, output_path: str) -> dict:
    """Compute data-quality summary stats and write them as JSON for the report.

    Returns the summary dict so the report generator can render it as a
    markdown table instead of an image.
    """
    subset = df.copy()
    snr_vals = subset["snr_db"].dropna()
    snr_finite = snr_vals[np.isfinite(snr_vals)]

    mean_snr = float(snr_finite.mean()) if len(snr_finite) else float("nan")
    n_total = len(subset)

    clip_col = subset.get("clipping_ratio")
    pct_clipped = 0.0
    if clip_col is not None and not clip_col.dropna().empty:
        pct_clipped = float((clip_col.dropna() > 0.01).mean() * 100)

    sil_col = subset.get("silence_ratio")
    pct_excessive_silence = 0.0
    if sil_col is not None and not sil_col.dropna().empty:
        pct_excessive_silence = float((sil_col.dropna() > 0.80).mean() * 100)

    passes_all = 0.0
    if not snr_finite.empty and clip_col is not None and sil_col is not None:
        mask = (
            (subset["snr_db"] >= get_audio_quality_config().snr_tier_low_db)
            & (subset["clipping_ratio"] <= 0.01)
            & (subset["silence_ratio"] <= 0.80)
        )
        valid = mask.dropna()
        passes_all = float(valid.mean() * 100) if len(valid) else 0.0

    summary = {
        "total_utterances": n_total,
        "mean_snr_db": round(mean_snr, 1),
        "pct_clipped": round(pct_clipped, 1),
        "pct_excessive_silence": round(pct_excessive_silence, 1),
        "pct_passing_all": round(passes_all, 1),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved quality summary: %s", output_path)
    return summary


def plot_wer_vs_mos_scatter(df: pd.DataFrame, output_path: str) -> None:
    """WER vs DNSMOS OVRL scatter with per-model regression lines."""
    if "dnsmos_ovrl" not in df.columns:
        logger.info("dnsmos_ovrl column missing — skipping WER vs MOS scatter.")
        return

    subset = df.dropna(subset=["dnsmos_ovrl", "wer"]).copy()
    subset = subset[subset["wer"] <= 1.0]
    if subset.empty:
        logger.warning("No data for WER-vs-MOS scatter; skipping.")
        return

    models = sorted(subset["model"].unique())
    colors = get_swarm_colors(len(models))

    plot = (
        p9.ggplot(subset, p9.aes(x="dnsmos_ovrl", y="wer", color="model"))
        + p9.geom_point(size=2.5, alpha=0.5)
        + p9.geom_smooth(method="lm", se=True, size=1.5, alpha=0.2)
        + p9.scale_color_manual(values=colors)
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.labs(
            title="WER vs Audio Quality (DNSMOS)",
            x="DNSMOS Overall MOS (1-5)",
            y="WER",
            color="Model",
        )
        + theme_swarm_lab(figure_size=(12, 8))
        + p9.theme(
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_background=p9.element_rect(fill="white"),
        )
    )
    save_plot(plot, output_path, width=12, height=8)


def plot_wer_by_mos_tier(df: pd.DataFrame, output_path: str) -> None:
    """Boxplot of WER per MOS quality tier, faceted by model."""
    if "mos_tier" not in df.columns:
        logger.info("mos_tier column missing — skipping WER by MOS tier.")
        return

    subset = df.dropna(subset=["mos_tier", "wer"]).copy()
    subset = subset[subset["wer"] <= 1.0]
    if subset.empty:
        logger.warning("No data for WER-by-MOS-tier; skipping.")
        return

    subset["mos_tier"] = pd.Categorical(subset["mos_tier"], categories=_TIER_ORDER, ordered=True)
    colors = [_TIER_COLORS[t] for t in _TIER_ORDER]

    plot = (
        p9.ggplot(subset, p9.aes(x="mos_tier", y="wer", fill="mos_tier"))
        + p9.geom_boxplot(alpha=0.8, outlier_alpha=0.3, width=0.6)
        + p9.scale_fill_manual(values=colors)
        + p9.scale_y_continuous(limits=(0, 1))
        + p9.facet_wrap("model", ncol=3)
        + p9.labs(
            title="WER by MOS Quality Tier",
            x="MOS Tier (DNSMOS OVRL)",
            y="WER",
        )
        + theme_swarm_lab(figure_size=(14, 8))
        + p9.theme(
            legend_position="none",
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_background=p9.element_rect(fill="white"),
        )
    )
    save_plot(plot, output_path, width=14, height=8)


def generate_audio_quality_plots(
    results_csvs: List[Tuple[str, str]],
    output_dir: str,
    language: str = "bengali",
    include_tier_plots: bool = False,
) -> None:
    """Generate audio-quality visualisations for a set of result CSVs.

    Always produces the SNR-vs-WER scatter, the SNR histogram, the quality
    summary JSON, and (when DNSMOS data is present) the WER-vs-MOS scatter.
    Set ``include_tier_plots=True`` to also generate tier-bucketed plots,
    useful for datasets with diverse audio quality ranges.

    Parameters
    ----------
    results_csvs : list of (model_name, csv_path) tuples
    output_dir   : directory under which plots are created
    language     : language label (for future per-language customisation)
    include_tier_plots : generate tier-bucketed plots (default False)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_tag_results(results_csvs)
    if df.empty:
        logger.warning("No result data available — skipping audio quality plots.")
        return

    required_cols = {"snr_db", "snr_tier", "wer"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.warning("Result CSVs missing columns %s — skipping audio quality plots.", missing)
        return

    logger.info("Generating audio quality plots (%d rows, %d models) ...", len(df), df["model"].nunique())

    plot_snr_vs_wer_scatter(df, str(out / "snr_vs_wer_scatter.png"))
    plot_snr_distribution(df, str(out / "snr_distribution.png"))
    compute_quality_summary(df, str(out / "quality_summary.json"))

    if "dnsmos_ovrl" in df.columns and df["dnsmos_ovrl"].notna().any():
        plot_wer_vs_mos_scatter(df, str(out / "wer_vs_mos_scatter.png"))

    if include_tier_plots:
        plot_wer_by_snr_tier(df, str(out / "wer_by_snr_tier.png"))
        plot_quality_tier_composition(df, str(out / "quality_tier_composition.png"))
        plot_model_tier_heatmap(df, str(out / "model_tier_heatmap.png"))
        if "dnsmos_ovrl" in df.columns and df["dnsmos_ovrl"].notna().any():
            plot_wer_by_mos_tier(df, str(out / "wer_by_mos_tier.png"))

    logger.info("Audio quality plots saved to %s", out)
