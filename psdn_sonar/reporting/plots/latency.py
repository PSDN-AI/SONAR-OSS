"""Inference latency visualisations for evaluation reports."""

import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plotnine as p9

from psdn_sonar.utils.plot_theme import get_swarm_colors, save_plot, theme_swarm_lab

logger = logging.getLogger(__name__)


def _prettify_model_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def _load_and_tag(results_csvs: List[Tuple[str, str]]) -> pd.DataFrame:
    frames = []
    for model_name, csv_path in results_csvs:
        try:
            df = pd.read_csv(csv_path)
            df["model"] = _prettify_model_name(model_name)
            frames.append(df)
        except Exception as exc:
            logger.warning("Could not load %s: %s", csv_path, exc)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    if "inference_latency_s" in combined.columns:
        combined["inference_latency_s"] = pd.to_numeric(combined["inference_latency_s"], errors="coerce")
    return combined


def plot_latency_boxplot(df: pd.DataFrame, output_path: str) -> None:
    """Boxplot comparing inference latency across models."""
    subset = df.dropna(subset=["inference_latency_s"]).copy()
    if subset.empty:
        logger.warning("No latency data for boxplot; skipping.")
        return

    colors = get_swarm_colors(subset["model"].nunique())

    plot = (
        p9.ggplot(subset, p9.aes(x="model", y="inference_latency_s", fill="model"))
        + p9.geom_boxplot(alpha=0.8, outlier_alpha=0.3, width=0.6)
        + p9.scale_fill_manual(values=colors)
        + p9.labs(
            title="Inference Latency per Model",
            x="Model",
            y="Latency (seconds)",
        )
        + theme_swarm_lab(figure_size=(12, 8))
        + p9.theme(
            legend_position="none",
            axis_text_x=p9.element_text(rotation=25, ha="right", size=9),
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            panel_background=p9.element_rect(fill="white"),
        )
    )
    save_plot(plot, output_path, width=12, height=8)


def generate_latency_plots(
    results_csvs: List[Tuple[str, str]],
    output_dir: str,
) -> None:
    """Generate all latency visualisations.

    Parameters
    ----------
    results_csvs : list of (model_name, csv_path) tuples
    output_dir   : directory to write plot PNGs
    """
    out = Path(output_dir)

    df = _load_and_tag(results_csvs)
    if df.empty or "inference_latency_s" not in df.columns:
        logger.warning("No latency data available — skipping latency plots.")
        return

    if df["inference_latency_s"].notna().sum() == 0:
        logger.warning("All latency values are null — skipping latency plots.")
        return

    out.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Generating latency plots (%d rows, %d models) ...",
        len(df),
        df["model"].nunique(),
    )

    plot_latency_boxplot(df, str(out / "latency_boxplot.png"))

    logger.info("Latency plots saved to %s", out)
