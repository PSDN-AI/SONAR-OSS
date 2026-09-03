"""Demographic performance analysis.

Analyzes ASR performance across demographic dimensions (age, gender, region)
from per-model results CSVs joined with per-recording speaker metadata.
Produces violin plots, per-group summary statistics, a text summary report,
and cross-model overview plots (overall benchmark and worst-group gaps).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotnine as p9

from psdn_sonar.utils.metrics import ensure_poseidon_score

logger = logging.getLogger(__name__)


def _model_name_from_csv(results_csv: Path) -> str:
    """Model name from a ``results_<model>_manifest.csv`` filename."""
    return results_csv.stem.replace("results_", "").replace("_manifest", "")


class DemographicAnalyzer:
    """ASR performance analysis across demographic dimensions."""

    METRICS = {
        "cer_conv": "CER",
        "wer_conv": "WER",
        "semantic_similarity_conv": "Semantic Similarity",
        "poseidon_score": "Poseidon Score",
    }

    DEMOGRAPHICS = {"age": "Age", "gender": "Gender", "region": "Region"}

    _ERROR_METRICS = ("cer_conv", "wer_conv")

    @staticmethod
    def load_data_with_metadata(results_csv: Path, dataset_dir: Path) -> pd.DataFrame:
        """Results CSV joined with speaker age/gender/region from per-recording metadata.

        Speaker attributes come from ``<dataset_dir>/<audio_id>/metadata.json``
        (``speaker_a`` / ``speaker_b`` keys); missing metadata yields NA columns.
        """
        df = pd.read_csv(results_csv)
        df = ensure_poseidon_score(df)

        metadata_list = []
        json_cache: Dict[str, dict] = {}
        for audio_id, speaker in zip(df["audio_id"], df["speaker"]):
            if audio_id not in json_cache:
                metadata_path = dataset_dir / audio_id / "metadata.json"
                json_cache[audio_id] = {}
                if metadata_path.exists():
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        json_cache[audio_id] = json.load(f)

            speaker_key = "speaker_a" if speaker == "A" else "speaker_b"
            speaker_meta = json_cache[audio_id].get(speaker_key, {})
            metadata_list.append(
                {
                    "age": speaker_meta.get("age"),
                    "gender": speaker_meta.get("gender"),
                    "region": speaker_meta.get("region"),
                }
            )

        return pd.concat([df, pd.DataFrame(metadata_list)], axis=1)

    @classmethod
    def create_violin_plot(cls, df: pd.DataFrame, demographic: str, metric: str, output_path: Path) -> None:
        """Violin + boxplot + jitter of *metric* grouped by *demographic*.

        Raises ValueError when no row carries both values: plotnine's own
        failure on an all-NA frame is an unrelated ``TypeError`` from axis
        expansion that names neither column (issue #234).
        """
        df_clean = df.dropna(subset=[demographic, metric]).copy()
        if df_clean.empty:
            raise ValueError(
                f"No rows have both '{demographic}' and '{metric}' — nothing to plot. "
                f"Demographic values come from <dataset_dir>/<audio_id>/metadata.json."
            )
        df_clean[f"{demographic}_str"] = df_clean[demographic].astype(str)

        plot = (
            p9.ggplot(df_clean, p9.aes(x=f"{demographic}_str", y=metric, fill=f"{demographic}_str"))
            + p9.geom_violin(alpha=0.5, scale="width", width=0.8)
            + p9.geom_boxplot(
                width=0.2, alpha=0.9, outlier_alpha=0, size=0.8, fatten=3, show_legend=False, color="black"
            )
            + p9.geom_jitter(width=0.15, height=0, alpha=0.4, size=1.2, color="black")
            + p9.labs(
                title=f"{cls.METRICS[metric]} by {cls.DEMOGRAPHICS[demographic]}",
                x=cls.DEMOGRAPHICS[demographic],
                y=cls.METRICS[metric],
            )
            + p9.scale_y_continuous(
                limits=(0, None) if metric in cls._ERROR_METRICS else None,  # ty: ignore[invalid-argument-type]
                expand=(0, 0, 0.1, 0),
            )
            + p9.theme_minimal()
            + p9.theme(
                figure_size=(14, 7),
                plot_title=p9.element_text(size=14, weight="bold"),
                axis_title=p9.element_text(size=12),
                axis_text=p9.element_text(size=10),
                legend_position="none",
                panel_grid_major_x=p9.element_blank(),
            )
        )

        plot.save(output_path, dpi=300, verbose=False)

    @staticmethod
    def generate_summary_stats(df: pd.DataFrame, demographic: str, metric: str) -> pd.DataFrame:
        """Count/mean/std/median/min/max of *metric* per demographic group."""
        df_clean = df.dropna(subset=[demographic, metric])
        stats = (
            df_clean.groupby(demographic)[metric]
            .agg(
                [
                    ("count", "count"),
                    ("mean", "mean"),
                    ("std", "std"),
                    ("median", "median"),
                    ("min", "min"),
                    ("max", "max"),
                ]
            )
            .reset_index()
        )
        return stats.sort_values("mean", ascending=False)

    @classmethod
    def run_full_analysis(
        cls, results_csv: Path, dataset_dir: Path, output_dir: Path, model_name: Optional[str] = None
    ) -> bool:
        """Generate violin plots, per-group stats CSVs, and a text summary for one model.

        Returns True when outputs were written. Demographics with no data
        are skipped, and when *no* recording has any metadata the whole
        analysis is skipped with a warning naming the expected layout and a
        False return, instead of crashing inside plotnine on the all-NA
        columns (issue #234: the shipped example fixture has no
        ``metadata.json``, and the resulting NA columns surfaced as an
        unrelated ``TypeError`` from axis expansion). Nothing is written and
        no directories are created for a skipped analysis.
        """
        if model_name is None:
            model_name = _model_name_from_csv(results_csv)

        df = cls.load_data_with_metadata(results_csv, dataset_dir)

        with_data = [d for d in cls.DEMOGRAPHICS if d in df.columns and df[d].notna().any()]
        if not with_data:
            logger.warning(
                "No demographic metadata found for any of the %d recording(s) in %s. "
                "Speaker attributes are read from %s/<audio_id>/metadata.json "
                "('speaker_a'/'speaker_b' objects with 'age', 'gender', 'region' keys); "
                "no such file exists for these recordings. Skipping demographic "
                "analysis for %s — the evaluation artifacts are unaffected.",
                df["audio_id"].nunique(),
                results_csv,
                dataset_dir,
                model_name,
            )
            return False

        plots_dir = output_dir / "demographic_plots" / model_name
        stats_dir = output_dir / "demographic_stats" / model_name
        plots_dir.mkdir(exist_ok=True, parents=True)
        stats_dir.mkdir(exist_ok=True, parents=True)

        for demographic in with_data:
            for metric in cls.METRICS:
                if metric not in df.columns:
                    continue
                if df.dropna(subset=[demographic, metric]).empty:
                    continue
                cls.create_violin_plot(df, demographic, metric, plots_dir / f"{demographic}_{metric}.png")
                stats = cls.generate_summary_stats(df, demographic, metric)
                stats.to_csv(stats_dir / f"{demographic}_{metric}.csv", index=False, float_format="%.4f")

        cls.create_summary_report(df, output_dir, output_dir / f"demographic_summary_{model_name}.txt")
        return True

    @classmethod
    def create_summary_report(cls, df: pd.DataFrame, output_dir: Path, report_path: Optional[Path] = None) -> None:
        """Best/worst group and gap per demographic and metric, written as plain text."""
        if report_path is None:
            report_path = output_dir / "demographic_analysis_summary.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("DEMOGRAPHIC PERFORMANCE ANALYSIS\n")
            f.write("=" * 80 + "\n\n")

            for demographic in cls.DEMOGRAPHICS:
                if demographic not in df.columns:
                    continue
                f.write(f"\n{cls.DEMOGRAPHICS[demographic].upper()}\n")
                f.write("-" * 80 + "\n\n")

                for metric in cls.METRICS:
                    if metric not in df.columns:
                        continue
                    df_clean = df.dropna(subset=[demographic, metric])
                    if len(df_clean) == 0:
                        continue

                    stats = df_clean.groupby(demographic)[metric].agg(["mean", "count"]).reset_index()
                    stats = stats.sort_values("mean", ascending=(metric in cls._ERROR_METRICS))

                    best = stats.iloc[0]
                    worst = stats.iloc[-1]

                    f.write(f"{cls.METRICS[metric]}:\n")
                    f.write(f"  Best:  {best[demographic]} = {best['mean']:.4f} (n={int(best['count'])})\n")
                    f.write(f"  Worst: {worst[demographic]} = {worst['mean']:.4f} (n={int(worst['count'])})\n")
                    if len(stats) > 1:
                        f.write(f"  Gap:   {abs(worst['mean'] - best['mean']):.4f}\n")
                    f.write("\n")

                f.write("\n")

    @classmethod
    def build_overall_benchmark_long_and_summary(
        cls, csv_list: List[Path]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Per-metric long-format data (one row per sample per model) and per-model mean/std/count."""
        long_rows: Dict[str, List[dict]] = {m: [] for m in cls.METRICS}
        for results_csv in csv_list:
            model_name = _model_name_from_csv(results_csv)
            try:
                df = pd.read_csv(results_csv)
                df = ensure_poseidon_score(df)
            except Exception as e:
                logger.warning(f"Skipping {results_csv}: {e}")
                continue
            for metric in cls.METRICS:
                if metric not in df.columns:
                    continue
                for val in df[metric].dropna():
                    long_rows[metric].append({"model": model_name, "value": val})

        long_out = {}
        summary_out = {}
        for metric, rows in long_rows.items():
            if not rows:
                continue
            long_out[metric] = pd.DataFrame(rows)
            summary_out[metric] = long_out[metric].groupby("model")["value"].agg(["mean", "std", "count"]).reset_index()
        return long_out, summary_out

    @classmethod
    def create_generic_benchmark_plots(
        cls,
        long_dfs: Dict[str, pd.DataFrame],
        summary_dfs: Dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> None:
        """Per-metric boxplots of the full dataset with models on the x-axis."""
        if not long_dfs or not summary_dfs:
            logger.warning("No benchmark data; skipping generic benchmark plots.")
            return
        plots_dir = output_dir / "demographic_plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        for metric in list(long_dfs.keys()):
            if metric not in summary_dfs:
                continue
            long_df = long_dfs[metric].copy()
            models = sorted(long_df["model"].unique())
            long_df["model"] = pd.Categorical(long_df["model"], categories=models, ordered=True)
            metric_label = cls.METRICS.get(metric, metric)
            plot = (
                p9.ggplot(long_df, p9.aes(x="model", y="value", fill="model"))
                + p9.geom_boxplot(outlier_alpha=0.4, width=0.7, alpha=0.85)
                + p9.labs(
                    title=f"{metric_label} by model (full dataset)",
                    x="Model",
                    y=metric_label,
                    fill="Model",
                )
                + p9.theme_minimal()
                + p9.theme(
                    figure_size=(14, 7),
                    plot_title=p9.element_text(size=14, weight="bold"),
                    axis_text_x=p9.element_text(rotation=45, ha="right"),
                    legend_position="right",
                )
            )
            if metric in cls._ERROR_METRICS:
                plot = plot + p9.scale_y_continuous(
                    limits=(0, None),  # ty: ignore[invalid-argument-type]
                    expand=(0, 0, 0.05, 0),
                )
            else:
                plot = plot + p9.scale_y_continuous(limits=(0, 1.05), expand=(0, 0, 0.05, 0))
            out_path = plots_dir / f"overall_benchmark_{metric}.png"
            plot.save(out_path, dpi=300, verbose=False)
            logger.info(f"Saved {out_path}")

    @classmethod
    def build_worst_demographic_table(
        cls, csv_list: List[Path], dataset_dir: Path
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Worst-group mean and worst-vs-overall gap per model, metric, and demographic.

        For error metrics the worst group has the highest mean; for
        similarity/composite scores it has the lowest.
        """
        worst_rows = []
        gap_rows = []
        for results_csv in csv_list:
            model_name = _model_name_from_csv(results_csv)
            try:
                df = cls.load_data_with_metadata(results_csv, dataset_dir)
            except Exception as e:
                logger.warning(f"Skipping {results_csv} for worst-demographic: {e}")
                continue
            for metric in cls.METRICS:
                if metric not in df.columns:
                    continue
                overall_mean = df[metric].mean()
                for dem in cls.DEMOGRAPHICS:
                    if dem not in df.columns:
                        continue
                    df_clean = df.dropna(subset=[dem, metric])
                    if df_clean.empty or df_clean[dem].nunique() < 2:
                        continue
                    group_means = df_clean.groupby(dem)[metric].mean()
                    worse_is_higher = metric in cls._ERROR_METRICS
                    worst_mean = group_means.max() if worse_is_higher else group_means.min()
                    gap = (worst_mean - overall_mean) if worse_is_higher else (overall_mean - worst_mean)
                    worst_rows.append(
                        {
                            "model": model_name,
                            "metric": metric,
                            "demographic": dem,
                            "worst_mean": worst_mean,
                            "overall_mean": overall_mean,
                        }
                    )
                    gap_rows.append({"model": model_name, "metric": metric, "demographic": dem, "gap": gap})
        return pd.DataFrame(worst_rows), pd.DataFrame(gap_rows)

    @classmethod
    def create_worst_demographic_plots(cls, worst_df: pd.DataFrame, gap_df: pd.DataFrame, output_dir: Path) -> None:
        """Bar charts of worst-group performance and worst-vs-overall gap per model."""
        if worst_df.empty:
            logger.warning("No worst-demographic data; skipping.")
            return
        plots_dir = output_dir / "demographic_plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        for metric in worst_df["metric"].unique():
            w = worst_df[worst_df["metric"] == metric]
            g = gap_df[gap_df["metric"] == metric] if not gap_df.empty else None
            metric_label = cls.METRICS.get(metric, metric)

            model_worst = w.groupby("model").agg(worst_mean=("worst_mean", "max")).reset_index()
            model_worst["model"] = pd.Categorical(
                model_worst["model"], categories=model_worst["model"].tolist(), ordered=True
            )
            model_worst["worst_label"] = model_worst["worst_mean"].apply(lambda x: f"{x:.3f}")
            plot = (
                p9.ggplot(model_worst, p9.aes(x="model", y="worst_mean", fill="model"))
                + p9.geom_col(show_legend=False)
                + p9.geom_text(p9.aes(label="worst_label"), va="bottom", nudge_y=0.02, size=9)
                + p9.labs(
                    title=f"Worst demographic performance by model ({metric_label})",
                    x="Model",
                    y=f"Worst-group mean {metric_label}",
                )
                + p9.theme_minimal()
                + p9.theme(
                    figure_size=(12, 6),
                    plot_title=p9.element_text(size=14, weight="bold"),
                    axis_text_x=p9.element_text(rotation=45, ha="right"),
                )
            )
            out_path = plots_dir / f"worst_demographic_{metric}.png"
            plot.save(out_path, dpi=300, verbose=False)
            logger.info(f"Saved {out_path}")

            if g is not None and not g.empty:
                model_gap = g.groupby("model").agg(gap=("gap", "max")).reset_index()
                model_gap["model"] = pd.Categorical(
                    model_gap["model"], categories=model_gap["model"].tolist(), ordered=True
                )
                plot_gap = (
                    p9.ggplot(model_gap, p9.aes(x="model", y="gap", fill="model"))
                    + p9.geom_col(show_legend=False)
                    + p9.labs(
                        title=f"Largest performance gap by model (worst demographic vs overall) — {metric_label}",
                        x="Model",
                        y="Gap (worst − overall)",
                    )
                    + p9.theme_minimal()
                    + p9.theme(
                        figure_size=(12, 6),
                        plot_title=p9.element_text(size=14, weight="bold"),
                        axis_text_x=p9.element_text(rotation=45, ha="right"),
                    )
                )
                out_gap = plots_dir / f"worst_demographic_gap_{metric}.png"
                plot_gap.save(out_gap, dpi=300, verbose=False)
                logger.info(f"Saved {out_gap}")
