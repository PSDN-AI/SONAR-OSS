"""
ASR evaluation result plotting utilities.
"""

import logging
from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    geom_abline,
    geom_col,
    geom_hline,
    geom_point,
    geom_text,
    geom_vline,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
)

from .plot_theme import save_plot, theme_swarm_lab

logger = logging.getLogger(__name__)


class ASRResultPlotter:
    CATEGORY_COLORS_WER = {"Excellent (<20%)": "#39AD48", "Good (20-50%)": "#3B5B92", "Fair (>50%)": "#D9544D"}

    CATEGORY_COLORS_CER = {"Excellent (<10%)": "#39AD48", "Good (10-30%)": "#3B5B92", "Fair (>30%)": "#D9544D"}

    MODEL_NAME_MAP = {
        "banglaspeech2text": "BanglaSpeech2Text",
        "khushids_bengali": "KhushiDS Bengali",
        "elevenlabs_api": "ElevenLabs API",
        "banglaasr": "BanglaASR",
        "wav2vec2_bengali": "Wav2Vec2-Bengali",
        "whisper_api": "Whisper API",
        "assemblyai_api": "AssemblyAI API",
        "banglaasr_v5": "BanglaASR v5",
        "tugstugi_bengali": "Tugstugi Bengali",
        "tugstugi_bengali_regional": "Tugstugi Regional",
    }

    @staticmethod
    def categorize_wer(wer: float) -> str:
        """Categorize WER performance."""
        if wer < 20:
            return "Excellent (<20%)"
        elif wer < 50:
            return "Good (20-50%)"
        else:
            return "Fair (>50%)"

    @staticmethod
    def categorize_cer(cer: float) -> str:
        """Categorize CER performance."""
        if cer < 10:
            return "Excellent (<10%)"
        elif cer < 30:
            return "Good (10-30%)"
        else:
            return "Fair (>30%)"

    @classmethod
    def prepare_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare dataframe for plotting."""
        df = df.copy()

        if "Avg WER (%)" in df.columns:
            df["Avg WER (%)"] = pd.to_numeric(df["Avg WER (%)"], errors="coerce")
        if "Avg CER (%)" in df.columns:
            df["Avg CER (%)"] = pd.to_numeric(df["Avg CER (%)"], errors="coerce")

        if "Model" in df.columns:
            df["Model_Display"] = df["Model"].map(lambda x: cls.MODEL_NAME_MAP.get(x, x))

        if "Avg WER (%)" in df.columns:
            df["WER_Category"] = df["Avg WER (%)"].apply(cls.categorize_wer)
        if "Avg CER (%)" in df.columns:
            df["CER_Category"] = df["Avg CER (%)"].apply(cls.categorize_cer)

        return df

    @classmethod
    def plot_wer_comparison(cls, df: pd.DataFrame, output_path: Path) -> None:
        """Create WER horizontal bar chart."""
        try:
            df = cls.prepare_data(df)
            df_sorted = df.sort_values("Avg WER (%)", ascending=True).reset_index(drop=True)

            plot = (
                ggplot(df_sorted, aes(x="Model_Display", y="Avg WER (%)", fill="WER_Category"))
                + geom_col(alpha=0.85, color="black", size=0.5)
                + coord_flip()
                + scale_fill_manual(values=cls.CATEGORY_COLORS_WER, name="Performance")
                + labs(title="ASR Model Performance - Word Error Rate (WER)", x="Model", y="Word Error Rate (%)")
                + theme_swarm_lab(figure_size=(12, 8))
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_plot(plot, str(output_path), dpi=600, width=12, height=8)
            logger.info(f"WER plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create WER plot: {str(e)}")
            raise

    @classmethod
    def plot_cer_comparison(cls, df: pd.DataFrame, output_path: Path) -> None:
        """Create CER horizontal bar chart."""
        try:
            df = cls.prepare_data(df)
            df_sorted = df.sort_values("Avg CER (%)", ascending=True).reset_index(drop=True)

            plot = (
                ggplot(df_sorted, aes(x="Model_Display", y="Avg CER (%)", fill="CER_Category"))
                + geom_col(alpha=0.85, color="black", size=0.5)
                + coord_flip()
                + scale_fill_manual(values=cls.CATEGORY_COLORS_CER, name="Performance")
                + labs(
                    title="ASR Model Performance - Character Error Rate (CER)", x="Model", y="Character Error Rate (%)"
                )
                + theme_swarm_lab(figure_size=(12, 8))
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_plot(plot, str(output_path), dpi=600, width=12, height=8)
            logger.info(f"CER plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create CER plot: {str(e)}")
            raise

    @classmethod
    def plot_wer_vs_cer_scatter(cls, df: pd.DataFrame, output_path: Path) -> None:
        """Create WER vs CER scatter plot."""
        try:
            df = cls.prepare_data(df)
            df["Performance"] = df["Avg WER (%)"].apply(cls.categorize_wer)

            plot = (
                ggplot(df, aes(x="Avg WER (%)", y="Avg CER (%)", color="Performance"))
                + geom_point(size=8, alpha=0.8, stroke=1.5)
                + geom_text(aes(label="Model_Display"), nudge_y=2, size=10, fontweight="bold", ha="center")
                + scale_color_manual(values=cls.CATEGORY_COLORS_WER, name="WER Performance")
                + labs(
                    title="ASR Model Performance: WER vs CER",
                    x="Word Error Rate (WER) %",
                    y="Character Error Rate (CER) %",
                )
                + theme_swarm_lab(figure_size=(14, 10))
            )

            max_wer = df["Avg WER (%)"].max()
            max_cer = df["Avg CER (%)"].max()

            if max_wer > 50:
                plot += geom_vline(xintercept=50, linetype="dashed", color="gray", alpha=0.5, size=1)
            if max_cer > 30:
                plot += geom_hline(yintercept=30, linetype="dashed", color="gray", alpha=0.5, size=1)

            plot += geom_abline(intercept=0, slope=1, linetype="dotted", color="black", alpha=0.3, size=1)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_plot(plot, str(output_path), dpi=600, width=14, height=10)
            logger.info(f"Scatter plot saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to create scatter plot: {str(e)}")
            raise

    @classmethod
    def create_all_plots(cls, summary_csv: Path, output_dir: Path) -> dict:
        """Create all standard ASR evaluation plots."""
        try:
            df = pd.read_csv(summary_csv)
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            plots = {}

            plots["wer"] = output_dir / "asr_wer_comparison.png"
            cls.plot_wer_comparison(df, plots["wer"])

            plots["cer"] = output_dir / "asr_cer_comparison.png"
            cls.plot_cer_comparison(df, plots["cer"])

            plots["scatter"] = output_dir / "asr_wer_vs_cer.png"
            cls.plot_wer_vs_cer_scatter(df, plots["scatter"])

            logger.info("All plots created successfully")
            return plots

        except Exception as e:
            logger.error(f"Failed to create plots: {str(e)}")
            raise
