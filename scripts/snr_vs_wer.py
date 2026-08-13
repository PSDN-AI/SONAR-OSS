#!/usr/bin/env python3
"""Compute audio-quality metrics for evaluation results and plot SNR vs WER.

Usage:
    python scripts/snr_vs_wer.py --csv whisper_api=results/whisper_api/commonvoice.csv \\
        --audio-root path/to/audio --language bengali --output-dir results/snr-analysis
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from psdn_sonar.audio_quality import compute_audio_quality_metrics
from psdn_sonar.reporting.plots.audio_quality import generate_audio_quality_plots

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_AUDIO_COLUMNS = ["audio_path", "path", "audio_file"]
_QUALITY_KEYS = ["snr_db", "clipping_ratio", "silence_ratio", "snr_tier"]


def parse_csv_specs(specs: list) -> dict:
    """Parse repeated ``model=path`` specs into ``{model: Path}``."""
    csvs = {}
    for spec in specs:
        model, sep, path = spec.partition("=")
        if not sep or not model or not path:
            raise ValueError(f"--csv expects MODEL=PATH, got {spec!r}")
        csvs[model] = Path(path)
    return csvs


def add_quality_metrics(df: pd.DataFrame, audio_root: Path) -> pd.DataFrame:
    """Append per-row audio-quality columns; missing/failed files get NaN."""
    audio_col = next((c for c in _AUDIO_COLUMNS if c in df.columns), None)
    if audio_col is None:
        raise ValueError(f"No audio column ({'/'.join(_AUDIO_COLUMNS)}) in results CSV")

    quality: dict = {key: [] for key in _QUALITY_KEYS}
    for idx, rel_path in enumerate(df[audio_col].astype(str)):
        audio_path = Path(rel_path) if Path(rel_path).is_absolute() else audio_root / rel_path
        metrics = {}
        if audio_path.exists():
            try:
                metrics = compute_audio_quality_metrics(str(audio_path), include_mos=False)
            except Exception as e:
                logger.warning("Audio quality failed for %s: %s", audio_path, e)
        for key in _QUALITY_KEYS:
            quality[key].append(metrics.get(key, None if key == "snr_tier" else np.nan))

        if (idx + 1) % 50 == 0:
            logger.info("Processed %d/%d files", idx + 1, len(df))

    for key, values in quality.items():
        df[key] = values

    for metric in ("wer", "cer"):
        if f"{metric}_conv" in df.columns and metric not in df.columns:
            df[metric] = df[f"{metric}_conv"]

    return df


def main():
    parser = argparse.ArgumentParser(description="Audio-quality metrics + SNR vs WER plots for evaluation results")
    parser.add_argument(
        "--csv", nargs="+", required=True, metavar="MODEL=PATH", help="Per-model evaluation results CSVs"
    )
    parser.add_argument("--audio-root", type=Path, default=Path("."), help="Root for relative audio paths")
    parser.add_argument("--language", default="bengali", help="Language label for plot titles")
    parser.add_argument("--output-dir", type=Path, default=Path("results/snr-analysis"), help="Output directory")
    args = parser.parse_args()

    csvs = parse_csv_specs(args.csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    enriched = []
    for model, csv_path in csvs.items():
        if not csv_path.exists():
            logger.warning("Skipping %s: %s not found", model, csv_path)
            continue

        df = pd.read_csv(csv_path)
        logger.info("%s: %d rows", model, len(df))
        df = add_quality_metrics(df, args.audio_root)

        out_csv = args.output_dir / f"{model}.csv"
        df.to_csv(out_csv, index=False)
        enriched.append((model, str(out_csv)))

        valid_snr = df["snr_db"].dropna()
        if not valid_snr.empty:
            logger.info(
                "%s SNR: min=%.1f max=%.1f mean=%.1f dB", model, valid_snr.min(), valid_snr.max(), valid_snr.mean()
            )

    if not enriched:
        logger.error("No data to plot.")
        sys.exit(1)

    plots_dir = args.output_dir / "plots"
    generate_audio_quality_plots(results_csvs=enriched, output_dir=str(plots_dir), language=args.language)
    logger.info("Plots saved to %s", plots_dir)


if __name__ == "__main__":
    main()
