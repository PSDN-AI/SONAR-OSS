"""Demographic analysis of multi-speaker evaluation results.

Analyzes ASR performance across demographic dimensions (age, gender,
region) and generates violin plots, boxplots, and summary statistics.
"""

import argparse
import logging
import sys
from pathlib import Path

from psdn_sonar.analysis.demographic_analyzer import DemographicAnalyzer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate demographic analysis plots from multi-speaker evaluation results"
    )
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Path to evaluation results CSV (e.g., asr_eval_results_model_manifest.csv)",
    )
    parser.add_argument("--dataset-dir", required=True, help="Path to dataset directory containing metadata.json files")
    parser.add_argument(
        "--output-dir",
        default="results/demographic-analysis",
        help="Output directory for plots and statistics (default: results/demographic-analysis)",
    )
    parser.add_argument(
        "--model-name", help="Model name for plot titles (optional, extracted from filename if not provided)"
    )

    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    if not results_csv.exists():
        logger.error(f"Results CSV not found: {results_csv}")
        sys.exit(1)

    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    model_name = args.model_name
    if not model_name:
        model_name = results_csv.stem.replace("asr_eval_results_", "").replace("_manifest", "")

    logger.info("Demographic analysis: results=%s dataset=%s model=%s", results_csv, dataset_dir, model_name)

    try:
        wrote_outputs = DemographicAnalyzer.run_full_analysis(
            results_csv=results_csv, dataset_dir=dataset_dir, output_dir=output_dir, model_name=model_name
        )
        if wrote_outputs:
            logger.info("Plots saved to: %s/demographic_plots/%s/", output_dir, model_name)
            logger.info("Statistics saved to: %s/demographic_stats/%s/", output_dir, model_name)
        # A metadata-less skip already logged a warning naming the expected
        # <dataset_dir>/<audio_id>/metadata.json layout (issue #234).
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
