"""Generate summary plots from evaluation results.

Creates WER/CER comparison charts and scatter plots from a summary CSV via
:class:`psdn_sonar.utils.plotting.ASRResultPlotter`. For the full analysis
pipeline (benchmarks, diversity, audio quality, markdown report), use
``psdn-sonar single ... --report`` instead.
"""

import argparse
import logging
from pathlib import Path

from psdn_sonar.utils.plotting import ASRResultPlotter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_asr_plots(summary_csv: Path, output_dir: Path) -> dict:
    """Generate WER/CER comparison and scatter plots; return name -> path."""
    plots = ASRResultPlotter.create_all_plots(summary_csv, output_dir)
    logger.info("Generated %d plots in %s", len(plots), output_dir)
    return plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary plots from an evaluation summary CSV")
    parser.add_argument("--summary-csv", type=Path, required=True, help="Summary CSV with per-model results")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/visualizations"), help="Directory to save plots"
    )

    args = parser.parse_args()
    generate_asr_plots(args.summary_csv, args.output_dir)
