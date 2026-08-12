"""Single-speaker dataset evaluation example.

Evaluates registered ASR models on a TSV dataset with ``audio_path`` and
``transcription`` columns. The core evaluator lives in
:mod:`psdn_sonar.evaluators.single_speaker`; this script is a minimal CLI
wrapper around it.
"""

import argparse
import logging

from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR models on a single-speaker audio dataset")
    parser.add_argument(
        "--tsv-path", type=str, required=True, help="Path to TSV file with audio_path and transcription columns"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/asr_evaluation",
        help="Output directory for results",
    )
    parser.add_argument("--models", nargs="+", default=["banglaasr_v5"], help="Models to evaluate (space-separated)")
    parser.add_argument("--max-samples", type=int, default=10, help="Maximum samples to evaluate (0 for all)")
    parser.add_argument("--compute-sem", action="store_true", help="Compute semantic similarity (slower)")

    args = parser.parse_args()

    SingleSpeakerEvaluator.run_evaluation(
        tsv_path=args.tsv_path,
        output_dir=args.output_dir,
        models=args.models,
        max_samples=args.max_samples,
        compute_sem=args.compute_sem,
    )

    logger.info("Evaluation completed successfully")
