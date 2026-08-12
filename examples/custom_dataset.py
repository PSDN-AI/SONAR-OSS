"""Custom dataset evaluation example.

Shows how to convert a dataset with arbitrary column names into the
psdn-sonar TSV format and evaluate registered models on it.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_dataset_format(tsv_path: Union[str, Path]) -> bool:
    """Check that the TSV has ``audio_path`` and ``transcription`` columns."""
    df = pd.read_csv(tsv_path, sep="\t")
    missing = {"audio_path", "transcription"} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    logger.info("Dataset validated: %d samples", len(df))
    return True


def prepare_custom_dataset(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    audio_column: str = "audio_path",
    text_column: str = "transcript",
) -> Path:
    """Convert a CSV with arbitrary column names into psdn-sonar TSV format."""
    df = pd.read_csv(input_path)
    prepared = pd.DataFrame({"audio_path": df[audio_column], "transcription": df[text_column]})
    prepared.to_csv(output_path, sep="\t", index=False)
    logger.info("Prepared %d samples -> %s", len(prepared), output_path)
    return Path(output_path)


def evaluate_on_custom_dataset(
    models: List[str],
    dataset_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    max_samples: int = 0,
) -> Dict:
    """Validate the dataset and evaluate *models* on it.

    Returns the result dict from
    :meth:`psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation`.
    """
    from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

    validate_dataset_format(dataset_path)
    output_dir = Path(output_dir) if output_dir else Path("results/custom_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    return SingleSpeakerEvaluator.run_evaluation(
        tsv_path=str(dataset_path),
        output_dir=str(output_dir),
        models=models,
        max_samples=max_samples,
        compute_sem=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ASR models on a custom TSV dataset")
    parser.add_argument("--dataset", required=True, help="Path to TSV with audio_path and transcription columns")
    parser.add_argument("--models", nargs="+", required=True, help="Registered model names to evaluate")
    parser.add_argument("--output-dir", default="results/custom_evaluation", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")

    args = parser.parse_args()

    evaluate_on_custom_dataset(
        models=args.models,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )
