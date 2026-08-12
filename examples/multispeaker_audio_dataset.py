"""Multi-speaker dataset evaluation example.

Evaluates an ASR model on a ``manifest.jsonl`` dataset where each entry
points to per-speaker audio, combined audio, and a transcript. This script
is a minimal CLI wrapper around
:func:`psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation`.
"""

import argparse
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_multispeaker_evaluation(
    manifest_path: str,
    model_name: str,
    output_dir: str = "results/multispeaker-eval",
    max_samples: int = 0,
    methods: Optional[List[str]] = None,
):
    """Run multi-speaker ASR evaluation and return the results CSV path."""
    from psdn_sonar.multispeaker_pipeline import run_multispeaker_evaluation as run_eval

    logger.info("Starting multi-speaker evaluation")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output: {output_dir}")

    result = run_eval(
        manifest_path=manifest_path,
        model_name=model_name,
        output_dir=output_dir,
        max_samples=max_samples,
        methods=methods,
    )

    logger.info("Multi-speaker evaluation completed")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-speaker ASR evaluation")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--output-dir", default="results/multispeaker-eval", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples (0=all)")
    parser.add_argument("--methods", nargs="+", help="Preprocessing methods")

    args = parser.parse_args()

    run_multispeaker_evaluation(
        manifest_path=args.manifest,
        model_name=args.model,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        methods=args.methods,
    )
