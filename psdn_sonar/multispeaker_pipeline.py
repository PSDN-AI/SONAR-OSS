"""Multi-speaker ASR evaluation pipeline.

Runs manifest-driven evaluation with per-speaker preprocessing method
selection and writes per-clip results plus a summary stats file.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def run_multispeaker_evaluation(
    manifest_path: str,
    model_name: str,
    output_dir: str = "Results",
    max_samples: int = 0,
    methods: Optional[List[str]] = None,
    sweep: bool = False,
    method: Optional[str] = None,
) -> Path:
    """Evaluate a manifest with one ASR model and return the results CSV path.

    Args:
        manifest_path: Path to a manifest.jsonl file.
        model_name: Registered model name (see :mod:`psdn_sonar.models.registry`).
        output_dir: Directory for output files.
        max_samples: Maximum samples to process (0 = all).
        methods: Preprocessing methods; ``None`` uses config defaults.
        sweep: Run all methods with oracle selection (inflates metrics).
        method: Explicit method name to use for all clips.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the model name is not registered.
    """
    from psdn_sonar.core import process_manifest_with_asr
    from psdn_sonar.models.registry import create_model
    from psdn_sonar.preprocessing.config_loader import load_multi_speaker_config

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    config = load_multi_speaker_config()
    if methods:
        config["methods"] = methods

    logger.info(
        "Multi-speaker evaluation: manifest=%s model=%s max_samples=%s",
        manifest_path,
        model_name,
        max_samples if max_samples > 0 else "all",
    )

    model = create_model(model_name)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir_path / f"asr_eval_results_{model_name}_{manifest_file.stem}.csv"

    process_manifest_with_asr(
        manifest_path=str(manifest_file),
        asr_model=model,
        output_csv=str(output_csv),
        max_samples=max_samples,
        asr_model_name=model_name,
        methods=config["methods"],
        config_settings=config,
        sweep=sweep,
        method=method,
    )

    logger.info("Results CSV: %s", output_csv)
    logger.info("Stats file: %s", output_csv.with_suffix(".txt"))
    return output_csv


def main() -> None:
    """CLI entry point for standalone multi-speaker evaluation."""
    parser = argparse.ArgumentParser(description="Multi-speaker ASR evaluation pipeline.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl file")
    parser.add_argument("--model", required=True, help="Registered ASR model name")
    parser.add_argument("--output-dir", default="Results", help="Output directory (default: Results)")
    parser.add_argument("--max-samples", type=int, default=0, help="Maximum samples to process (0 = all)")
    parser.add_argument("--methods", help="Comma-separated preprocessing methods (default: from config)")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")] if args.methods else None
    run_multispeaker_evaluation(
        manifest_path=args.manifest,
        model_name=args.model,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        methods=methods,
    )


if __name__ == "__main__":
    main()
