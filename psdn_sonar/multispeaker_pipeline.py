"""Multi-speaker ASR evaluation pipeline.

Runs manifest-driven evaluation with per-speaker preprocessing method
selection and writes per-clip results plus a summary stats file.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from psdn_sonar.config import load_env

logger = logging.getLogger(__name__)


def run_multispeaker_evaluation(
    manifest_path: str,
    model_name: str,
    output_dir: str = "Results",
    max_samples: int = 0,
    methods: Optional[List[str]] = None,
    sweep: bool = False,
    method: Optional[str] = None,
    language: str = "bn",
    custom_hf_model: Optional[str] = None,
    config_path: Optional[str] = None,
) -> Path:
    """Evaluate a manifest with one ASR model and return the results CSV path.

    Args:
        manifest_path: Path to a manifest.jsonl file.
        model_name: Registered model name (see :mod:`psdn_sonar.models.registry`).
        output_dir: Directory for output files.
        max_samples: Maximum samples to process (0 = all).
        methods: Preprocessing methods; ``None`` uses config defaults.
        sweep: Score every active method against ground truth and keep the best
            per clip. With more than one active method this is oracle selection
            and inflates the reported metrics.
        method: Explicit method name to use for all clips.
        language: ISO 639-1 code used for WER/CER normalization.
        custom_hf_model: HuggingFace repo id; when set, ``model_name`` is only
            used as the results-file stem.
        config_path: Multi-speaker preprocessing YAML; ``None`` uses the packaged
            ``psdn_sonar/multi_speaker_config.yaml``. ``methods`` still wins over
            whatever method list the file carries.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError: If the model name is not registered.
        RuntimeError: If an explicitly requested method needs pyannote.audio
            and it is not installed, or if no clip could be processed.
    """
    # Same credential sources as the single-speaker path (issue #167): hosted
    # API adapters and the pyannote HF_TOKEN reads all go through os.getenv,
    # so .env must be loaded before create_model / manifest processing.
    load_env()

    from psdn_sonar.core import process_manifest_with_asr
    from psdn_sonar.models.registry import create_model
    from psdn_sonar.preprocessing.config_loader import KNOWN_METHODS, load_multi_speaker_config
    from psdn_sonar.preprocessing.methods import PYANNOTE_METHODS
    from psdn_sonar.preprocessing.pyannote_utils import PYANNOTE_AVAILABLE

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    if method in PYANNOTE_METHODS and not PYANNOTE_AVAILABLE:
        # Fail fast: the explicit method applies to every clip, so the whole
        # run is doomed. Surface the actionable install hint instead of one
        # generic per-clip warning per speaker.
        raise RuntimeError(
            f"Preprocessing method '{method}' requires pyannote.audio, which is not installed. "
            "Install with: pip install 'psdn-sonar[pyannote]' "
            "(pyannote models are gated on HuggingFace — set HF_TOKEN after accepting the model terms)."
        )

    # An override replaces the file's method list, so the file is not required
    # to carry a usable one — its settings still are.
    config = load_multi_speaker_config(config_path, methods_required=not (methods or method))
    if methods:
        # An explicit list bypasses the config file, and with it the file
        # loader's KNOWN_METHODS validation — so validate here rather than let
        # an unknown name reach the strategy table as a KeyError per clip.
        unknown = [m for m in methods if m not in KNOWN_METHODS]
        if unknown:
            raise ValueError(
                f"Unknown preprocessing method(s): {', '.join(unknown)}. "
                f"Known methods: {', '.join(sorted(KNOWN_METHODS))}."
            )
        config["methods"] = methods

    logger.info(
        "Multi-speaker evaluation: manifest=%s model=%s language=%s max_samples=%s",
        manifest_path,
        model_name,
        language,
        max_samples if max_samples > 0 else "all",
    )
    logger.info(
        "Scope: multi-speaker WER/CER measures the preprocessing + ASR pipeline end to end "
        "(trimming/VAD/diarization errors are charged to the run), not isolated model "
        "capability. See docs/SCORE_INTERPRETATION.md."
    )

    model = create_model(model_name, custom_hf_model=custom_hf_model, language=language)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir_path / f"asr_eval_results_{model_name}_{manifest_file.stem}.csv"

    process_manifest_with_asr(
        manifest_path=str(manifest_file),
        asr_model=model,
        output_csv=str(output_csv),
        max_samples=max_samples,
        asr_model_name=model_name,
        language=language,
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
