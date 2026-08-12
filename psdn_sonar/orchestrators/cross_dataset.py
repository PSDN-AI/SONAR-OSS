"""Cross-dataset ASR comparison: run multiple models across local datasets.

Writes ``{model_key}_{dataset_name}_n{sample_size}.csv`` plus a ``.txt``
stats summary per (model, dataset) pair under the results directory.
"""

import argparse
import logging
import os
import traceback
from typing import Callable, List, Optional, Tuple

from psdn_sonar.config import load_env
from psdn_sonar.core import process_dataset_with_asr
from psdn_sonar.loaders.resolution import resolve_dataset_dir
from psdn_sonar.models.base import ASRModel
from psdn_sonar.models.registry import create_model, get_language_defaults

logger = logging.getLogger(__name__)

DEFAULT_BASE_DIR = "Datasets"
DEFAULT_RESULTS_DIR = os.path.join("Results", "cross-dataset-comparison")

DATASET_NAMES = [
    "commonvoice",
    "fleurs",
    "openslr37_bd",
    "openslr37_in",
    "openslr53",
]

_API_ENV_KEYS = {
    "elevenlabs_api": "ELEVENLABS_API_KEY",
    "whisper_api": "OPENAI_API_KEY",
    "assemblyai_api": "ASSEMBLYAI_API_KEY",
}

ModelConfig = Tuple[str, Callable[[], ASRModel], Optional[str]]


def default_models_config(language: str = "bengali") -> List[ModelConfig]:
    """``(model_key, factory, env_key)`` tuples for a language's default roster.

    Models are resolved through the registry; API models carry the env var
    that must be set for them to run.
    """
    names = get_language_defaults(language) or []
    return [(name, lambda name=name: create_model(name), _API_ENV_KEYS.get(name)) for name in names]


class CrossDatasetComparison:
    """Runs multiple ASR models on all available datasets and writes per-pair results."""

    @staticmethod
    def parse_sample_size(samples_arg: Optional[int] = None, prompt_input: Optional[str] = None) -> int:
        """Resolve the per-dataset sample size from CLI arg or prompt text.

        Falls back to 100 when neither source yields a positive integer.
        """
        if samples_arg is not None and samples_arg > 0:
            return samples_arg
        if prompt_input is not None:
            try:
                n = int(prompt_input.strip())
                if n > 0:
                    return n
            except (ValueError, TypeError):
                pass
        return 100

    @staticmethod
    def get_available_datasets(base_dir: str, dataset_names: List[str]) -> List[str]:
        """Dataset names from *dataset_names* that resolve to a directory under *base_dir*."""
        return [name for name in dataset_names if resolve_dataset_dir(base_dir, name)]

    @staticmethod
    def build_models_to_run(
        models_config: List[ModelConfig],
        env_getter: Optional[Callable[[str], Optional[str]]] = None,
    ) -> List[Tuple[str, ASRModel]]:
        """Instantiate models whose required env vars are present.

        Models with a missing env var or a failing factory are skipped.

        Returns:
            List of ``(model_key, model_instance)``.
        """
        if env_getter is None:
            env_getter = os.getenv
        result = []
        for model_key, factory, env_key in models_config:
            if env_key and not env_getter(env_key):
                logger.info("Skipping %s: %s not set", model_key, env_key)
                continue
            try:
                result.append((model_key, factory()))
            except Exception as e:
                logger.warning("Skipping %s: %s", model_key, e)
        return result

    @staticmethod
    def run_single(
        model_key: str,
        model: ASRModel,
        dataset_name: str,
        dataset_dir: str,
        results_dir: str,
        sample_size: int,
    ) -> bool:
        """Run one model on one dataset; return True on success."""
        output_csv = os.path.join(results_dir, f"{model_key}_{dataset_name}_n{sample_size}.csv")
        try:
            process_dataset_with_asr(
                dataset_name=dataset_name,
                dataset_dir=dataset_dir,
                asr_model=model,
                output_tsv=output_csv,
                max_samples=sample_size,
                asr_model_name=model_key,
            )
            return True
        except Exception:
            traceback.print_exc()
            return False

    @staticmethod
    def run_all(
        models_to_run: List[Tuple[str, ASRModel]],
        datasets: List[str],
        base_dir: str,
        results_dir: str,
        sample_size: int,
    ) -> Tuple[List[str], List[str]]:
        """Run every (model, dataset) pair.

        Returns:
            ``(completed, failed)`` lists of ``"model_key/dataset_name"`` entries.
        """
        os.makedirs(results_dir, exist_ok=True)
        completed = []
        failed = []
        for model_key, model in models_to_run:
            for dataset_name in datasets:
                dataset_dir = resolve_dataset_dir(base_dir, dataset_name)
                pair = f"{model_key}/{dataset_name}"
                if dataset_dir and CrossDatasetComparison.run_single(
                    model_key, model, dataset_name, dataset_dir, results_dir, sample_size
                ):
                    completed.append(pair)
                else:
                    failed.append(pair)
        return completed, failed

    @staticmethod
    def run(
        sample_size: int,
        base_dir: Optional[str] = None,
        results_dir: Optional[str] = None,
        dataset_names: Optional[List[str]] = None,
        models_config: Optional[List[ModelConfig]] = None,
        language: str = "bengali",
    ) -> Tuple[List[str], List[str], str]:
        """Resolve datasets and models, then run the full comparison.

        Returns:
            ``(completed, failed, results_dir)``. Both lists are empty when no
            datasets resolve or no models can be instantiated.
        """
        base_dir = base_dir or DEFAULT_BASE_DIR
        results_dir = results_dir or DEFAULT_RESULTS_DIR
        dataset_names = dataset_names or DATASET_NAMES
        if models_config is None:
            models_config = default_models_config(language)

        datasets = CrossDatasetComparison.get_available_datasets(base_dir, dataset_names)
        if not datasets:
            logger.error("No datasets found under %s", os.path.abspath(base_dir))
            return [], [], results_dir

        models_to_run = CrossDatasetComparison.build_models_to_run(models_config)
        if not models_to_run:
            logger.error("No models could be instantiated; check API keys and installed extras")
            return [], [], results_dir

        completed, failed = CrossDatasetComparison.run_all(models_to_run, datasets, base_dir, results_dir, sample_size)
        return completed, failed, results_dir


def main() -> None:
    """CLI entry point: parse args, run the comparison, log a short summary."""
    load_env()
    parser = argparse.ArgumentParser(description="Run multiple ASR models across datasets for comparison.")
    parser.add_argument("samples", nargs="?", type=int, default=None, help="Sample size per dataset (default: 100)")
    parser.add_argument("--language", default="bengali", help="Language whose default model roster to run")
    parser.add_argument("--base-dir", default=None, help=f"Dataset root directory (default: {DEFAULT_BASE_DIR})")
    parser.add_argument("--results-dir", default=None, help=f"Output directory (default: {DEFAULT_RESULTS_DIR})")
    args = parser.parse_args()

    sample_size = CrossDatasetComparison.parse_sample_size(samples_arg=args.samples)
    completed, failed, results_dir = CrossDatasetComparison.run(
        sample_size=sample_size,
        base_dir=args.base_dir,
        results_dir=args.results_dir,
        language=args.language,
    )

    logger.info("Completed: %d | Failed: %d", len(completed), len(failed))
    logger.info("Results: %s", results_dir)


if __name__ == "__main__":
    main()
