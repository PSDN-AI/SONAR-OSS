"""Discover which public HuggingFace datasets are available for a language."""

from __future__ import annotations

import logging
from typing import Optional

from psdn_sonar.language_codes import LANG_CODE_TO_NAME

from .registry import (
    _DATASET_LANG_GATES,
    DATASET_REGISTRY,
    AvailableDataset,
    resolve_config,
)

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Discovers available public datasets for a given language code."""

    @staticmethod
    def discover(
        language: str,
        dataset_filter: Optional[list[str]] = None,
        validate_remote: bool = False,
    ) -> list[AvailableDataset]:
        """Return datasets available for *language*.

        Parameters
        ----------
        language:
            ISO 639-1 language code (e.g. ``"ur"``).
        dataset_filter:
            If given, only check these dataset names (e.g. ``["common_voice", "fleurs"]``).
        validate_remote:
            If ``True``, attempt to load dataset metadata from HuggingFace Hub
            to confirm the config actually exists.  Requires network access.
        """
        language = language.lower()
        if language not in LANG_CODE_TO_NAME:
            logger.warning(
                "Language code '%s' is not in the known language list. "
                "Discovery will still attempt matching but results may be incomplete.",
                language,
            )

        results: list[AvailableDataset] = []

        for ds_name, spec in DATASET_REGISTRY.items():
            if dataset_filter and ds_name not in dataset_filter:
                continue

            lang_gate = _DATASET_LANG_GATES.get(ds_name)
            if lang_gate is not None and language not in lang_gate:
                continue

            config = resolve_config(spec, language)
            if config is None:
                continue

            if validate_remote and config and not _remote_config_exists(spec.hf_id, config):
                logger.info("  %s/%s: not found on HuggingFace Hub", spec.hf_id, config)
                continue

            num_examples = None
            if validate_remote:
                num_examples = _get_split_sizes(spec.hf_id, config, list(spec.splits))

            results.append(
                AvailableDataset(
                    name=ds_name,
                    hf_id=spec.hf_id,
                    config=config,
                    splits=list(spec.splits),
                    num_examples=num_examples,
                    text_column=spec.text_column,
                    audio_column=spec.audio_column,
                )
            )

        return results

    @staticmethod
    def print_summary(datasets: list[AvailableDataset], language: str) -> None:
        """Print a human-readable table of discovered datasets."""
        lang_name = LANG_CODE_TO_NAME.get(language, language)
        print(f"\nDatasets available for {lang_name} ({language}):\n")

        if not datasets:
            print("  (none found)\n")
            return

        header = f"  {'Dataset':<30} {'HuggingFace ID':<45} {'Config':<20} {'Splits'}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for ds in datasets:
            splits_str = ", ".join(ds.splits)
            sizes = ""
            if ds.num_examples:
                sizes = " (" + ", ".join(f"{s}={n}" for s, n in ds.num_examples.items()) + ")"
            print(f"  {ds.name:<30} {ds.hf_id:<45} {ds.config or '(none)':<20} {splits_str}{sizes}")

        print()


def _remote_config_exists(hf_id: str, config: str) -> bool:
    """Check whether a HF dataset config actually exists (lightweight check)."""
    if not config:
        return True
    try:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(hf_id)
        return config in configs
    except Exception:
        return False


def _get_split_sizes(hf_id: str, config: str, splits: list[str]) -> Optional[dict[str, int]]:
    """Try to get the number of examples per split without downloading data."""
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(hf_id, config if config else None)
        info = builder.info
        if info.splits is None:
            return None
        return {s: info.splits[s].num_examples for s in splits if s in info.splits}
    except Exception:
        return None
