"""Discover which public HuggingFace datasets are available for a language."""

from __future__ import annotations

import logging
from typing import Optional

from psdn_sonar.language_codes import LANG_CODE_TO_NAME

from .registry import (
    _BENCHMARK_CATALOG,
    _DATASET_LANG_GATES,
    DATASET_REGISTRY,
    FLEURS_CONFIG,
    AvailableDataset,
    resolve_config,
)

logger = logging.getLogger(__name__)


def validate_dataset_filter(dataset_filter: list[str]) -> None:
    """Reject filter entries that can never match, with one reason per entry.

    A typo, a catalogued-but-disabled benchmark, and a catalogued non-HF
    source used to behave identically (silently matching nothing), so the
    caller could not tell them apart. Raises ``ValueError`` naming each bad
    entry and why it cannot be discovered.
    """
    discoverable = ", ".join(sorted(DATASET_REGISTRY))
    problems: list[str] = []
    for name in dataset_filter:
        if name in DATASET_REGISTRY:
            continue
        benchmark = _BENCHMARK_CATALOG.benchmarks.get(name)
        if benchmark is None:
            problems.append(f"'{name}': unknown dataset name. Discoverable datasets: {discoverable}.")
        elif not benchmark.enabled:
            problems.append(
                f"'{name}' ({benchmark.display_name}): catalogued but disabled "
                f"(review decision: {benchmark.review.decision}), so it cannot be discovered or prepared."
            )
        elif benchmark.source.kind != "huggingface":
            problems.append(
                f"'{name}' ({benchmark.display_name}): catalogued as an {benchmark.source.kind} source; "
                "`discover` covers HuggingFace-hosted sources only. "
                "Load it with the psdn_sonar.core library loaders instead."
            )
        else:
            problems.append(
                f"'{name}' ({benchmark.display_name}): catalogued but not wired into `discover`. "
                f"Discoverable datasets: {discoverable}."
            )
    if problems:
        raise ValueError("Invalid --datasets entries:\n  " + "\n  ".join(problems))


def dataset_language_support(ds_name: str) -> str:
    """Short human-readable hint of which languages a discoverable dataset serves."""
    gate = _DATASET_LANG_GATES.get(ds_name)
    if gate is not None:
        return f"{ds_name} supports: {', '.join(sorted(gate))}"
    spec = DATASET_REGISTRY.get(ds_name)
    if spec is not None and "{fleurs}" in spec.config_template:
        return f"{ds_name} supports: {', '.join(sorted(FLEURS_CONFIG))}"
    return f"{ds_name} has no language gate"


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
            If given, only check these dataset names (e.g. ``["fleurs", "voxpopuli"]``).
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

        if dataset_filter:
            validate_dataset_filter(dataset_filter)

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

            if validate_remote and config and not _remote_config_exists(spec.hf_id, config, spec.revision):
                logger.info("  %s/%s: not found on HuggingFace Hub", spec.hf_id, config)
                continue

            num_examples = None
            if validate_remote:
                num_examples = _get_split_sizes(spec.hf_id, config, list(spec.splits), spec.revision)

            results.append(
                AvailableDataset(
                    name=ds_name,
                    hf_id=spec.hf_id,
                    config=config,
                    revision=spec.revision,
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
        print(f"\nDatasets available for {lang_name} ({language}) via `discover` (HuggingFace-hosted sources):\n")

        if not datasets:
            print("  (none found)\n")
        else:
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

        _print_catalog_scope_note()


def _print_catalog_scope_note() -> None:
    """Name the catalogued benchmarks `discover` cannot reach.

    Without this, ``discover --language bn`` listing only FLEURS reads as
    "FLEURS is all there is for Bengali" while the catalog also holds the
    three OpenSLR Bengali corpora used on the public leaderboard.
    """
    other = [(name, spec) for name, spec in _BENCHMARK_CATALOG.benchmarks.items() if name not in DATASET_REGISTRY]
    if not other:
        return
    print("  The benchmark catalog also holds entries this command cannot discover or prepare:")
    for name, spec in sorted(other, key=lambda item: item[0]):
        if not spec.enabled:
            reason = f"disabled in the catalog (review decision: {spec.review.decision})"
        elif spec.source.kind != "huggingface":
            reason = f"{spec.source.kind} source — use the psdn_sonar.core library loaders"
        else:
            reason = "not wired into `discover`"
        print(f"    {name:<28} {spec.display_name:<38} ({reason})")
    print()


def _remote_config_exists(hf_id: str, config: str, revision: str) -> bool:
    """Check whether a HF dataset config actually exists (lightweight check)."""
    if not config:
        return True
    try:
        from datasets import get_dataset_config_names

        configs = get_dataset_config_names(hf_id, revision=revision)
        return config in configs
    except Exception:
        return False


def _get_split_sizes(hf_id: str, config: str, splits: list[str], revision: str) -> Optional[dict[str, int]]:
    """Try to get the number of examples per split without downloading data."""
    try:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(hf_id, config if config else None, revision=revision)
        info = builder.info
        if info.splits is None:
            return None
        return {s: info.splits[s].num_examples for s in splits if s in info.splits}
    except Exception:
        return None
