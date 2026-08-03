"""Verify that an installed psdn-sonar wheel contains its runtime resources."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

import yaml

import psdn_sonar
from psdn_sonar.config_loader import ConfigManager
from psdn_sonar.preprocessing import load_multi_speaker_config


def main() -> None:
    package_root = Path(psdn_sonar.__file__).resolve().parent
    try:
        source_root = Path(os.environ["SONAR_SOURCE_ROOT"]).resolve()
    except KeyError as exc:
        raise RuntimeError("SONAR_SOURCE_ROOT must identify the source checkout") from exc

    if package_root.is_relative_to(source_root):
        raise RuntimeError(f"Imported psdn_sonar from the source checkout: {package_root}")

    installed_version = version("psdn-sonar")
    if psdn_sonar.__version__ != installed_version:
        raise RuntimeError(
            f"Version mismatch: psdn_sonar.__version__={psdn_sonar.__version__!r}, metadata={installed_version!r}"
        )

    # Every runtime YAML in the package tree, not just conf/: files like
    # psdn_sonar/multi_speaker_config.yaml have silent-fallback loaders, so
    # a wheel that drops them still imports fine but changes behavior.
    source_package_root = source_root / "psdn_sonar"
    source_configs = sorted(path for pattern in ("*.yaml", "*.yml") for path in source_package_root.rglob(pattern))
    if not source_configs:
        raise RuntimeError(f"No source YAML configuration found under {source_package_root}")

    for source_config in source_configs:
        relative_path = source_config.relative_to(source_package_root)
        installed_config = package_root / relative_path
        if not installed_config.is_file():
            raise RuntimeError(f"Installed wheel is missing configuration: {relative_path}")
        if installed_config.read_bytes() != source_config.read_bytes():
            raise RuntimeError(f"Installed configuration differs from source: {relative_path}")
        if not isinstance(yaml.safe_load(installed_config.read_text(encoding="utf-8")), dict):
            raise RuntimeError(f"Installed configuration is not a YAML mapping: {relative_path}")

    source_cache_root = source_root / "config" / "language"
    source_caches = sorted(source_cache_root.glob("*/loanword_cache.json"))
    if not source_caches:
        raise RuntimeError(f"No source loanword caches found under {source_cache_root}")

    for source_cache in source_caches:
        language = source_cache.parent.name
        installed_cache = package_root / "resources" / "language" / language / "loanword_cache.json"
        if not installed_cache.is_file():
            raise RuntimeError(f"Installed wheel is missing the {language} loanword cache")
        if installed_cache.read_bytes() != source_cache.read_bytes():
            raise RuntimeError(f"Installed {language} loanword cache differs from source")
        cache = json.loads(installed_cache.read_text(encoding="utf-8"))
        if not isinstance(cache, dict) or not cache or not all(isinstance(value, str) for value in cache.values()):
            raise RuntimeError(f"Installed {language} loanword cache is invalid")

    config = ConfigManager().load(language="bn", backend="huggingface", validation="strict")
    if (
        config.language.code != "bn"
        or config.backend.name != "huggingface"
        or config.validation.schema.mode != "strict"
    ):
        raise RuntimeError("Installed package could not load the expected merged configuration")

    # Behavioral check: load_multi_speaker_config falls back to defaults with
    # only a log warning when its YAML is missing, so byte-comparison alone is
    # not enough — assert the installed loader returns what the file declares.
    declared_methods = yaml.safe_load((package_root / "multi_speaker_config.yaml").read_text(encoding="utf-8"))[
        "methods"
    ]
    loaded_methods = load_multi_speaker_config()["methods"]
    if loaded_methods != declared_methods:
        raise RuntimeError(
            f"Installed multi-speaker config loaded methods {loaded_methods}, "
            f"expected {declared_methods} — the loader fell back to defaults"
        )

    print(
        f"Verified psdn-sonar {installed_version} at {package_root}: "
        f"{len(source_configs)} YAML configs and {len(source_caches)} loanword caches"
    )


if __name__ == "__main__":
    main()
