"""Verify that an installed psdn-sonar wheel contains its runtime resources."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

import yaml

import psdn_sonar
from psdn_sonar.config_loader import ConfigManager


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

    source_config_root = source_root / "psdn_sonar" / "conf"
    source_configs = sorted(source_config_root.rglob("*.yaml"))
    if not source_configs:
        raise RuntimeError(f"No source YAML configuration found under {source_config_root}")

    for source_config in source_configs:
        relative_path = source_config.relative_to(source_config_root)
        installed_config = package_root / "conf" / relative_path
        if not installed_config.is_file():
            raise RuntimeError(f"Installed wheel is missing configuration: {relative_path}")
        if installed_config.read_bytes() != source_config.read_bytes():
            raise RuntimeError(f"Installed configuration differs from source: {relative_path}")
        if not isinstance(yaml.safe_load(installed_config.read_text(encoding="utf-8")), dict):
            raise RuntimeError(f"Installed configuration is not a YAML mapping: {relative_path}")

    for language in ("bn", "hi", "ko"):
        source_cache = source_root / "config" / "language" / language / "loanword_cache.json"
        installed_cache = package_root / "resources" / "language" / language / "loanword_cache.json"
        if not installed_cache.is_file():
            raise RuntimeError(f"Installed wheel is missing the {language} loanword cache")
        if installed_cache.read_bytes() != source_cache.read_bytes():
            raise RuntimeError(f"Installed {language} loanword cache differs from source")
        cache = json.loads(installed_cache.read_text(encoding="utf-8"))
        if not cache or not all(isinstance(key, str) and isinstance(value, str) for key, value in cache.items()):
            raise RuntimeError(f"Installed {language} loanword cache is invalid")

    config = ConfigManager().load(language="bn", backend="huggingface", validation="strict")
    if (
        config.language.code != "bn"
        or config.backend.name != "huggingface"
        or config.validation.schema.mode != "strict"
    ):
        raise RuntimeError("Installed package could not load the expected merged configuration")

    print(
        f"Verified psdn-sonar {installed_version} at {package_root}: "
        f"{len(source_configs)} YAML configs and 3 loanword caches"
    )


if __name__ == "__main__":
    main()
