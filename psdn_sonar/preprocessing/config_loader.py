"""Config loader for multi-speaker preprocessing trim/diarization settings.

Run-level options (``run.seed``, pipeline steps) come from
``psdn_sonar.config_loader`` / ``conf/config.yaml``. POSEIDON score weights
remain in ``psdn_sonar.config`` (environment variables).
"""

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

KNOWN_METHODS = {
    "energy_trim",
    "timestamp_trim",
    "no_trim",
    "pyannote_vad",
    "scribe_diarize",
    "pyannote_diarize",
}

DEFAULT_METHODS = ["energy_trim", "timestamp_trim", "no_trim"]

DEFAULT_SETTINGS = {
    "silence": {"max_silence_ms": 400, "min_silence_len": 500, "silence_thresh": -40},
    "timestamp": {"padding_ms": 100},
    "pyannote": {"vad_min_duration_on": 0.3, "vad_min_duration_off": 0.3, "vad_gap_ms": 400},
}


def load_multi_speaker_config(config_path: Optional[str] = None) -> dict:
    """Load multi-speaker preprocessing config from a YAML file.

    Defaults to ``psdn_sonar/multi_speaker_config.yaml``. Unknown methods are
    skipped with a warning; missing files or settings fall back to defaults.
    Returns a dict with ``methods`` plus per-method settings sections.
    """
    if config_path is None:
        # __file__ is psdn_sonar/preprocessing/config_loader.py
        # multi_speaker_config.yaml lives at psdn_sonar/multi_speaker_config.yaml
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_root, "multi_speaker_config.yaml")

    if not os.path.exists(config_path):
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {"methods": list(DEFAULT_METHODS), **DEFAULT_SETTINGS}

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed, using default config")
        return {"methods": list(DEFAULT_METHODS), **DEFAULT_SETTINGS}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Error reading config {config_path}: {e}, using defaults")
        return {"methods": list(DEFAULT_METHODS), **DEFAULT_SETTINGS}

    methods = data.get("methods", DEFAULT_METHODS)
    validated_methods = []
    for m in methods:
        if m in KNOWN_METHODS:
            validated_methods.append(m)
        else:
            logger.warning(f"Unknown method '{m}' in config, skipping")

    if not validated_methods:
        logger.warning("No valid methods in config, using defaults")
        validated_methods = list(DEFAULT_METHODS)

    return {
        "methods": validated_methods,
        "silence": {**DEFAULT_SETTINGS["silence"], **data.get("silence", {})},
        "timestamp": {**DEFAULT_SETTINGS["timestamp"], **data.get("timestamp", {})},
        "pyannote": {**DEFAULT_SETTINGS["pyannote"], **data.get("pyannote", {})},
    }
