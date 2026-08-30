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


def load_multi_speaker_config(config_path: Optional[str] = None, *, methods_required: bool = True) -> dict:
    """Load multi-speaker preprocessing config from a YAML file.

    ``config_path=None`` resolves to the packaged
    ``psdn_sonar/multi_speaker_config.yaml`` and tolerates a missing or
    unreadable file by falling back to defaults, so a damaged install still
    runs. A caller that *names* a file gets that file or an error: falling back
    there would evaluate with a configuration nobody asked for and report the
    result as if it were the requested one.

    Malformed structure is an error either way — ``methods: 5`` and
    ``silence: oops`` used to escape as a raw ``TypeError`` from the iteration
    and the merge. Unknown method *names* are still skipped with a warning.

    ``methods_required=False`` says the caller is replacing the method list
    anyway (``--methods`` / ``--method``), so a file whose own list holds
    nothing usable returns ``methods: []`` — with its settings intact — rather
    than failing. Blocking there would make an override unusable against a
    config with a stale method list, which is one of the things an override is
    for. Structural validation still applies.

    Returns a dict with ``methods`` plus per-method settings sections.
    """
    explicit = config_path is not None
    if config_path is None:
        # __file__ is psdn_sonar/preprocessing/config_loader.py
        # multi_speaker_config.yaml lives at psdn_sonar/multi_speaker_config.yaml
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(package_root, "multi_speaker_config.yaml")

    def _defaults_or_raise(reason: str) -> dict:
        if explicit:
            raise ValueError(f"Cannot use preprocessing config {config_path}: {reason}")
        logger.warning(f"{reason}, using defaults")
        return {"methods": list(DEFAULT_METHODS), **DEFAULT_SETTINGS}

    if not os.path.exists(config_path):
        if explicit:
            raise FileNotFoundError(f"Preprocessing config not found: {config_path}")
        logger.warning(f"Config not found at {config_path}, using defaults")
        return {"methods": list(DEFAULT_METHODS), **DEFAULT_SETTINGS}

    try:
        import yaml
    except ImportError:
        return _defaults_or_raise("pyyaml is not installed")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return _defaults_or_raise(f"Error reading config {config_path}: {e}")

    # Only an empty document is an empty config. A blanket ``or {}`` also
    # swallowed ``[]``, ``false``, ``0`` and ``""``, which then sailed past the
    # mapping check below and silently produced the default configuration —
    # the same "named a file, evaluated with something else" failure this
    # function exists to prevent.
    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise ValueError(f"{config_path}: top level must be a mapping, got {type(data).__name__}")

    sections = {}
    for name in ("silence", "timestamp", "pyannote"):
        section = data.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{config_path}: '{name}' must be a mapping, got {type(section).__name__}")
        sections[name] = {**DEFAULT_SETTINGS[name], **section}

    methods = data.get("methods", DEFAULT_METHODS)
    if not isinstance(methods, list) or not all(isinstance(m, str) for m in methods):
        raise ValueError(f"{config_path}: 'methods' must be a list of strings, got {methods!r}")

    validated_methods = []
    for m in methods:
        if m in KNOWN_METHODS:
            validated_methods.append(m)
        else:
            logger.warning(f"Unknown method '{m}' in config, skipping")

    if not validated_methods and methods_required:
        return _defaults_or_raise(f"no known methods in {config_path}")

    return {"methods": validated_methods, **sections}
