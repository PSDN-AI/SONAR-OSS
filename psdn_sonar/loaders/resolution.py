"""Resolve dataset root directories under a base directory.

Each dataset is tried against its known folder-name variants and validated
against the files/dirs its loader expects before being accepted.
"""

import logging
import os
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _has_common_voice_layout(d: str) -> bool:
    return os.path.exists(os.path.join(d, "bn", "test.tsv")) and os.path.isdir(os.path.join(d, "bn", "clips"))


def _has_fleurs_layout(d: str) -> bool:
    return os.path.exists(os.path.join(d, "test", "test.tsv")) and os.path.isdir(os.path.join(d, "test", "audio"))


def _has_line_index_layout(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "line_index.tsv")) and os.path.isdir(os.path.join(d, "wavs"))


def _has_utt_spk_text_layout(d: str) -> bool:
    return os.path.exists(os.path.join(d, "utt_spk_text.tsv"))


# dataset name -> (candidate folder names, layout validator)
_DATASET_DIRS: Dict[str, Tuple[List[str], Callable[[str], bool]]] = {
    "commonvoice": (["Common_Voice", "commonvoice", "Common-Voice"], _has_common_voice_layout),
    "fleurs": (["Fleurs", "fleurs", "fleurs_bn_in", "Fleurs_bn_in"], _has_fleurs_layout),
    "openslr37_bd": (["OpenSLR37_BD", "openslr37_bd", "OpenSLR37-BD", "openslr37-bd"], _has_line_index_layout),
    "openslr37_in": (["OpenSLR37_IN", "openslr37_in", "OpenSLR37-IN", "openslr37-in"], _has_line_index_layout),
    "openslr53": (["OpenSLR53", "openslr53", "OpenSLR-53", "openslr-53"], _has_utt_spk_text_layout),
}

_EXPECTED_STRUCTURES: Dict[str, str] = {
    "commonvoice": "Common_Voice/bn/test.tsv, bn/clips/*.mp3",
    "fleurs": "Fleurs/test/test.tsv, test/audio/*.wav",
    "openslr37_bd": "OpenSLR37_BD/line_index.tsv, wavs/*.wav (or utt_spk_text + asr_bengali_*)",
    "openslr37_in": "OpenSLR37_IN/line_index.tsv, wavs/*.wav",
    "openslr53": "OpenSLR53/utt_spk_text.tsv, asr_bengali_0..8/**/*.flac",
}


def resolve_dataset_dir(base_dir: str, dataset_name: str) -> Optional[str]:
    """Return the resolved dataset root under ``base_dir``, or ``None``.

    Tries the dataset's known folder-name variants and validates the
    expected on-disk layout. Common Voice additionally accepts any
    ``cv-corpus*`` folder.
    """
    base_dir = os.path.abspath(base_dir)
    if not os.path.isdir(base_dir):
        return None

    entry = _DATASET_DIRS.get(dataset_name)
    if entry is None:
        return None

    candidates, is_valid = entry
    for name in candidates:
        d = os.path.join(base_dir, name)
        if os.path.isdir(d) and is_valid(d):
            return d

    if dataset_name == "commonvoice":
        for item in os.listdir(base_dir):
            p = os.path.join(base_dir, item)
            if os.path.isdir(p) and "cv-corpus" in item.lower() and _has_common_voice_layout(p):
                return p

    return None


def print_expected_structure(dataset_name: str) -> None:
    """Log the expected folder structure for a dataset."""
    msg = _EXPECTED_STRUCTURES.get(dataset_name, "?")
    logger.info("  Expected: {base}/%s", msg)
