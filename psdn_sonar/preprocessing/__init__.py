"""Audio preprocessing for multi-speaker evaluation."""

from psdn_sonar.preprocessing.audio_utils import (
    get_audio_duration,
    get_combined_audio_path,
    parse_timestamp,
    trim_by_timestamps,
    trim_silence,
)
from psdn_sonar.preprocessing.config_loader import (
    DEFAULT_METHODS,
    KNOWN_METHODS,
    load_multi_speaker_config,
)
from psdn_sonar.preprocessing.methods import (
    PER_CHANNEL_METHODS,
    PER_CLIP_METHODS,
    PYANNOTE_METHODS,
    dual_assignment_score,
    preprocess_energy_trim,
    preprocess_no_trim,
    preprocess_pyannote_vad,
    preprocess_timestamp_trim,
    run_pyannote_diarize,
    run_scribe_diarize,
)
from psdn_sonar.preprocessing.preprocessing_selector import (
    run_single_method,
    run_sweep,
    select_best_preprocessing,
)
from psdn_sonar.preprocessing.pyannote_utils import (
    PYANNOTE_AVAILABLE,
    assign_words_to_speakers,
    extract_and_concat_segments,
)

__all__ = [
    "DEFAULT_METHODS",
    "KNOWN_METHODS",
    "PER_CHANNEL_METHODS",
    "PER_CLIP_METHODS",
    "PYANNOTE_AVAILABLE",
    "PYANNOTE_METHODS",
    "assign_words_to_speakers",
    "dual_assignment_score",
    "extract_and_concat_segments",
    "get_audio_duration",
    "get_combined_audio_path",
    "load_multi_speaker_config",
    "parse_timestamp",
    "preprocess_energy_trim",
    "preprocess_no_trim",
    "preprocess_pyannote_vad",
    "preprocess_timestamp_trim",
    "run_pyannote_diarize",
    "run_scribe_diarize",
    "run_single_method",
    "run_sweep",
    "select_best_preprocessing",
    "trim_by_timestamps",
    "trim_silence",
]
