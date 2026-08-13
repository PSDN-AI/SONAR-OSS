"""Preprocessing methods for multi-speaker evaluation.

Per-channel methods preprocess one speaker's audio and return a path, so they
work with every ASR model. Per-clip methods transcribe the combined audio and
require diarization support from the model.
"""

import logging
from pathlib import Path

from .audio_utils import get_audio_duration, trim_by_timestamps, trim_silence

logger = logging.getLogger(__name__)

# Method-name sets are derived from the strategy registries at the bottom of
# this module; adding a method means adding one registry entry.


def preprocess_energy_trim(
    audio_path,
    max_silence_ms: int = 400,
    min_silence_len: int = 500,
    silence_thresh: int = -40,
) -> tuple:
    """Energy-threshold silence removal.

    Returns ``(processed_path, original_duration_s, trimmed_duration_s)``.
    """
    return trim_silence(
        Path(audio_path),
        max_silence_ms=max_silence_ms,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )


def preprocess_timestamp_trim(
    audio_path,
    segments: list,
    speaker: str,
    padding_ms: int = 100,
) -> tuple:
    """Extract audio segments by transcript timestamps for speaker "A" or "B".

    Returns ``(processed_path, original_duration_s, trimmed_duration_s)``.
    """
    return trim_by_timestamps(
        Path(audio_path),
        segments,
        speaker,
        padding_ms=padding_ms,
    )


def preprocess_no_trim(audio_path) -> tuple:
    """No preprocessing — return ``(audio_path, duration_s, duration_s)``."""
    dur = get_audio_duration(audio_path)
    return audio_path, dur, dur


def preprocess_pyannote_vad(audio_path, gap_ms: int = 400) -> tuple:
    """Neural VAD (pyannote/segmentation-3.0) on a single channel.

    Returns ``(processed_path, original_duration_s, trimmed_duration_s)``.
    """
    from .pyannote_utils import extract_and_concat_segments, run_vad_on_channel

    speech_segments = run_vad_on_channel(Path(audio_path))
    return extract_and_concat_segments(Path(audio_path), speech_segments, gap_ms=gap_ms)


def run_scribe_diarize(combined_audio, asr_model) -> dict:
    """Diarized transcription of combined audio via the model's own diarization.

    Requires ``asr_model.transcribe_diarized()``. Returns ``{speaker_id: text}``.
    """
    return asr_model.transcribe_diarized(str(combined_audio), num_speakers=2)


def run_pyannote_diarize(combined_audio, asr_model) -> dict:
    """Pyannote speaker diarization + ASR word timestamps on combined audio.

    Requires ``asr_model.transcribe_with_word_timestamps()``.
    Returns ``{speaker_id: text}``.
    """
    from .pyannote_utils import assign_words_to_speakers, run_diarization

    words = asr_model.transcribe_with_word_timestamps(str(combined_audio))
    diar_segments = run_diarization(Path(combined_audio), num_speakers=2)
    return assign_words_to_speakers(words, diar_segments)


def dual_assignment_score(
    speaker_texts: dict,
    ref_a: str,
    ref_b: str,
    metric_fn,
) -> tuple:
    """Try both speaker-to-reference assignments and keep the better one.

    *metric_fn* is a ``Callable(ref, hyp) -> (cer, wer, similarity, poseidon)``.
    Returns ``(result_a, result_b)`` dicts with ``text``/``cer``/``wer``/
    ``similarity`` keys (plus ``error`` when a side has no assigned speaker).
    """
    sids = list(speaker_texts.keys())

    def _empty_result(error_msg):
        return {"text": "", "cer": None, "wer": None, "similarity": None, "error": error_msg}

    def _score(ref, hyp):
        cer, wer, sim, poseidon = metric_fn(ref, hyp)
        return {"cer": cer, "wer": wer, "similarity": sim}

    def _heuristic(scores):
        """Selection heuristic: avg((1-cer) + (1-wer) + sim) / 3."""
        cer = scores.get("cer") or 1.0
        wer = scores.get("wer") or 1.0
        sim = scores.get("similarity") or 0.0
        return ((1 - cer) + (1 - wer) + sim) / 3

    if len(sids) == 0:
        return (
            _empty_result("No speakers detected"),
            _empty_result("No speakers detected"),
        )

    if len(sids) == 1:
        text = speaker_texts[sids[0]]
        scores_a = _score(ref_a, text) if ref_a else {"cer": None, "wer": None, "similarity": None}
        scores_b = _score(ref_b, text) if ref_b else {"cer": None, "wer": None, "similarity": None}
        sim_a = scores_a.get("similarity") or 0.0
        sim_b = scores_b.get("similarity") or 0.0

        if sim_a >= sim_b:
            return (
                {"text": text, **scores_a},
                _empty_result("Single speaker detected, assigned to A"),
            )
        else:
            return (
                _empty_result("Single speaker detected, assigned to B"),
                {"text": text, **scores_b},
            )

    # 2+ speakers: try both assignments
    t0 = speaker_texts[sids[0]]
    t1 = speaker_texts[sids[1]]

    # Assignment 1: sids[0]=A, sids[1]=B
    a1_scores_a = _score(ref_a, t0)
    a1_scores_b = _score(ref_b, t1)
    a1_avg = (_heuristic(a1_scores_a) + _heuristic(a1_scores_b)) / 2

    # Assignment 2: sids[0]=B, sids[1]=A
    a2_scores_a = _score(ref_a, t1)
    a2_scores_b = _score(ref_b, t0)
    a2_avg = (_heuristic(a2_scores_a) + _heuristic(a2_scores_b)) / 2

    if a1_avg >= a2_avg:
        return (
            {"text": t0, **a1_scores_a},
            {"text": t1, **a1_scores_b},
        )
    else:
        return (
            {"text": t1, **a2_scores_a},
            {"text": t0, **a2_scores_b},
        )


# ---------------------------------------------------------------------------
# Strategy registries (Strategy pattern)
#
# Per-channel strategies share the signature
#   (audio_path, speaker, segments, silence_settings, timestamp_settings,
#    pyannote_settings) -> (processed_path, original_duration_s, trimmed_duration_s)
# and per-clip strategies share
#   (combined_audio, asr_model) -> {speaker_id: text}
# The method-name sets are derived from the registries, so registering a new
# strategy here is the only change needed to make it selectable.
# ---------------------------------------------------------------------------


def _energy_trim_strategy(audio_path, speaker, segments, silence_settings, timestamp_settings, pyannote_settings):
    return preprocess_energy_trim(
        audio_path,
        max_silence_ms=silence_settings.get("max_silence_ms", 400),
        min_silence_len=silence_settings.get("min_silence_len", 500),
        silence_thresh=silence_settings.get("silence_thresh", -40),
    )


def _timestamp_trim_strategy(audio_path, speaker, segments, silence_settings, timestamp_settings, pyannote_settings):
    if not segments:
        dur = get_audio_duration(audio_path)
        return audio_path, dur, dur
    return preprocess_timestamp_trim(
        audio_path,
        segments,
        speaker,
        padding_ms=timestamp_settings.get("padding_ms", 100),
    )


def _no_trim_strategy(audio_path, speaker, segments, silence_settings, timestamp_settings, pyannote_settings):
    return preprocess_no_trim(audio_path)


def _pyannote_vad_strategy(audio_path, speaker, segments, silence_settings, timestamp_settings, pyannote_settings):
    return preprocess_pyannote_vad(
        audio_path,
        gap_ms=pyannote_settings.get("vad_gap_ms", 400),
    )


PER_CHANNEL_STRATEGIES = {
    "energy_trim": _energy_trim_strategy,
    "timestamp_trim": _timestamp_trim_strategy,
    "no_trim": _no_trim_strategy,
    "pyannote_vad": _pyannote_vad_strategy,
}

PER_CLIP_STRATEGIES = {
    "scribe_diarize": run_scribe_diarize,
    "pyannote_diarize": run_pyannote_diarize,
}

# Methods that operate on per-channel audio (one API call per speaker)
PER_CHANNEL_METHODS = frozenset(PER_CHANNEL_STRATEGIES)
# Methods that operate on combined audio (one API call for both speakers)
PER_CLIP_METHODS = frozenset(PER_CLIP_STRATEGIES)
# Methods requiring pyannote
PYANNOTE_METHODS = frozenset({"pyannote_vad", "pyannote_diarize"})
