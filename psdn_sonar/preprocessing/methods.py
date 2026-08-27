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

    Raises when either half produces nothing: a run that cannot attribute
    words must fail with the reason rather than hand every word to one
    speaker and drop the other from the evaluation (issue #189).
    """
    from .pyannote_utils import assign_words_to_speakers, run_diarization

    words = asr_model.transcribe_with_word_timestamps(str(combined_audio))
    if not words:
        cause = getattr(asr_model, "last_transcribe_error", None)
        raise RuntimeError(
            "pyannote_diarize needs word timestamps to attribute words to speakers, but "
            f"{type(asr_model).__name__} returned none for {Path(combined_audio).name}"
            + (f": {cause}" if cause else " (no error was recorded by the adapter)")
        )

    diar_segments = run_diarization(Path(combined_audio), num_speakers=2)
    if not diar_segments:
        raise RuntimeError(
            f"pyannote diarization found no speech turns in {Path(combined_audio).name}, so the "
            f"{len(words)} transcribed word(s) cannot be attributed to speakers. Check that the "
            "combined audio actually contains speech."
        )

    speaker_texts = assign_words_to_speakers(words, diar_segments)
    speakers_found = len({seg["speaker"] for seg in diar_segments})
    if speakers_found < 2:
        raise RuntimeError(
            f"pyannote diarization separated only {speakers_found} speaker(s) in "
            f"{Path(combined_audio).name} where 2 were requested, so one speaker's reference "
            "would be scored against the other's words. Not scoring this clip rather than "
            "charging the whole transcript to one speaker."
        )
    return speaker_texts


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
        """Selection heuristic: avg((1-cer) + (1-wer) + sim) / 3.

        Missing metrics count as worst case via explicit None checks so a
        legitimate 0.0 survives — ``or`` turned a perfect CER/WER of 0.0
        into 1.0, which could flip the winning assignment and charge both
        speakers the swapped pairing's error rates (issue #106).
        """
        cer = scores.get("cer")
        wer = scores.get("wer")
        sim = scores.get("similarity")
        cer = 1.0 if cer is None else cer
        wer = 1.0 if wer is None else wer
        sim = 0.0 if sim is None else sim
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
        sim_a = scores_a.get("similarity")
        sim_b = scores_b.get("similarity")
        sim_a = 0.0 if sim_a is None else sim_a
        sim_b = 0.0 if sim_b is None else sim_b

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

# Model capability each per-clip method needs, checked before the method runs.
# Without this, an adapter that does not implement the required method failed
# on the bare NotImplementedError, whose str() is "" — so the run reported
# `pyannote_diarize failed:` with no reason at all (issue #189).
PER_CLIP_REQUIRED_CAPABILITY = {
    "scribe_diarize": "supports_diarization",
    "pyannote_diarize": "supports_word_timestamps",
}

# Methods that operate on per-channel audio (one API call per speaker)
PER_CHANNEL_METHODS = frozenset(PER_CHANNEL_STRATEGIES)
# Methods that operate on combined audio (one API call for both speakers)
PER_CLIP_METHODS = frozenset(PER_CLIP_STRATEGIES)
# Methods requiring pyannote
PYANNOTE_METHODS = frozenset({"pyannote_vad", "pyannote_diarize"})
