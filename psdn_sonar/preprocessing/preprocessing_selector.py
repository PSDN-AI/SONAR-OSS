"""Preprocessing method selection for multi-speaker evaluation.

- auto (default): all per-channel methods run in parallel, scored by audio-signal
  quality (no ASR, no ground truth); the best is sent to ASR once per speaker.
- ``--method <name>``: use one explicit method for all clips (skips scoring).
- ``--sweep``: oracle mode — ground-truth scoring; inflates metrics, ablation only.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

from .audio_utils import get_audio_duration
from .methods import (
    PER_CHANNEL_METHODS,
    PER_CHANNEL_STRATEGIES,
    PER_CLIP_METHODS,
    PER_CLIP_REQUIRED_CAPABILITY,
    PER_CLIP_STRATEGIES,
    PYANNOTE_METHODS,
    dual_assignment_score,
)
from .pyannote_utils import PYANNOTE_AVAILABLE

logger = logging.getLogger(__name__)

_MAX_TRIM_RATIO = 0.85


def _exc_reason(exc: Exception) -> str:
    """Readable reason for *exc*, never the empty string.

    ``str(NotImplementedError())`` is ``""``, so a failure on an unimplemented
    adapter method used to be reported as "<method> failed:" with nothing after
    the colon (issue #189).
    """
    return str(exc) or type(exc).__name__


def _unsupported_capability_error(method_name: str, asr_model) -> Optional[str]:
    """Why *asr_model* cannot run *method_name*, or ``None`` when it can.

    The capability properties existed but nothing read them, so an
    unsupported model reached the strategy and failed on a bare
    ``NotImplementedError`` (issue #189).
    """
    capability = PER_CLIP_REQUIRED_CAPABILITY.get(method_name)
    if capability is None or getattr(asr_model, capability, False):
        return None
    return (
        f"{type(asr_model).__name__} does not support {method_name}: it requires "
        f"{capability}, which this adapter does not implement. Use a model that does "
        "(elevenlabs_api is the only registered adapter with word timestamps), or pick "
        "a per-channel method such as --method timestamp_trim."
    )


def _preprocess_only(
    method_name: str,
    audio_path,
    speaker: str,
    segments: list,
    silence_settings: dict,
    timestamp_settings: dict,
    pyannote_settings: dict,
) -> tuple:
    """Preprocess audio with one per-channel strategy — no ASR involved.

    Returns ``(processed_path, original_duration_s, trimmed_duration_s)``.
    ``timestamp_trim`` without segments degrades to a passthrough.
    """
    strategy = PER_CHANNEL_STRATEGIES.get(method_name)
    if strategy is None:
        raise ValueError(f"Unknown per-channel method: {method_name}")
    return strategy(audio_path, speaker, segments, silence_settings, timestamp_settings, pyannote_settings)


def _score_preprocessed(processed_path, orig_dur: float, trim_dur: float) -> float:
    """Score preprocessed audio: ``(1 - silence_ratio) × (1 - duration_change)``.

    Higher is better. Trimming more than ``_MAX_TRIM_RATIO`` of the original
    duration scores 0; an unanalysable file assumes a 0.5 silence ratio.
    """
    if orig_dur <= 0:
        return 0.0

    duration_change = (orig_dur - trim_dur) / orig_dur
    if duration_change > _MAX_TRIM_RATIO:
        return 0.0

    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_file(str(processed_path))
        total_ms = len(audio)
        if total_ms == 0:
            return 0.0
        nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-40)
        speech_ms = sum(end - start for start, end in nonsilent)
        # Absolute, VAD-based silence fraction used only for internal method
        # scoring — a different definition from the frame-relative
        # audio_quality.calculate_silence_ratio() reported in results CSVs.
        silence_ratio = 1.0 - (speech_ms / total_ms)
    except Exception:
        silence_ratio = 0.5

    return (1.0 - silence_ratio) * (1.0 - duration_change)


def _remove_temp_file(path, original) -> None:
    """Delete *path* if it is a temp artifact distinct from *original*."""
    if str(path) != str(original) and str(path).startswith("/tmp"):
        try:
            if os.path.exists(str(path)):
                os.remove(str(path))
        except OSError:
            pass


def select_best_preprocessing(
    audio_path,
    speaker: str,
    segments: list,
    active_methods: List[str],
    silence_settings: dict,
    timestamp_settings: dict,
    pyannote_settings: dict,
    max_workers: int = 4,
) -> tuple:
    """Run all per-channel preprocessing methods in parallel and return the best.

    Selection is based purely on audio-signal quality (silence ratio + duration
    preservation) — no ASR calls, no ground truth. Falls back to the untouched
    input when no per-channel method is available or none succeeds.

    Returns ``(best_processed_path, method_name, orig_dur, trim_dur)``.
    """
    per_channel = [m for m in active_methods if m in PER_CHANNEL_METHODS]
    if not per_channel:
        dur = get_audio_duration(str(audio_path))
        return str(audio_path), "no_trim", dur, dur

    def _run(method_name):
        try:
            processed_path, orig_dur, trim_dur = _preprocess_only(
                method_name,
                audio_path,
                speaker,
                segments,
                silence_settings,
                timestamp_settings,
                pyannote_settings,
            )
            score = _score_preprocessed(processed_path, orig_dur, trim_dur)
            return method_name, processed_path, orig_dur, trim_dur, score, None
        except Exception as e:
            return method_name, audio_path, 0.0, 0.0, -1.0, str(e)

    with ThreadPoolExecutor(max_workers=min(max_workers, len(per_channel))) as ex:
        results = list(ex.map(_run, per_channel))

    valid = [(m, path, od, td, score) for m, path, od, td, score, err in results if err is None and score >= 0]

    if not valid:
        dur = get_audio_duration(str(audio_path))
        return str(audio_path), "no_trim", dur, dur

    best_method, best_path, best_orig, best_trim, best_score = max(valid, key=lambda x: x[4])

    logger.debug(
        "select_best_preprocessing [%s/%s]: %s → %s (score=%.3f)",
        getattr(audio_path, "stem", audio_path),
        speaker,
        {m: f"{s:.3f}" for m, _, _, _, s, _ in results if s >= 0},
        best_method,
        best_score,
    )

    for m, path, od, td, score, err in results:
        if m != best_method:
            _remove_temp_file(path, audio_path)

    return str(best_path), best_method, best_orig, best_trim


def run_single_method(
    entry,
    asr_model,
    ref_a: str,
    ref_b: str,
    segments: list,
    audio_a,
    audio_b,
    combined_audio,
    metric_fn,
    transcribe_fn,
    active_methods: List[str],
    method_name: Optional[str] = None,
    silence_settings: Optional[dict] = None,
    timestamp_settings: Optional[dict] = None,
    pyannote_settings: Optional[dict] = None,
) -> tuple:
    """Select preprocessing and transcribe once per speaker.

    ``method_name=None`` scores all per-channel methods and sends the best to
    ASR; a given *method_name* is used directly. Per-clip (diarization) methods
    run only when explicitly requested or no per-channel method is active.
    Metrics are left ``None`` (selection never sees ground truth; the caller
    scores). Returns ``(all_results, best_a, best_b)`` like :func:`run_sweep`.
    """
    silence_settings = silence_settings or {}
    timestamp_settings = timestamp_settings or {}
    pyannote_settings = pyannote_settings or {}
    # timestamp_trim's offsets are combined-timeline; give the strategy the
    # combined recording to cut from when the channel file cannot hold them
    # (issue #205).
    timestamp_settings = {**timestamp_settings, "combined_audio_path": combined_audio}

    all_results = {"A": [], "B": []}

    if method_name in PYANNOTE_METHODS and not PYANNOTE_AVAILABLE:
        # Same actionable message run_sweep emits; recorded as the per-speaker
        # error so the CSV names the missing extra instead of a generic
        # "No module named 'pyannote'" per clip.
        err = f"Skipping {method_name}: pyannote.audio not installed. Install with: pip install 'psdn-sonar[pyannote]'"
        logger.warning(f"{entry.audio_id}: {err}")
        for speaker in ("A", "B"):
            all_results[speaker].append(_error_result(method_name, err))
        return all_results, None, None

    per_channel = [m for m in active_methods if m in PER_CHANNEL_METHODS]
    per_clip = [m for m in active_methods if m in PER_CLIP_METHODS]

    if method_name in PER_CLIP_METHODS or (not per_channel and per_clip):
        clip_method = method_name if method_name in PER_CLIP_METHODS else per_clip[0]
        capability_error = _unsupported_capability_error(clip_method, asr_model)
        if capability_error is not None:
            logger.warning(f"Skipping {clip_method} for {entry.audio_id}: {capability_error}")
            for speaker in ("A", "B"):
                all_results[speaker].append(_error_result(clip_method, capability_error))
        elif combined_audio is None:
            err = "combined audio not found"
            logger.warning(f"Skipping {clip_method} for {entry.audio_id}: {err}")
            for speaker in ("A", "B"):
                all_results[speaker].append(_error_result(clip_method, err))
        else:
            try:
                result_a, result_b = _run_per_clip_method(
                    clip_method, combined_audio, asr_model, ref_a, ref_b, metric_fn
                )
                result_a["method"] = clip_method
                result_b["method"] = clip_method
                all_results["A"].append(result_a)
                all_results["B"].append(result_b)
            except Exception as e:
                reason = _exc_reason(e)
                logger.warning(f"{entry.audio_id} {clip_method} failed: {reason}")
                for speaker in ("A", "B"):
                    all_results[speaker].append(_error_result(clip_method, reason))

        best_a = _first_valid(all_results["A"])
        best_b = _first_valid(all_results["B"])
        return all_results, best_a, best_b

    for speaker, audio_path, ref_text in [("A", audio_a, ref_a), ("B", audio_b, ref_b)]:
        if not per_channel:
            all_results[speaker].append(_error_result("none", "No per-channel methods available"))
            continue

        try:
            if method_name is not None:
                processed_path, orig_dur, trim_dur = _preprocess_only(
                    method_name,
                    audio_path,
                    speaker,
                    segments,
                    silence_settings,
                    timestamp_settings,
                    pyannote_settings,
                )
                selected_method = method_name
            else:
                processed_path, selected_method, orig_dur, trim_dur = select_best_preprocessing(
                    audio_path,
                    speaker,
                    segments,
                    per_channel,
                    silence_settings,
                    timestamp_settings,
                    pyannote_settings,
                )

            text = transcribe_fn(str(processed_path))

            all_results[speaker].append(
                {
                    "text": text,
                    "method": selected_method,
                    "original_duration_s": round(orig_dur, 2),
                    "trimmed_duration_s": round(trim_dur, 2),
                    "cer": None,
                    "wer": None,
                    "similarity": None,
                }
            )

            _remove_temp_file(processed_path, audio_path)

        except Exception as e:
            reason = _exc_reason(e)
            logger.warning(f"{entry.audio_id}/{speaker}: failed: {reason}")
            all_results[speaker].append(_error_result(method_name or "auto", reason))

    best_a = _first_valid(all_results["A"])
    best_b = _first_valid(all_results["B"])
    return all_results, best_a, best_b


def _heuristic(cer, wer, sim):
    """Oracle heuristic ``((1-cer) + (1-wer) + sim) / 3`` — ground-truth-based,
    sweep/ablation mode only.
    """
    cer = cer if cer is not None else 1.0
    wer = wer if wer is not None else 1.0
    sim = sim if sim is not None else 0.0
    return ((1 - cer) + (1 - wer) + sim) / 3


def run_sweep(
    entry,
    asr_model,
    ref_a: str,
    ref_b: str,
    segments: list,
    audio_a,
    audio_b,
    combined_audio,
    metric_fn,
    transcribe_fn,
    methods: List[str],
    silence_settings: Optional[dict] = None,
    timestamp_settings: Optional[dict] = None,
    pyannote_settings: Optional[dict] = None,
) -> tuple:
    """Run all methods with oracle (ground-truth) selection.

    WARNING: selecting the winner by ground truth inflates all reported metrics;
    use only via ``--sweep`` for ablation. Returns ``(all_results, best_a,
    best_b)`` like :func:`run_single_method`.
    """
    silence_settings = silence_settings or {}
    timestamp_settings = timestamp_settings or {}
    pyannote_settings = pyannote_settings or {}
    # Same combined-recording injection as run_single_method (issue #205).
    timestamp_settings = {**timestamp_settings, "combined_audio_path": combined_audio}

    all_results = {"A": [], "B": []}

    for method_name in methods:
        if method_name in PYANNOTE_METHODS and not PYANNOTE_AVAILABLE:
            logger.warning(
                f"Skipping {method_name}: pyannote.audio not installed. "
                "Install with: pip install 'psdn-sonar[pyannote]'"
            )
            continue

        if method_name in PER_CHANNEL_METHODS:
            for speaker, audio_path, ref_text in [("A", audio_a, ref_a), ("B", audio_b, ref_b)]:
                if method_name == "timestamp_trim" and not segments:
                    all_results[speaker].append(_error_result(method_name, "No segments available"))
                    continue
                try:
                    processed_path, orig_dur, trim_dur = _preprocess_only(
                        method_name,
                        audio_path,
                        speaker,
                        segments,
                        silence_settings,
                        timestamp_settings,
                        pyannote_settings,
                    )
                    result = _transcribe_and_score(
                        processed_path, ref_text, transcribe_fn, metric_fn, orig_dur, trim_dur
                    )
                    result["method"] = method_name
                    all_results[speaker].append(result)
                    _remove_temp_file(processed_path, audio_path)
                except Exception as e:
                    reason = _exc_reason(e)
                    logger.warning(f"{entry.audio_id}/{speaker} {method_name} failed: {reason}")
                    all_results[speaker].append(_error_result(method_name, reason))

        elif method_name in PER_CLIP_METHODS:
            # Each per-clip method needs its own capability: scribe_diarize
            # needs the model's diarization, pyannote_diarize needs word
            # timestamps. Checking only the former let a model without word
            # timestamps through to a bare NotImplementedError (issue #189).
            capability_error = _unsupported_capability_error(method_name, asr_model)
            if capability_error is not None:
                logger.warning(f"Skipping {method_name}: {capability_error}")
                continue
            if combined_audio is None:
                logger.warning(f"Skipping {method_name} for {entry.audio_id}: combined audio not found")
                continue
            try:
                result_a, result_b = _run_per_clip_method(
                    method_name, combined_audio, asr_model, ref_a, ref_b, metric_fn
                )
                result_a["method"] = method_name
                result_b["method"] = method_name
                all_results["A"].append(result_a)
                all_results["B"].append(result_b)
            except Exception as e:
                reason = _exc_reason(e)
                logger.warning(f"{entry.audio_id} {method_name} failed: {reason}")
                for speaker in ("A", "B"):
                    all_results[speaker].append(_error_result(method_name, reason))

    best_a = _select_best_oracle(all_results["A"])
    best_b = _select_best_oracle(all_results["B"])
    return all_results, best_a, best_b


def _transcribe_and_score(processed_path, ref_text, transcribe_fn, metric_fn, orig_dur, trim_dur) -> dict:
    """Transcribe preprocessed audio and score it against the reference."""
    text = transcribe_fn(str(processed_path))
    cer, wer, sim, _ = metric_fn(ref_text, text)
    return {
        "text": text,
        "cer": cer,
        "wer": wer,
        "similarity": sim,
        "original_duration_s": round(orig_dur, 2),
        "trimmed_duration_s": round(trim_dur, 2),
    }


def _select_best_oracle(results: list) -> Optional[dict]:
    """Select the result with the highest oracle heuristic score."""
    valid = [r for r in results if r.get("error") is None and r.get("cer") is not None]
    if not valid:
        return results[0] if results else None
    return max(valid, key=lambda r: _heuristic(r.get("cer"), r.get("wer"), r.get("similarity")))


def _error_result(method_name: str, error: str) -> dict:
    return {
        "method": method_name,
        "error": error,
        "cer": None,
        "wer": None,
        "similarity": None,
        "text": "",
        "original_duration_s": 0.0,
        "trimmed_duration_s": 0.0,
    }


def _first_valid(results: list) -> Optional[dict]:
    for r in results:
        if r.get("error") is None:
            return r
    return results[0] if results else None


def _run_per_clip_method(method_name, combined_audio, asr_model, ref_a, ref_b, metric_fn):
    """Diarize-transcribe combined audio and dual-assign the speaker texts."""
    strategy = PER_CLIP_STRATEGIES.get(method_name)
    if strategy is None:
        raise ValueError(f"Unknown per-clip method: {method_name}")
    speaker_texts = strategy(combined_audio, asr_model)

    result_a, result_b = dual_assignment_score(speaker_texts, ref_a, ref_b, metric_fn)
    for r in (result_a, result_b):
        r.setdefault("original_duration_s", 0.0)
        r.setdefault("trimmed_duration_s", 0.0)
    return result_a, result_b
