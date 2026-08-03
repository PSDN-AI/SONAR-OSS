"""Audio utilities for multi-speaker preprocessing."""

import atexit
import logging
import tempfile
from pathlib import Path
from typing import Optional

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

logger = logging.getLogger(__name__)

_temp_dirs: list[tempfile.TemporaryDirectory] = []


def _get_temp_dir(prefix: str) -> Path:
    """Create a managed temp directory that is cleaned up at process exit."""
    td = tempfile.TemporaryDirectory(prefix=prefix)
    _temp_dirs.append(td)
    return Path(td.name)


@atexit.register
def _cleanup_temp_dirs():
    for td in _temp_dirs:
        try:
            td.cleanup()
        except Exception:
            pass
    _temp_dirs.clear()


def trim_silence(
    audio_path: Path,
    output_path: Optional[Path] = None,
    max_silence_ms: int = 400,
    min_silence_len: int = 500,
    silence_thresh: int = -40,
) -> tuple:
    """Remove long silences, keeping at most *max_silence_ms* between chunks.

    Returns ``(output_path, original_duration_s, trimmed_duration_s)``.
    When no non-silent ranges are detected the input path is returned unchanged.
    """
    audio = AudioSegment.from_file(audio_path)
    original_duration = len(audio) / 1000.0

    nonsilent_ranges = detect_nonsilent(audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh)

    if not nonsilent_ranges:
        return audio_path, original_duration, original_duration

    result = AudioSegment.empty()
    padding = AudioSegment.silent(duration=max_silence_ms)

    for i, (start, end) in enumerate(nonsilent_ranges):
        chunk = audio[start:end]
        result += chunk
        if i < len(nonsilent_ranges) - 1:
            result += padding

    trimmed_duration = len(result) / 1000.0

    if output_path is None:
        tmpdir = _get_temp_dir("sonar_trim_")
        output_path = tmpdir / f"trimmed_{Path(audio_path).stem}.wav"

    result.export(str(output_path), format="wav")
    return output_path, original_duration, trimmed_duration


def get_audio_duration(audio_path) -> float:
    """Get duration of audio file in seconds."""
    audio = AudioSegment.from_file(str(audio_path))
    return len(audio) / 1000.0


def parse_timestamp(timestamp: str) -> float:
    """Parse ``"HH:MM:SS"``, ``"MM:SS"``, or float-seconds timestamps to seconds."""
    if ":" not in str(timestamp):
        return float(timestamp)
    parts = str(timestamp).split(":")
    if len(parts) == 3:
        hours, minutes, seconds = map(float, parts)
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:
        minutes, seconds = map(float, parts)
        return minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")


def trim_by_timestamps(
    audio_path: Path,
    segments: list,
    speaker: str,
    padding_ms: int = 100,
    output_path: Optional[Path] = None,
) -> tuple:
    """Extract the segments belonging to *speaker* ("A" or "B") from *audio_path*.

    Segment dicts need ``speaker``, ``start``, and ``end`` keys; each kept
    segment is padded by *padding_ms* on both sides.

    Returns ``(output_path, original_duration_s, trimmed_duration_s)``.
    When no segments match the speaker the input path is returned unchanged.
    """
    speaker_filter = "speaker_a" if speaker == "A" else "speaker_b"

    filtered_segments = [seg for seg in segments if seg.get("speaker", "").lower() == speaker_filter]

    if not filtered_segments:
        original_duration = get_audio_duration(audio_path)
        return audio_path, original_duration, original_duration

    audio = AudioSegment.from_file(str(audio_path))
    original_duration = len(audio) / 1000.0

    result = AudioSegment.empty()
    padding_audio = AudioSegment.silent(duration=padding_ms)

    for i, segment in enumerate(filtered_segments):
        start_time = parse_timestamp(segment["start"])
        end_time = parse_timestamp(segment["end"])

        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        start_ms = max(0, start_ms)
        end_ms = min(len(audio), end_ms)

        if start_ms >= end_ms:
            continue

        if i > 0:
            result += padding_audio

        pad_start = max(0, start_ms - padding_ms)
        pad_end = min(len(audio), end_ms + padding_ms)

        segment_audio = audio[pad_start:pad_end]
        result += segment_audio

    if filtered_segments:
        result += padding_audio

    trimmed_duration = len(result) / 1000.0

    if output_path is None:
        tmpdir = _get_temp_dir("sonar_ts_trim_")
        output_path = tmpdir / f"trimmed_timestamp_{Path(audio_path).stem}.wav"

    result.export(str(output_path), format="wav")
    return output_path, original_duration, trimmed_duration


def get_combined_audio_path(entry) -> Optional[Path]:
    """Return the combined stereo audio path for a multi-speaker clip, if present.

    Follows the layout convention ``{clip_dir}/{audio_id}_Combined_Audio.wav``.
    """
    if "speaker_a" in entry.audio_filepaths:
        clip_dir = (entry.base_dir / entry.audio_filepaths["speaker_a"]).resolve().parent
    else:
        clip_dir = entry.base_dir / "data" / entry.audio_id

    combined_path = clip_dir / f"{entry.audio_id}_Combined_Audio.wav"
    if combined_path.exists():
        return combined_path
    return None
