"""Loader for multi-speaker datasets described by a ``manifest.jsonl``.

Each manifest line points at per-speaker audio files and a transcript JSON.
Transcripts come in two shapes: a ``{"segments": [...]}`` object with
``speaker``/``text`` entries, or a bare array of such entries (legacy).
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ManifestEntry:
    """Entry from manifest.jsonl."""

    audio_id: str
    audio_filepaths: dict  # {"speaker_a": "path/to/a.wav", "speaker_b": "path/to/b.wav"}
    transcript_filepath: str
    num_speakers: int
    base_dir: Path  # Directory containing manifest.jsonl; anchors relative paths


def load_manifest(manifest_path: str) -> List[ManifestEntry]:
    """Load ``manifest.jsonl``, skipping blank lines."""
    path = Path(manifest_path)
    base_dir = path.parent
    entries = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(
                ManifestEntry(
                    audio_id=data["audio_id"],
                    audio_filepaths=data["audio_filepaths"],
                    transcript_filepath=data["transcript_filepath"],
                    num_speakers=data["num_speakers"],
                    base_dir=base_dir,
                )
            )

    return entries


def _parse_transcript(data) -> Tuple[List[str], List[str], list]:
    """Split transcript entries into speaker-A/B lines plus raw segments.

    Segment-format speakers match ``speaker_a`` or any ``*_a`` suffix. The
    legacy array format uses a looser heuristic (label shorter than 20 chars
    containing "speaker", or at most 2 chars) before suffix matching, and
    synthesises zero timestamps for the returned segments.
    """
    speaker_a_lines: List[str] = []
    speaker_b_lines: List[str] = []
    segments: list = []

    if "segments" in data:
        segments = data["segments"]
        for segment in segments:
            speaker = segment.get("speaker", "").lower().strip()
            text = segment.get("text", "")
            if speaker == "speaker_a" or speaker.endswith("_a"):
                speaker_a_lines.append(text)
            elif speaker == "speaker_b" or speaker.endswith("_b"):
                speaker_b_lines.append(text)
    else:
        for entry_item in data:
            speaker_raw = entry_item.get("speaker", "")
            speaker = speaker_raw.lower().strip()
            text = entry_item.get("text", "")
            is_speaker_label = len(speaker_raw) < 20 and ("speaker" in speaker or len(speaker) <= 2)
            if is_speaker_label:
                if speaker.endswith("a") or speaker == "a":
                    speaker_a_lines.append(text)
                elif speaker.endswith("b") or speaker == "b":
                    speaker_b_lines.append(text)
            segments.append(
                {
                    "speaker": speaker_raw,
                    "text": text,
                    "start": "00:00:00",
                    "end": "00:00:00",
                }
            )

    return speaker_a_lines, speaker_b_lines, segments


def load_transcript(transcript_path) -> Tuple[str, str]:
    """Return ``(speaker_a_text, speaker_b_text)`` from a transcript JSON."""
    speaker_a_text, speaker_b_text, _ = load_transcript_with_segments(transcript_path)
    return speaker_a_text, speaker_b_text


def load_transcript_with_segments(transcript_path) -> Tuple[str, str, list]:
    """Return ``(speaker_a_text, speaker_b_text, segments)`` from a transcript JSON."""
    with open(str(transcript_path), encoding="utf-8") as f:
        data = json.load(f)

    speaker_a_lines, speaker_b_lines, segments = _parse_transcript(data)
    return " ".join(speaker_a_lines), " ".join(speaker_b_lines), segments


def get_clip_files(entry: ManifestEntry) -> Tuple[Optional[Path], Optional[Path], Path]:
    """Resolve ``(audio_a_path, audio_b_path, transcript_path)`` against the manifest dir."""
    base_dir = entry.base_dir
    audio_filepaths = entry.audio_filepaths

    audio_a_path = None
    audio_b_path = None

    if "speaker_a" in audio_filepaths:
        audio_a_path = (base_dir / audio_filepaths["speaker_a"]).resolve()
    if "speaker_b" in audio_filepaths:
        audio_b_path = (base_dir / audio_filepaths["speaker_b"]).resolve()

    transcript_path = (base_dir / entry.transcript_filepath).resolve()

    return audio_a_path, audio_b_path, transcript_path


def get_output_fieldnames() -> List[str]:
    """Return CSV column names for multi-speaker evaluation output."""
    return [
        "audio_id",
        "speaker",
        "best_method",
        "path",
        "transcription",
        "transcription_norm",
        "asr_transcription",
        "asr_transcription_norm_non",
        "asr_transcription_norm_conv",
        "cer_non",
        "wer_non",
        "semantic_similarity_non",
        "poseidon_score_non",
        "cer_conv",
        "wer_conv",
        "semantic_similarity_conv",
        "poseidon_score_conv",
        "original_duration_s",
        "trimmed_duration_s",
        "snr_db",
        "clipping_ratio",
        "silence_ratio",
        "snr_tier",
        "quality_warnings",
        "inference_latency_s",
        "all_method_scores",
        "error",
    ]


def create_output_row(
    audio_id: str,
    speaker: str,
    best_method: str,
    path: str,
    transcription: str,
    transcription_norm: str,
    asr_transcription: str,
    asr_norm_non: str,
    asr_norm_conv: str,
    cer_non,
    wer_non,
    sem_non,
    poseidon_non,
    cer_conv,
    wer_conv,
    sem_conv,
    poseidon_conv,
    original_duration_s: float,
    trimmed_duration_s: float,
    snr_db=None,
    clipping_ratio=None,
    silence_ratio=None,
    snr_tier=None,
    quality_warnings: Optional[str] = None,
    inference_latency_s=None,
    all_method_scores: Optional[dict] = None,
    error: Optional[str] = None,
) -> dict:
    """Create an output row dict matching :func:`get_output_fieldnames`."""

    def _fmt(val):
        return f"{val:.4f}" if val is not None else ""

    return {
        "audio_id": audio_id,
        "speaker": speaker,
        "best_method": best_method or "",
        "path": path,
        "transcription": transcription,
        "transcription_norm": transcription_norm,
        "asr_transcription": asr_transcription,
        "asr_transcription_norm_non": asr_norm_non,
        "asr_transcription_norm_conv": asr_norm_conv,
        "cer_non": _fmt(cer_non),
        "wer_non": _fmt(wer_non),
        "semantic_similarity_non": _fmt(sem_non),
        "poseidon_score_non": _fmt(poseidon_non),
        "cer_conv": _fmt(cer_conv),
        "wer_conv": _fmt(wer_conv),
        "semantic_similarity_conv": _fmt(sem_conv),
        "poseidon_score_conv": _fmt(poseidon_conv),
        "original_duration_s": f"{original_duration_s:.2f}" if original_duration_s else "",
        "trimmed_duration_s": f"{trimmed_duration_s:.2f}" if trimmed_duration_s else "",
        "snr_db": f"{snr_db:.2f}" if snr_db is not None else "",
        "clipping_ratio": f"{clipping_ratio:.6f}" if clipping_ratio is not None else "",
        "silence_ratio": f"{silence_ratio:.4f}" if silence_ratio is not None else "",
        "snr_tier": snr_tier or "",
        "quality_warnings": quality_warnings or "",
        "inference_latency_s": f"{inference_latency_s:.4f}" if inference_latency_s is not None else "",
        "all_method_scores": json.dumps(all_method_scores) if all_method_scores else "",
        "error": error or "",
    }
