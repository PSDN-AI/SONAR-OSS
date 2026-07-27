"""Optional pyannote-based VAD and diarization utilities.

All pyannote imports are behind try/except so the main pipeline works
without pyannote installed. Check PYANNOTE_AVAILABLE before calling any function.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _import_pyannote() -> bool:
    """Import ``pyannote.audio`` under a scoped ``torch.load`` compatibility patch.

    PyTorch 2.6 changed ``torch.load`` to ``weights_only=True`` by default,
    which breaks pyannote/lightning checkpoint loading. ``torch.load`` is
    restored after the import so safe loading is not disabled globally; the
    lightning loader stays patched because checkpoints load later, when the
    pipelines are first created.
    """
    try:
        import torch
    except ImportError:
        return False

    try:
        import lightning_fabric.utilities.cloud_io as cloud_io

        original_pl_load = cloud_io._load

        def _patched_pl_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_pl_load(*args, **kwargs)

        cloud_io._load = _patched_pl_load
    except (ImportError, AttributeError):
        pass

    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load  # ty: ignore[invalid-assignment]
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    finally:
        torch.load = original_torch_load
    return True


PYANNOTE_AVAILABLE = _import_pyannote()

_vad_pipeline = None
_diarization_pipeline = None


def get_vad_pipeline(hf_token: Optional[str] = None, min_duration_on: float = 0.3, min_duration_off: float = 0.3):
    """Get or create the VAD pipeline singleton (pyannote/segmentation-3.0)."""
    global _vad_pipeline
    if _vad_pipeline is not None:
        return _vad_pipeline

    import os

    from pyannote.audio import Model
    from pyannote.audio.pipelines import VoiceActivityDetection

    token = hf_token or os.getenv("HF_TOKEN")
    model = Model.from_pretrained("pyannote/segmentation-3.0", use_auth_token=token)
    _vad_pipeline = VoiceActivityDetection(segmentation=model)
    _vad_pipeline.instantiate(
        {
            "min_duration_on": min_duration_on,
            "min_duration_off": min_duration_off,
        }
    )
    return _vad_pipeline


def get_diarization_pipeline(hf_token: Optional[str] = None):
    """Get or create the diarization pipeline singleton (pyannote/speaker-diarization-3.1)."""
    global _diarization_pipeline
    if _diarization_pipeline is not None:
        return _diarization_pipeline

    import os

    import torch
    from pyannote.audio import Pipeline

    token = hf_token or os.getenv("HF_TOKEN")
    _diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _diarization_pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        _diarization_pipeline.to(torch.device("cuda"))

    return _diarization_pipeline


def run_vad_on_channel(audio_path: Path) -> list:
    """Run VAD on a single-channel audio file.

    Returns a list of ``(start_seconds, end_seconds)`` speech segments.
    """
    vad = get_vad_pipeline()
    vad_result = vad(str(audio_path))
    return [(seg.start, seg.end) for seg in vad_result.get_timeline()]


def extract_and_concat_segments(
    audio_path: Path,
    segments: list,
    gap_ms: int = 400,
) -> tuple:
    """Extract ``(start_s, end_s)`` speech segments and concatenate with gaps.

    Returns ``(output_path, original_duration_s, trimmed_duration_s)``.
    When *segments* is empty the input path is returned unchanged.
    """
    from pydub import AudioSegment
    from pydub.effects import normalize

    audio = AudioSegment.from_file(str(audio_path))
    original_duration = len(audio) / 1000.0

    if not segments:
        return audio_path, original_duration, original_duration

    chunks = []
    for start_s, end_s in segments:
        start_ms = max(0, int(start_s * 1000))
        end_ms = min(len(audio), int(end_s * 1000))
        if end_ms > start_ms:
            chunks.append(audio[start_ms:end_ms])

    if not chunks:
        return audio_path, original_duration, original_duration

    silence = AudioSegment.silent(duration=gap_ms)
    trimmed = chunks[0]
    for chunk in chunks[1:]:
        trimmed = trimmed + silence + chunk
    trimmed = normalize(trimmed)

    trimmed_duration = len(trimmed) / 1000.0

    from .audio_utils import _get_temp_dir

    tmpdir = _get_temp_dir("sonar_vad_")
    output_path = tmpdir / f"trimmed_{Path(audio_path).stem}.wav"
    trimmed.export(str(output_path), format="wav")

    return output_path, original_duration, trimmed_duration


def run_diarization(audio_path: Path, num_speakers: int = 2) -> list:
    """Run speaker diarization; returns dicts with ``speaker``/``start``/``end`` keys."""
    pipeline = get_diarization_pipeline()
    diarization = pipeline(str(audio_path), num_speakers=num_speakers)
    segments = []
    if hasattr(diarization, "itertracks"):
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "speaker": speaker,
                    "start": float(turn.start),
                    "end": float(turn.end),
                }
            )
    return segments


def assign_words_to_speakers(words: list, diar_segments: list) -> dict:
    """Assign word-level timestamps to diarization speakers.

    Each word goes to the segment containing its midpoint, falling back to
    maximum time overlap, then to ``"unknown"``. Words and segments are dicts
    with ``start``/``end`` (words also ``text``, segments also ``speaker``).
    Returns ``{speaker_id: concatenated text}``.
    """
    speaker_texts: dict = {}
    for word in words:
        w_mid = (word["start"] + word["end"]) / 2.0
        best_speaker = None
        best_overlap = 0.0
        for seg in diar_segments:
            if seg["start"] <= w_mid <= seg["end"]:
                best_speaker = seg["speaker"]
                break
            overlap_start = max(word["start"], seg["start"])
            overlap_end = min(word["end"], seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = seg["speaker"]
        if best_speaker is None:
            best_speaker = "unknown"
        speaker_texts.setdefault(best_speaker, []).append(word["text"])
    return {sid: " ".join(texts) for sid, texts in speaker_texts.items()}
