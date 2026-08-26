"""Optional pyannote-based VAD and diarization utilities.

All pyannote imports are behind try/except so the main pipeline works
without pyannote installed. Check PYANNOTE_AVAILABLE before calling any function.

Requires ``pyannote.audio`` >= 4.0 (the ``[pyannote]`` extra). The 3.x series
cannot import against the modern torchaudio the ``[ml]`` extra locks
(``torchaudio.AudioMetaData`` was removed in torchaudio 2.9 — issue #129);
4.x decodes audio through torchcodec instead, which needs a system ``ffmpeg``
— the same binary the pipeline-based ASR adapters already require.
"""

import logging
import re
from pathlib import Path
from typing import NoReturn, Optional

logger = logging.getLogger(__name__)

GATED_MODEL_HINT = (
    "pyannote models are gated on HuggingFace: a valid HF_TOKEN alone is not "
    "enough. The token's account must also accept each model's user conditions "
    "(a one-time form on the model page) at "
    "https://huggingface.co/pyannote/segmentation-3.0, "
    "https://huggingface.co/pyannote/speaker-diarization-3.1, and "
    "https://huggingface.co/pyannote/speaker-diarization-community-1 — the "
    "diarization pipeline downloads the third as a gated dependency under "
    "pyannote.audio 4.x, so it needs accepting even though no command names it "
    "(issue #190). Until then HuggingFace rejects the request even though the "
    "token itself is valid. Also check that HF_TOKEN is set (in .env or the "
    "environment) and not expired."
)

_AUTH_ERROR_MARKERS = ("401", "403", "gated", "authorized", "forbidden", "restricted")

# Repos a HuggingFace error names, as a URL
# (https://huggingface.co/<org>/<name>/resolve/...) or in prose
# ("Access to model <org>/<name> is restricted").
_HF_REPO_IN_ERROR_RE = re.compile(
    r"huggingface\.co/([\w.-]+/[\w.-]+)|(?:model|repo(?:sitory)?)\s+'?([\w.-]+/[\w.-]+)'?",
    re.IGNORECASE,
)


def _refused_repo(error_text: str, model_id: str) -> Optional[str]:
    """The repo the error actually refuses, when it is not *model_id* itself.

    Loading a pipeline can fetch gated dependency repos the caller never named:
    under pyannote.audio 4.x, ``pyannote/speaker-diarization-3.1`` pulls
    ``pyannote/speaker-diarization-community-1`` (issue #190). A headline that
    repeats the requested model sends the reader back to re-check an
    authorization that is already granted, so surface the repo named by the
    refusal itself. Candidates outside the requested model's org are ignored —
    HuggingFace errors also link non-repo pages (docs, settings/tokens).
    """
    org = model_id.split("/", 1)[0].lower()
    for match in _HF_REPO_IN_ERROR_RE.finditer(error_text):
        repo = (match.group(1) or match.group(2)).rstrip(".")
        if repo.lower() != model_id.lower() and repo.lower().startswith(f"{org}/"):
            return repo
    return None


def _raise_load_error(model_id: str, exc: Exception) -> NoReturn:
    """Re-raise a ``from_pretrained`` failure, attaching the gated-model guidance
    when it looks like an auth/gating rejection (issue #171): the raw HuggingFace
    401/403 (e.g. ``403 ... not in the authorized list``) carries no hint that
    accepting the model's user conditions is a required step beyond HF_TOKEN.
    When the refusal is for a dependency repo rather than *model_id* itself,
    the headline names that repo (issue #190).
    """
    text = str(exc).lower()
    if any(marker in text for marker in _AUTH_ERROR_MARKERS):
        refused = _refused_repo(str(exc), model_id)
        if refused is not None:
            raise RuntimeError(
                f"Could not load '{model_id}' from HuggingFace: access was refused for "
                f"'{refused}', a gated repo this pipeline depends on: {exc}. {GATED_MODEL_HINT}"
            ) from exc
        raise RuntimeError(f"Could not load '{model_id}' from HuggingFace: {exc}. {GATED_MODEL_HINT}") from exc
    raise exc


def _import_pyannote() -> bool:
    """Import ``pyannote.audio``, working around PyTorch 2.6's ``weights_only=True``
    default that breaks pyannote/lightning checkpoint loading. ``torch.load`` is
    restored after import; the lightning loader stays patched for lazy loads.
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

        setattr(cloud_io, "_load", _patched_pl_load)
    except (ImportError, AttributeError):
        pass

    original_torch_load = torch.load

    def _patched_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    # Patched through setattr, not by assignment: a type checker reads the
    # assignment as an incompatible attribute write, and the suppression it
    # needs is only "used" when torch's types are visible — so the gate's
    # verdict flipped with whether torch was installed (issue #172).
    setattr(torch, "load", _patched_torch_load)
    try:
        import pyannote.audio  # noqa: F401
    except ImportError:
        return False
    finally:
        setattr(torch, "load", original_torch_load)
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

    # pyannote.audio 4.x renamed ``use_auth_token`` to ``token`` (issue #129).
    token = hf_token or os.getenv("HF_TOKEN")
    try:
        model = Model.from_pretrained("pyannote/segmentation-3.0", token=token)
    except Exception as e:
        _raise_load_error("pyannote/segmentation-3.0", e)
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

    # pyannote.audio 4.x renamed ``use_auth_token`` to ``token`` (issue #129).
    token = hf_token or os.getenv("HF_TOKEN")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=token)
    except Exception as e:
        _raise_load_error("pyannote/speaker-diarization-3.1", e)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        pipeline.to(torch.device("mps"))
    elif torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))

    _diarization_pipeline = pipeline
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
    """Assign each word to the speaker segment containing its midpoint, falling
    back to max time overlap, then ``"unknown"``. Returns ``{speaker_id: text}``.
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
