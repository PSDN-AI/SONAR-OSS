"""ASR model base interface and protocol-aware latency types."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class LatencyMetrics:
    """Protocol-aware latency for a single transcription.

    - ``complete_s``: total wall-clock to the final transcript (defined for
      both batch and streaming protocols).
    - ``ttft_s``: wall-clock to the first partial transcript; streaming
      protocols only, ``None`` for batch.
    """

    complete_s: float
    ttft_s: Optional[float] = None


# Backwards-compatible return type for ``ASRModel.transcribe``: legacy adapters
# return a bare ``str``; latency-aware adapters may return ``(text, metrics)``.
TranscribeResult = Union[str, Tuple[str, LatencyMetrics], None]


def unpack_transcription(
    result: TranscribeResult,
    *,
    fallback_complete_s: Optional[float] = None,
) -> Tuple[Optional[str], Optional[LatencyMetrics]]:
    """Normalise a ``transcribe`` return into ``(text, LatencyMetrics | None)``.

    ``(text, metrics)`` tuples pass through unchanged. A bare ``str`` /
    ``None`` is paired with metrics synthesised from ``fallback_complete_s``
    (typically the caller's own wall-clock measurement), or with ``None``
    when no fallback is supplied.
    """
    if isinstance(result, tuple):
        text, metrics = result
        return text, metrics
    if fallback_complete_s is not None:
        return result, LatencyMetrics(complete_s=fallback_complete_s, ttft_s=None)
    return result, None


class ASRModel:
    """Base class for all ASR model adapters. Subclasses must implement ``transcribe``.

    ``supports_latency_metrics`` advertises that ``transcribe`` may return a
    ``(text, LatencyMetrics)`` tuple instead of a bare ``str``. It does NOT
    promise ``ttft_s`` is populated (batch protocols have no first-partial
    event). The flag is for capability display only — callers should use
    :func:`unpack_transcription`, which handles both return shapes.
    """

    supports_latency_metrics: bool = False

    def transcribe(self, audio_path: str) -> TranscribeResult:
        raise NotImplementedError

    @property
    def supports_diarization(self) -> bool:
        return False

    @property
    def supports_word_timestamps(self) -> bool:
        return False

    def transcribe_diarized(self, audio_path: str, num_speakers: int = 2) -> dict:
        raise NotImplementedError

    def transcribe_with_word_timestamps(self, audio_path: str) -> list:
        raise NotImplementedError
