"""ASR model base interface and protocol-aware latency types."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass(frozen=True)
class LatencyMetrics:
    """Protocol-aware latency for a single transcription.

    A single scalar latency conflates two operationally distinct numbers,
    so we keep them apart:

    - ``complete_s``: total wall-clock from request start to *final*
      transcript. Defined for **both** batch and streaming protocols.
    - ``ttft_s``: time-to-first-token — wall-clock until the *first partial*
      transcript arrives. Defined **only** for streaming protocols; ``None``
      for batch adapters (which have no notion of a partial result).

    ``complete_s`` is the required field; ``ttft_s`` defaults to ``None`` so
    batch adapters can construct ``LatencyMetrics(complete_s=x)`` directly.
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

    Adapters may return either a bare ``str`` (legacy / batch) or a
    ``(text, LatencyMetrics)`` tuple (latency-aware). This helper lets callers
    treat both uniformly:

    - ``(text, metrics)`` tuple → returned as-is.
    - bare ``str`` / ``None`` → paired with a synthesised ``LatencyMetrics``
      whose ``complete_s`` is ``fallback_complete_s`` (typically the caller's
      own wall-clock measurement) and whose ``ttft_s`` is ``None``. If no
      fallback is supplied, the metrics half is ``None``.
    """
    if isinstance(result, tuple):
        text, metrics = result
        return text, metrics
    if fallback_complete_s is not None:
        return result, LatencyMetrics(complete_s=fallback_complete_s, ttft_s=None)
    return result, None


class ASRModel:
    """Base class for all ASR model adapters. Subclasses must implement ``transcribe``.

    ``supports_latency_metrics`` advertises only that an adapter *may* return a
    ``(text, LatencyMetrics)`` tuple from :meth:`transcribe` (vs a bare
    ``str``). It is **not** a promise that ``ttft_s`` is populated: a batch
    adapter can set this ``True`` and still report ``ttft_s=None`` because batch
    protocols have no first-partial event. In other words, the flag means
    "returns a ``LatencyMetrics`` object", not "is TTFT-capable" — whether TTFT
    is actually measured depends on the adapter's protocol/mode (e.g.
    ``AssemblyAIAPIModel(streaming=True)``). It is ``False`` by default.

    Callers should not rely on this flag for correctness — use
    :func:`unpack_transcription`, which normalises both return shapes and
    leaves ``ttft_s`` as whatever the adapter reported (possibly ``None``). The
    flag is for reporting / capability display only.
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
