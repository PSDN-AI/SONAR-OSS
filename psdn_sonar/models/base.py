"""ASR model base interface and protocol-aware latency types."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MissingFfmpegError(RuntimeError):
    """The ffmpeg binary is required by the selected adapter but not on PATH.

    Raised at model-load time by adapters that hand audio *file paths* to the
    ``transformers`` ASR pipeline, which shells out to ffmpeg to decode them —
    WAV included (issue #109). Defined here (ML-free module) so the CLI can
    catch it without importing the torch-heavy adapter module.
    """


class MissingDependencyError(RuntimeError):
    """A Python package required by the selected adapter is not installed.

    Raised at model-load time with a message naming the missing package and
    the install extra that provides it (issue #108: ``khushids_bengali``
    failed with a bare ``ModuleNotFoundError: No module named 'peft'`` and
    took the whole multi-model run down with it).
    """


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

    #: Id of the service that serves this adapter's inference, recorded in the
    #: ``scores.json`` submission block. Hosted-API adapters override this
    #: ("openai", "elevenlabs", "assemblyai"); everything that runs in-process
    #: keeps "local". Exists because the submission block used to fill the
    #: field from an undocumented env var with a "local" default, so a
    #: hosted-API run recorded ``provider: local`` (issue #184).
    provider: str = "local"

    #: The provider-side model identifier actually requested (e.g. "whisper-1",
    #: "scribe_v2"), or ``None`` when the adapter pins no provider model id.
    #: Recorded as ``model_snapshot`` so an API result is pinned to the model
    #: string sent to the provider rather than to this package's registry alias.
    provider_model_id: Optional[str] = None

    #: Cause of the most recent ``transcribe`` failure, or ``None``. Set by
    #: :meth:`_record_transcribe_failure`; cleared by the evaluator before each
    #: call. Exists because ``transcribe`` returns ``None`` on failure by design
    #: (one bad clip must not abort a run), which used to reduce every failure —
    #: including a 401 from an invalid API key — to "Empty prediction" in the
    #: CSV/scores artifacts the CLI points at (issue #170).
    last_transcribe_error: Optional[str] = None

    def _record_transcribe_failure(self, audio_path: str, exc: Exception) -> None:
        """Log a per-clip transcription failure and keep its cause on the adapter.

        Adapters call this from their catch-and-return-``None`` blocks so the
        evaluator can write the real cause into the per-row ``error`` column
        instead of the downstream "Empty prediction" symptom.
        """
        self.last_transcribe_error = str(exc)
        logger.error("Transcription failed for %s: %s", audio_path, exc)

    def get_hf_lineage(self) -> Tuple[Optional[str], Optional[str]]:
        """Best-effort ``(model_id, revision)`` of the loaded HF checkpoint.

        Introspects the ``transformers`` config of the loaded model
        (``config._name_or_path`` / ``config._commit_hash``) via the
        conventional adapter attributes ``self.model`` or ``self.pipe``.
        Returns ``(None, None)`` for adapters without a local HuggingFace
        checkpoint (hosted APIs) or when the attributes are absent.

        Recorded in ``scores.json`` so a run can be reproduced against the
        exact checkpoint it used — the registry pins no revisions, so
        without this the checkpoint behind a published number is
        unrecoverable (issue #120). For adapters that merge PEFT weights
        onto a base model, the lineage reflects the loaded base config and
        is therefore approximate.
        """
        for attr in ("model", "pipe"):
            wrapper = getattr(self, attr, None)
            if wrapper is None:
                continue
            for candidate in (wrapper, getattr(wrapper, "model", None)):
                config = getattr(candidate, "config", None) if candidate is not None else None
                if config is None:
                    continue
                model_id = getattr(config, "_name_or_path", None)
                revision = getattr(config, "_commit_hash", None)
                model_id = model_id if isinstance(model_id, str) and model_id else None
                revision = revision if isinstance(revision, str) and revision else None
                if model_id or revision:
                    return (model_id, revision)
        return (None, None)

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
