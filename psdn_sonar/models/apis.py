"""Hosted ASR vendor API adapters (OpenAI Whisper, ElevenLabs, AssemblyAI).

Vendor SDKs (``openai``, ``assemblyai``) are imported lazily inside each
adapter's ``__init__`` and live behind the ``[apis]`` extra:

    pip install "psdn-sonar[apis]"

API keys are read from environment variables (``OPENAI_API_KEY``,
``ELEVENLABS_API_KEY`` / ``XI_API_KEY``, ``ASSEMBLYAI_API_KEY``) unless
passed explicitly — never hardcoded and never logged.

Like the HuggingFace adapters, ``transcribe`` returns ``None`` on failure
rather than raising, so one bad clip or transient API error does not abort
a long evaluation run; transient network errors are additionally retried
with exponential backoff via :func:`_retry`.
"""

import functools
import logging
import os
import time
from typing import Optional

import requests

from .base import ASRModel, LatencyMetrics

logger = logging.getLogger(__name__)


def _retry(max_retries: int = 3, backoff: float = 1.0):
    """Retry decorator with exponential backoff for transient API errors."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_exc = e
                    if attempt < max_retries:
                        wait = backoff * (2 ** (attempt - 1))
                        logger.warning("Attempt %d/%d failed (%s), retrying in %.1fs…", attempt, max_retries, e, wait)
                        time.sleep(wait)
            logger.error("All %d attempts failed", max_retries)
            # Keep the cause on the adapter so the artifacts carry it (issue
            # #170) — the decorated method is a bound ASRModel.transcribe.
            if args and isinstance(args[0], ASRModel):
                args[0].last_transcribe_error = f"All {max_retries} attempts failed: {last_exc}"
            return None

        return wrapper

    return decorator


class WhisperAPIModel(ASRModel):
    """OpenAI hosted Whisper transcription (``audio.transcriptions``).

    Pass ``language`` (ISO 639-1) to pin the transcription language;
    otherwise the API auto-detects it.
    """

    provider = "openai"

    def __init__(self, api_key=None, model="whisper-1", language=None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                'openai package is required for WhisperAPIModel. Install with: pip install "psdn-sonar[apis]"'
            )

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.provider_model_id = model
        self.language = language

    @_retry()
    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            with open(audio_path, "rb") as f:
                params = {"model": self.model, "file": f}
                if self.language:
                    params["language"] = self.language
                return self.client.audio.transcriptions.create(**params).text
        except Exception as e:
            self._record_transcribe_failure(audio_path, e)
            return None


class ElevenLabsAPIModel(ASRModel):
    """ElevenLabs Speech-to-Text via REST API (xi-api-key header). Uses requests to avoid SDK auth issues.

    ``language`` is the toolkit's ISO 639-1 code (what ``--language``
    carries); it is converted to the vendor's ISO 639-3 form via
    ``LANG_CODE_TO_ELEVENLABS`` when a mapping exists, otherwise passed
    through — the endpoint accepts both forms. An explicit
    ``language_code`` (already in vendor format) wins over ``language``;
    with neither, the historical Bengali default applies.
    """

    provider = "elevenlabs"

    def __init__(self, api_key=None, model_id="scribe_v2", language_code=None, language=None):
        if language_code is None:
            if language:
                from psdn_sonar.language_codes import LANG_CODE_TO_ELEVENLABS

                language_code = LANG_CODE_TO_ELEVENLABS.get(language.lower(), language)
            else:
                language_code = "ben"
        key = (api_key or os.getenv("ELEVENLABS_API_KEY") or os.getenv("XI_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "ElevenLabs API key not found. Set ELEVENLABS_API_KEY (or XI_API_KEY) in .env or as env var. "
                "Ensure .env exists in the script directory and contains ELEVENLABS_API_KEY=sk_..."
            )
        self._api_key = key
        self._model_id = model_id
        self.provider_model_id = model_id
        self._language_code = language_code
        self._session = requests.Session()
        self._session.headers["xi-api-key"] = self._api_key

    @property
    def supports_diarization(self) -> bool:
        return True

    @property
    def supports_word_timestamps(self) -> bool:
        return True

    @staticmethod
    def _detect_content_type(audio_path: str) -> str:
        return "audio/mpeg" if audio_path.lower().endswith(".mp3") else "audio/wav"

    def _handle_error(self, e):
        err = getattr(e, "response", None)
        if err is not None:
            try:
                body = err.json()
                d = body.get("detail")
                msg = d.get("message", str(d)) if isinstance(d, dict) else (d or str(body))
            except Exception:
                msg = f"{err.status_code} {err.text[:200]}"
        else:
            msg = str(e)
        # Keep the cause on the adapter so the artifacts carry it (issue #170).
        self.last_transcribe_error = str(msg)
        logger.error("API error: %s", msg)

    @_retry()
    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            ctype = self._detect_content_type(audio_path)
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, ctype)}
                data = {"model_id": self._model_id, "language_code": self._language_code}
                r = self._session.post(url, files=files, data=data, timeout=60)
            r.raise_for_status()
            out = r.json()
            return (out.get("text") or "").strip()
        except Exception as e:
            self._handle_error(e)
            return None

    def transcribe_diarized(self, audio_path: str, num_speakers: int = 2) -> dict:
        """Transcribe with diarization, returning dict mapping speaker_id to text."""
        try:
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            ctype = self._detect_content_type(audio_path)
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, ctype)}
                data = {
                    "model_id": self._model_id,
                    "language_code": self._language_code,
                    "diarize": "true",
                    "num_speakers": str(num_speakers),
                    "timestamps_granularity": "word",
                }
                r = self._session.post(url, files=files, data=data, timeout=120)
            r.raise_for_status()
            out = r.json()

            speaker_texts = {}
            words = out.get("words", [])
            for word in words:
                if word.get("type") == "word":
                    sid = word.get("speaker_id") or "unknown"
                    speaker_texts.setdefault(sid, []).append(word.get("text", ""))

            return {sid: " ".join(texts) for sid, texts in speaker_texts.items()}
        except Exception as e:
            self._handle_error(e)
            return {}

    def transcribe_with_word_timestamps(self, audio_path: str) -> list:
        """Transcribe with word-level timestamps, returning list of {text, start, end}."""
        try:
            url = "https://api.elevenlabs.io/v1/speech-to-text"
            ctype = self._detect_content_type(audio_path)
            with open(audio_path, "rb") as f:
                files = {"file": (os.path.basename(audio_path), f, ctype)}
                data = {
                    "model_id": self._model_id,
                    "language_code": self._language_code,
                    "timestamps_granularity": "word",
                }
                r = self._session.post(url, files=files, data=data, timeout=120)
            r.raise_for_status()
            out = r.json()

            words = []
            for w in out.get("words", []):
                if w.get("type") == "word":
                    words.append(
                        {
                            "text": w.get("text", ""),
                            "start": w.get("start", 0.0),
                            "end": w.get("end", 0.0),
                        }
                    )
            return words
        except Exception as e:
            self._handle_error(e)
            return []


class AssemblyAIAPIModel(ASRModel):
    """AssemblyAI adapter with optional streaming TTFT measurement.

    Batch mode (``streaming=False``, default) returns the final transcript
    with ``ttft_s=None``. Streaming mode drives the SDK's ``streaming.v3``
    client (the replacement for ``RealtimeTranscriber``, which the SDK
    removed — issue #208: the adapter kept calling it, so every utterance
    fell back to batch while scores.json recorded ``protocol: streaming``)
    and records wall-clock to the first non-empty transcript as ``ttft_s``.

    A runtime streaming failure logs a warning and falls back to batch for
    that utterance, and the fallback is *counted*: ``streaming_fallbacks``,
    ``streamed_utterances`` and ``last_streaming_error`` let the evaluator
    record the protocol that actually ran and put the reason in the
    scores.json ``warnings`` array. Branch on ``self.streaming`` (not
    ``supports_latency_metrics``) to know whether TTFT will be measured.
    """

    supports_latency_metrics = True
    provider = "assemblyai"

    def __init__(
        self,
        api_key=None,
        language_code=None,
        streaming: bool = False,
        sample_rate: int = 16000,
        language=None,
    ):
        # ``language`` is the toolkit's ISO 639-1 code, which is also what
        # AssemblyAI's language_code expects; an explicit ``language_code``
        # wins, and the historical Bengali default applies with neither.
        if language_code is None:
            language_code = language or "bn"
        try:
            import assemblyai as aai
        except ImportError:
            raise ImportError(
                'assemblyai package is required for AssemblyAIAPIModel. Install with: pip install "psdn-sonar[apis]"'
            )

        aai.settings.api_key = api_key or os.getenv("ASSEMBLYAI_API_KEY")
        self.config = aai.TranscriptionConfig(punctuate=True, format_text=True, language_code=language_code)
        # This adapter pins no speech model — the SDK's server-side default
        # serves the request — but record it if the config carries one.
        speech_model = getattr(self.config, "speech_model", None)
        self.provider_model_id = str(speech_model) if speech_model else None
        self.transcriber = aai.Transcriber(config=self.config)
        self.streaming = streaming
        self._sample_rate = sample_rate
        self._language_code = language_code
        # Executed-protocol bookkeeping (issue #208): the evaluator records
        # the protocol that ran, not the one that was requested.
        self.streamed_utterances = 0
        self.streaming_fallbacks = 0
        self.last_streaming_error: Optional[str] = None
        if streaming:
            self._require_streaming_support()

    @staticmethod
    def _require_streaming_support() -> None:
        """Fail at construction when the SDK has no ``streaming.v3`` client.

        The class this adapter used to call, ``aai.RealtimeTranscriber``,
        is gone from current SDK releases, so streaming failed on every
        utterance and silently ran batch (issue #208). Failing here costs
        milliseconds, not a whole run recorded under the wrong protocol.
        """
        try:
            import assemblyai.streaming.v3  # noqa: F401
        except ImportError as exc:
            import assemblyai as aai

            version = getattr(aai, "__version__", "unknown")
            raise ImportError(
                f"--streaming needs the assemblyai SDK's streaming.v3 client, which the "
                f"installed version ({version}) does not provide. Upgrade the SDK "
                "(pip install -U assemblyai — recent releases including 1.x ship it) "
                "or drop --streaming to run the batch protocol."
            ) from exc

    @_retry()
    def transcribe(self, audio_path: str):
        if self.streaming:
            try:
                text, ttft_s, complete_s = self._transcribe_streaming(audio_path)
                self.streamed_utterances += 1
                return (text or ""), LatencyMetrics(complete_s=complete_s, ttft_s=ttft_s)
            except Exception as e:
                self.streaming_fallbacks += 1
                self.last_streaming_error = str(e) or type(e).__name__
                logger.warning(
                    "Streaming transcription failed (%s); falling back to batch for %s "
                    "(the fallback is recorded in scores.json — issue #208)",
                    e,
                    audio_path,
                )
        try:
            t0 = time.perf_counter()
            transcript = self.transcriber.transcribe(audio_path)
            text = transcript.text or ""
            complete_s = round(time.perf_counter() - t0, 4)
            if not text and getattr(transcript, "error", None):
                # The SDK reports some failures (e.g. rejected audio) as an
                # errored transcript object rather than an exception.
                self.last_transcribe_error = str(transcript.error)
                logger.error("Transcription failed for %s: %s", audio_path, transcript.error)
                return None
            return text, LatencyMetrics(complete_s=complete_s, ttft_s=None)
        except Exception as e:
            self._record_transcribe_failure(audio_path, e)
            return None

    def _make_streaming_client(self):
        """Build the SDK's v3 streaming client. Overridable in tests."""
        import assemblyai as aai
        from assemblyai.streaming.v3 import StreamingClient, StreamingClientOptions

        return StreamingClient(StreamingClientOptions(api_key=aai.settings.api_key))

    def _iter_pcm_frames(self, audio_path: str, frame_ms: int = 300):
        """Yield mono int16 PCM frames from ``audio_path`` at ``self._sample_rate``.

        The streaming client is told the stream rate, so frames MUST be
        emitted at that rate (downmix to mono, resample when the file's
        native rate differs) or the server mis-times every frame.
        Overridable in tests.
        """
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if getattr(data, "ndim", 1) > 1:  # downmix to mono
            data = data.mean(axis=1)
        if sr != self._sample_rate:
            import librosa

            data = librosa.resample(np.asarray(data, dtype="float32"), orig_sr=sr, target_sr=self._sample_rate)
        # Float [-1, 1] -> little-endian int16 PCM, which is what the streaming
        # transcriber expects on the wire.
        pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
        frame_len = max(1, int(self._sample_rate * frame_ms / 1000))
        for start in range(0, len(pcm), frame_len):
            yield pcm[start : start + frame_len].tobytes()

    def _transcribe_streaming(self, audio_path: str):
        """Stream ``audio_path`` through the v3 client and return
        ``(text, ttft_s, complete_s)``.

        ``ttft_s`` is the wall-clock to the first non-empty transcript event
        (partial or final); ``complete_s`` is the total wall-clock through
        session close. Turn transcripts are joined in turn order; when the
        session ends mid-turn, the last unfinished transcript stands in.

        Raises when the session produced no transcript and reported an
        error, so the caller counts the fallback instead of scoring an
        empty streaming result (issue #208).
        """
        from assemblyai.streaming.v3 import StreamingEvents, StreamingParameters

        ttft: Optional[float] = None
        turns: dict = {}
        last_partial = ""
        errors: list = []
        t0 = time.perf_counter()

        def on_turn(_client, event):
            nonlocal ttft, last_partial
            text = (getattr(event, "transcript", "") or "").strip()
            if not text:
                return
            if ttft is None:
                ttft = round(time.perf_counter() - t0, 4)
            if getattr(event, "end_of_turn", False):
                turns[getattr(event, "turn_order", len(turns))] = text
            else:
                last_partial = text

        def on_error(_client, error):
            errors.append(error)
            logger.error("Streaming error: %s", error)

        client = self._make_streaming_client()
        client.on(StreamingEvents.Turn, on_turn)
        client.on(StreamingEvents.Error, on_error)
        client.connect(
            StreamingParameters(
                sample_rate=self._sample_rate,
                format_turns=True,
                language_code=self._language_code,
            )
        )
        try:
            client.stream(self._iter_pcm_frames(audio_path))
        finally:
            client.disconnect(terminate=True)

        text = " ".join(turns[k] for k in sorted(turns)).strip() or last_partial
        if not text and errors:
            raise RuntimeError(f"streaming session produced no transcript: {errors[-1]}")
        complete_s = round(time.perf_counter() - t0, 4)
        return text, ttft, complete_s
