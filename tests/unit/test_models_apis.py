"""Hermetic tests for the vendor API adapters.

No network calls and no vendor SDKs required: ``assemblyai`` is stubbed via
``sys.modules``, ElevenLabs speaks plain HTTP through a mocked
``requests.Session``, and the retry decorator is exercised directly. This
file runs fully in CI, which installs only ``[dev]``.

Covers:
  * ``_retry`` — retries transient network errors, gives up after
    ``max_retries``, and does not retry non-network exceptions.
  * ``ElevenLabsAPIModel`` — missing-key error, transcribe/diarization/word
    timestamp response parsing.
  * ``AssemblyAIAPIModel`` — streaming TTFT capture (via a fake v3
    streaming client), fallback to batch on streaming failure with the
    fallback counted for the artifact (issue #208), fail-fast construction
    when the SDK lacks ``streaming.v3``, batch path leaving ``ttft_s``
    None, and PCM framing (resample / passthrough / downmix).
"""

from __future__ import annotations

import sys
import time
import types as _types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

# ---------------------------------------------------------------------------
# _retry
# ---------------------------------------------------------------------------


class TestRetryDecorator:
    def test_retries_transient_errors_then_succeeds(self, monkeypatch):
        from psdn_sonar.models.apis import _retry

        monkeypatch.setattr(time, "sleep", lambda s: None)
        calls = {"n": 0}

        @_retry(max_retries=3, backoff=0.01)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.exceptions.Timeout("slow")
            return "ok"

        assert flaky() == "ok"
        assert calls["n"] == 3

    def test_gives_up_after_max_retries(self, monkeypatch):
        from psdn_sonar.models.apis import _retry

        monkeypatch.setattr(time, "sleep", lambda s: None)

        @_retry(max_retries=2, backoff=0.01)
        def always_down():
            raise requests.exceptions.ConnectionError("refused")

        assert always_down() is None

    def test_non_network_errors_propagate(self):
        from psdn_sonar.models.apis import _retry

        @_retry()
        def broken():
            raise ValueError("not a network problem")

        with pytest.raises(ValueError):
            broken()

    def test_exhaustion_records_cause_on_the_adapter(self, monkeypatch):
        # Issue #170: a run that exhausted its retries used to leave only
        # "Empty prediction" in the artifacts.
        from psdn_sonar.models.apis import _retry
        from psdn_sonar.models.base import ASRModel

        monkeypatch.setattr(time, "sleep", lambda s: None)

        class _Down(ASRModel):
            @_retry(max_retries=2, backoff=0.01)
            def transcribe(self, audio_path):
                raise requests.exceptions.ConnectionError("connection refused")

        model = _Down()
        assert model.transcribe("clip.wav") is None
        assert model.last_transcribe_error == "All 2 attempts failed: connection refused"


# ---------------------------------------------------------------------------
# ElevenLabs (plain HTTP via requests — no SDK involved)
# ---------------------------------------------------------------------------


def _elevenlabs_with_mock_session(response_json: dict):
    from psdn_sonar.models.apis import ElevenLabsAPIModel

    model = ElevenLabsAPIModel(api_key="test-key")
    response = MagicMock()
    response.json.return_value = response_json
    response.raise_for_status.return_value = None
    model._session = MagicMock()
    model._session.post.return_value = response
    return model


class TestElevenLabsAPIModel:
    def test_missing_api_key_raises(self, monkeypatch):
        from psdn_sonar.models.apis import ElevenLabsAPIModel

        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("XI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key not found"):
            ElevenLabsAPIModel()

    def test_capability_flags(self):
        model = _elevenlabs_with_mock_session({})
        assert model.supports_diarization is True
        assert model.supports_word_timestamps is True

    def test_transcribe_parses_text(self, tmp_path):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"\x00")
        model = _elevenlabs_with_mock_session({"text": "  hello world  "})
        assert model.transcribe(str(clip)) == "hello world"

    def test_transcribe_diarized_groups_by_speaker(self, tmp_path):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"\x00")
        model = _elevenlabs_with_mock_session(
            {
                "words": [
                    {"type": "word", "speaker_id": "A", "text": "hello"},
                    {"type": "spacing", "speaker_id": "A", "text": " "},
                    {"type": "word", "speaker_id": "B", "text": "hi"},
                    {"type": "word", "speaker_id": "A", "text": "there"},
                ]
            }
        )
        assert model.transcribe_diarized(str(clip)) == {"A": "hello there", "B": "hi"}

    def test_auth_failure_cause_recorded_for_artifacts(self, tmp_path):
        # Issue #170: the API's own message (parsed from the response body)
        # must survive into last_transcribe_error, not just the terminal log.
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"\x00")
        model = _elevenlabs_with_mock_session({})
        response = MagicMock()
        response.json.return_value = {"detail": {"message": "Invalid API key"}}
        model._session.post.side_effect = requests.HTTPError(response=response)

        assert model.transcribe(str(clip)) is None
        assert model.last_transcribe_error == "Invalid API key"

    def test_word_timestamps_filter_non_words(self, tmp_path):
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"\x00")
        model = _elevenlabs_with_mock_session(
            {
                "words": [
                    {"type": "word", "text": "hello", "start": 0.1, "end": 0.5},
                    {"type": "spacing", "text": " ", "start": 0.5, "end": 0.6},
                    {"type": "word", "text": "world", "start": 0.6, "end": 1.0},
                ]
            }
        )
        words = model.transcribe_with_word_timestamps(str(clip))
        assert words == [
            {"text": "hello", "start": 0.1, "end": 0.5},
            {"text": "world", "start": 0.6, "end": 1.0},
        ]


# ---------------------------------------------------------------------------
# AssemblyAI streaming TTFT (fake v3 streaming client) + batch path
# ---------------------------------------------------------------------------


def _ensure_streaming_v3_stub() -> None:
    """Give the assemblyai stub the ``streaming.v3`` submodule the adapter
    imports (issue #208: the adapter now uses the v3 client — the SDK ships
    no ``RealtimeTranscriber`` anymore)."""
    if "assemblyai.streaming.v3" in sys.modules:
        return

    class _StreamingEvents:
        Turn = "Turn"
        Error = "Error"

    class _StreamingClientOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _StreamingClient:  # pragma: no cover — replaced via monkeypatch
        def __init__(self, options):
            self.options = options

    class _StreamingParameters:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    streaming = _types.ModuleType("assemblyai.streaming")
    v3 = _types.ModuleType("assemblyai.streaming.v3")
    v3.StreamingClient = _StreamingClient
    v3.StreamingClientOptions = _StreamingClientOptions
    v3.StreamingParameters = _StreamingParameters
    v3.StreamingEvents = _StreamingEvents
    streaming.v3 = v3
    sys.modules["assemblyai"].streaming = streaming
    sys.modules["assemblyai.streaming"] = streaming
    sys.modules["assemblyai.streaming.v3"] = v3


def _install_assemblyai_stub() -> None:
    if "assemblyai" in sys.modules:
        _ensure_streaming_v3_stub()
        return

    class _Settings:
        api_key = None

    class _TranscriptionConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _Transcriber:
        def __init__(self, config=None):
            self.config = config

        def transcribe(self, audio_path):  # overridden per-test
            return SimpleNamespace(text="")

    aai = _types.ModuleType("assemblyai")
    aai.settings = _Settings()
    aai.TranscriptionConfig = _TranscriptionConfig
    aai.Transcriber = _Transcriber
    sys.modules["assemblyai"] = aai
    _ensure_streaming_v3_stub()


class _FakeStreamingClient:
    """Simulates the v3 streaming client: a partial turn then a final."""

    def __init__(self):
        self.handlers = {}
        self.params = None
        self.disconnected_with = None

    def on(self, event, handler):
        self.handlers[event] = handler

    def connect(self, params):
        self.params = params

    def stream(self, frames):
        on_turn = self.handlers["Turn"]
        for i, _frame in enumerate(frames):
            if i == 0:
                time.sleep(0.005)  # ensure a measurable, non-zero TTFT
                on_turn(self, SimpleNamespace(transcript="hello", end_of_turn=False, turn_order=0))
            elif i == 1:
                on_turn(self, SimpleNamespace(transcript="hello world", end_of_turn=True, turn_order=0))

    def disconnect(self, terminate=False):
        self.disconnected_with = terminate


class TestAssemblyAIStreaming:
    def test_supports_latency_metrics_flag(self):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        assert AssemblyAIAPIModel.supports_latency_metrics is True

    def test_streaming_captures_ttft(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=True)
        fake = _FakeStreamingClient()
        monkeypatch.setattr(model, "_make_streaming_client", lambda: fake)
        monkeypatch.setattr(model, "_iter_pcm_frames", lambda audio_path: [b"\x00\x00", b"\x00\x00"])

        result = model.transcribe("clip.wav")
        assert isinstance(result, tuple)
        text, lm = result
        assert text == "hello world"
        assert lm.ttft_s is not None
        assert lm.complete_s >= lm.ttft_s
        # Executed-protocol bookkeeping (issue #208).
        assert model.streamed_utterances == 1
        assert model.streaming_fallbacks == 0
        assert fake.disconnected_with is True  # session terminated cleanly

    def test_streaming_session_params_carry_rate_and_language(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=True, sample_rate=16000, language="en")
        fake = _FakeStreamingClient()
        monkeypatch.setattr(model, "_make_streaming_client", lambda: fake)
        monkeypatch.setattr(model, "_iter_pcm_frames", lambda audio_path: [b"\x00\x00", b"\x00\x00"])

        model.transcribe("clip.wav")
        assert fake.params.sample_rate == 16000
        assert fake.params.language_code == "en"
        assert fake.params.format_turns is True

    def test_streaming_failure_falls_back_to_batch_and_is_counted(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=True)

        def _boom(*a, **kw):
            raise RuntimeError("ws connect failed")

        monkeypatch.setattr(model, "_make_streaming_client", _boom)
        monkeypatch.setattr(model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="batch fallback"))

        text, lm = model.transcribe("clip.wav")
        assert text == "batch fallback"
        assert lm.ttft_s is None  # batch path → no TTFT
        # Issue #208: the fallback used to be a terminal warning only; now
        # the adapter counts it so the artifact can record what ran.
        assert model.streaming_fallbacks == 1
        assert model.streamed_utterances == 0
        assert "ws connect failed" in model.last_streaming_error

    def test_constructor_fails_fast_when_sdk_has_no_v3_client(self, monkeypatch):
        # Issue #208: the adapter called aai.RealtimeTranscriber, which no
        # released SDK ships anymore, so streaming failed per utterance at
        # transcription time. An SDK without streaming.v3 must fail at
        # construction with the reason, not run a whole batch of fallbacks.
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        monkeypatch.delitem(sys.modules, "assemblyai.streaming.v3", raising=False)
        monkeypatch.delitem(sys.modules, "assemblyai.streaming", raising=False)
        monkeypatch.delattr(sys.modules["assemblyai"], "streaming", raising=False)

        with pytest.raises(ImportError, match="streaming.v3"):
            AssemblyAIAPIModel(api_key="x", streaming=True)
        # Batch mode does not need the streaming client and must still work.
        assert AssemblyAIAPIModel(api_key="x", streaming=False).streaming is False

    def test_streaming_session_with_no_transcript_and_errors_raises(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        class _SilentErroringClient(_FakeStreamingClient):
            def stream(self, frames):
                list(frames)
                self.handlers["Error"](self, RuntimeError("auth rejected"))

        model = AssemblyAIAPIModel(api_key="x", streaming=True)
        monkeypatch.setattr(model, "_make_streaming_client", lambda: _SilentErroringClient())
        monkeypatch.setattr(model, "_iter_pcm_frames", lambda audio_path: [b"\x00\x00"])

        with pytest.raises(RuntimeError, match="no transcript"):
            model._transcribe_streaming("clip.wav")

    def test_batch_path_returns_complete_only(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=False)
        monkeypatch.setattr(model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="batch text"))

        text, lm = model.transcribe("clip.wav")
        assert text == "batch text"
        assert lm.ttft_s is None
        assert lm.complete_s is not None

    def test_batch_exception_cause_recorded_for_artifacts(self, monkeypatch):
        # Issue #170: "Failed to upload audio file: Invalid API key" used to
        # exist only in the terminal; the artifacts said "Empty prediction".
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=False)

        def _boom(ap):
            raise RuntimeError("Failed to upload audio file: Invalid API key")

        monkeypatch.setattr(model.transcriber, "transcribe", _boom)

        assert model.transcribe("clip.wav") is None
        assert model.last_transcribe_error == "Failed to upload audio file: Invalid API key"

    def test_errored_transcript_object_cause_recorded(self, monkeypatch):
        # The SDK reports some failures as an errored transcript object rather
        # than an exception; that cause must be kept too.
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=False)
        monkeypatch.setattr(
            model.transcriber, "transcribe", lambda ap: SimpleNamespace(text=None, error="Invalid API key")
        )

        assert model.transcribe("clip.wav") is None
        assert model.last_transcribe_error == "Invalid API key"


class TestAssemblyAIModelSnapshot:
    """Issue #212: the adapter pins no speech model, so model_snapshot in
    scores.json recorded the registry alias — the one hosted adapter whose
    artifact could not be tied to a server-side model id. The response names
    the model that served the request; the adapter records the first one."""

    def _model(self):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        return AssemblyAIAPIModel(api_key="x", streaming=False)

    def test_served_speech_model_recorded_from_response(self, monkeypatch):
        model = self._model()
        assert model.provider_model_id is None
        monkeypatch.setattr(
            model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="hi", speech_model="universal")
        )

        model.transcribe("clip.wav")
        assert model.provider_model_id == "universal"

    def test_served_speech_model_read_from_raw_response_too(self, monkeypatch):
        # Some SDK versions expose the field only on the raw JSON payload.
        model = self._model()
        monkeypatch.setattr(
            model.transcriber,
            "transcribe",
            lambda ap: SimpleNamespace(text="hi", json_response={"speech_model": "slam-1"}),
        )

        model.transcribe("clip.wav")
        assert model.provider_model_id == "slam-1"

    def test_response_without_the_field_keeps_alias_fallback(self, monkeypatch):
        model = self._model()
        monkeypatch.setattr(model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="hi"))

        model.transcribe("clip.wav")
        assert model.provider_model_id is None  # scores.json falls back to the alias

    def test_config_pinned_model_is_not_overwritten(self, monkeypatch):
        model = self._model()
        model.provider_model_id = "pinned-by-config"
        monkeypatch.setattr(
            model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="hi", speech_model="universal")
        )

        model.transcribe("clip.wav")
        assert model.provider_model_id == "pinned-by-config"


def _install_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    class _OpenAI:
        def __init__(self, api_key=None):
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kw: SimpleNamespace(text="")))

    mod = _types.ModuleType("openai")
    mod.OpenAI = _OpenAI
    sys.modules["openai"] = mod


class TestWhisperAPIModel:
    def test_api_error_cause_recorded_for_artifacts(self, tmp_path):
        # Issue #170: the 401 from an invalid OPENAI_API_KEY existed only in
        # the terminal; the CSV/scores artifacts said "Empty prediction".
        _install_openai_stub()
        from psdn_sonar.models.apis import WhisperAPIModel

        model = WhisperAPIModel(api_key="qa-invalid-key-for-testing")
        model.client = MagicMock()
        model.client.audio.transcriptions.create.side_effect = Exception(
            "Error code: 401 - Incorrect API key provided: qa-inval***"
        )
        clip = tmp_path / "clip.wav"
        clip.write_bytes(b"\x00")

        assert model.transcribe(str(clip)) is None
        assert model.last_transcribe_error == "Error code: 401 - Incorrect API key provided: qa-inval***"


class TestIterPcmFrames:
    """The realtime transcriber is told the stream is ``self._sample_rate``, so
    the frames it receives must actually be at that rate — regardless of the
    source file's native rate."""

    def _write_wav(self, path, sr, seconds=1.0, channels=1):
        import numpy as np
        import soundfile as sf

        n = int(sr * seconds)
        data = np.zeros((n, channels) if channels > 1 else n, dtype="float32")
        sf.write(str(path), data, sr)

    def test_resamples_to_configured_rate(self, tmp_path):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        wav = tmp_path / "src_48k.wav"
        self._write_wav(wav, sr=48000, seconds=1.0)
        model = AssemblyAIAPIModel(api_key="x", streaming=True, sample_rate=16000)

        frames = list(model._iter_pcm_frames(str(wav), frame_ms=300))
        total_samples = sum(len(f) // 2 for f in frames)  # int16 -> 2 bytes/sample
        # 1s @ 48 kHz resampled to 16 kHz -> ~16000 samples, NOT ~48000.
        assert abs(total_samples - 16000) <= 5
        # Non-final frames are exactly frame_len (4800 samples) * 2 bytes.
        assert len(frames[0]) == 4800 * 2
        assert all(isinstance(f, (bytes, bytearray)) for f in frames)

    def test_native_rate_passthrough(self, tmp_path):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        wav = tmp_path / "src_16k.wav"
        self._write_wav(wav, sr=16000, seconds=0.6)
        model = AssemblyAIAPIModel(api_key="x", streaming=True, sample_rate=16000)

        frames = list(model._iter_pcm_frames(str(wav), frame_ms=300))
        total_samples = sum(len(f) // 2 for f in frames)
        assert total_samples == int(16000 * 0.6)

    def test_stereo_is_downmixed_to_mono(self, tmp_path):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        wav = tmp_path / "src_stereo.wav"
        self._write_wav(wav, sr=16000, seconds=0.5, channels=2)
        model = AssemblyAIAPIModel(api_key="x", streaming=True, sample_rate=16000)

        frames = list(model._iter_pcm_frames(str(wav), frame_ms=300))
        total_samples = sum(len(f) // 2 for f in frames)
        # Mono after downmix: 0.5s @ 16 kHz = 8000 samples (not 16000 interleaved).
        assert total_samples == 8000
