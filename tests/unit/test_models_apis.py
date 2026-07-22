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
  * ``AssemblyAIAPIModel`` — streaming TTFT capture (via a fake realtime
    transcriber), fallback to batch on streaming failure, batch path leaving
    ``ttft_s`` None, and PCM framing (resample / passthrough / downmix).
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
# AssemblyAI streaming TTFT (fake realtime transcriber) + batch path
# ---------------------------------------------------------------------------


def _install_assemblyai_stub() -> None:
    if "assemblyai" in sys.modules:
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

    class _RealtimeTranscriber:  # pragma: no cover — replaced via monkeypatch
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    aai = _types.ModuleType("assemblyai")
    aai.settings = _Settings()
    aai.TranscriptionConfig = _TranscriptionConfig
    aai.Transcriber = _Transcriber
    aai.RealtimeTranscriber = _RealtimeTranscriber
    sys.modules["assemblyai"] = aai


class _FakeRealtime:
    """Simulates AssemblyAI's realtime transcriber: a partial then a final."""

    def __init__(self, on_data, on_error):
        self.on_data = on_data
        self.on_error = on_error
        self._frames = 0

    def connect(self):
        pass

    def stream(self, frame):
        self._frames += 1
        if self._frames == 1:
            time.sleep(0.005)  # ensure a measurable, non-zero TTFT
            self.on_data(SimpleNamespace(text="hello", is_final=False))
        elif self._frames == 2:
            self.on_data(SimpleNamespace(text="hello world", is_final=True))

    def close(self):
        pass


class TestAssemblyAIStreaming:
    def test_supports_latency_metrics_flag(self):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        assert AssemblyAIAPIModel.supports_latency_metrics is True

    def test_streaming_captures_ttft(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=True)
        monkeypatch.setattr(
            model, "_make_realtime_transcriber", lambda on_data, on_error: _FakeRealtime(on_data, on_error)
        )
        monkeypatch.setattr(model, "_iter_pcm_frames", lambda audio_path: [b"\x00\x00", b"\x00\x00"])

        result = model.transcribe("clip.wav")
        assert isinstance(result, tuple)
        text, lm = result
        assert text == "hello world"
        assert lm.ttft_s is not None
        assert lm.complete_s >= lm.ttft_s

    def test_streaming_failure_falls_back_to_batch(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=True)

        def _boom(*a, **kw):
            raise RuntimeError("ws connect failed")

        monkeypatch.setattr(model, "_make_realtime_transcriber", _boom)
        monkeypatch.setattr(model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="batch fallback"))

        text, lm = model.transcribe("clip.wav")
        assert text == "batch fallback"
        assert lm.ttft_s is None  # batch path → no TTFT

    def test_batch_path_returns_complete_only(self, monkeypatch):
        _install_assemblyai_stub()
        from psdn_sonar.models.apis import AssemblyAIAPIModel

        model = AssemblyAIAPIModel(api_key="x", streaming=False)
        monkeypatch.setattr(model.transcriber, "transcribe", lambda ap: SimpleNamespace(text="batch text"))

        text, lm = model.transcribe("clip.wav")
        assert text == "batch text"
        assert lm.ttft_s is None
        assert lm.complete_s is not None


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
