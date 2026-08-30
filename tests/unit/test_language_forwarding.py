"""Issue #186: --language must reach a registered model's constructor, and
AssemblyAI's streaming/ttft mode must be reachable.

``create_model`` used its ``language`` argument only on the
``custom_hf_model`` branch; registered models were built from registry
kwargs alone, and the three hosted-API adapters are registered with empty
kwargs — so their constructor defaults always won. Every AssemblyAI request
said Bengali (``--language en`` runs included), every ElevenLabs request
sent ``ben``, and ``streaming=True`` (the only way to measure ``ttft_s``)
had no entry point anywhere.

Hermetic like tests/unit/test_models_apis.py: vendor SDKs are stubbed via
``sys.modules`` when absent, ElevenLabs needs only ``requests`` (never
constructed with a real session here), and no network is touched.
"""

from __future__ import annotations

import sys
import types as _types
from types import SimpleNamespace

import pytest

from psdn_sonar.models import registry

# ---------------------------------------------------------------------------
# SDK stubs (installed only when the real package is absent)
# ---------------------------------------------------------------------------


def _ensure_streaming_v3_stub() -> None:
    """The adapter's streaming mode imports ``assemblyai.streaming.v3``
    (issue #208); the stub must provide it like the real SDK does."""
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


def _install_openai_stub() -> None:
    if "openai" in sys.modules:
        return

    class _OpenAI:
        def __init__(self, api_key=None):
            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=lambda **kw: SimpleNamespace(text="")))

    mod = _types.ModuleType("openai")
    mod.OpenAI = _OpenAI
    sys.modules["openai"] = mod


# ---------------------------------------------------------------------------
# create_model forwarding mechanics (stub adapter classes, no SDKs at all)
# ---------------------------------------------------------------------------


class _LanguageAwareStub:
    def __init__(self, language=None, streaming=False):
        self.language = language
        self.streaming = streaming


class _LanguageBlindStub:
    def __init__(self):
        pass


class TestCreateModelForwarding:
    @pytest.fixture
    def register(self, monkeypatch):
        """Register a stub class under a fake name with given default kwargs."""

        def _register(cls, default_kwargs=None):
            monkeypatch.setitem(registry._MODEL_CONFIGS, "stub_model", ("ignored.by.monkeypatch", default_kwargs or {}))
            monkeypatch.setattr(registry, "_import_class", lambda path: cls)
            return "stub_model"

        return _register

    def test_language_reaches_a_constructor_that_accepts_it(self, register):
        name = register(_LanguageAwareStub)
        model = registry.create_model(name, language="ko")
        assert model.language == "ko"

    def test_registry_pinned_language_wins(self, register):
        """whisper_small_hi pins language='hi' in the registry; a mismatched
        --language must not override what the entry knows about the model."""
        name = register(_LanguageAwareStub, {"language": "hi"})
        model = registry.create_model(name, language="ko")
        assert model.language == "hi"

    def test_language_blind_constructor_is_not_passed_language(self, register):
        name = register(_LanguageBlindStub)
        model = registry.create_model(name, language="ko")  # must not TypeError
        assert isinstance(model, _LanguageBlindStub)

    def test_streaming_reaches_a_constructor_that_accepts_it(self, register):
        name = register(_LanguageAwareStub)
        model = registry.create_model(name, streaming=True)
        assert model.streaming is True

    def test_streaming_request_on_incapable_model_warns_and_runs_batch(self, register, caplog):
        name = register(_LanguageBlindStub)
        with caplog.at_level("WARNING"):
            model = registry.create_model(name, streaming=True)
        assert isinstance(model, _LanguageBlindStub)
        assert "no streaming mode" in caplog.text
        assert "ttft_s" in caplog.text

    def test_no_streaming_request_no_warning(self, register, caplog):
        name = register(_LanguageBlindStub)
        with caplog.at_level("WARNING"):
            registry.create_model(name, streaming=None)
            registry.create_model(name)
        assert "streaming" not in caplog.text

    def test_custom_hf_model_with_streaming_warns(self, monkeypatch, caplog):
        class _CustomStub:
            def __init__(self, model_id, language=None):
                self.model_id = model_id
                self.language = language

        fake_hf = _types.ModuleType("psdn_sonar.models.huggingface")
        fake_hf.CustomHuggingFaceModel = _CustomStub
        monkeypatch.setitem(sys.modules, "psdn_sonar.models.huggingface", fake_hf)
        with caplog.at_level("WARNING"):
            model = registry.create_model("ignored", custom_hf_model="org/model", language="hi", streaming=True)
        assert "no streaming mode" in caplog.text
        assert model.language == "hi"


# ---------------------------------------------------------------------------
# The real registry entries end to end (the issue's Effect 1)
# ---------------------------------------------------------------------------


class TestApiAdaptersReceiveTheLanguage:
    def test_elevenlabs_language_forwarded_and_converted(self):
        model = registry.create_model("elevenlabs_api", language="hi", api_key="test-key")
        assert model._language_code == "hin"  # ISO 639-1 -> vendor's 639-3

    def test_elevenlabs_default_stays_bengali(self):
        model = registry.create_model("elevenlabs_api", api_key="test-key")
        assert model._language_code == "ben"

    def test_elevenlabs_explicit_language_code_wins(self):
        from psdn_sonar.models.apis import ElevenLabsAPIModel

        model = ElevenLabsAPIModel(api_key="test-key", language_code="kor", language="hi")
        assert model._language_code == "kor"

    def test_elevenlabs_unmapped_language_passes_through(self):
        # The endpoint accepts ISO 639-1 as well, so an unmapped code is sent
        # as-is rather than dropped.
        from psdn_sonar.models.apis import ElevenLabsAPIModel

        model = ElevenLabsAPIModel(api_key="test-key", language="sw")
        assert model._language_code in ("sw", "swa")

    def test_assemblyai_language_and_streaming_forwarded(self):
        _install_assemblyai_stub()
        model = registry.create_model("assemblyai_api", language="ko", streaming=True, api_key="test-key")
        assert model.config.language_code == "ko"
        assert model.streaming is True

    def test_assemblyai_default_stays_bengali(self):
        _install_assemblyai_stub()
        model = registry.create_model("assemblyai_api", api_key="test-key")
        assert model.config.language_code == "bn"
        assert model.streaming is False

    def test_whisper_api_language_forwarded(self):
        _install_openai_stub()
        model = registry.create_model("whisper_api", language="ko", api_key="test-key")
        assert model.language == "ko"


# ---------------------------------------------------------------------------
# scores.json protocol follows the model that ran (ties into #184's contract)
# ---------------------------------------------------------------------------


class TestSubmissionProtocolFollowsTheModel:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in ("SONAR_PROTOCOL", "SONAR_PROVIDER", "SONAR_REGION"):
            monkeypatch.delenv(var, raising=False)

    def test_streaming_model_records_streaming_protocol(self):
        from psdn_sonar.evaluators.single_speaker import _default_submission_for_model

        model = SimpleNamespace(provider="assemblyai", provider_model_id=None, streaming=True)
        cfg = _default_submission_for_model(model, "assemblyai_api", language="en")
        assert cfg.protocol == "streaming"

    def test_batch_model_records_batch_protocol(self):
        from psdn_sonar.evaluators.single_speaker import _default_submission_for_model

        model = SimpleNamespace(provider="assemblyai", provider_model_id=None, streaming=False)
        cfg = _default_submission_for_model(model, "assemblyai_api", language="en")
        assert cfg.protocol == "batch"

    def test_env_override_still_wins(self, monkeypatch):
        from psdn_sonar.evaluators.single_speaker import _default_submission_for_model

        monkeypatch.setenv("SONAR_PROTOCOL", "batch")
        model = SimpleNamespace(provider="assemblyai", provider_model_id=None, streaming=True)
        cfg = _default_submission_for_model(model, "assemblyai_api", language="en")
        assert cfg.protocol == "batch"

    def test_streaming_run_that_fell_back_records_batch(self):
        # Issue #208: protocol used to come from the requested mode, so a
        # run whose every utterance fell back to batch recorded streaming.
        from psdn_sonar.evaluators.single_speaker import _default_submission_for_model

        model = SimpleNamespace(
            provider="assemblyai",
            provider_model_id=None,
            streaming=True,
            streaming_fallbacks=2,
            streamed_utterances=0,
        )
        cfg = _default_submission_for_model(model, "assemblyai_api", language="en")
        assert cfg.protocol == "batch"

    def test_streaming_run_with_no_fallbacks_records_streaming(self):
        from psdn_sonar.evaluators.single_speaker import _default_submission_for_model

        model = SimpleNamespace(
            provider="assemblyai",
            provider_model_id=None,
            streaming=True,
            streaming_fallbacks=0,
            streamed_utterances=2,
        )
        cfg = _default_submission_for_model(model, "assemblyai_api", language="en")
        assert cfg.protocol == "streaming"


class TestStreamingFallbackWarning:
    """Issue #208: the per-utterance fallback existed only in the terminal;
    scores.json said protocol streaming with empty warnings and null TTFT."""

    def test_warning_names_counts_and_reason(self):
        from psdn_sonar.evaluators.single_speaker import _streaming_fallback_warning

        model = SimpleNamespace(
            streaming=True,
            streaming_fallbacks=2,
            streamed_utterances=1,
            last_streaming_error="module 'assemblyai' has no attribute 'RealtimeTranscriber'",
        )
        warning = _streaming_fallback_warning(model, "assemblyai_api")
        assert warning is not None
        assert "2 of 3" in warning
        assert "assemblyai_api" in warning
        assert "RealtimeTranscriber" in warning
        assert "batch" in warning

    def test_silent_when_nothing_fell_back(self):
        from psdn_sonar.evaluators.single_speaker import _streaming_fallback_warning

        clean = SimpleNamespace(streaming=True, streaming_fallbacks=0, streamed_utterances=3)
        assert _streaming_fallback_warning(clean, "m") is None

    def test_silent_for_batch_models_without_the_attributes(self):
        from psdn_sonar.evaluators.single_speaker import _streaming_fallback_warning

        assert _streaming_fallback_warning(SimpleNamespace(), "whisper_base_en") is None


class TestFallbackRecordedInArtifact:
    """End-to-end through run_evaluation: a streaming run that fell back on
    every utterance must produce an artifact saying protocol batch, with the
    fallback warning in the warnings array (issue #208)."""

    def _run(self, tmp_path, monkeypatch, fallbacks):
        import json

        from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        for var in ("SONAR_PROTOCOL", "SONAR_PROVIDER", "SONAR_REGION"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *a, **k: [{"audio_path": "clip.wav", "ground_truth": "hello world"}],
        )
        model = SimpleNamespace(
            provider="assemblyai",
            provider_model_id=None,
            streaming=True,
            streaming_fallbacks=fallbacks,
            streamed_utterances=1 - min(fallbacks, 1),
            last_streaming_error="ws connect failed" if fallbacks else None,
        )
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", lambda *a, **k: model)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *a, **k: {
                "model_name": "assemblyai_api",
                "results": [],
                "summary": {
                    "total_samples": 1,
                    "successful": 1,
                    "failed": 0,
                    "avg_wer": 0.1,
                    "avg_cer": 0.05,
                    "elapsed_time": 0.1,
                    "avg_latency_s": None,
                    "median_latency_s": None,
                    "p95_latency_s": None,
                },
            },
        )
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["assemblyai_api"],
            language="en",
            write_scores=True,
            compute_sem=False,
        )
        return json.loads((tmp_path / "scores_assemblyai_api.json").read_text(encoding="utf-8"))

    def test_fallback_run_records_batch_and_warns(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path, monkeypatch, fallbacks=1)
        assert payload["submission"]["protocol"] == "batch"
        fallback_warnings = [w for w in payload["warnings"] if "fell back to the batch protocol" in w]
        assert len(fallback_warnings) == 1
        assert "ws connect failed" in fallback_warnings[0]
        assert "fell back to the batch protocol" in caplog.text

    def test_clean_streaming_run_records_streaming_with_no_warning(self, tmp_path, monkeypatch, caplog):
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path, monkeypatch, fallbacks=0)
        assert payload["submission"]["protocol"] == "streaming"
        assert payload["warnings"] == []
