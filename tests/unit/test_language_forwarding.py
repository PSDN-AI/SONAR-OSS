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
