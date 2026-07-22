"""Tests for the ASR backend layer.

Setup/config resolution and registry wiring are exercised against the real
``conf/`` tree via ``load_config``; the transcribe path is hermetic (the
transformers pipeline is mocked), so nothing here needs the ``[ml]`` extra.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from psdn_sonar.config_loader import load_config


def _setup_backend(language="bn"):
    import psdn_sonar.backends  # noqa: F401 — triggers registration
    from psdn_sonar.registry import get_asr_backend

    backend = get_asr_backend("huggingface")()
    backend.setup(load_config(language=language, backend="huggingface"))
    return backend


class TestBackendRegistry:
    def test_huggingface_backend_registered(self):
        import psdn_sonar.backends  # noqa: F401
        from psdn_sonar.backends.base import ASRBackend
        from psdn_sonar.registry import get_asr_backend

        cls = get_asr_backend("huggingface")
        assert issubclass(cls, ASRBackend)

    def test_unknown_backend_raises(self):
        from psdn_sonar.registry import get_asr_backend

        with pytest.raises(ValueError, match="Unknown ASR backend"):
            get_asr_backend("nonexistent_backend")


class TestHuggingFaceBackendSetup:
    @pytest.mark.parametrize(
        ("language", "expected_model"),
        [
            ("bn", "bangla-speech-processing/BanglaASR"),
            ("ko", "kresnik/wav2vec2-large-xlsr-korean"),
            ("hi", "vasista22/whisper-hindi-large-v2"),
            ("en", "openai/whisper-base"),
        ],
    )
    def test_resolves_default_model_per_language(self, language, expected_model):
        backend = _setup_backend(language)
        assert backend.model_name == expected_model

    def test_missing_default_model_raises(self):
        backend = _setup_backend("bn")
        config = load_config(language="bn", backend="huggingface")
        config.language.code = "zz"  # no default_zz key in the backend config
        with pytest.raises(ValueError, match="No default model configured"):
            backend.setup(config)

    def test_supports_language_reflects_config_keys(self):
        backend = _setup_backend("bn")
        assert backend.supports_language("bn") is True
        assert backend.supports_language("ko") is True
        assert backend.supports_language("zz") is False

    def test_supports_language_false_before_setup(self):
        import psdn_sonar.backends  # noqa: F401
        from psdn_sonar.registry import get_asr_backend

        backend = get_asr_backend("huggingface")()
        assert backend.supports_language("bn") is False


class TestHuggingFaceBackendTranscribe:
    def test_lazy_pipeline_load_and_reuse(self):
        backend = _setup_backend("bn")

        fake_pipe = MagicMock(return_value={"text": "  hello  "})
        fake_pipeline_factory = MagicMock(return_value=fake_pipe)
        with patch.dict("sys.modules", {"transformers": MagicMock(pipeline=fake_pipeline_factory)}):
            out1 = backend.transcribe(Path("a.wav"), "bn")
            out2 = backend.transcribe(Path("b.wav"), "bn")

        assert out1 == "  hello  "
        assert out2 == "  hello  "
        # The pipeline is constructed once and reused across calls.
        fake_pipeline_factory.assert_called_once()
        assert fake_pipe.call_count == 2

    def test_teardown_releases_pipeline(self):
        backend = _setup_backend("bn")
        backend.pipe = MagicMock()
        backend.teardown()
        assert backend.pipe is None
