"""Tests for the model registry and factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from psdn_sonar.models.registry import (
    _MODEL_CONFIGS,
    LANGUAGE_DEFAULT_MODELS,
    UnknownModelError,
    create_model,
    get_language_defaults,
    list_models,
)


class TestListModels:
    def test_returns_sorted_list(self):
        models = list_models()
        assert models == sorted(models)

    def test_contains_known_models(self):
        models = list_models()
        assert "whisper_api" in models
        assert "elevenlabs_api" in models
        assert "assemblyai_api" in models
        assert "banglaasr_v5" in models

    def test_matches_config_keys(self):
        models = list_models()
        assert set(models) == set(_MODEL_CONFIGS.keys())


class TestGetLanguageDefaults:
    def test_bengali(self):
        defaults = get_language_defaults("bn")
        assert defaults is not None
        assert "banglaspeech2text" in defaults

    def test_bengali_alias(self):
        assert get_language_defaults("bengali") == get_language_defaults("bn")

    def test_korean(self):
        defaults = get_language_defaults("ko")
        assert defaults is not None

    def test_unknown_language(self):
        assert get_language_defaults("zz") is None

    def test_case_insensitive(self):
        assert get_language_defaults("BN") == get_language_defaults("bn")


class TestCreateModel:
    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            create_model("nonexistent_model_xyz")

    def test_unknown_model_raises_dedicated_type(self):
        # Issue #168: callers must be able to tell "no such model" apart from
        # other ValueErrors an adapter constructor raises (e.g. a missing API
        # key), so the unknown-model error has its own type.
        with pytest.raises(UnknownModelError):
            create_model("nonexistent_model_xyz")

    def test_unknown_model_error_is_a_value_error(self):
        # Backwards compatibility: existing `except ValueError` callers keep working.
        assert issubclass(UnknownModelError, ValueError)

    @patch("psdn_sonar.models.registry._import_class")
    def test_creates_registered_model(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        create_model("whisper_api")
        mock_cls.assert_called_once()

    @patch("psdn_sonar.models.registry._import_class")
    def test_kwargs_override(self, mock_import):
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        create_model("whisper_base_en", device=-1)
        mock_cls.assert_called_once_with(model_id="openai/whisper-base", device=-1)

    # NOTE: the custom_hf_model path is covered in the HuggingFace adapter
    # tests — patching it requires psdn_sonar.models.huggingface to exist.


class TestModelConfigs:
    def test_all_configs_have_class_path_and_kwargs(self):
        for name, config in _MODEL_CONFIGS.items():
            assert len(config) == 2, f"Model {name} config should be (class_path, kwargs)"
            class_path, kwargs = config
            assert isinstance(class_path, str), f"Model {name} class_path should be str"
            assert isinstance(kwargs, dict), f"Model {name} kwargs should be dict"

    def test_language_aliases_resolved(self):
        assert LANGUAGE_DEFAULT_MODELS["bengali"] is LANGUAGE_DEFAULT_MODELS["bn"]
        assert LANGUAGE_DEFAULT_MODELS["korean"] is LANGUAGE_DEFAULT_MODELS["ko"]
        assert LANGUAGE_DEFAULT_MODELS["hindi"] is LANGUAGE_DEFAULT_MODELS["hi"]
        assert LANGUAGE_DEFAULT_MODELS["english"] is LANGUAGE_DEFAULT_MODELS["en"]
