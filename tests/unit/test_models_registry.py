"""Tests for the model registry and factory functions."""

from unittest.mock import MagicMock, patch

import pytest

from psdn_sonar.models.base import MissingDependencyError
from psdn_sonar.models.registry import (
    _MODEL_CONFIGS,
    LANGUAGE_DEFAULT_MODELS,
    UnknownModelError,
    _raise_adapter_import_error,
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

    def test_banglaasr_v5_is_reachable_by_default(self):
        # Issue #212: registered but in no language default list, so one of
        # the strongest Bengali checkpoints of the dev5 pass never ran
        # unless named explicitly with --models.
        assert "banglaasr_v5" in get_language_defaults("bn")

    def test_korean_alias_duplicates_config_but_not_the_defaults(self):
        # wav2vec2_xlsr_korean is a deliberate backwards-compatibility alias
        # of kresnik_wav2vec2_large_xlsr_korean (issue #212); it must stay
        # constructible but never make the defaults run the same checkpoint
        # twice.
        from psdn_sonar.models.registry import _MODEL_CONFIGS

        assert _MODEL_CONFIGS["wav2vec2_xlsr_korean"] == _MODEL_CONFIGS["kresnik_wav2vec2_large_xlsr_korean"]
        defaults = get_language_defaults("ko")
        assert "kresnik_wav2vec2_large_xlsr_korean" in defaults
        assert "wav2vec2_xlsr_korean" not in defaults

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


class TestMissingExtraNamedOnImportFailure:
    """Issue #169: a core-only install asking for any local model failed with a
    bare 'No module named torch' and no mention of the [ml] extra that ships it."""

    @patch(
        "psdn_sonar.models.registry._import_class",
        side_effect=ModuleNotFoundError("No module named 'torch'", name="torch"),
    )
    def test_local_model_without_ml_extra_names_the_extra(self, mock_import):
        with pytest.raises(MissingDependencyError) as excinfo:
            create_model("whisper_base_en")

        text = str(excinfo.value)
        assert "No module named 'torch'" in text  # the original error is preserved
        assert 'pip install "psdn-sonar[ml]"' in text
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)

    def test_missing_dependency_error_is_a_runtime_error(self):
        # The multi CLI handler catches RuntimeError for a clean, traceback-free
        # exit; this pins the contract that registry import failures ride it.
        assert issubclass(MissingDependencyError, RuntimeError)

    def test_submodule_name_maps_through_its_top_level_package(self):
        exc = ModuleNotFoundError("No module named 'pyannote.audio'", name="pyannote.audio")
        with pytest.raises(MissingDependencyError, match=r"psdn-sonar\[pyannote\]"):
            _raise_adapter_import_error(exc)

    def test_peft_maps_to_the_bengali_extra(self):
        exc = ModuleNotFoundError("No module named 'peft'", name="peft")
        with pytest.raises(MissingDependencyError, match=r"psdn-sonar\[bengali\]"):
            _raise_adapter_import_error(exc)

    def test_module_without_a_known_extra_reraises_unchanged(self):
        exc = ModuleNotFoundError("No module named 'somethingelse'", name="somethingelse")
        with pytest.raises(ModuleNotFoundError) as excinfo:
            _raise_adapter_import_error(exc)
        assert excinfo.value is exc

    def test_broken_internal_class_path_reraises_unchanged(self):
        # A typo'd psdn_sonar class path must not be blamed on an extra.
        exc = ModuleNotFoundError(
            "No module named 'psdn_sonar.models.nonexistent'", name="psdn_sonar.models.nonexistent"
        )
        with pytest.raises(ModuleNotFoundError) as excinfo:
            _raise_adapter_import_error(exc)
        assert excinfo.value is exc


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
