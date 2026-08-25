"""Centralized ASR model registry.

All model name -> class mappings live here. Both single-speaker and
multi-speaker pipelines resolve models through this registry.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class UnknownModelError(ValueError):
    """Raised by :func:`create_model` when the model name is not registered.

    Subclasses ``ValueError`` for backwards compatibility, but lets callers
    distinguish "no such model" from other ``ValueError``\\s an adapter's
    constructor may raise — e.g. a missing API key (issue #168), which must
    reach the user instead of being reported as an unknown model.
    """


# Maps model name -> (class_path_string, default_kwargs)
# Using strings for class paths avoids importing heavy ML libraries at import time.
_MODEL_CONFIGS: Dict[str, Tuple[str, dict]] = {
    # Bengali HuggingFace models
    "banglaspeech2text": (
        "psdn_sonar.models.huggingface.BanglaSpeech2TextModel",
        {"model_id": "anuragshas/whisper-large-v2-bn"},
    ),
    "khushids_bengali": (
        "psdn_sonar.models.huggingface.KhushiDSBengaliModel",
        {"model_id": "KhushiDS/whisper-large-v3-Bengali"},
    ),
    "tugstugi_bengali": (
        "psdn_sonar.models.huggingface.StandardHuggingFaceASR",
        {"model_id": "bengaliAI/tugstugi_bengaliai-asr_whisper-medium"},
    ),
    "tugstugi_bengali_regional": (
        "psdn_sonar.models.huggingface.StandardHuggingFaceASR",
        {"model_id": "bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium"},
    ),
    "banglaasr": ("psdn_sonar.models.huggingface.BanglaASRModel", {}),
    "wav2vec2_bengali": ("psdn_sonar.models.huggingface.Wav2Vec2BengaliModel", {}),
    # Korean HuggingFace models
    "kresnik_wav2vec2_large_xlsr_korean": (
        "psdn_sonar.models.huggingface.Wav2Vec2KoreanModel",
        {"model_id": "kresnik/wav2vec2-large-xlsr-korean"},
    ),
    "wav2vec2_xlsr_korean": (
        "psdn_sonar.models.huggingface.Wav2Vec2KoreanModel",
        {"model_id": "kresnik/wav2vec2-large-xlsr-korean"},
    ),
    "wav2vec2_base_korean": (
        "psdn_sonar.models.huggingface.Wav2Vec2KoreanModel",
        {"model_id": "Kkonjeong/wav2vec2-base-korean"},
    ),
    "whisper_small_ko": ("psdn_sonar.models.huggingface.WhisperKoreanModel", {"model_id": "SungBeom/whisper-small-ko"}),
    # Hindi HuggingFace models
    "whisper_hindi_large_v2": (
        "psdn_sonar.models.huggingface.StandardHuggingFaceASR",
        {"model_id": "vasista22/whisper-hindi-large-v2"},
    ),
    "whisper_small_hi": (
        "psdn_sonar.models.huggingface.StandardHuggingFaceASR",
        {"model_id": "openai/whisper-small", "language": "hi"},
    ),
    # English HuggingFace models
    "whisper_base_en": ("psdn_sonar.models.huggingface.StandardHuggingFaceASR", {"model_id": "openai/whisper-base"}),
    "whisper_small_en": ("psdn_sonar.models.huggingface.StandardHuggingFaceASR", {"model_id": "openai/whisper-small"}),
    # API models
    "elevenlabs_api": ("psdn_sonar.models.apis.ElevenLabsAPIModel", {}),
    "whisper_api": ("psdn_sonar.models.apis.WhisperAPIModel", {}),
    "assemblyai_api": ("psdn_sonar.models.apis.AssemblyAIAPIModel", {}),
    # Special-case models
    "banglaasr_v5": ("psdn_sonar.models.huggingface.BanglaASRV5Model", {}),
}

LANGUAGE_DEFAULT_MODELS: Dict[str, List[str]] = {
    "bn": [
        "banglaspeech2text",
        "khushids_bengali",
        "tugstugi_bengali",
        "tugstugi_bengali_regional",
        "banglaasr",
        "wav2vec2_bengali",
        "elevenlabs_api",
        "whisper_api",
        "assemblyai_api",
    ],
    "ko": [
        "kresnik_wav2vec2_large_xlsr_korean",
        "wav2vec2_base_korean",
        "whisper_small_ko",
        "elevenlabs_api",
        "whisper_api",
        "assemblyai_api",
    ],
    "hi": [
        "whisper_hindi_large_v2",
        "whisper_small_hi",
        "elevenlabs_api",
        "whisper_api",
        "assemblyai_api",
    ],
    "en": [
        "whisper_base_en",
        "whisper_small_en",
        "elevenlabs_api",
        "whisper_api",
        "assemblyai_api",
    ],
}

# Long-name aliases share the same list objects as their ISO 639-1 codes.
LANGUAGE_DEFAULT_MODELS["bengali"] = LANGUAGE_DEFAULT_MODELS["bn"]
LANGUAGE_DEFAULT_MODELS["korean"] = LANGUAGE_DEFAULT_MODELS["ko"]
LANGUAGE_DEFAULT_MODELS["hindi"] = LANGUAGE_DEFAULT_MODELS["hi"]
LANGUAGE_DEFAULT_MODELS["english"] = LANGUAGE_DEFAULT_MODELS["en"]


def _import_class(class_path: str):
    """Import a class from a dotted path like 'psdn_sonar.models.apis.WhisperAPIModel'."""
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def create_model(name: str, *, custom_hf_model: Optional[str] = None, language: Optional[str] = None, **kwargs) -> Any:
    """Create an ASR model by name.

    Args:
        name: Registered model name (e.g. 'wav2vec2_bengali', 'elevenlabs_api')
        custom_hf_model: If set, creates a CustomHuggingFaceModel with this model ID
        language: Language code for language-aware models
        **kwargs: Override default kwargs for the model class

    Returns:
        An ASRModel instance

    Raises:
        UnknownModelError: If model name is not registered and no custom_hf_model
            given (a ``ValueError`` subclass)
    """
    if custom_hf_model:
        from psdn_sonar.models.huggingface import CustomHuggingFaceModel

        logger.info(f"Loading custom HuggingFace model: {custom_hf_model}")
        return CustomHuggingFaceModel(model_id=custom_hf_model, language=language)

    config = _MODEL_CONFIGS.get(name)
    if config is None:
        raise UnknownModelError(f"Unknown model '{name}'. Available: {list_models()}")

    class_path, default_kwargs = config
    merged_kwargs = {**default_kwargs, **kwargs}
    cls = _import_class(class_path)
    return cls(**merged_kwargs)


def list_models() -> List[str]:
    """Return all registered model names."""
    return sorted(_MODEL_CONFIGS.keys())


def get_model_config(name: str) -> Optional[Tuple[str, dict]]:
    """``(class_path, kwargs)`` for a registered model name, or None."""
    return _MODEL_CONFIGS.get(name)


def get_language_defaults(language: str) -> Optional[List[str]]:
    """Return default model names for a language, or None if unknown."""
    return LANGUAGE_DEFAULT_MODELS.get(language.lower())
