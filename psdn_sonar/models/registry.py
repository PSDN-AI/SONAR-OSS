"""Centralized ASR model registry.

All model name -> class mappings live here. Both single-speaker and
multi-speaker pipelines resolve models through this registry.
"""

import inspect
import logging
from typing import Any, Dict, List, NoReturn, Optional, Tuple

from psdn_sonar.models.base import MissingDependencyError

logger = logging.getLogger(__name__)

# Top-level module name -> the install extra that provides it. Used to turn a
# bare ModuleNotFoundError raised while importing an adapter module into an
# actionable "install this extra" message (issue #169: a core-only install
# asking for any local model got only "No module named 'torch'").
_EXTRA_FOR_MODULE = {
    "torch": "ml",
    "torchaudio": "ml",
    "transformers": "ml",
    "sentence_transformers": "ml",
    "speechmos": "ml",
    "onnxruntime": "ml",
    "peft": "bengali",
    "pyannote": "pyannote",
    "openai": "apis",
    "assemblyai": "apis",
}


def _raise_adapter_import_error(exc: ModuleNotFoundError) -> NoReturn:
    """Re-raise an adapter-module import failure, naming the install extra.

    Only translates modules that ship with a known extra; anything else —
    including a broken ``psdn_sonar``-internal class path — re-raises
    unchanged rather than guessing.
    """
    top_level = (exc.name or "").partition(".")[0]
    extra = _EXTRA_FOR_MODULE.get(top_level)
    if extra is None:
        raise exc
    raise MissingDependencyError(
        f"{exc} — the '{top_level}' package ships with the [{extra}] extra. "
        f'Install with: pip install "psdn-sonar[{extra}]"'
    ) from exc


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
    # Intentional alias of kresnik_wav2vec2_large_xlsr_korean (same class and
    # model_id), kept for backwards compatibility with existing --models
    # invocations; deliberately not in any language default list so the
    # checkpoint is never evaluated twice in one run (issue #212).
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
        # Registered but reachable only via --models until issue #212 noted
        # it was among the strongest Bengali results in the dev5 pass.
        "banglaasr_v5",
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


def create_model(
    name: str,
    *,
    custom_hf_model: Optional[str] = None,
    language: Optional[str] = None,
    streaming: Optional[bool] = None,
    **kwargs,
) -> Any:
    """Create an ASR model by name.

    Args:
        name: Registered model name (e.g. 'wav2vec2_bengali', 'elevenlabs_api')
        custom_hf_model: If set, creates a CustomHuggingFaceModel with this model ID
        language: Language code (ISO 639-1) forwarded to any constructor that
            declares a ``language`` parameter, unless the registry entry pins
            one (e.g. ``whisper_small_hi``). Before issue #186 this argument
            was used only on the ``custom_hf_model`` branch, so registered
            models — the hosted API adapters above all — were always built
            with their constructor defaults and every AssemblyAI/ElevenLabs
            request went out saying Bengali regardless of ``--language``.
        streaming: Forwarded to constructors that declare a ``streaming``
            parameter (``assemblyai_api``, which records ``ttft_s`` in that
            mode). Requesting it for a model without a streaming mode logs a
            warning and runs batch — one incapable model must not abort a
            multi-model run. ``None`` means "not requested".
        **kwargs: Override default kwargs for the model class

    Returns:
        An ASRModel instance

    Raises:
        UnknownModelError: If model name is not registered and no custom_hf_model
            given (a ``ValueError`` subclass)
        MissingDependencyError: If the adapter module needs a package that ships
            with an install extra which is not installed (e.g. torch / ``[ml]``)
    """
    if custom_hf_model:
        if streaming:
            logger.warning(
                "Custom HuggingFace model '%s' has no streaming mode; running in the "
                "batch protocol (ttft_s stays null).",
                custom_hf_model,
            )
        try:
            from psdn_sonar.models.huggingface import CustomHuggingFaceModel
        except ModuleNotFoundError as exc:
            _raise_adapter_import_error(exc)

        logger.info(f"Loading custom HuggingFace model: {custom_hf_model}")
        return CustomHuggingFaceModel(model_id=custom_hf_model, language=language)

    config = _MODEL_CONFIGS.get(name)
    if config is None:
        raise UnknownModelError(f"Unknown model '{name}'. Available: {list_models()}")

    class_path, default_kwargs = config
    merged_kwargs = {**default_kwargs, **kwargs}
    try:
        cls = _import_class(class_path)
    except ModuleNotFoundError as exc:
        _raise_adapter_import_error(exc)

    try:
        init_params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover — C-implemented __init__
        init_params = {}
    if language is not None and "language" in init_params and "language" not in merged_kwargs:
        merged_kwargs["language"] = language
    if streaming is not None and "streaming" not in merged_kwargs:
        if "streaming" in init_params:
            merged_kwargs["streaming"] = streaming
        elif streaming:
            logger.warning(
                "Model '%s' has no streaming mode; running in the batch protocol (ttft_s stays null).",
                name,
            )
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
