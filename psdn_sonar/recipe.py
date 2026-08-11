"""Language-agnostic ASR evaluation recipes.

A recipe bundles everything needed to evaluate a language: the default model
roster (from the model registry), the public datasets, and canonical text
normalization/tokenization. Users provide the language and optionally a path
to their own dataset.

Example:
    recipe = get_recipe("bengali", "path/to/my/dataset")
    normalized = recipe.normalize(text)
    tokens = recipe.tokenize(normalized)
"""

import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from psdn_sonar.models.registry import get_language_defaults, get_model_config
from psdn_sonar.utils.text_processing import normalize_text_unified

logger = logging.getLogger(__name__)

_LANGUAGE_CODES = {"bengali": "bn", "hindi": "hi", "english": "en", "korean": "ko"}

_LANGUAGE_ALIASES = {
    "bn": "bengali",
    "hi": "hindi",
    "en": "english",
    "ko": "korean",
    **{name: name for name in _LANGUAGE_CODES},
}

_LANGUAGE_DATASETS: Dict[str, List[Dict]] = {
    "bengali": [
        {"name": "common_voice", "path": "mozilla-foundation/common_voice_13_0"},
        {"name": "fleurs", "path": "google/fleurs"},
        {"name": "openslr37_bd", "path": "openslr/SLR37"},
        {"name": "openslr37_in", "path": "openslr/SLR37"},
        {"name": "openslr53", "path": "openslr/SLR53"},
    ],
    "hindi": [
        {"name": "common_voice", "path": "mozilla-foundation/common_voice_13_0"},
        {"name": "fleurs", "path": "google/fleurs"},
    ],
    "english": [
        {"name": "librispeech", "path": "librispeech_asr"},
        {"name": "common_voice", "path": "mozilla-foundation/common_voice_13_0"},
    ],
    "korean": [
        {"name": "common_voice", "path": "mozilla-foundation/common_voice_13_0"},
        {"name": "fleurs", "path": "google/fleurs"},
    ],
}

_API_PROVIDERS = {
    "elevenlabs_api": "elevenlabs",
    "whisper_api": "openai",
    "assemblyai_api": "assemblyai",
}


def _model_entry(name: str) -> Dict:
    """Recipe model descriptor for a registry model name."""
    if name in _API_PROVIDERS:
        return {"name": name, "type": "api", "provider": _API_PROVIDERS[name]}
    entry = {"name": name, "type": "finetuned", "provider": "huggingface"}
    _, kwargs = get_model_config(name) or (None, {})
    if "model_id" in kwargs:
        entry["model_id"] = kwargs["model_id"]
    return entry


@dataclass
class Recipe:
    """Evaluation configuration for one language: models, datasets, normalization."""

    language: str
    models: List[Dict]
    datasets: List[Dict]

    def normalize(self, text: str) -> str:
        """Normalize text with the canonical pipeline used during evaluation."""
        return normalize_text_unified(text, language=_LANGUAGE_CODES[self.language])

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Whitespace tokenization."""
        return text.split()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def display_models(self) -> str:
        return json.dumps(self.models, indent=2, ensure_ascii=False)

    def display_datasets(self) -> str:
        return json.dumps(self.datasets, indent=2, ensure_ascii=False)


class RecipeFactory:
    """Creates language-specific recipes from the model registry and dataset table."""

    @classmethod
    def create(cls, language: str, dataset_path: Optional[str] = None) -> Recipe:
        """Recipe for *language* (name or ISO 639-1 code).

        Raises ValueError for unsupported languages.
        """
        canonical = _LANGUAGE_ALIASES.get(language.lower())
        if canonical is None:
            supported = ", ".join(sorted(_LANGUAGE_CODES))
            raise ValueError(f"Language '{language}' not supported. Supported: {supported}")

        models = [_model_entry(name) for name in get_language_defaults(canonical) or []]
        datasets = list(_LANGUAGE_DATASETS[canonical])
        if dataset_path:
            datasets.append({"name": "user_dataset", "path": dataset_path})

        return Recipe(language=canonical, models=models, datasets=datasets)


def get_recipe(language: str, dataset_path: Optional[str] = None) -> Recipe:
    """Recipe for *language*, optionally registering a user dataset path."""
    return RecipeFactory.create(language, dataset_path)
