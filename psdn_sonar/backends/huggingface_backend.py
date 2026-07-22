"""HuggingFace ``transformers``-pipeline ASR backend.

Resolves the model from the run config's ``backend.model.default_<lang>``
key (see ``conf/backend/huggingface.yaml``) rather than taking a model name
directly — use the adapters in :mod:`psdn_sonar.models` when you want to
pin a specific checkpoint. ``transformers`` is imported only on the first
``transcribe`` call, so constructing and setting up the backend works
without the ``[ml]`` extra installed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from ..registry import register_asr
from .base import ASRBackend

logger = logging.getLogger(__name__)


@register_asr("huggingface")
class HuggingFaceBackend(ASRBackend):
    def __init__(self):
        self.pipe = None
        self.model_name: Optional[str] = None
        self.device: Optional[str] = None
        self.config = None

    def setup(self, config: Any) -> None:
        self.config = config
        backend_config = config.backend
        language_code = config.language.code

        model_key = f"default_{language_code}"
        self.model_name = getattr(backend_config.model, model_key, None)

        if not self.model_name:
            raise ValueError(
                f"No default model configured for language: {language_code}. "
                f"Add 'default_{language_code}' to conf/backend/huggingface.yaml"
            )

        self.device = backend_config.pipeline.device
        logger.info(f"HuggingFace backend configured: model={self.model_name}, device={self.device}")

    def transcribe(self, audio_path: Path, language: str) -> str:
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers")

        if self.pipe is None:
            logger.info(f"Loading HuggingFace ASR model: {self.model_name}")
            self.pipe = hf_pipeline("automatic-speech-recognition", model=self.model_name, device=self.device)

        result = self.pipe(str(audio_path))
        # Single-input calls return {"text": ...}; the isinstance check
        # narrows the pipeline's union return type (batched calls return a
        # list, which this backend never issues).
        return str(result["text"]) if isinstance(result, dict) else ""

    def teardown(self) -> None:
        self.pipe = None
        logger.debug("HuggingFace backend torn down")

    def supports_language(self, language: str) -> bool:
        if not self.config:
            return False
        model_key = f"default_{language}"
        return hasattr(self.config.backend.model, model_key)
