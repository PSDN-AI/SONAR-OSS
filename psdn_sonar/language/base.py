"""Abstract base class for per-language text processors.

A language processor owns the text-normalization and tokenization rules used
to prepare BOTH the reference and the ASR hypothesis before scoring. Concrete
implementations register themselves with
:func:`psdn_sonar.registry.register_language` and are looked up by ISO 639-1
code via :func:`psdn_sonar.registry.get_language_processor`.
"""

from abc import ABC, abstractmethod
from typing import Any, List


class LanguageProcessor(ABC):
    """Base contract for language-specific normalization and tokenization.

    Args:
        config: Loaded run configuration (see ``psdn_sonar.config_loader``);
            processors read their behavior flags from
            ``config.language.normalization`` and ``config.language.tokenizer``.
    """

    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return *text* normalized for WER/CER scoring."""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split normalized text into the units WER is computed over."""

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit tokens to spoken words. Default: no-op."""
        return text

    def validate_text(self, text: str) -> bool:
        """Return whether *text* plausibly belongs to this language. Default: accept."""
        return True
