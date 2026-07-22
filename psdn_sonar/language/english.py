"""English language processor."""

from typing import Dict, List

from ..registry import register_language
from .base import LanguageProcessor


@register_language("en")
class EnglishProcessor(LanguageProcessor):
    """English normalization and tokenization.

    Uses the shared :meth:`LanguageProcessor.normalize` pipeline with the
    English symbol map and ``num2words``-based digit verbalization.

    The default config (``conf/language/en.yaml``) sets
    ``verbalize_numbers: true``; changing it shifts WER/CER on digit-bearing
    references, so keep it fixed within any set of runs you compare.
    """

    def _symbol_map(self) -> Dict[str, str]:
        from ..utils.symbols import ENGLISH_SYMBOL_MAP

        return ENGLISH_SYMBOL_MAP

    def verbalize_numbers(self, text: str) -> str:
        """Convert 1-4 digit runs to English words (``50`` → ``fifty``).

        Digit runs glued to Latin letters are left intact (``"v2"``,
        ``"H2O"``) — see ``psdn_sonar.utils.numbers`` for the contract.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "en")

    def tokenize(self, text: str) -> List[str]:
        if self.config.language.tokenizer == "char":
            return list(text.replace(" ", ""))
        return text.split()

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        return any(c.isascii() and c.isalpha() for c in text)
