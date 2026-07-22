"""Hindi language processor."""

import logging
import unicodedata
from typing import Dict, List

from ..registry import register_language
from ..utils.numbers import to_ascii_digits
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("hi")
class HindiProcessor(LanguageProcessor):
    """Hindi normalization and tokenization.

    Uses the shared :meth:`LanguageProcessor.normalize` pipeline, preceded by
    loanword replacement (Latin → Devanagari), Indic NLP Unicode
    normalization (plain ``unicodedata`` fallback when ``indic-nlp-library``
    from the ``[hindi]`` extra is absent), and Devanagari → ASCII digit
    translation.
    """

    _loanword_language = "hi"

    def __init__(self, config):
        super().__init__(config)
        self._indic_normalizer = None

    def _pre_normalize(self, text: str) -> str:
        text = self._replace_loanwords(text)
        text = self._normalize_unicode(text)
        return to_ascii_digits(text, "hi")

    def _symbol_map(self) -> Dict[str, str]:
        from ..utils.symbols import HINDI_SYMBOL_MAP

        return HINDI_SYMBOL_MAP

    def _normalize_unicode(self, text: str) -> str:
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

            if self._indic_normalizer is None:
                self._indic_normalizer = IndicNormalizerFactory().get_normalizer("hi")
            return self._indic_normalizer.normalize(text)
        except Exception:
            logger.debug("Indic NLP not available, using basic Unicode normalization")
            return unicodedata.normalize(self.config.language.normalization.unicode_form, text)

    def verbalize_numbers(self, text: str) -> str:
        """Convert 1-4 digit runs to Hindi words (``50`` → ``पचास``).

        Delegates to ``psdn_sonar.utils.numbers.verbalize_digits``
        (``indic-numtowords``, a core dependency, so output is deterministic
        across installs). Digit runs glued to Latin letters are left intact.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "hi")

    def tokenize(self, text: str) -> List[str]:
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "indic":
            return self._tokenize_indic(text)
        if tokenizer_type == "subword":
            logger.debug("Subword tokenization requires a trained SentencePiece model")
            return text.split()
        if tokenizer_type == "char":
            return list(text.replace(" ", ""))
        return text.split()

    @staticmethod
    def _tokenize_indic(text: str) -> List[str]:
        try:
            from indicnlp.tokenize import indic_tokenize

            return indic_tokenize.trivial_tokenize(text)
        except Exception:
            logger.debug("indicnlp not available, falling back to space tokenization")
            return text.split()

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        min_code, max_code = self.config.language.unicode_range
        return any(min_code <= ord(c) <= max_code for c in text)
