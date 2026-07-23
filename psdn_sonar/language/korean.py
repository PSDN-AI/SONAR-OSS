"""Korean language processor."""

import logging
import unicodedata
from typing import Dict, List

from ..registry import register_language
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("ko")
class KoreanProcessor(LanguageProcessor):
    """Korean normalization and tokenization.

    Uses the shared :meth:`LanguageProcessor.normalize` pipeline, preceded by
    loanword replacement (Latin → Hangul), Unicode normalization, and either
    G2P phonemization or optional spacing normalization.

    G2P and number verbalization are mutually exclusive: G2P already
    phonemizes digits into Hangul, so the regex-based verbalizer is skipped
    when ``use_g2p`` is set.

    Optional dependencies (``[korean]`` extra: ``g2pk``, ``konlpy``,
    ``jamo``, ``pykospacing``) degrade gracefully to fallback behavior when
    absent.
    """

    _loanword_language = "ko"

    def __init__(self, config):
        super().__init__(config)
        self._g2p = None
        self._mecab = None

    def _pre_normalize(self, text: str) -> str:
        text = self._replace_loanwords(text)
        text = unicodedata.normalize(self.config.language.normalization.unicode_form, text)

        if self._use_g2p():
            return self._apply_g2p(text)
        if getattr(self.config.language.normalization, "normalize_spacing", False):
            return self._normalize_spacing(text)
        return text

    def _use_g2p(self) -> bool:
        return getattr(self.config.language.normalization, "use_g2p", False)

    def _should_verbalize_numbers(self) -> bool:
        return not self._use_g2p() and super()._should_verbalize_numbers()

    def _symbol_map(self) -> Dict[str, str]:
        from ..utils.symbols import KOREAN_SYMBOL_MAP

        return KOREAN_SYMBOL_MAP

    def _apply_g2p(self, text: str) -> str:
        try:
            from g2pk import G2p

            if self._g2p is None:
                self._g2p = G2p()
            return self._g2p(text)
        except Exception:
            logger.debug("g2pK not available, skipping G2P normalization")
            return text

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        try:
            from pykospacing import Spacing

            return Spacing()(text)
        except Exception:
            return text

    def verbalize_numbers(self, text: str) -> str:
        """Convert 1-4 digit runs to Sino-Korean words (``100원`` → ``백원``).

        Delegates to ``psdn_sonar.utils.numbers.verbalize_digits``
        (``num2words``, a core dependency). Digit runs glued to Latin
        letters are left intact (``"v2"``); runs adjacent to Hangul/CJK
        ARE matched.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "ko")

    def tokenize(self, text: str) -> List[str]:
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "jamo":
            return self._tokenize_jamo(text)
        if tokenizer_type == "mecab":
            return self._tokenize_mecab(text)
        if tokenizer_type in ("okt", "komoran"):
            tokens = self._tokenize_konlpy(tokenizer_type, text)
            if tokens is not None:
                return tokens
        elif tokenizer_type == "char":
            return list(text.replace(" ", ""))
        return text.split()

    @staticmethod
    def _tokenize_konlpy(tagger: str, text: str) -> List[str] | None:
        try:
            from konlpy import tag

            tokenizer = tag.Okt() if tagger == "okt" else tag.Komoran()
            return tokenizer.morphs(text)
        except Exception:
            return None

    @staticmethod
    def _tokenize_jamo(text: str) -> List[str]:
        """Decompose Hangul syllables into jamo; fall back to char mode."""
        try:
            import jamo

            tokens: List[str] = []
            for char in text:
                if jamo.is_hangul_char(char):
                    tokens.extend(list(jamo.h2j(char)))
                elif char != " ":
                    tokens.append(char)
            return tokens
        except Exception:
            logger.debug("jamo not available, falling back to character tokenization")
            return list(text.replace(" ", ""))

    def _tokenize_mecab(self, text: str) -> List[str]:
        try:
            from konlpy.tag import Mecab

            if self._mecab is None:
                self._mecab = Mecab()
            return self._mecab.morphs(text)
        except Exception:
            logger.debug("MeCab not available, falling back to space tokenization")
            return text.split()

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        ranges = [tuple(self.config.language.unicode_range)]
        ranges.extend(tuple(r) for r in getattr(self.config.language, "additional_ranges", []))
        return any(lo <= ord(c) <= hi for c in text for lo, hi in ranges)
