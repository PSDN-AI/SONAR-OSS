import logging
import re
import unicodedata
from typing import List

from ..registry import register_language
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("bn")
class BengaliProcessor(LanguageProcessor):
    """Bengali text normalization and tokenization.

    Uses ``bnlp`` (``[bengali]`` extra) for cleaning and word tokenization
    when available, degrading gracefully to Unicode normalization and
    whitespace splitting when it is not.

    Note: WER scoring for Bengali normally goes through the canonical
    pipeline in ``psdn_sonar.utils.text_processing.normalize_bengali_for_wer``
    (loanword replacement, suffix splitting, nasal normalization); this
    processor provides the general config-driven normalize/tokenize contract
    used elsewhere (validation, dataset preparation).
    """

    def normalize(self, text: str) -> str:
        """Normalize Bengali text: digits → words, Unicode form, cleaning,
        punctuation strip, whitespace collapse (each step config-gated)."""
        if not text or not text.strip():
            return ""

        if self.config.language.normalization.verbalize_numbers:
            text = self.verbalize_numbers(text)

        norm_form = self.config.language.normalization.unicode_form
        text = unicodedata.normalize(norm_form, text)

        try:
            from bnlp import CleanText

            cleaner = CleanText(
                fix_unicode=True,
                unicode_norm=True,
                unicode_norm_form=norm_form,
                remove_url=True,
                remove_email=True,
                remove_emoji=True,
                remove_number=False,
                remove_digits=False,
                remove_punct=False,
            )
            text = cleaner(text)
        except Exception:
            logger.debug("bnlp CleanText failed in BengaliProcessor, using fallback", exc_info=True)

        if self.config.language.normalization.remove_punctuation:
            text = self._remove_punctuation(text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize with bnlp's word tokenizer, char mode, or whitespace,
        per ``config.language.tokenizer`` ("bnlp" / "char" / anything else)."""
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "bnlp":
            try:
                from bnlp.tokenize import Tokenizer

                tokenizer = Tokenizer()
                return tokenizer.word_tokenize(text)
            except Exception:
                pass
        elif tokenizer_type == "char":
            return list(text.replace(" ", ""))

        return text.split()

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit runs to Bengali words (Bengali numerals first
        mapped to ASCII via the configured ``numeral_map``)."""
        text = self._convert_bengali_digits(text)

        try:
            from num2words import num2words

            def replace_match(match):
                try:
                    number = int(match.group())
                    return num2words(number, lang="bn")
                except Exception:
                    return match.group()

            return re.sub(r"\d+", replace_match, text)
        except Exception:
            return text

    def _convert_bengali_digits(self, text: str) -> str:
        numeral_map = self.config.language.numeral_map
        for bn_digit, en_digit in numeral_map.items():
            text = text.replace(bn_digit, en_digit)
        return text

    def _remove_punctuation(self, text: str) -> str:
        return "".join(c for c in text if not unicodedata.category(c).startswith("P"))

    def validate_text(self, text: str) -> bool:
        """Accept text containing at least one code point in the configured
        Bengali Unicode range."""
        if not text:
            return False
        unicode_range = self.config.language.unicode_range
        min_code, max_code = unicode_range
        bengali_chars = sum(1 for c in text if min_code <= ord(c) <= max_code)
        return bengali_chars > 0
