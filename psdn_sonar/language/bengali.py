"""Bengali language processor."""

import logging
import re
import unicodedata
from typing import Dict, List

from ..registry import register_language
from .base import LanguageProcessor

logger = logging.getLogger(__name__)

# Same rule as normalize_bengali_for_wer's digit-comma stripping step
# (২,০০০ → ২০০০): a comma between digits is a thousands separator, not
# content. Applied after Bengali→ASCII digit mapping, so ASCII lookarounds
# suffice. Without it "১,০০০" verbalized as two runs, "এক" + num2words(000).
_RE_DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")


@register_language("bn")
class BengaliProcessor(LanguageProcessor):
    """Bengali normalization and tokenization.

    Overrides :meth:`normalize` with a Bengali-specific pipeline (digit
    verbalization first, ``bnlp`` cleaning, punctuation-only strip) rather
    than the shared template. Uses ``bnlp`` (``[bengali]`` extra) when
    available, degrading gracefully to Unicode normalization and whitespace
    splitting.

    Note: this processor is currently NOT on any production path (issue
    #115). WER scoring goes through the canonical pipeline in
    ``psdn_sonar.utils.text_processing.normalize_bengali_for_wer`` — the
    ``"bn"`` branch of ``normalize_text_unified`` short-circuits to it
    before the processor registry is consulted — and dataset preparation
    calls ``normalize_text_unified``, so it routes the same way. The class
    is registered to satisfy the config-driven normalize/tokenize contract
    that every supported language exposes. Its rules deliberately differ
    from the canonical pipeline (no loanword replacement, suffix splitting,
    or nasal normalization), so wiring it into scoring later would shift
    Bengali WER and requires a normalization-contract bump plus
    re-baselining.
    """

    def normalize(self, text: str) -> str:
        """Normalize Bengali text: symbols → words, digits → words, Unicode
        form, cleaning, punctuation strip, whitespace collapse (config-gated)."""
        if not text or not text.strip():
            return ""

        # Same gating and ordering contract as the base template: symbols
        # verbalize before digits so the "50" in "50%" survives into number
        # verbalization, and before the punctuation strip so the inserted
        # words survive it (issue #136).
        if self.config.language.normalization.remove_punctuation:
            text = self._verbalize_symbols(text)
        if self.config.language.normalization.verbalize_numbers:
            text = self.verbalize_numbers(text)

        norm_form = self.config.language.normalization.unicode_form
        text = unicodedata.normalize(norm_form, text)
        text = self._clean_with_bnlp(text, norm_form)

        if self.config.language.normalization.remove_punctuation:
            text = self._remove_punctuation(text)

        return re.sub(r"\s+", " ", text).strip()

    def _symbol_map(self) -> Dict[str, str]:
        from ..utils.symbols import BENGALI_SYMBOL_MAP

        return BENGALI_SYMBOL_MAP

    @staticmethod
    def _clean_with_bnlp(text: str, norm_form: str) -> str:
        try:
            # Through the guard, not `from bnlp import ...`: importing bnlp
            # directly runs its unbounded import-time NLTK download (#204).
            from psdn_sonar.utils.bnlp_compat import import_bnlp

            bnlp = import_bnlp()
            if bnlp is None:
                return text
            CleanText = bnlp.CleanText

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
            return cleaner(text)
        except Exception:
            logger.debug("bnlp CleanText failed in BengaliProcessor, using fallback", exc_info=True)
            return text

    def tokenize(self, text: str) -> List[str]:
        """Tokenize with bnlp's word tokenizer, char mode, or whitespace,
        per ``config.language.tokenizer``."""
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "bnlp":
            try:
                from psdn_sonar.utils.bnlp_compat import import_bnlp

                if import_bnlp() is None:
                    raise ImportError("bnlp is not installed")
                # Safe now: the guarded import above already executed the
                # package body, so this submodule import has no side effects.
                from bnlp.tokenize import Tokenizer

                return Tokenizer().word_tokenize(text)
            except Exception:
                pass
        elif tokenizer_type == "char":
            return list(text.replace(" ", ""))
        return text.split()

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit runs to Bengali words (Bengali numerals first mapped
        to ASCII via the configured ``numeral_map``, thousands commas
        stripped so ১,০০০ verbalizes like ১০০০)."""
        text = self._convert_bengali_digits(text)
        text = _RE_DIGIT_COMMA.sub("", text)

        try:
            from num2words import num2words

            def replace_match(match: re.Match) -> str:
                try:
                    return num2words(int(match.group()), lang="bn")
                except Exception:
                    return match.group()

            return re.sub(r"\d+", replace_match, text)
        except Exception:
            return text

    def _convert_bengali_digits(self, text: str) -> str:
        for bn_digit, en_digit in self.config.language.numeral_map.items():
            text = text.replace(bn_digit, en_digit)
        return text

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        # Bengali strips only P* (keeps S*), unlike the shared pipeline.
        return "".join(c for c in text if not unicodedata.category(c).startswith("P"))

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        min_code, max_code = self.config.language.unicode_range
        return any(min_code <= ord(c) <= max_code for c in text)
