import logging
import re
import unicodedata
from typing import List

from ..registry import register_language
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("en")
class EnglishProcessor(LanguageProcessor):
    def __init__(self, config):
        super().__init__(config)

    def normalize(self, text: str) -> str:
        """Normalize English text for WER/CER evaluation.

        Pipeline (each step is gated by its config flag):
          1. Lowercase
          2. Symbol verbalization (``%`` → ``percent`` etc.)
          3. Number verbalization (``50`` → ``fifty``)
          4. P*+S* punctuation/symbol stripping
          5. Whitespace collapse

        Order matters between steps 2-4:
          * Symbols are verbalized FIRST so ``"50%"`` becomes
            ``"50 percent"`` while the digit is still present.
          * Numbers are verbalized SECOND so they pick up the digit
            from the just-expanded symbol form.
          * Stripping happens LAST so the inserted spoken-word content
            survives — and so ``"3.14"`` reads as ``"three.fourteen"``
            → ``"threefourteen"`` rather than ``"314"`` → ``"three
            hundred fourteen"`` (digits glued by punctuation stripping).

        Note on number verbalization:
          The default English config (``conf/language/en.yaml``) sets
          ``verbalize_numbers: true``, matching the Hindi/Korean
          processors, so digit-bearing references are spelled out
          before scoring. Set ``verbalize_numbers: false`` in your
          config (or override at load time) for digit-preserving
          behavior — but note that changing it shifts WER/CER on
          digit-bearing references, so keep it fixed within any set
          of runs you intend to compare.
        """
        if not text or not text.strip():
            return ""

        text = text.lower()

        if self.config.language.normalization.remove_punctuation:
            text = self._verbalize_symbols(text)

        if getattr(self.config.language.normalization, "verbalize_numbers", False):
            text = self.verbalize_numbers(text)

        if self.config.language.normalization.remove_punctuation:
            text = self._remove_punctuation(text)

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _verbalize_symbols(self, text: str) -> str:
        """Expand semantic symbols (%, +, =, &, @, #, <, >, /) into spoken words.

        Runs at step 2 of ``normalize`` — BEFORE ``verbalize_numbers``
        (so the digit in ``"50%"`` survives into number verbalization)
        and BEFORE ``_remove_punctuation`` (so the inserted English word
        ``percent`` survives the broad P*+S* strip). See
        ``psdn_sonar.utils.symbols`` for the rationale and symbol table.
        """
        from ..utils.symbols import ENGLISH_SYMBOL_MAP, verbalize_symbols

        return verbalize_symbols(text, ENGLISH_SYMBOL_MAP)

    def tokenize(self, text: str) -> List[str]:
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "char":
            return list(text.replace(" ", ""))

        return text.split()

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit tokens (1-4 digits) to English words.

        Runs at step 3 of ``normalize`` — AFTER ``_verbalize_symbols``
        (so ``"50%"`` arrives here as ``"50 percent"`` with the digit
        intact) and BEFORE ``_remove_punctuation`` (so adjacent
        punctuation does not silently glue separate digit runs).

        Delegates to ``psdn_sonar.utils.numbers.verbalize_digits``,
        which wraps ``num2words(lang='en')``. Digit runs adjacent to
        ASCII Latin letters are intentionally skipped (so ``"v2"``
        stays ``"v2"``, ``"H2O"`` stays ``"h2o"``, etc.) — see
        ``psdn_sonar.utils.numbers`` for the regex contract.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "en")

    def _remove_punctuation(self, text: str) -> str:
        """Strip Unicode punctuation (P*) and remaining symbols (S*).

        Runs at step 4 of ``normalize`` — AFTER both
        ``_verbalize_symbols`` (semantic glyphs → English words) AND
        ``verbalize_numbers`` (digits → English words). What's left at
        this point is symbols whose glyph is either not spoken or whose
        spoken form is highly context-dependent (and so isn't a safe
        global expansion):
          * currency: ``$ ₹ ₩`` — speakers say "dollars / rupees / won"
            but the position relative to the number is language-dependent
          * modifiers / other symbols: ``™ © ® ÷ ± × ≠`` — typically not
            spoken in everyday speech
        Stripping these from both the reference and the ASR prediction
        keeps WER/CER from being inflated by glyph-vs-word mismatches.
        """
        cat = unicodedata.category
        return "".join(c for c in text if not (cat(c).startswith("P") or cat(c).startswith("S")))

    def validate_text(self, text: str) -> bool:
        if not text:
            return False
        return any(c.isascii() and c.isalpha() for c in text)
