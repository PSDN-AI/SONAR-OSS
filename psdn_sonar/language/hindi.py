import logging
import re
import unicodedata
from typing import List

from ..registry import register_language
from ..utils.numbers import to_ascii_digits
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("hi")
class HindiProcessor(LanguageProcessor):
    def __init__(self, config):
        super().__init__(config)
        self._indic_normalizer = None
        self._loanword_cache = None
        self._loanword_cache_loaded = False

    def normalize(self, text: str) -> str:
        """Normalize Hindi text for WER/CER evaluation.

        Pipeline (each step is gated by its config flag):
          1. Loanword replacement (Latin → Devanagari via cache)
          2. Unicode normalization (IndicNormalizerFactory if available,
             plain NFC otherwise)
          3. Devanagari → ASCII digit translation (``५००`` → ``500``)
          4. Symbol verbalization (``%`` → ``प्रतिशत`` etc.)
          5. Number verbalization (``50`` → ``पचास``)
          6. P*+S* punctuation/symbol stripping
          7. Lowercase + whitespace collapse

        Order matters between steps 4-6:
          * Symbols are verbalized FIRST so ``"50%"`` becomes
            ``"50 प्रतिशत"`` while the digit is still present.
          * Numbers are verbalized SECOND so they pick up the digit
            from the just-expanded symbol form.
          * Stripping happens LAST so the just-introduced spoken-word
            content survives — and so ``"3.14"`` reads as
            ``तीन.चौदह`` → ``तीनचौदह`` (digits handled separately)
            rather than ``"314"`` → ``तीन सौ चौदह`` (digits glued by
            punctuation stripping).
        """
        if not text or not text.strip():
            return ""

        cache = self._get_loanword_cache()
        if cache:
            from ..utils.loanword import replace_latin_tokens

            text, _, _ = replace_latin_tokens(text, cache)

        text = self._normalize_unicode(text)
        text = to_ascii_digits(text, "hi")

        if self.config.language.normalization.remove_punctuation:
            text = self._verbalize_symbols(text)

        if getattr(self.config.language.normalization, "verbalize_numbers", False):
            text = self.verbalize_numbers(text)

        if self.config.language.normalization.remove_punctuation:
            text = self._remove_punctuation(text)

        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _verbalize_symbols(self, text: str) -> str:
        """Expand semantic symbols (%, +, =, &, @, #, <, >, /) into Hindi words.

        Runs at step 4 of ``normalize`` — BEFORE ``verbalize_numbers``
        (so the digit in ``"50%"`` survives into number verbalization)
        and BEFORE ``_remove_punctuation`` (so the inserted Hindi word
        ``प्रतिशत`` survives the broad P*+S* strip). See
        ``psdn_sonar.utils.symbols`` for the rationale and symbol table.
        """
        from ..utils.symbols import HINDI_SYMBOL_MAP, verbalize_symbols

        return verbalize_symbols(text, HINDI_SYMBOL_MAP)

    def _get_loanword_cache(self) -> dict:
        if not self._loanword_cache_loaded:
            self._loanword_cache_loaded = True
            try:
                from ..utils.loanword import get_cache_path, load_cache

                cache_path = get_cache_path("hi")
                self._loanword_cache = load_cache(cache_path)
                if self._loanword_cache:
                    logger.debug(
                        "Loaded %d Hindi loanword entries from %s",
                        len(self._loanword_cache),
                        cache_path,
                    )
            except Exception:
                logger.debug("Could not load Hindi loanword cache", exc_info=True)
                self._loanword_cache = None
        return self._loanword_cache or {}

    def _normalize_unicode(self, text: str) -> str:
        try:
            from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

            if self._indic_normalizer is None:
                factory = IndicNormalizerFactory()
                self._indic_normalizer = factory.get_normalizer("hi")
            return self._indic_normalizer.normalize(text)
        except Exception:
            logger.debug("Indic NLP not available, using basic Unicode normalization")
            norm_form = self.config.language.normalization.unicode_form
            return unicodedata.normalize(norm_form, text)

    def tokenize(self, text: str) -> List[str]:
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "indic":
            return self._tokenize_indic(text)
        elif tokenizer_type == "subword":
            return self._tokenize_subword(text)
        elif tokenizer_type == "char":
            return list(text.replace(" ", ""))

        return text.split()

    def _tokenize_indic(self, text: str) -> List[str]:
        try:
            from indicnlp.tokenize import indic_tokenize

            return indic_tokenize.trivial_tokenize(text)
        except Exception:
            logger.debug("indicnlp not available, falling back to space tokenization")
            return text.split()

    def _tokenize_subword(self, text: str) -> List[str]:
        logger.debug("Subword tokenization requires trained SentencePiece model")
        return text.split()

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit tokens (1-4 digits) to Hindi words.

        Runs at step 5 of ``normalize`` — AFTER ``_verbalize_symbols``
        (so ``"50%"`` arrives here as ``"50 प्रतिशत"`` with the digit
        intact) and BEFORE ``_remove_punctuation`` (so adjacent
        punctuation does not silently glue separate digit runs).

        Delegates to ``psdn_sonar.utils.numbers.verbalize_digits``,
        which wraps ``indic_numtowords.num2words(lang='hi')`` (a hard
        core dependency, so the output is deterministic across
        installed environments). Skips 5+ digit sequences (likely phone
        numbers / IDs) and leading-zero tokens. Digit runs adjacent to
        ASCII Latin letters are intentionally skipped (so ``"v2"``
        stays ``"v2"``); see ``psdn_sonar.utils.numbers`` for the regex
        contract.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "hi")

    def _remove_punctuation(self, text: str) -> str:
        """Strip Unicode punctuation (P*) and remaining symbols (S*).

        Runs at step 6 of ``normalize`` — AFTER both
        ``_verbalize_symbols`` (semantic glyphs → Hindi words) AND
        ``verbalize_numbers`` (digits → Hindi words). What's left at
        this point is currency (``₹ $``), modifier and rare math glyphs
        (``™ © ÷ ±``), and ASCII punctuation (``। , ! ?``) — none of
        which are spoken as the glyph itself. Stripping these from both
        sides of the WER/CER comparison keeps glyph-vs-word mismatches
        from inflating the metric.
        """
        cat = unicodedata.category
        return "".join(c for c in text if not (cat(c).startswith("P") or cat(c).startswith("S")))

    def validate_text(self, text: str) -> bool:
        if not text:
            return False

        unicode_range = self.config.language.unicode_range
        min_code, max_code = unicode_range
        hindi_chars = sum(1 for c in text if min_code <= ord(c) <= max_code)

        return hindi_chars > 0
