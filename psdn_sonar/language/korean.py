import logging
import re
import unicodedata
from typing import List

from ..registry import register_language
from .base import LanguageProcessor

logger = logging.getLogger(__name__)


@register_language("ko")
class KoreanProcessor(LanguageProcessor):
    def __init__(self, config):
        super().__init__(config)
        self._g2p = None
        self._mecab = None
        self._loanword_cache = None
        self._loanword_cache_loaded = False

    def normalize(self, text: str) -> str:
        """Normalize Korean text for WER/CER evaluation.

        Pipeline (each step is gated by its config flag):
          1. Loanword replacement (Latin → Hangul via cache)
          2. Unicode normalization (NFC by default)
          3. EITHER G2P phonemization OR (optional spacing normalization).
             G2P and ``verbalize_numbers`` are mutually exclusive — G2P
             already phonemizes digits into Hangul, so running the
             regex-based verbalizer on G2P output would be a no-op at
             best and incorrect at worst.
          4. Symbol verbalization (``%`` → ``퍼센트`` etc.) — runs
             BEFORE number verbalization so the digit in ``"50%"``
             survives into number verbalization.
          5. Number verbalization (digits → Sino-Korean words) — runs
             only when ``not use_g2p and verbalize_numbers``.
          6. P*+S* punctuation/symbol stripping — runs LAST so the
             inserted spoken-word content survives.
          7. Lowercase + whitespace collapse

        **Why this exact symbol → digit → strip order**:
        Identical to ``EnglishProcessor.normalize`` /
        ``HindiProcessor.normalize`` AND to the fallback path in
        ``psdn_sonar.utils.text_processing.normalize_text_unified``,
        so all four code paths produce byte-identical output without
        depending on latent invariants like "every symbol-map value is
        space-padded" or "the digit-run regex requires Latin-letter
        glue". An earlier version ran ``verbalize_numbers`` BEFORE
        ``_verbalize_symbols`` here while the fallback ran them in the
        opposite order; they happened to commute on every input we
        could construct, but a future symbol-map entry whose value
        starts/ends with a Latin letter would have silently broken
        Korean-only happy/fallback parity. Aligning the orders removes
        the implicit dependency entirely.

        Numbers-before-strip ordering is what makes ``"3.14"`` read as
        ``삼.십사`` → ``삼십사`` (digits handled separately) rather than
        ``"314"`` → ``삼백십사`` (digits glued by punctuation stripping).
        """
        if not text or not text.strip():
            return ""

        cache = self._get_loanword_cache()
        if cache:
            from ..utils.loanword import replace_latin_tokens

            text, _, _ = replace_latin_tokens(text, cache)

        norm_form = self.config.language.normalization.unicode_form
        text = unicodedata.normalize(norm_form, text)

        use_g2p = getattr(self.config.language.normalization, "use_g2p", False)
        if use_g2p:
            text = self._apply_g2p(text)
        elif getattr(self.config.language.normalization, "normalize_spacing", False):
            text = self._normalize_spacing(text)

        if self.config.language.normalization.remove_punctuation:
            text = self._verbalize_symbols(text)

        if not use_g2p and self.config.language.normalization.verbalize_numbers:
            text = self.verbalize_numbers(text)

        if self.config.language.normalization.remove_punctuation:
            text = self._remove_punctuation(text)

        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _verbalize_symbols(self, text: str) -> str:
        """Expand semantic symbols (%, +, =, &, @, #, <, >, /) into Korean words.

        Runs at step 4 of ``normalize`` — BEFORE ``verbalize_numbers``
        (so the digit in ``"50%"`` survives into number verbalization)
        and BEFORE ``_remove_punctuation`` (so the inserted Korean word
        ``퍼센트`` survives the broad P*+S* strip). See
        ``psdn_sonar.utils.symbols`` for the rationale and the symbol
        table (and why some Korean readings — e.g. ``=`` → ``이콜`` instead
        of ``는`` — were chosen for neutrality).
        """
        from ..utils.symbols import KOREAN_SYMBOL_MAP, verbalize_symbols

        return verbalize_symbols(text, KOREAN_SYMBOL_MAP)

    def _get_loanword_cache(self) -> dict:
        if not self._loanword_cache_loaded:
            self._loanword_cache_loaded = True
            try:
                from ..utils.loanword import get_cache_path, load_cache

                cache_path = get_cache_path("ko")
                self._loanword_cache = load_cache(cache_path)
                if self._loanword_cache:
                    logger.debug(
                        "Loaded %d Korean loanword entries from %s",
                        len(self._loanword_cache),
                        cache_path,
                    )
            except Exception:
                logger.debug("Could not load Korean loanword cache", exc_info=True)
                self._loanword_cache = None
        return self._loanword_cache or {}

    def tokenize(self, text: str) -> List[str]:
        tokenizer_type = self.config.language.tokenizer

        if tokenizer_type == "jamo":
            return self._tokenize_jamo(text)
        elif tokenizer_type == "mecab":
            return self._tokenize_mecab(text)
        elif tokenizer_type == "okt":
            try:
                from konlpy.tag import Okt

                tokenizer = Okt()
                return tokenizer.morphs(text)
            except Exception:
                pass
        elif tokenizer_type == "komoran":
            try:
                from konlpy.tag import Komoran

                tokenizer = Komoran()
                return tokenizer.morphs(text)
            except Exception:
                pass
        elif tokenizer_type == "char":
            return list(text.replace(" ", ""))

        return text.split()

    def _apply_g2p(self, text: str) -> str:
        try:
            from g2pk import G2p

            if self._g2p is None:
                self._g2p = G2p()
            return self._g2p(text)
        except Exception:
            logger.debug("g2pK not available, skipping G2P normalization")
            return text

    def _tokenize_jamo(self, text: str) -> List[str]:
        try:
            import jamo

            tokens = []
            for char in text:
                if jamo.is_hangul_char(char):
                    decomposed = jamo.h2j(char)
                    tokens.extend(list(decomposed))
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

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit tokens (1-4 digits) to Sino-Korean words.

        Delegates to ``psdn_sonar.utils.numbers.verbalize_digits`` (which
        wraps ``num2words(lang='ko')``) so that the fallback path in
        ``normalize_text_unified`` produces byte-identical output for the
        same input — single source of truth for digit-to-word rules
        across the happy and fallback paths. ``num2words`` is a hard core
        dependency, so the produced output is deterministic across
        installed environments.

        Digit runs adjacent to ASCII Latin letters are intentionally
        skipped (so ``"v2"`` stays ``"v2"``, not ``"v이"``). Digit runs
        adjacent to Hangul / CJK / Devanagari ARE matched, which is why
        ``"100원"`` correctly becomes ``"백원"``. See
        ``psdn_sonar.utils.numbers`` for the regex contract.
        """
        from ..utils.numbers import verbalize_digits

        return verbalize_digits(text, "ko")

    def _normalize_spacing(self, text: str) -> str:
        try:
            from pykospacing import Spacing

            spacing = Spacing()
            return spacing(text)
        except Exception:
            return text

    def _remove_punctuation(self, text: str) -> str:
        """Strip Unicode punctuation (P*) and remaining symbols (S*).

        Runs as the LAST step before lowercasing, AFTER both
        ``verbalize_numbers`` (digits → Sino-Korean words) and
        ``_verbalize_symbols`` (semantic symbols → Korean words) have
        replaced their respective glyphs. What's left at this point is
        currency (``₩ $``), modifier and rare math glyphs
        (``™ © ÷ ±``), and ASCII punctuation — none of which are spoken
        as the glyph itself. Stripping them from both sides of the
        WER/CER comparison keeps glyph-vs-word mismatches from inflating
        the metric.
        """
        cat = unicodedata.category
        return "".join(c for c in text if not (cat(c).startswith("P") or cat(c).startswith("S")))

    def validate_text(self, text: str) -> bool:
        if not text:
            return False

        unicode_range = self.config.language.unicode_range
        min_code, max_code = unicode_range
        korean_chars = sum(1 for c in text if min_code <= ord(c) <= max_code)

        if hasattr(self.config.language, "additional_ranges"):
            for add_range in self.config.language.additional_ranges:
                add_min, add_max = add_range
                korean_chars += sum(1 for c in text if add_min <= ord(c) <= add_max)

        return korean_chars > 0
