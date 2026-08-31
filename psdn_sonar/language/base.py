"""Abstract base class for per-language text processors.

A language processor owns the text-normalization and tokenization rules used
to prepare BOTH the reference and the ASR hypothesis before scoring. Concrete
implementations register themselves with
:func:`psdn_sonar.registry.register_language` and are looked up by ISO 639-1
code via :func:`psdn_sonar.registry.get_language_processor`.
"""

import logging
import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LanguageProcessor(ABC):
    """Base contract for language-specific normalization and tokenization.

    Args:
        config: Loaded run configuration (see ``psdn_sonar.config_loader``);
            processors read their behavior flags from
            ``config.language.normalization`` and ``config.language.tokenizer``.
    """

    #: ISO 639-1 code of the loanword cache to load, or ``None`` when the
    #: language has no Latin-script loanword replacement step.
    _loanword_language: Optional[str] = None

    def __init__(self, config: Any):
        self.config = config
        self._loanword_cache: Optional[dict] = None
        self._loanword_cache_loaded = False

    def normalize(self, text: str) -> str:
        """Return *text* normalized for WER/CER scoring.

        Template method implementing the shared pipeline:

        1. :meth:`_pre_normalize` (language-specific: loanwords, Unicode form)
        2. symbol verbalization (``%`` → spoken word)
        3. number verbalization (digits → spoken words)
        4. punctuation/symbol stripping
        5. lowercase + whitespace collapse

        Steps 2-4 must run in exactly this order: symbols first so the digit
        in ``"50%"`` survives into number verbalization, stripping last so
        the just-inserted spoken words survive the strip. Subclasses with a
        different pipeline shape override ``normalize`` entirely.
        """
        if not text or not text.strip():
            return ""

        text = self._pre_normalize(text)

        remove_punctuation = self.config.language.normalization.remove_punctuation
        if remove_punctuation:
            text = self._verbalize_symbols(text)
        if self._should_verbalize_numbers():
            text = self.verbalize_numbers(text)
        if remove_punctuation:
            text = self._remove_punctuation(text)

        return re.sub(r"\s+", " ", text.lower()).strip()

    def _pre_normalize(self, text: str) -> str:
        """Language-specific preprocessing before the shared pipeline. Default: no-op."""
        return text

    def _replace_loanwords(self, text: str) -> str:
        """Replace Latin-script loanwords via the language's cache, if any."""
        cache = self._get_loanword_cache()
        if not cache:
            return text

        from ..utils.loanword import replace_latin_tokens

        text, _, _ = replace_latin_tokens(text, cache)
        return text

    def _get_loanword_cache(self) -> dict:
        if self._loanword_language is None:
            return {}
        if not self._loanword_cache_loaded:
            self._loanword_cache_loaded = True
            try:
                from ..utils.loanword import get_cache_path, load_cache

                cache_path = get_cache_path(self._loanword_language)
                self._loanword_cache = load_cache(cache_path)
                if self._loanword_cache:
                    logger.debug(
                        "Loaded %d loanword entries for %s from %s",
                        len(self._loanword_cache),
                        self._loanword_language,
                        cache_path,
                    )
            except Exception:
                logger.debug("Could not load loanword cache for %s", self._loanword_language, exc_info=True)
                self._loanword_cache = None
        return self._loanword_cache or {}

    def _should_verbalize_numbers(self) -> bool:
        return getattr(self.config.language.normalization, "verbalize_numbers", False)

    def _symbol_map(self) -> Optional[Dict[str, str]]:
        """Symbol → spoken-word map for this language. ``None`` disables step 2."""
        return None

    def _verbalize_symbols(self, text: str) -> str:
        symbol_map = self._symbol_map()
        if not symbol_map:
            return text

        from ..utils.symbols import verbalize_symbols

        return verbalize_symbols(text, symbol_map)

    @staticmethod
    def _remove_punctuation(text: str) -> str:
        """Strip Unicode punctuation (P*) and symbols (S*).

        Anything reaching this step is either not spoken as its glyph
        (``™ ©``) or has a context-dependent spoken form (currency), so
        stripping both sides of the comparison keeps glyph-vs-word
        mismatches from inflating WER/CER.
        """
        cat = unicodedata.category
        return "".join(c for c in text if not (cat(c).startswith("P") or cat(c).startswith("S")))

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split normalized text into word-level tokens.

        Not currently on the scoring path (issue #223): WER/CER are computed
        by ``jiwer`` over the normalized string, which splits on whitespace
        (Bengali additionally pre-splits via ``_tokenize_bengali`` in
        ``psdn_sonar.utils.text_processing``). This method is part of the
        config-driven contract every language exposes for library callers;
        wiring it into scoring would change token counts and therefore WER,
        so it requires a normalization-contract bump plus re-baselining.
        """

    def verbalize_numbers(self, text: str) -> str:
        """Convert digit tokens to spoken words. Default: no-op."""
        return text

    def validate_text(self, text: str) -> bool:
        """Return whether *text* plausibly belongs to this language. Default: accept."""
        return True
