import logging
import re
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bengali numeral conversion
# ---------------------------------------------------------------------------

_BENGALI_TO_ARABIC = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def convert_bengali_digits_to_english(text: str) -> str:
    """Convert Bengali digits (০-৯) to English digits (0-9)."""
    if not text:
        return ""
    return text.translate(_BENGALI_TO_ARABIC)


def verbalize_numerals(text: str) -> str:
    """Convert all numerals (Bengali & English) in text to Bengali spoken words.

    Example: "2024" or "২০২৪" -> "দুই হাজার চব্বিশ"
    """
    if not text or not text.strip():
        return ""

    text = convert_bengali_digits_to_english(text)

    try:
        from num2words import num2words
    except ImportError:
        return text

    def replace_match(match):
        try:
            number = int(match.group())
            return num2words(number, lang="bn")
        except Exception:
            return match.group()

    return re.sub(r"\d+", replace_match, text)


def remove_punctuation_rule_based(text: str) -> str:
    """Remove punctuation from Bengali text using Unicode category 'P*'."""
    if not text or not text.strip():
        return ""

    result = []
    for char in text:
        category = unicodedata.category(char)
        if category.startswith("P"):
            continue
        result.append(char)

    return "".join(result)


def _verbalize_semantic_symbols(text: str, language: str) -> str:
    """Expand semantic symbols (``%``, ``+``, ``=`` …) into spoken-word
    forms. No-op for languages without a symbol map.

    Kept separate from ``_strip_punctuation_and_symbols`` so that the
    fallback path in ``normalize_text_unified`` can interleave the
    pipeline as ``verbalize_symbols`` -> ``verbalize_digits`` ->
    ``strip``, matching the per-language processors. Combining the
    two steps was the root cause of a happy/fallback divergence on
    inputs like ``"3.14"`` in Korean, where the fallback used to
    strip the ``.`` first and end up reading ``314`` as a single
    number.
    """
    from .symbols import (
        ENGLISH_SYMBOL_MAP,
        HINDI_SYMBOL_MAP,
        KOREAN_SYMBOL_MAP,
        verbalize_symbols,
    )

    symbol_maps = {"en": ENGLISH_SYMBOL_MAP, "hi": HINDI_SYMBOL_MAP, "ko": KOREAN_SYMBOL_MAP}
    symbol_map = symbol_maps.get(language)
    if symbol_map:
        text = verbalize_symbols(text, symbol_map)
    return text


def _strip_punctuation_and_symbols(text: str) -> str:
    """Strip Unicode P* (punctuation) and S* (symbols) glyphs.

    Mirrors what ``EnglishProcessor._remove_punctuation`` /
    ``HindiProcessor._remove_punctuation`` /
    ``KoreanProcessor._remove_punctuation`` do, so that the fallback
    path produces byte-identical output for any input that doesn't
    require the language-specific Unicode normalizer.
    """
    cat = unicodedata.category
    return "".join(c for c in text if not (cat(c).startswith("P") or cat(c).startswith("S")))


def _apply_loanword_replacement_for_fallback(text: str, language: str) -> str:
    """Run the language's loanword cache (Latin -> native script) if available.

    The happy path does this as the very first step in ``normalize`` for
    Hindi and Korean. The fallback used to skip it entirely, which broke
    parity for any reference containing English loanwords like
    ``"computer"``, ``"phone"``, ``"meeting"`` (in Hindi/Korean refs).
    No-op for languages without a loanword cache (English, Bengali,
    anything else).
    """
    if language not in ("hi", "ko"):
        return text
    try:
        from .loanword import get_cache_path, load_cache, replace_latin_tokens

        cache_path = get_cache_path(language)
        cache = load_cache(cache_path)
        if cache:
            text, _, _ = replace_latin_tokens(text, cache)
    except Exception:
        logger.debug("Loanword cache load failed for '%s' in fallback", language, exc_info=True)
    return text


def _apply_symbol_normalization(text: str, language: str) -> str:
    """Backwards-compat shim: verbalize-then-strip in one call.

    Kept so any external caller that imported the old helper still
    works. The fallback path in ``normalize_text_unified`` no longer
    uses it, because it needs to slot ``verbalize_digits`` *between*
    the two halves to match the per-language processor order.
    """
    text = _verbalize_semantic_symbols(text, language)
    return _strip_punctuation_and_symbols(text)


# ---------------------------------------------------------------------------
# Canonical Bengali normalization helpers
#
# The exact rule set and rule ORDER below are a compatibility contract:
# reference and hypothesis text must be normalized identically (and stably
# across releases) for WER numbers to be comparable between runs and against
# previously published results. Treat any change here as a scoring-breaking
# change that requires re-baselining benchmarks.
# ---------------------------------------------------------------------------

_RE_ZWJ = re.compile(r"[\u200c\u200d]")
_RE_DIGIT_COMMA = re.compile(r"(?<=[০-৯\d]),(?=[০-৯\d])")
_RE_PUNCTUATION_BN = re.compile(r"[।॥৳,.!?;:\"'\-\(\)\[\]\{\}<>…—–]")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_ANUNASIKA = re.compile(r"\u0999\u09CD([\u0995-\u09B9])")

_BENGALI_SUFFIXES = [
    "গুলোতে",
    "গুলোর",
    "গুলোকে",
    "গুলো",
    "টাকে",
    "টিকে",
    "টায়",
    "টার",
    "টিতে",
    "টির",
    "টাতে",
    "টা",
    "টি",
    "তে",
    "কে",
    "এর",
    "য়ে",
    "এ",
]

_EKAR = "\u09c7"

_NUMBER_VARIANTS: dict[str, str] = {
    "পনের": "পনেরো",
    "ষোল": "ষোলো",
    "সতের": "সতেরো",
    "বিশ": "কুড়ি",
    "তিরিশ": "ত্রিশ",
    "দুশো": "দুইশো",
    "ছশো": "ছয়শো",
    "নশো": "নয়শো",
}


def _has_bengali(s: str) -> bool:
    return any("\u0980" <= ch <= "\u09ff" for ch in s)


def _split_suffixes(text: str, protected_tokens: set[str] | None = None) -> str:
    """Split common Bengali suffixes for consistent tokenization.

    Both ASR and reference text may attach/detach suffixes differently
    (e.g. 'প্যাকেটটা' vs 'প্যাকেট টা'). Splitting them in both ensures
    these orthographic differences don't count as WER errors.
    """
    tokens = text.split()
    result = []
    for token in tokens:
        if protected_tokens and token in protected_tokens:
            result.append(token)
            continue

        split = False

        for suffix in _BENGALI_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                stem = token[: -len(suffix)]
                if _has_bengali(stem):
                    result.append(stem)
                    result.append(suffix)
                    split = True
                    break

        if not split:
            if len(token) >= 3 and token[-1] == "র" and token[-2] == _EKAR and _has_bengali(token[:-2]):
                result.append(token[:-2])
                result.append("এর")
                split = True
            elif len(token) >= 3 and token[-1] == _EKAR and _has_bengali(token[:-1]):
                result.append(token[:-1])
                result.append("এ")
                split = True

        if not split:
            result.append(token)

    return " ".join(result)


def _digits_to_words_bn(text: str) -> str:
    """Convert standalone digit tokens (1-4 digits) to Bengali words.

    Skips 5+ digit sequences (phone numbers) and leading-zero tokens.
    Applies শত→শো post-processing on num2words output only.
    """
    from num2words import num2words

    tokens = text.split()
    result = []
    for token in tokens:
        if token.isdigit() and len(token) <= 4 and not (len(token) > 1 and token[0] == "0"):
            word = num2words(int(token), lang="bn").strip()
            word = word.replace("শত", "শো")
            result.append(word)
        else:
            result.append(token)
    return " ".join(result)


def _normalize_number_variants(text: str) -> str:
    tokens = text.split()
    return " ".join(_NUMBER_VARIANTS.get(t, t) for t in tokens)


def _normalize_nasals(text: str) -> str:
    """Normalize ঙ্ + consonant → ং + consonant (same sound, different spelling)."""
    return _RE_ANUNASIKA.sub("ং\\1", text)


# ---------------------------------------------------------------------------
# Loanword cache (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_LOANWORD_CACHE: dict[str, str] | None = None
_PROTECTED_TOKENS: set[str] | None = None


def _get_loanword_cache() -> dict[str, str]:
    global _LOANWORD_CACHE, _PROTECTED_TOKENS
    if _LOANWORD_CACHE is None:
        from .loanword import get_cache_path, load_cache

        cache_path = get_cache_path("bn")
        _LOANWORD_CACHE = load_cache(cache_path)
        _PROTECTED_TOKENS = set(_LOANWORD_CACHE.values()) if _LOANWORD_CACHE else None
        if _LOANWORD_CACHE:
            logger.debug("Loaded %d loanword cache entries from %s", len(_LOANWORD_CACHE), cache_path)
    return _LOANWORD_CACHE


def _get_protected_tokens() -> set[str] | None:
    _get_loanword_cache()
    return _PROTECTED_TOKENS


# ---------------------------------------------------------------------------
# Bengali tokenizer (bnlp BasicTokenizer, optional)
# ---------------------------------------------------------------------------

try:
    from bnlp import BasicTokenizer as _BnlpBasicTokenizer

    _BNLP_TOKENIZER_AVAILABLE = True
    _bengali_tokenizer = _BnlpBasicTokenizer()
except ImportError:
    _BNLP_TOKENIZER_AVAILABLE = False
    _bengali_tokenizer = None


def _tokenize_bengali(text: str) -> list[str]:
    if _BNLP_TOKENIZER_AVAILABLE and _bengali_tokenizer is not None:
        try:
            return _bengali_tokenizer.tokenize(text)
        except Exception:
            return text.split()
    return text.split()


# ---------------------------------------------------------------------------
# WER normalization contract versions
# ---------------------------------------------------------------------------

# Version of each language's WER normalization rule set. Bump a language's
# number whenever its rule set or rule order changes, so scores produced under
# different rule sets are never compared as if they were like-for-like.
# Recorded per run in ``scores.json`` (see issue #120: published numbers could
# not be compared against a reproduction because the rule set in force at
# publish time was not recorded anywhere).
WER_NORMALIZATION_CONTRACTS: dict[str, int] = {"bn": 1, "en": 1, "hi": 1, "ko": 1}


def wer_normalization_contract(language: str) -> str:
    """Identifier of the WER normalization rule set in force for *language*.

    Format ``"<lang>:v<version>"`` for languages with a versioned rule set,
    ``"<lang>:unversioned"`` otherwise. Bengali additionally carries
    ``"+bnlp"`` or ``"-bnlp"`` because :func:`_tokenize_bengali` silently
    falls back to whitespace splitting when the optional ``bnlp`` package is
    not installed — two environments running identical code can tokenize
    (and therefore score) Bengali differently, so the marker is part of the
    contract.
    """
    lang = (language or "").lower()
    version = WER_NORMALIZATION_CONTRACTS.get(lang)
    if version is None:
        return f"{lang}:unversioned"
    contract = f"{lang}:v{version}"
    if lang == "bn":
        contract += "+bnlp" if _BNLP_TOKENIZER_AVAILABLE else "-bnlp"
    return contract


# ---------------------------------------------------------------------------
# Canonical Bengali normalization (for WER)
# ---------------------------------------------------------------------------


def normalize_bengali_for_wer(text: str) -> str:
    """Normalize Bengali text for WER computation (canonical pipeline).

    This is the single canonical normalization applied to BOTH the reference
    and the hypothesis before scoring (it produces the
    ``normalized_reference`` / ``normalized_hypothesis`` columns in
    results.csv), so WER numbers are directly comparable across runs and
    against published results. The rule set and rule order are a stability
    contract — see the section comment above the helpers.

    Pipeline:
    1. Loanword replacement (Latin-script → Bengali via cache)
    2. Unicode NFKC normalization
    3. ZWJ/ZWNJ removal
    4. Digit comma stripping (২,০০০ → ২০০০)
    5. Punctuation → space
    6. Bengali numerals → Arabic digits
    7. Digit tokens → Bengali words (1-4 digits; শত→শো)
    8. Number word variant canonicalization
    9. Nasal normalization (ঙ্→ং before consonant)
    10. Whitespace collapse
    11. Suffix splitting (skip loanword tokens)
    12. Lowercase
    13. Tokenize (bnlp BasicTokenizer) + rejoin
    """
    if not text or not text.strip():
        return ""

    cache = _get_loanword_cache()
    protected = None

    if cache:
        from .loanword import replace_latin_tokens

        text, _, _ = replace_latin_tokens(text, cache)
        protected = _get_protected_tokens()

    text = unicodedata.normalize("NFKC", text)
    text = _RE_ZWJ.sub("", text)
    text = _RE_DIGIT_COMMA.sub("", text)
    text = _RE_PUNCTUATION_BN.sub(" ", text)
    text = text.translate(_BENGALI_TO_ARABIC)
    text = _digits_to_words_bn(text)
    text = _normalize_number_variants(text)
    text = _normalize_nasals(text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    text = _split_suffixes(text, protected)
    text = text.lower()

    tokens = _tokenize_bengali(text)
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Unified normalization entry point
# ---------------------------------------------------------------------------


def _normalize_with_processor(text: str, language: str) -> Optional[str]:
    """Normalize via the registered language processor; None when unavailable.

    Returning None signals the caller to use its rule-based fallback (e.g.
    missing language config, registry import failure, heavy dependency absent).
    """
    try:
        import psdn_sonar.language  # noqa: F401 — triggers @register_language decorators

        from ..config_loader import load_config
        from ..registry import get_language_processor

        config = load_config(language=language, backend="huggingface")
        processor_cls = get_language_processor(language)
        return processor_cls(config).normalize(text)
    except Exception:
        logger.debug("Language processor normalization failed for '%s', using fallback", language, exc_info=True)
        return None


def normalize_text_unified(text: str, language: str = "en") -> str:
    """Unified Normalization Pipeline for both Reference and Hypothesis.

    For Bengali (language="bn"), uses the canonical Bengali normalization
    (:func:`normalize_bengali_for_wer`) that includes loanword replacement,
    suffix splitting, nasal normalization, and number variant
    canonicalization, so WER numbers are directly comparable across runs.

    For English/Hindi/Korean, runs the registered language processor
    (``EnglishProcessor`` / ``HindiProcessor`` / ``KoreanProcessor``)
    when available, and falls back to a rule-based pipeline if processor
    initialization fails (e.g. missing language config or registry import).

    **Happy / fallback parity contract:**
    The fallback intentionally mirrors the per-language processor for:
      * Loanword replacement (Hindi / Korean Latin -> native script)
      * Symbol verbalization (``%``, ``+``, ``=``, ...) running BEFORE
        digit verbalization, which runs BEFORE P*+S* stripping.
      * Number verbalization via the shared
        ``psdn_sonar.utils.numbers.verbalize_digits`` (script-native
        digits like Devanagari ``५००`` are translated to ASCII inside
        ``verbalize_digits`` so this works on both paths).
      * Lowercasing and whitespace collapse.

    Known acceptable divergence: when the happy path uses a richer
    script-aware Unicode normalizer than the fallback's plain NFC
    (Hindi's ``IndicNormalizerFactory`` may map ASCII ``:`` -> Devanagari
    visarga ``ः``, etc.), the two paths can produce different output for
    inputs that exercise that normalizer's specific transformations.
    This is by design — the fallback can't replicate language-specific
    Unicode normalization without also pulling in the heavy dependency
    that the fallback exists to avoid.

    Args:
        text: Input text to normalize
        language: Language code ('bn' for Bengali, 'ko' for Korean, 'en' for English)

    Returns:
        Normalized text
    """
    if not text or not text.strip():
        return ""

    if language == "en":
        result = _normalize_with_processor(text, "en")
        if result is not None:
            return result

        from .numbers import verbalize_digits

        # Order mirrors EnglishProcessor.normalize:
        #   lower -> verbalize_symbols -> verbalize_numbers -> strip P+S
        # Doing verbalize_numbers BEFORE strip is what makes "3.14"
        # produce "three.fourteen" -> "threefourteen" instead of the
        # period being stripped first into "314" -> "three hundred fourteen".
        text = text.lower()
        text = _verbalize_semantic_symbols(text, "en")
        text = verbalize_digits(text, "en")
        text = _strip_punctuation_and_symbols(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    elif language == "bn":
        return normalize_bengali_for_wer(text)

    elif language in ["ko", "hi"]:
        result = _normalize_with_processor(text, language)
        if result is not None:
            return result

        from .numbers import verbalize_digits

        # Order mirrors HindiProcessor.normalize / KoreanProcessor.normalize:
        #   loanword cache -> NFC -> verbalize_symbols -> verbalize_numbers
        #   -> strip P+S -> lower
        # Loanword replacement was previously skipped in the fallback,
        # which silently changed reference text for Hindi/Korean inputs
        # containing English loanwords. Adding it here closes that
        # parity gap. The verbalize-before-strip ordering is what
        # makes "3.14" produce identical output across both paths.
        text = _apply_loanword_replacement_for_fallback(text, language)
        text = unicodedata.normalize("NFC", text)
        text = _verbalize_semantic_symbols(text, language)
        text = verbalize_digits(text, language)
        text = _strip_punctuation_and_symbols(text)
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        return text

    else:
        text = remove_punctuation_rule_based(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


# ---------------------------------------------------------------------------
# Devanagari helpers
# ---------------------------------------------------------------------------


def is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari script characters."""
    if not text:
        return False
    for char in text:
        code_point = ord(char)
        if 0x0900 <= code_point <= 0x097F:
            return True
    return False


def convert_devanagari_to_bengali(text: str) -> str:
    """Convert Devanagari script to Bengali script."""
    if not text:
        return text

    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        try:
            bengali_text = transliterate(text, sanscript.DEVANAGARI, sanscript.BENGALI)
            return bengali_text
        except Exception:
            logger.debug("indic_transliteration failed, using Unicode offset fallback", exc_info=True)
    except ImportError:
        logger.debug("indic_transliteration not installed, using Unicode offset fallback")

    result = []
    for char in text:
        code_point = ord(char)
        if 0x0900 <= code_point <= 0x097F:
            bengali_code_point = code_point + 0x80
            if 0x0980 <= bengali_code_point <= 0x09FF:
                result.append(chr(bengali_code_point))
            else:
                result.append(char)
        else:
            result.append(char)
    return "".join(result)
