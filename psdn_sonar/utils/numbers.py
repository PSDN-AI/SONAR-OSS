"""Language-aware digit-to-word verbalization for ASR text normalization.

Used both by the language processors (happy path) and by the
``normalize_text_unified`` fallback path (when a processor cannot be
initialized) so that ``"100"`` -> ``"one hundred"`` (or its Hindi /
Korean equivalent) regardless of which code path runs. Without this
shared helper the fallback would emit raw digits while the happy
path emitted words, which is a real reproducibility hole for an
evaluation framework that computes WER/CER.

Behavior contract (kept identical to the per-language processors):
  - Thousands separators inside well-formed grouped numbers are
    stripped before digit-run extraction (``"1,000"`` -> ``"1000"`` ->
    ``"one thousand"``), matching what the canonical Bengali pipeline
    has always done — see ``strip_group_separators`` (issue #135).
  - Only digit runs of length 1-4 are converted; longer runs
    (phone numbers, IDs) and leading-zero runs are preserved as-is.
  - Digit runs glued to a **Latin letter** on either side are NOT
    matched (``"v2"``, ``"iPhone15"``, ``"H2O"`` stay intact). This
    prevents the verbalizer from shredding mixed Latin alphanumerics
    that a speaker would read as a single token. Digits adjacent to
    non-Latin letters (Hangul ``원``, Devanagari ``रुपये``, CJK) ARE
    still matched, because in those scripts numbers commonly sit
    next to logograms in compound words like ``100원``.
  - English uses ``num2words(lang='en')`` and replaces the hyphen with
    a space so ``"twenty-one"`` becomes ``"twenty one"`` for tokenization.
  - Hindi uses ``indic_numtowords.num2words(lang='hi')`` (a hard core
    dependency, so this path is deterministic across environments).
  - Korean uses ``num2words(lang='ko')``.

Dependency contract:
  ``num2words`` and ``indic-numtowords`` are declared in the **core**
  ``[project] dependencies`` of ``pyproject.toml``, so a properly
  installed ``psdn-sonar`` always has them available. If an import
  here fails (only possible in a deliberately stripped-down install),
  the converter logs a **WARNING** (loud, default-visible) before
  returning the text unchanged — a missing core dep silently changing
  WER/CER would be a real reproducibility hole, so the missed
  verbalization is reported rather than swallowed.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# One-shot suppression so a stripped-down install doesn't spam the log
# with the same WARNING for every line of text that contains a digit.
_WARNED_MISSING: set[str] = set()


def _warn_missing_once(library: str, language: str) -> None:
    key = f"{library}:{language}"
    if key in _WARNED_MISSING:
        return
    _WARNED_MISSING.add(key)
    logger.warning(
        "Number verbalization disabled for language=%r: required core dependency "
        "%r is not importable in this environment. Reinstall with `pip install "
        "psdn-sonar` (the dep is declared in [project] dependencies) to restore "
        "deterministic WER/CER. Digits will be left as-is until then.",
        language,
        library,
    )


# Latin alphabet incl. Latin-1 supplement letters (À-Ö, Ø-ö, ø-ÿ),
# explicitly skipping the multiplication (×, U+00D7) and division
# (÷, U+00F7) operators which sit in the same Latin-1 block but are
# Sm category, not letters. This is the set we treat as "alphanumeric
# token glue" — digits adjacent to one of these are left alone.
_LATIN_LETTER = r"[A-Za-z\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u00FF]"
_DIGIT_RUN_RE = re.compile(rf"(?<!{_LATIN_LETTER})\d+(?!{_LATIN_LETTER})")

# Thousands-separator handling (issue #135). A separator is presentation,
# not content: "1,000" and "1000" denote the same number, but the digit-run
# regex used to see two runs — "1" and "000" — and emit "one000" (the "000"
# hits the leading-zero skip). Separators are stripped only inside
# well-formed grouped numbers, so enumerations like "1,2,3" and genuinely
# separate runs like "2020 100" are left alone:
#   - Indian comma grouping (incl. the Western single-group case):
#     1,000 / 1,00,000 / 12,34,567 — 2-digit groups, final group of 3
#   - Western comma grouping: 123,456 / 1,234,567 — groups of 3
#   - Space grouping: 1 000 / 10 000 (ASCII space, NBSP, thin space,
#     narrow NBSP). An ASCII space between a 1-3 digit run and a 3-digit
#     run is read as grouping; "5 100 dollar bills" is the known
#     trade-off, accepted because grouped numerals are far more common
#     in transcript text than that construction.
# Numbers whose joined form exceeds 4 digits (1,000,000 -> 1000000) then
# fall under the existing long-run skip and stay as digits — the same
# behavior the canonical Bengali pipeline has always had for ২,০০,০০০.
_GROUP_SEPARATOR_SPACES = " \u00a0\u2009\u202f"
# Alternation order matters: the Western pattern must be tried before the
# Indian one, and the trailing (?!,?\d) guard rejects a partial match that
# stops before a ",digit" continuation — otherwise "1,000,000" would match
# only its "1,000" prefix and leave ",000" behind (regex alternation is
# first-match, not longest-match).
_GROUPED_NUMBER_RE = re.compile(
    rf"(?<!\d)"
    rf"(?:\d{{1,3}}(?:,\d{{3}})+"
    rf"|\d{{1,2}}(?:,\d{{2}})+,\d{{3}}"
    rf"|\d{{1,3}}(?:[{_GROUP_SEPARATOR_SPACES}]\d{{3}})+)"
    rf"(?!,?\d)"
)
_GROUP_SEPARATOR_RE = re.compile(rf"[,{_GROUP_SEPARATOR_SPACES}]")


def strip_group_separators(text: str) -> str:
    """Remove thousands separators inside well-formed grouped numbers.

    ``1,000`` -> ``1000``; ``1 000`` -> ``1000``; ``12,34,567`` -> ``1234567``.
    Text that isn't a grouped number (``1,2,3``, ``2020 100``) is unchanged.
    """
    return _GROUPED_NUMBER_RE.sub(lambda m: _GROUP_SEPARATOR_RE.sub("", m.group()), text)


# Per-script digit translation tables. The processor / fallback paths apply
# the matching one BEFORE digit-run extraction so that script-native digits
# (Devanagari ``५००``, Bengali ``৫০০``) get verbalized identically to
# ASCII ``500`` instead of being silently left as glyphs.
_NATIVE_TO_ASCII_DIGITS: dict[str, dict[int, int]] = {
    "hi": str.maketrans("०१२३४५६७८९", "0123456789"),
    "bn": str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789"),
}


def to_ascii_digits(text: str, language: str) -> str:
    """Translate script-native digits in ``text`` to ASCII 0-9.

    No-op for languages that don't have a script-native digit set in
    ``_NATIVE_TO_ASCII_DIGITS`` (currently English, Korean).
    """
    table = _NATIVE_TO_ASCII_DIGITS.get(language)
    if table is None:
        return text
    return text.translate(table)


def _should_skip(token: str) -> bool:
    return len(token) > 4 or (len(token) > 1 and token[0] == "0")


def _english_converter() -> Optional[Callable[[str], str]]:
    try:
        from num2words import num2words
    except ImportError:
        _warn_missing_once("num2words", "en")
        return None

    def convert(token: str) -> str:
        try:
            return num2words(int(token), lang="en").replace("-", " ")
        except Exception:
            return token

    return convert


def _hindi_converter() -> Optional[Callable[[str], str]]:
    try:
        from indic_numtowords import num2words as indic_num2words
    except ImportError:
        _warn_missing_once("indic_numtowords", "hi")
        return None

    def convert(token: str) -> str:
        try:
            return indic_num2words(int(token), lang="hi")
        except Exception:
            return token

    return convert


def _korean_converter() -> Optional[Callable[[str], str]]:
    try:
        from num2words import num2words
    except ImportError:
        _warn_missing_once("num2words", "ko")
        return None

    def convert(token: str) -> str:
        try:
            return num2words(int(token), lang="ko")
        except Exception:
            return token

    return convert


_CONVERTERS: dict[str, Callable[[], Optional[Callable[[str], str]]]] = {
    "en": _english_converter,
    "hi": _hindi_converter,
    "ko": _korean_converter,
}


def verbalize_digits(text: str, language: str) -> str:
    """Verbalize 1-4 digit runs into the spoken words of ``language``.

    Performs script-native digit translation first (Devanagari -> ASCII
    for Hindi, Bengali -> ASCII for Bengali) so that ``५००`` and ``500``
    produce the same verbalization. This means the same call works for
    both the language-processor happy path AND the rule-based fallback
    path in ``normalize_text_unified`` — there's no separate digit-script
    conversion step the caller can forget.

    Returns ``text`` unchanged for unsupported languages or when the
    underlying number-words library is missing in this environment.
    """
    if not text or language not in _CONVERTERS:
        return text

    convert = _CONVERTERS[language]()
    if convert is None:
        return text

    text = to_ascii_digits(text, language)
    text = strip_group_separators(text)

    def _replace(match: re.Match) -> str:
        token = match.group()
        if _should_skip(token):
            return token
        return convert(token)

    return _DIGIT_RUN_RE.sub(_replace, text)
