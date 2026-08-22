"""Detect a mismatch between reference text script and the selected language.

Issue #148: ``--language`` validation rejects unknown codes and warns when a
recognized code has no dedicated normalizer, but a *supported* code applied
to data in a different language (``--language ko`` on English references)
used to run silently and produce a complete, healthy-looking scorecard. The
language selects the normalization branch, so the scores in that scorecard
are computed with the wrong rules and are not comparable to anything — yet
nothing in the log or the artifacts said so.

The four dedicated normalizer languages each imply one writing system, so
the dominant Unicode script of the reference transcriptions is a cheap,
dependency-free signal. This is deliberately a *warning*, not an error:
Latin-script references cannot distinguish English from, say, Swahili, and
code-switched corpora legitimately mix scripts. The check only fires when a
clear majority of script-bearing characters belong to a script other than
the one the selected language is written in.
"""

from __future__ import annotations

from typing import Iterable, Optional

# Writing system implied by each dedicated-normalizer language.
EXPECTED_SCRIPTS = {
    "bn": "bengali",
    "hi": "devanagari",
    "ko": "hangul",
    "en": "latin",
}

_SCRIPT_DISPLAY = {
    "bengali": "Bengali script",
    "devanagari": "Devanagari",
    "hangul": "Hangul",
    "latin": "Latin script",
}

_SCRIPT_LANGUAGE_HINT = {
    "bengali": "bn",
    "devanagari": "hi",
    "hangul": "ko",
    "latin": "en (or another Latin-script language)",
}

# A clear majority of script-bearing characters must disagree before the
# warning fires, so legitimately code-switched references do not trip it.
_MISMATCH_THRESHOLD = 0.5
# Below this many script-bearing characters the sample is too small to call.
_MIN_SCRIPT_CHARS = 10
# Enough signal for a whole corpus; keeps the scan O(1) on huge datasets.
_MAX_SCAN_CHARS = 20_000


def _classify_char(ch: str) -> Optional[str]:
    """Map a character to one of the four scripts, or None if neutral."""
    code = ord(ch)
    if 0x0041 <= code <= 0x005A or 0x0061 <= code <= 0x007A or 0x00C0 <= code <= 0x024F:
        return "latin"
    if 0x0900 <= code <= 0x097F:
        return "devanagari"
    if 0x0980 <= code <= 0x09FF:
        return "bengali"
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return "hangul"
    return None


def script_mismatch_warning(references: Iterable[str], language: str) -> Optional[str]:
    """Warning text when the references' dominant script contradicts *language*.

    Returns ``None`` when the language implies no single script (not one of
    the four dedicated-normalizer languages — those already get the
    no-normalizer warning), when there is too little script-bearing text to
    call, or when no other script holds a clear majority.
    """
    expected = EXPECTED_SCRIPTS.get(language.lower())
    if expected is None:
        return None

    counts: dict[str, int] = {}
    scanned = 0
    for text in references:
        if not text:
            continue
        for ch in text:
            script = _classify_char(ch)
            if script is not None:
                counts[script] = counts.get(script, 0) + 1
                scanned += 1
        if scanned >= _MAX_SCAN_CHARS:
            break

    total = sum(counts.values())
    if total < _MIN_SCRIPT_CHARS:
        return None

    dominant, dominant_count = max(counts.items(), key=lambda item: item[1])
    if dominant == expected or dominant_count / total < _MISMATCH_THRESHOLD:
        return None

    share_pct = round(100 * dominant_count / total)
    return (
        f"Reference transcriptions look like {_SCRIPT_DISPLAY[dominant]} "
        f"({share_pct}% of script-bearing characters), but --language "
        f"'{language}' expects {_SCRIPT_DISPLAY[expected]}. The "
        f"'{language}' normalizer will still be applied, so WER/CER are "
        "computed with the wrong rules and are not comparable to correctly "
        "configured runs. If the data is actually "
        f"{_SCRIPT_LANGUAGE_HINT[dominant]}, rerun with that --language. "
        "This warning is recorded in scores.json under 'warnings'."
    )
