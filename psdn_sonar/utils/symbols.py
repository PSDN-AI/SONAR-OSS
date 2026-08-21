"""Verbalize semantically meaningful symbols into their spoken-word forms.

ASR transcripts compare a written reference against a model's audio
prediction. When the reference contains a symbol that the speaker
*verbalized* in audio (e.g. "50%" is spoken as "fifty percent",
"C++" as "c plus plus", "x=y" as "x equals y"), the symbol must be
expanded to its spoken form before any punctuation/symbol stripping
runs — otherwise the expanded word is silently dropped from the
reference and the WER/CER comparison is wrong:

  Without verbalization:
    ref  "50%"           -> strip '%' -> "50"           -> "fifty"
    pred "fifty percent" ->              "fifty percent"
    WER mismatch on the missing word "percent".

  With verbalization (this module):
    ref  "50%"           -> "50 percent" -> "fifty percent"
    pred "fifty percent" -> "fifty percent"
    Match.

Currency markers ($, ₹, ₩) and modifier symbols (™, ©, ®) are
intentionally NOT verbalized here — speakers don't always read currency
glyphs as words (e.g. "$10" can be "ten dollars" or "ten bucks") and
the position depends on language, so we let the broad S* strip remove
them and accept that the unit word may differ between ref and pred.
"""

from __future__ import annotations

import re

# Symbols that carry meaning a speaker is highly likely to verbalize.
# Keep these maps small and high-confidence — the goal is to avoid
# silently dropping spoken content, not to hand-verbalize every glyph.
ENGLISH_SYMBOL_MAP: dict[str, str] = {
    "%": " percent ",
    "+": " plus ",
    "=": " equals ",
    "&": " and ",
    "@": " at ",
    "#": " number ",
    "<": " less than ",
    ">": " greater than ",
    "/": " slash ",
}

HINDI_SYMBOL_MAP: dict[str, str] = {
    "%": " प्रतिशत ",
    "+": " जोड़ ",
    "=": " बराबर ",
    "&": " और ",
    "@": " ऐट ",
    "#": " नंबर ",
    "<": " छोटा ",
    ">": " बड़ा ",
    "/": " स्लैश ",
}

BENGALI_SYMBOL_MAP: dict[str, str] = {
    # Mirrors the other three maps key-for-key (issue #136: Bengali was the
    # only supported language without a symbol map, so "%" survived
    # normalization as a literal and ৫০% could never match ৫০ শতাংশ).
    # Conservative, widely-used readings; loanword transliterations where
    # that is what Bengali ASR ground truth tends to contain.
    "%": " শতাংশ ",
    "+": " যোগ ",
    "=": " সমান ",
    "&": " এবং ",
    "@": " অ্যাট ",  # transliteration of "at"; standard for email-address readings
    "#": " হ্যাশ ",
    "<": " ছোট ",
    ">": " বড় ",
    "/": " স্ল্যাশ ",
}

KOREAN_SYMBOL_MAP: dict[str, str] = {
    # Conservative, widely-used readings only. Avoid colloquialisms
    # (e.g. "골뱅이" for @ is informal/dated; "는" for = is a topic
    # marker, not the equals sign). When a symbol has multiple natural
    # readings, prefer the loanword/transliteration that matches what
    # Korean ASR ground-truth pipelines tend to produce.
    "%": " 퍼센트 ",
    "+": " 더하기 ",
    "=": " 이콜 ",  # transliteration of "equal"; standard in math/programming readings
    "&": " 그리고 ",
    "@": " 앳 ",  # transliteration of "at"; standard for email-address readings
    "#": " 샵 ",
    "<": " 보다 작은 ",
    ">": " 보다 큰 ",
    "/": " 슬래시 ",
}


def verbalize_symbols(text: str, symbol_map: dict[str, str]) -> str:
    """Replace each symbol in ``text`` with its spoken-word equivalent.

    A single regex character class handles all keys in one pass so we
    don't apply replacements left-to-right (which would risk replacing
    a word inserted by an earlier replacement). The replacement strings
    are wrapped in spaces so adjacent tokens stay separable after the
    later whitespace-collapse step.
    """
    if not symbol_map:
        return text
    pattern = re.compile("|".join(re.escape(k) for k in symbol_map.keys()))
    return pattern.sub(lambda m: symbol_map[m.group(0)], text)
