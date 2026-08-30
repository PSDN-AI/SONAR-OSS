"""Multilingual text normalization test suite.

Extracted from ``tests/test_cli.py`` to keep CLI tests focused on CLI
behavior and normalization tests focused on the language-processor /
fallback contract.

Covers, per language (en / hi / ko + bn regressions):

  - Loanword cache lookup and case insensitivity
  - Number verbalization (digit-to-word) and long-number / leading-zero skip
  - Punctuation and Unicode-symbol removal
  - Symbol verbalization (``%`` -> "percent" / "प्रतिशत" / "퍼센트", etc.)
  - Symbol-strip scope: which S* glyphs are intentionally dropped vs. which
    get verbalized first (the "no silent loss of spoken content" contract)
  - Happy-path / fallback-path parity for both symbols AND numbers,
    including Devanagari-digit input on the Hindi side
  - Loanword cache integrity validation (catches typos and casing slips
    in the shipped JSON files)
  - Case normalization, empty-string handling, whitespace collapse
"""

from pathlib import Path

import pytest


def _force_processor_failure(monkeypatch, language: str) -> None:
    """Make ``get_language_processor(language)`` raise so the fallback runs."""
    from psdn_sonar import registry

    original = registry.get_language_processor

    def fake(code, *args, **kwargs):
        if code == language:
            raise RuntimeError("forced failure for fallback test")
        return original(code, *args, **kwargs)

    monkeypatch.setattr(registry, "get_language_processor", fake)


class TestLoanwordNormalizationContract:
    """Loanword **contract** invariants — what we actually promise consumers.

    These tests assert ONLY the structural contract:
      1. The Latin source token is no longer present in normalized output.
      2. The output contains characters in the target script (Devanagari /
         Hangul / Bengali) — i.e. *some* native-script content arrived.
      3. Case-insensitive lookup: ``"phone"`` and ``"PHONE"`` map to the
         same normalized form.

    These tests deliberately do NOT assert the specific transliteration
    (``"google"`` -> ``"गूगल"``), because that's a heuristic choice baked
    into the loanword cache JSON. If a linguist later prefers ``"गूगुल"``,
    the cache update should not break the contract tests — only the
    paired heuristic tests in ``TestLoanwordNormalizationHeuristics``
    below need to update with the cache.
    """

    @staticmethod
    def _has_devanagari(s: str) -> bool:
        return any("\u0900" <= ch <= "\u097f" for ch in s)

    @staticmethod
    def _has_hangul(s: str) -> bool:
        return any("\uac00" <= ch <= "\ud7a3" for ch in s)

    @staticmethod
    def _has_bengali(s: str) -> bool:
        return any("\u0980" <= ch <= "\u09ff" for ch in s)

    def test_hindi_latin_token_replaced_with_devanagari(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Google पर search करो", language="hi")
        assert "google" not in result.lower(), f"latin token survived: {result!r}"
        assert "search" not in result.lower(), f"latin token survived: {result!r}"
        assert self._has_devanagari(result), f"no devanagari in output: {result!r}"

    def test_hindi_loanword_case_insensitive(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        lower = normalize_text_unified("phone दो", language="hi")
        upper = normalize_text_unified("PHONE दो", language="hi")
        assert lower == upper
        assert "phone" not in lower.lower()

    def test_hindi_mixed_loanword_and_native(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("मेरा account number 42 है।", language="hi")
        assert "account" not in result
        assert "number" not in result
        assert "।" not in result

    def test_korean_latin_token_replaced_with_hangul(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Google에서 search 해주세요", language="ko")
        assert "google" not in result.lower(), f"latin token survived: {result!r}"
        assert "search" not in result.lower(), f"latin token survived: {result!r}"
        assert self._has_hangul(result), f"no hangul in output: {result!r}"

    def test_korean_loanword_case_insensitive(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        lower = normalize_text_unified("phone 번호", language="ko")
        upper = normalize_text_unified("PHONE 번호", language="ko")
        assert lower == upper
        assert "phone" not in lower.lower()

    def test_bengali_loanword_regression(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("customer সেবা", language="bn")
        assert "customer" not in result
        assert self._has_bengali(result), f"no bengali in output: {result!r}"


class TestLoanwordNormalizationHeuristics:
    """Loanword **heuristic** assertions — specific transliterations.

    These tests pin the EXACT current state of the loanword caches
    (``config/language/{hi,ko,bn}/loanword_cache.json``). Their purpose
    is two-fold:
      1. Catch regressions where a normalization-pipeline change alters
         transliteration unexpectedly.
      2. Make cache changes reviewable — if a linguist updates the cache
         to prefer ``"गूगुल"`` over ``"गूगल"``, the test failure here
         signals the assertion needs to be updated alongside the cache
         JSON in the same PR.

    These are NOT contractual. The contract lives in
    ``TestLoanwordNormalizationContract`` above. Future cache updates
    that change the specific transliteration should update the
    assertion here in the same commit, not be blocked by it.
    """

    def test_hindi_google_transliteration(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Google पर search करो", language="hi")
        assert "गूगल" in result, f"current cache maps Google->गूगल; got {result!r}"
        assert "सर्च" in result, f"current cache maps search->सर्च; got {result!r}"

    def test_hindi_phone_transliteration(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("phone दो", language="hi")
        assert "फोन" in result, f"current cache maps phone->फोन; got {result!r}"

    def test_hindi_account_number_transliterations(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("मेरा account number 42 है।", language="hi")
        assert "अकाउंट" in result, f"current cache maps account->अकाउंट; got {result!r}"
        assert "नंबर" in result, f"current cache maps number->नंबर; got {result!r}"

    def test_korean_google_transliteration(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Google에서 search 해주세요", language="ko")
        assert "구글" in result, f"current cache maps google->구글; got {result!r}"
        assert "서치" in result, f"current cache maps search->서치; got {result!r}"

    def test_korean_phone_transliteration(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("phone 번호", language="ko")
        assert "폰" in result, f"current cache maps phone->폰; got {result!r}"

    def test_bengali_customer_transliteration(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("customer সেবা", language="bn")
        assert "কাস্টমার" in result, f"current cache maps customer->কাস্টমার; got {result!r}"


class TestNumberVerbalization:
    """Number-to-word conversion across all supported languages.

    The contract is "digits are converted to words"
    (``"100" not in result`` after normalization). The specific
    spellings (``"एक सौ"``, ``"one hundred"``, ``"백"``) come from
    upstream stable libraries — ``num2words`` for English/Korean and
    ``indic-numtowords`` for Hindi — and are not heuristics maintained
    in this repo. If an upstream version bump changes the spelling
    (rare, since these libraries pin natural-language rules) the
    assertions below will catch it.
    """

    def test_hindi_number_verbalization(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("मुझे 100 रुपये चाहिए", language="hi")
        assert "100" not in result
        assert "एक सौ" in result

    def test_hindi_devanagari_digits(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("यह ५०० का है", language="hi")
        assert "५०० " not in result
        assert "500" not in result
        assert "पाँच सौ" in result

    def test_hindi_skips_long_numbers(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("फोन 98765 है", language="hi")
        assert "98765" in result

    def test_korean_number_verbalization(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("100원입니다", language="ko")
        assert "100" not in result
        assert "백" in result

    def test_english_number_verbalization(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("I have 100 dollars.", language="en")
        assert "100" not in result
        assert "one hundred" in result

    def test_english_number_hyphen_handled(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("42 items", language="en")
        assert "42" not in result
        assert "forty two" in result
        assert "-" not in result

    def test_english_skips_long_numbers(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Phone 12345", language="en")
        assert "12345" in result

    def test_bengali_number_regression(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("২০২৪ সালে", language="bn")
        assert "2024" not in result
        assert "২০২৪" not in result


class TestThousandsSeparatorHandling:
    """Issue #135: a thousands separator used to split number verbalization,
    so "1,000 dollars" became "one000 dollars" — the "1" verbalized, the
    "000" hit the leading-zero skip, and the comma was stripped later,
    gluing them. A separator is presentation, not content: the separated
    and unseparated spellings of the same number must normalize identically.
    """

    def _assert_same_as_unseparated(self, lang: str, separated: str, unseparated: str):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        sep = normalize_text_unified(separated, language=lang)
        unsep = normalize_text_unified(unseparated, language=lang)
        assert sep == unsep, f"{lang}: {separated!r} -> {sep!r} but {unseparated!r} -> {unsep!r}"
        return sep

    def test_english_comma_separator(self):
        result = self._assert_same_as_unseparated("en", "1,000 dollars", "1000 dollars")
        assert result == "one thousand dollars"

    def test_english_space_separator(self):
        self._assert_same_as_unseparated("en", "1 000 dollars", "1000 dollars")

    def test_english_thin_space_separator(self):
        self._assert_same_as_unseparated("en", "1\u202f000 dollars", "1000 dollars")

    def test_hindi_comma_separator(self):
        result = self._assert_same_as_unseparated("hi", "1,000 रुपये", "1000 रुपये")
        assert "000" not in result

    def test_hindi_devanagari_digits_with_comma(self):
        self._assert_same_as_unseparated("hi", "१,००० रुपये", "१००० रुपये")

    def test_korean_comma_separator(self):
        result = self._assert_same_as_unseparated("ko", "1,000원", "1000원")
        assert "000" not in result

    def test_korean_multi_digit_groups(self):
        self._assert_same_as_unseparated("ko", "5,500원", "5500원")

    def test_million_stays_digits_like_unseparated(self):
        # Joined form exceeds the 4-digit verbalization cap, so it stays as
        # digits (phone/ID skip) — but crucially as ONE clean run, not
        # "one000000".
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("1,000,000 dollars", language="en")
        assert "1000000" in result
        assert "one000000" not in result

    def test_indian_grouping_stays_digits(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("10,00,000 रुपये", language="hi")
        assert "1000000" in result

    def test_enumeration_not_merged(self):
        # "1,2,3" is a list, not a grouped number — the digits must not be
        # joined into one hundred twenty-three.
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("options 1,2,3 available", language="en")
        assert "hundred" not in result

    def test_adjacent_independent_runs_not_merged(self):
        # A 4-digit year followed by a 3-digit count is NOT space grouping
        # (leading group would be 4 digits) and must verbalize separately.
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("in 2020 100 people came", language="en")
        assert "one hundred" in result
        assert "2020100" not in result
        assert "2020" not in result  # year itself still verbalizes

    def test_fallback_parity_for_separators(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        happy = normalize_text_unified("1,000 dollars", language="en")
        _force_processor_failure(monkeypatch, "en")
        fallback = normalize_text_unified("1,000 dollars", language="en")
        assert happy == fallback == "one thousand dollars"


class TestPunctuationAndSymbolRemoval:
    """Punctuation and symbol removal across languages."""

    def test_hindi_danda_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("यह एक परीक्षण है।", language="hi")
        assert "।" not in result

    def test_hindi_currency_symbol_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("कीमत ₹500 है", language="hi")
        assert "₹" not in result

    def test_korean_punctuation_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("안녕하세요! 만나서 반가워요.", language="ko")
        assert "!" not in result
        assert "." not in result

    def test_english_all_punctuation_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Hello, World! How are you?", language="en")
        assert "," not in result
        assert "!" not in result
        assert "?" not in result

    def test_english_currency_symbol_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("The price is $42!", language="en")
        assert "$" not in result

    def test_english_math_and_modifier_symbols_removed(self):
        """Glyphs absent from final output.

        Math operators, percent, and modifiers either get verbalized into
        words (``+`` -> 'plus', ``%`` -> 'percent') or stripped (``™``).
        Either way no glyph should survive into the WER/CER comparison.
        """
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("2+2=4 with 50% growth ™", language="en")
        for glyph in ["+", "=", "%", "™"]:
            assert glyph not in result, f"expected {glyph!r} stripped, got {result!r}"

    def test_hindi_math_symbol_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("लाभ 50% बढ़ा", language="hi")
        assert "%" not in result

    def test_korean_currency_symbol_removed(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("가격은 ₩500입니다", language="ko")
        assert "₩" not in result


class TestSymbolVerbalization:
    """Lock in that semantic symbols become *spoken words*, not silently dropped.

    Addresses the failure mode where stripping Unicode S* could silently
    delete spoken content (e.g. ``50%`` -> ``50``, losing the word "percent"
    that the speaker actually uttered). The verbalize_symbols step runs
    before _remove_punctuation specifically to prevent that loss.
    """

    def test_english_percent_becomes_percent_word(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("growth was 50%", language="en")
        assert "percent" in result
        assert "%" not in result

    def test_english_plus_in_cpp_preserved_as_word(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("I write C++ code", language="en")
        assert "plus plus" in result
        assert "+" not in result

    def test_english_equals_preserved_as_word(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("the formula is x=y", language="en")
        assert "equals" in result
        assert "=" not in result

    def test_english_ampersand_and_at_preserved(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("AT&T at user@host", language="en")
        assert "and" in result
        assert "at" in result
        assert "&" not in result
        assert "@" not in result

    def test_hindi_percent_becomes_pratishat(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("लाभ 50% बढ़ा", language="hi")
        assert "प्रतिशत" in result
        assert "%" not in result

    def test_hindi_plus_and_equals_become_words(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("5+5=10", language="hi")
        assert "जोड़" in result
        assert "बराबर" in result
        assert "+" not in result
        assert "=" not in result

    def test_korean_percent_becomes_percent_word(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("가격은 50% 올랐다", language="ko")
        assert "퍼센트" in result
        assert "%" not in result

    def test_korean_plus_becomes_deohagi(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("5+5", language="ko")
        assert "더하기" in result
        assert "+" not in result

    def test_korean_equals_uses_neutral_transliteration(self):
        """Locks in conservative Korean mappings.

        '=' was previously mapped to '는' (a topic-marker particle, not the
        equals sign — that mapping was wrong). The standard math/programming
        reading in Korean is the loanword '이콜', which is what we use now.
        """
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("x=y", language="ko")
        assert "이콜" in result
        assert "=" not in result

    def test_korean_at_uses_neutral_transliteration(self):
        """Locks in conservative Korean mapping.

        '@' was previously mapped to '골뱅이' (literally 'snail' — a dated
        colloquialism for the email '@' symbol). The neutral modern reading
        in spoken Korean is the loanword '앳' (transliteration of 'at').
        """
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("user@host", language="ko")
        assert "앳" in result
        assert "@" not in result


class TestSymbolStripScopeAcknowledgment:
    """Document and lock in *which* unlisted S* symbols get silently dropped.

    The S* category in Unicode is huge (currency, math, modifier, geometric,
    musical, etc.). We deliberately verbalize a small set of high-confidence
    semantic glyphs (``% + = & @ # < > /``) and drop everything else. That's
    a real trade-off — for an evaluation framework it's the right call (a
    glyph the speaker doesn't utter should not survive in either the
    reference or the prediction), but the trade-off needs to be visible in
    tests so a future reader knows it's deliberate.

    These tests assert each of the following glyphs is dropped (not preserved
    or expanded). A future change that wants to verbalize one of these should
    delete the corresponding assertion AND add a positive test in
    TestSymbolVerbalization:
      - currency: ``$ ¥ ₹ ₩ €``  (numeric value preserved by verbalize_numbers,
        but the unit word position is language-dependent so we don't expand
        the glyph itself)
      - math / scientific: ``± ÷ × ≠ ≤ ≥ √ ∞ °`` (rare in spoken transcripts)
      - modifier / mark: ``™ © ® ° µ`` (typically not spoken)
      - geometric / arrow: ``→ ← ▲ ●``
      - musical: ``♭ ♯``

    If a transcript pipeline starts emitting any of these as words and the
    ASR predictions also do, that's the signal to move them into
    psdn_sonar.utils.symbols.
    """

    # Each glyph below is verified to be in Unicode category P* or S* (audited
    # via unicodedata.category) — that's the contract of _remove_punctuation.
    # ``µ`` (MICRO SIGN, U+00B5) is intentionally NOT here because it's
    # category Ll (Lowercase Letter), not S*, so it survives normalization;
    # the audit caught that mismatch and would catch any future addition.
    INTENTIONALLY_DROPPED = [
        # currency (Sc)
        "$",
        "¥",
        "₹",
        "₩",
        "€",
        # math / scientific (Sm)
        "±",
        "÷",
        "×",
        "≠",
        "≤",
        "≥",
        "√",
        "∞",
        "♯",
        # modifier / mark (So)
        "°",
        "™",
        "©",
        "®",
        # geometric / arrow (Sm / So)
        "→",
        "←",
        "▲",
        "●",
        # musical (So)
        "♭",
    ]

    def test_intentionally_dropped_list_only_contains_p_or_s(self):
        """Meta-check: every glyph in ``INTENTIONALLY_DROPPED`` must be in
        Unicode category P* or S*, since that's exactly what
        ``_remove_punctuation`` strips. This catches mistakes like adding
        ``µ`` (which is Ll, Lowercase Letter — not stripped).
        """
        import unicodedata

        bad = [
            (g, unicodedata.category(g))
            for g in self.INTENTIONALLY_DROPPED
            if not (unicodedata.category(g).startswith("P") or unicodedata.category(g).startswith("S"))
        ]
        assert not bad, (
            "INTENTIONALLY_DROPPED contains glyph(s) not in P*/S* — "
            "_remove_punctuation will not strip them: " + ", ".join(f"{g!r} ({cat})" for g, cat in bad)
        )

    @pytest.mark.parametrize("glyph", INTENTIONALLY_DROPPED)
    def test_glyph_is_dropped_in_english(self, glyph):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified(f"foo {glyph} bar", language="en")
        assert glyph not in result, (
            f"Glyph {glyph!r} survived English normalization. Either add it to "
            f"ENGLISH_SYMBOL_MAP in psdn_sonar/utils/symbols.py and add a "
            f"positive test in TestSymbolVerbalization, or update "
            f"INTENTIONALLY_DROPPED here to acknowledge the new behavior."
        )

    @pytest.mark.parametrize("glyph", INTENTIONALLY_DROPPED)
    def test_glyph_is_dropped_in_hindi(self, glyph):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified(f"foo {glyph} bar", language="hi")
        assert glyph not in result

    @pytest.mark.parametrize("glyph", INTENTIONALLY_DROPPED)
    def test_glyph_is_dropped_in_korean(self, glyph):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified(f"foo {glyph} bar", language="ko")
        assert glyph not in result


class TestSymbolNormalizationFallbackParity:
    """Lock in that the rule-based fallback handles symbols the same way as
    the full language processor.

    The fallback runs whenever the language processor cannot be initialized
    (missing optional language deps, broken config, etc.). Without parity,
    the same input would normalize to different strings in different
    environments — a real reproducibility hole for an evaluation framework.
    """

    def test_english_fallback_verbalizes_percent(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        _force_processor_failure(monkeypatch, "en")
        result = normalize_text_unified("growth was 50%", language="en")
        assert "percent" in result
        assert "%" not in result

    def test_english_fallback_strips_currency(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        _force_processor_failure(monkeypatch, "en")
        result = normalize_text_unified("price is $10", language="en")
        assert "$" not in result

    def test_hindi_fallback_verbalizes_percent(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        _force_processor_failure(monkeypatch, "hi")
        result = normalize_text_unified("लाभ 50% बढ़ा", language="hi")
        assert "प्रतिशत" in result
        assert "%" not in result

    def test_korean_fallback_verbalizes_percent(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        _force_processor_failure(monkeypatch, "ko")
        result = normalize_text_unified("가격은 50% 올랐다", language="ko")
        assert "퍼센트" in result
        assert "%" not in result

    def test_korean_fallback_strips_currency(self, monkeypatch):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        _force_processor_failure(monkeypatch, "ko")
        result = normalize_text_unified("가격은 ₩500입니다", language="ko")
        assert "₩" not in result


class TestNumberVerbalizationFallbackParity:
    """Lock in that digit-to-word verbalization is byte-identical between
    the language-processor happy path and the rule-based fallback path.

    Both paths now delegate to ``psdn_sonar.utils.numbers.verbalize_digits``
    so divergence is structurally prevented; these tests catch any
    regression. The Hindi case includes both ASCII-digit AND
    Devanagari-digit input so script-native digit handling is also covered
    on the fallback side, not just on the happy path.
    """

    def _assert_parity(self, monkeypatch, lang: str, text: str):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        happy = normalize_text_unified(text, language=lang)
        _force_processor_failure(monkeypatch, lang)
        fallback = normalize_text_unified(text, language=lang)
        assert happy == fallback, f"{lang}: happy={happy!r} fallback={fallback!r}"

    def test_english_number_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "en", "I have 100 dollars and 5 apples")

    def test_english_long_number_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "en", "the phone is 98765")

    def test_hindi_ascii_number_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "hi", "मुझे 100 रुपये चाहिए")

    def test_hindi_devanagari_digits_parity(self, monkeypatch):
        """Devanagari digits MUST verbalize identically on both paths.

        Previously the fallback skipped Devanagari->ASCII translation, so
        ``५००`` survived as a glyph while the happy path produced
        ``पाँच सौ``. ``verbalize_digits`` now performs the script
        translation internally, removing the divergence.
        """
        self._assert_parity(monkeypatch, "hi", "यह ५०० का है")

    def test_hindi_combined_devanagari_digits_and_symbol_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "hi", "लाभ ५०% बढ़ा")

    def test_korean_number_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "100원입니다")

    def test_korean_combined_number_and_symbol_parity(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "가격은 50% 올랐다")


class TestDigitsAdjacentToPunctuationParity:
    """Round-4 review regression: ``"3.14"`` style inputs (digits glued to
    punctuation) used to diverge between the happy path and the fallback.

    Reproduction of the bug the reviewer flagged:
      * Happy (Korean): ``verbalize_numbers`` ran BEFORE punctuation strip,
        so ``"3.14"`` -> ``"삼.십사"`` -> ``"삼십사"``.
      * Fallback: punctuation was stripped FIRST, so ``"3.14"`` ->
        ``"314"`` -> ``"삼백십사"``. Different word, different WER.

    The previous parity tests above don't catch this because none of
    their inputs (``"100원입니다"``, ``"가격은 50% 올랐다"``) had digits
    adjacent to punctuation. These tests do.
    """

    def _assert_parity(self, monkeypatch, lang: str, text: str):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        happy = normalize_text_unified(text, language=lang)
        _force_processor_failure(monkeypatch, lang)
        fallback = normalize_text_unified(text, language=lang)
        assert happy == fallback, f"{lang}: happy={happy!r} fallback={fallback!r}"

    @pytest.mark.parametrize("text", ["3.14", "version 2.0", "1,000 dollars", "items: 5,10,15"])
    def test_english_digits_adjacent_to_punctuation_parity(self, monkeypatch, text):
        self._assert_parity(monkeypatch, "en", text)

    @pytest.mark.parametrize("text", ["3.14", "लाभ 5%", "१००.५० रुपये", "कीमत 1,000 रुपये"])
    def test_hindi_digits_adjacent_to_punctuation_parity(self, monkeypatch, text):
        # NOTE: ASCII ``:`` is intentionally avoided here. The happy-path
        # IndicNormalizerFactory converts ``":"`` to the Devanagari
        # visarga ``"ः"`` as part of its script-aware cleanup, while the
        # fallback's plain ``unicodedata.normalize("NFC", text)`` does
        # not — that's an acknowledged richer-vs-poorer Unicode
        # normalizer divergence, not a digit/punctuation parity bug.
        self._assert_parity(monkeypatch, "hi", text)

    @pytest.mark.parametrize("text", ["3.14", "version 2.0", "₩100원", "가격: 1,000원", "100.5%"])
    def test_korean_digits_adjacent_to_punctuation_parity(self, monkeypatch, text):
        self._assert_parity(monkeypatch, "ko", text)


class TestMixedAlphanumericPreservation:
    """Round-4 review regression: ``verbalize_digits`` must NOT shred
    Latin-alphanumeric tokens like ``"v2"`` / ``"iPhone15"`` / ``"H2O"``
    / ``"MP3"`` / ``"2nd"``.

    Pre-fix English used the bare ``\\d+`` regex from the shared
    ``verbalize_digits`` helper, which matched any digit run regardless
    of context, so ``"v2"`` became ``"vtwo"``. The fix tightens the
    regex to ``(?<!latin)\\d+(?!latin)`` so digits adjacent to a Latin
    letter are left alone. Digits adjacent to non-Latin letters
    (Hangul ``원``, Devanagari ``रुपये``, CJK) are still matched —
    that's the ``"100원" -> "백원"`` behavior we want to preserve.

    Issue #209: that pattern still matched a **proper sub-run** of a
    glued digit run, from either side, so a run of two or more digits
    was half-verbalized (``"15m" -> "one5m"``, ``"iPhone15" ->
    "iphone1five"``). Every case the class shipped with used a
    single-digit run, which has no sub-run to fall back on, so none of
    them could catch it. The guards now name a digit as glue too.
    """

    @pytest.mark.parametrize(
        "input_text,expected_substring",
        [
            ("v2", "v2"),
            ("H2O", "h2o"),
            ("MP3", "mp3"),
            ("2nd", "2nd"),
            ("iPhone", "iphone"),
            ("iPhone15", "iphone15"),
            ("10GB", "10gb"),
        ],
    )
    def test_english_mixed_alphanumeric_preserved(self, input_text, expected_substring):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified(input_text, language="en")
        assert expected_substring in result, f"{input_text!r} -> {result!r} lost mixed alphanumeric token"

    def test_korean_digit_glued_to_hangul_still_verbalizes(self):
        """Positive case: Korean ``"100원"`` MUST still verbalize the
        ``100`` because Hangul is not a Latin letter — that's the whole
        point of doing per-script lookarounds rather than ``\\b``."""
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("100원", language="ko")
        assert "100" not in result, f"100 was not verbalized: {result!r}"
        assert "백" in result, f"expected '백' in {result!r}"

    def test_hindi_digit_glued_to_devanagari_still_verbalizes(self):
        """Same positive case for Hindi: digits next to Devanagari
        letters must still be verbalized."""
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("100रुपये", language="hi")
        assert "100" not in result, f"100 was not verbalized: {result!r}"

    @pytest.mark.parametrize("language", ["en", "ko", "hi", "bn"])
    @pytest.mark.parametrize("input_text", ["15m", "30km", "100MB", "abc123def"])
    def test_multi_digit_run_glued_to_a_latin_letter_left_intact(self, language, input_text):
        """Issue #209: a run of two or more digits glued to a Latin letter.

        Normalization runs on the reference and the hypothesis before WER
        and CER, so a token the normalizer half-verbalizes on one side only
        widens the distance it exists to close. Each input here was matched
        by the pre-fix pattern (``"15m" -> "one5m"`` / ``"일5m"`` /
        ``"एक5m"``); Bengali was unaffected and is the control.
        """
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified(input_text, language=language)
        assert result.strip() == input_text.lower(), f"{input_text!r} ({language}) -> {result!r}"

    @pytest.mark.parametrize(
        "text,expected_runs",
        [
            # Glued on the right — the engine used to backtrack into a sub-run.
            ("15m", []),
            ("30km", []),
            ("100MB", []),
            ("10GB", []),
            # Glued on the left — the engine used to start inside the run.
            ("iPhone15", []),
            ("abc123def", []),
            # Single-digit runs: correct before the fix, must stay correct.
            ("v2", []),
            ("2nd", []),
            ("H2O", []),
            # Not glued at all — these must still be verbalized.
            ("15", ["15"]),
            ("1000", ["1000"]),
            ("15미터", ["15"]),
            ("100원", ["100"]),
            ("2020 100", ["2020", "100"]),
            ("3.14", ["3", "14"]),
        ],
    )
    def test_digit_run_regex_matches_no_sub_run_of_a_glued_run(self, text, expected_runs):
        """Pin the mechanism at the pattern, not just its visible output.

        Both lookarounds have to reject an adjacent digit: the lookahead
        stops a greedy ``\\d+`` backtracking to a shorter run, the lookbehind
        stops the match starting further into one.
        """
        from psdn_sonar.utils.numbers import _DIGIT_RUN_RE

        assert [m.group() for m in _DIGIT_RUN_RE.finditer(text)] == expected_runs


class TestLoanwordReplacementFallbackParity:
    """Round-4 review regression: the Hindi/Korean fallback used to skip
    loanword replacement entirely, so a reference like ``"Google पर"``
    normalized to ``"google पर"`` on the fallback path but ``"गूगल पर"``
    on the happy path. The fix calls
    ``_apply_loanword_replacement_for_fallback`` at the start of the
    fallback so both paths apply the cache.
    """

    def _assert_parity(self, monkeypatch, lang: str, text: str):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        happy = normalize_text_unified(text, language=lang)
        _force_processor_failure(monkeypatch, lang)
        fallback = normalize_text_unified(text, language=lang)
        assert happy == fallback, f"{lang}: happy={happy!r} fallback={fallback!r}"

    def test_hindi_loanword_parity_in_fallback(self, monkeypatch):
        self._assert_parity(monkeypatch, "hi", "Google पर search करो")

    def test_korean_loanword_parity_in_fallback(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "computer로 작업해요")


class TestSymbolDigitOrderingParity:
    """Pin the happy/fallback ordering contract for symbol vs digit
    verbalization in ALL three languages.

    Previously, ``KoreanProcessor.normalize`` ran ``verbalize_numbers``
    BEFORE ``_verbalize_symbols`` while the fallback in
    ``normalize_text_unified`` ran them in the OPPOSITE order. They
    happened to produce byte-identical output for every input we
    could construct, because:

      1. Every symbol-map value is space-padded
         (``"%" -> " 퍼센트 "``), and
      2. The digit-run regex requires non-Latin glue on both sides.

    Both invariants are real but latent — a future symbol-map entry
    whose value starts or ends with a Latin letter would break Korean
    parity ONLY (since English/Hindi already use the
    symbols-then-digits order in both happy and fallback).

    The fix aligned Korean's happy path with the symbols->digits->strip
    order used by the other three code paths. These tests pin that
    contract so a future reorder doesn't regress it silently.
    """

    def _assert_parity(self, monkeypatch, lang: str, text: str) -> None:
        from psdn_sonar.utils.text_processing import normalize_text_unified

        happy = normalize_text_unified(text, language=lang)
        _force_processor_failure(monkeypatch, lang)
        fallback = normalize_text_unified(text, language=lang)
        assert happy == fallback, (
            f"{lang}: input={text!r}  happy={happy!r}  fallback={fallback!r}  (symbol/digit ordering parity broken)"
        )

    def test_korean_percent_with_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "100% 할인")

    def test_korean_plus_between_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "2+3=5")

    def test_korean_digits_glued_to_hangul_with_symbol(self, monkeypatch):
        self._assert_parity(monkeypatch, "ko", "100원 + 50원")

    def test_hindi_percent_with_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "hi", "50% छूट")

    def test_hindi_plus_between_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "hi", "10+20")

    def test_english_percent_with_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "en", "50% off today")

    def test_english_equals_with_digits(self, monkeypatch):
        self._assert_parity(monkeypatch, "en", "2+3=5")


class TestLoanwordCacheIntegrity:
    """Hard guard against silent normalization drift in the shipped caches.

    The Bengali / Hindi / Korean loanword caches are large hand-curated
    JSON files (~48 KB / ~12 KB / ~5 KB) that a typo or casing mistake can
    quietly break — the cache lookup just misses, and the offending Latin
    token survives normalization with no error raised.
    """

    @staticmethod
    def _cache_paths():
        repo_root = Path(__file__).resolve().parent.parent
        return [
            ("bn", repo_root / "config" / "language" / "bn" / "loanword_cache.json"),
            ("hi", repo_root / "config" / "language" / "hi" / "loanword_cache.json"),
            ("ko", repo_root / "config" / "language" / "ko" / "loanword_cache.json"),
        ]

    def test_all_shipped_caches_validate(self):
        from psdn_sonar.utils.loanword import validate_cache_file

        for lang, path in self._cache_paths():
            issues = validate_cache_file(path, language=lang)
            assert not issues, f"loanword cache {path} has {len(issues)} issue(s):\n  - " + "\n  - ".join(issues)

    def test_validator_catches_uppercase_keys(self):
        from psdn_sonar.utils.loanword import validate_cache

        bad = {"Phone": "ফোন"}
        issues = validate_cache(bad)
        assert any("not lowercase" in i for i in issues)

    def test_validator_catches_case_insensitive_collision(self):
        from psdn_sonar.utils.loanword import validate_cache

        bad = {"phone": "ফোন", "Phone": "টেলিফোন"}
        issues = validate_cache(bad)
        assert any("case-insensitive duplicate" in i or "case-insensitive" in i for i in issues)

    def test_validator_catches_pure_ascii_value(self):
        from psdn_sonar.utils.loanword import validate_cache

        bad = {"phone": "phone"}  # forgot to transliterate
        issues = validate_cache(bad)
        assert any("pure ASCII" in i for i in issues)

    def test_validator_catches_whitespace_in_value(self):
        from psdn_sonar.utils.loanword import validate_cache

        bad = {"phone": "ফোন "}
        issues = validate_cache(bad)
        assert any("whitespace" in i for i in issues)

    def test_validator_catches_empty_value(self):
        from psdn_sonar.utils.loanword import validate_cache

        bad = {"phone": ""}
        issues = validate_cache(bad)
        assert any("empty" in i for i in issues)


class TestCaseNormalization:
    """Lowercase normalization across languages."""

    def test_hindi_lowercase(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Hello दुनिया", language="hi")
        assert result == result.lower()

    def test_korean_lowercase(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("ABC 테스트", language="ko")
        assert result == result.lower()

    def test_english_lowercase(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("Hello World", language="en")
        assert result == "hello world"


class TestEmptyAndEdgeCases:
    """Edge cases for normalization."""

    def test_empty_string_all_languages(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        for lang in ["en", "hi", "ko", "bn"]:
            assert normalize_text_unified("", language=lang) == ""
            assert normalize_text_unified("   ", language=lang) == ""

    def test_whitespace_collapse(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("hello    world   test", language="en")
        assert "  " not in result

    def test_digits_only_preserved(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("42", language="hi")
        assert result.strip() != ""

    def test_loanword_not_in_cache_preserved(self):
        from psdn_sonar.utils.text_processing import normalize_text_unified

        result = normalize_text_unified("xyzabc123 है", language="hi")
        assert "xyzabc" in result.lower() or result.strip() != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
