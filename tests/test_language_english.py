"""Tests for the English language processor.

Covers registration, the normalize pipeline ordering contract
(symbols → numbers → punctuation strip), tokenization modes, and
text validation.
"""

import pytest


def _make_processor():
    from psdn_sonar.config_loader import load_config
    from psdn_sonar.language.english import EnglishProcessor

    config = load_config(language="en", backend="huggingface")
    return EnglishProcessor(config)


class TestEnglishRegistration:
    def test_registered_under_en(self):
        import psdn_sonar.language  # noqa: F401 — triggers registration
        from psdn_sonar.language.english import EnglishProcessor
        from psdn_sonar.registry import get_language_processor

        assert get_language_processor("en") is EnglishProcessor


class TestEnglishNormalize:
    def test_lowercase_and_whitespace_collapse(self):
        proc = _make_processor()
        assert proc.normalize("  Hello   WORLD  ") == "hello world"

    def test_empty_and_blank_input(self):
        proc = _make_processor()
        assert proc.normalize("") == ""
        assert proc.normalize("   ") == ""

    def test_symbol_verbalized_before_number(self):
        # "50%" must become "fifty percent": the symbol expands first
        # (while the digit still exists), then the digit verbalizes.
        proc = _make_processor()
        assert proc.normalize("50%") == "fifty percent"

    def test_decimal_not_glued_into_single_number(self):
        # Verbalize-before-strip: "3.14" → "three.fourteen" → "threefourteen",
        # never "314" → "three hundred fourteen".
        proc = _make_processor()
        assert proc.normalize("3.14") == "threefourteen"

    def test_alphanumeric_tokens_preserved(self):
        # Digit runs adjacent to letters are not verbalized ("v2" stays "v2").
        proc = _make_processor()
        assert proc.normalize("release v2") == "release v2"

    def test_punctuation_stripped(self):
        proc = _make_processor()
        assert proc.normalize("hello, world!") == "hello world"


class TestEnglishTokenize:
    def test_word_tokenization_default(self):
        proc = _make_processor()
        assert proc.tokenize("hello world") == ["hello", "world"]

    def test_char_tokenization_when_configured(self):
        from psdn_sonar.config_loader import load_config
        from psdn_sonar.language.english import EnglishProcessor

        config = load_config(
            language="en",
            backend="huggingface",
            overrides={"language": {"tokenizer": "char"}},
        )
        proc = EnglishProcessor(config)
        assert proc.tokenize("ab c") == ["a", "b", "c"]


class TestEnglishValidateText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("hello", True),
            ("", False),
            ("1234 !?", False),
            ("mixed বাংলা text", True),
        ],
    )
    def test_validate_text(self, text, expected):
        proc = _make_processor()
        assert proc.validate_text(text) is expected
