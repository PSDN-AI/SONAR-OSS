"""Registry-level tests plus Hindi and Korean processor coverage.

Per-language English and Bengali coverage lives in
``test_language_english.py`` / ``test_language_bengali.py``; this module
verifies the registry contract across all four processors and exercises the
Hindi and Korean pipelines. Runs without the ``[hindi]`` / ``[korean]``
extras: both processors degrade gracefully to their fallback paths.
"""

import pytest


def _make_processor(language):
    from psdn_sonar.config_loader import load_config
    from psdn_sonar.registry import get_language_processor

    config = load_config(language=language, backend="huggingface")
    return get_language_processor(language)(config)


class TestLanguageRegistry:
    def test_all_languages_registered(self):
        import psdn_sonar.language  # noqa: F401 — triggers registration
        from psdn_sonar.registry import list_language_processors

        registered = set(list_language_processors())
        assert {"bn", "en", "hi", "ko"} <= registered

    @pytest.mark.parametrize("code", ["bn", "en", "hi", "ko"])
    def test_get_processor_returns_class(self, code):
        import psdn_sonar.language  # noqa: F401
        from psdn_sonar.language.base import LanguageProcessor
        from psdn_sonar.registry import get_language_processor

        cls = get_language_processor(code)
        assert issubclass(cls, LanguageProcessor)

    def test_unknown_language_raises_with_available_list(self):
        from psdn_sonar.registry import get_language_processor

        with pytest.raises(ValueError, match="Unknown language"):
            get_language_processor("unknown_lang")


class TestHindiProcessor:
    def test_empty_and_blank_input(self):
        proc = _make_processor("hi")
        assert proc.normalize("") == ""
        assert proc.normalize("   ") == ""

    def test_devanagari_digits_verbalized(self):
        # ५०० → ASCII 500 → Hindi words; no digit of either script survives.
        proc = _make_processor("hi")
        normalized = proc.normalize("५००")
        assert "५००" not in normalized
        assert "500" not in normalized
        assert normalized.strip() != ""

    def test_symbol_verbalized_before_number(self):
        # "50%" → "50 प्रतिशत" → "पचास प्रतिशत": the percent word must
        # survive stripping and the digit must be verbalized.
        proc = _make_processor("hi")
        normalized = proc.normalize("50%")
        assert "प्रतिशत" in normalized
        assert "50" not in normalized

    def test_punctuation_stripped(self):
        proc = _make_processor("hi")
        normalized = proc.normalize("नमस्ते, दुनिया!")
        assert not any(ch in normalized for ch in ",!")
        assert normalized.strip() != ""

    def test_loanword_replaced_with_devanagari(self):
        # "customer" is in the Hindi loanword cache shipped with the wheel.
        proc = _make_processor("hi")
        normalized = proc.normalize("customer सेवा")
        assert "customer" not in normalized

    def test_tokenize_falls_back_to_words(self):
        proc = _make_processor("hi")
        tokens = proc.tokenize("नमस्ते दुनिया")
        assert isinstance(tokens, list)
        assert len(tokens) == 2

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("नमस्ते", True),
            ("mixed नमस्ते text", True),
            ("english only", False),
            ("", False),
        ],
    )
    def test_validate_text(self, text, expected):
        proc = _make_processor("hi")
        assert proc.validate_text(text) is expected


class TestKoreanProcessor:
    def test_empty_and_blank_input(self):
        proc = _make_processor("ko")
        assert proc.normalize("") == ""
        assert proc.normalize("   ") == ""

    def test_punctuation_stripped(self):
        proc = _make_processor("ko")
        normalized = proc.normalize("안녕하세요!")
        assert "!" not in normalized
        assert normalized.strip() != ""

    def test_numbers_verbalized_to_sino_korean(self):
        proc = _make_processor("ko")
        normalized = proc.normalize("123")
        assert "123" not in normalized
        assert normalized.strip() != ""

    def test_digit_adjacent_to_hangul_verbalized(self):
        # "100원" → "백원": Hangul-adjacent digit runs ARE matched (unlike
        # Latin-adjacent runs such as "v2", which stay untouched).
        proc = _make_processor("ko")
        normalized = proc.normalize("100원")
        assert "100" not in normalized
        assert "원" in normalized

    def test_symbol_verbalized_before_number(self):
        proc = _make_processor("ko")
        normalized = proc.normalize("50%")
        assert "퍼센트" in normalized
        assert "50" not in normalized

    def test_loanword_replaced_with_hangul(self):
        # "phone" is in the Korean loanword cache shipped with the wheel.
        proc = _make_processor("ko")
        normalized = proc.normalize("phone 번호")
        assert "phone" not in normalized

    def test_tokenize_word_mode(self):
        proc = _make_processor("ko")
        tokens = proc.tokenize("안녕하세요 세계")
        assert isinstance(tokens, list)
        assert len(tokens) == 2

    def test_char_tokenization_when_configured(self):
        from psdn_sonar.config_loader import load_config
        from psdn_sonar.registry import get_language_processor

        config = load_config(
            language="ko",
            backend="huggingface",
            overrides={"language": {"tokenizer": "char"}},
        )
        proc = get_language_processor("ko")(config)
        assert proc.tokenize("안녕 하") == ["안", "녕", "하"]

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("안녕하세요", True),
            ("mixed 안녕 text", True),
            ("english only", False),
            ("", False),
        ],
    )
    def test_validate_text(self, text, expected):
        proc = _make_processor("ko")
        assert proc.validate_text(text) is expected
