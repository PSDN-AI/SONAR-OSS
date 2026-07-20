"""Tests for the Bengali language processor.

Covers registration, config-driven normalization (digit verbalization,
Bengali-numeral mapping, punctuation strip), tokenization fallbacks, and
Unicode-range text validation. Runs without the ``[bengali]`` extra: the
processor degrades gracefully when ``bnlp`` is unavailable.
"""

import pytest


def _make_processor():
    from psdn_sonar.config_loader import load_config
    from psdn_sonar.language.bengali import BengaliProcessor

    config = load_config(language="bn", backend="huggingface")
    return BengaliProcessor(config)


class TestBengaliRegistration:
    def test_registered_under_bn(self):
        import psdn_sonar.language  # noqa: F401 — triggers registration
        from psdn_sonar.language.bengali import BengaliProcessor
        from psdn_sonar.registry import get_language_processor

        assert get_language_processor("bn") is BengaliProcessor


class TestBengaliNormalize:
    def test_empty_and_blank_input(self):
        proc = _make_processor()
        assert proc.normalize("") == ""
        assert proc.normalize("   ") == ""

    def test_punctuation_stripped(self):
        proc = _make_processor()
        normalized = proc.normalize("এটি, একটি! পরীক্ষা?")
        assert not any(ch in normalized for ch in ",!?")
        assert normalized.strip() != ""

    def test_bengali_digits_verbalized(self):
        # ২ (Bengali two) must map to ASCII via numeral_map and then
        # verbalize to the Bengali word — no digit survives.
        proc = _make_processor()
        normalized = proc.normalize("২")
        assert "২" not in normalized
        assert "2" not in normalized
        assert normalized.strip() != ""

    def test_ascii_digits_verbalized(self):
        proc = _make_processor()
        normalized = proc.normalize("123")
        assert "123" not in normalized
        assert normalized.strip() != ""

    def test_whitespace_collapsed(self):
        proc = _make_processor()
        assert proc.normalize("এটি   একটি  পরীক্ষা") == "এটি একটি পরীক্ষা"


class TestBengaliTokenize:
    def test_tokenize_returns_words(self):
        # bn.yaml configures the bnlp tokenizer; without the [bengali]
        # extra this transparently falls back to whitespace splitting.
        proc = _make_processor()
        tokens = proc.tokenize("এটি একটি পরীক্ষা")
        assert isinstance(tokens, list)
        assert len(tokens) == 3

    def test_char_tokenization_when_configured(self):
        from psdn_sonar.config_loader import load_config
        from psdn_sonar.language.bengali import BengaliProcessor

        config = load_config(
            language="bn",
            backend="huggingface",
            overrides={"language": {"tokenizer": "char"}},
        )
        proc = BengaliProcessor(config)
        assert proc.tokenize("এটি") == list("এটি")


class TestBengaliValidateText:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("এটি একটি পরীক্ষা", True),
            ("mixed এটি text", True),
            ("english only", False),
            ("", False),
            ("1234", False),
        ],
    )
    def test_validate_text(self, text, expected):
        proc = _make_processor()
        assert proc.validate_text(text) is expected
