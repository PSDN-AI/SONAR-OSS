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

    def test_thousands_separator_stripped(self):
        # Issue #135 (same defect class on the processor's own num2words
        # path): ১,০০০ used to verbalize as two runs, "এক" + words(000).
        proc = _make_processor()
        assert proc.normalize("১,০০০") == proc.normalize("১০০০")
        assert "," not in proc.normalize("১,০০০")

    def test_percent_symbol_verbalized(self):
        # Issue #136: Bengali was the only language without a symbol map,
        # so "%" survived normalization while en/hi/ko all verbalize it.
        proc = _make_processor()
        normalized = proc.normalize("৫০%")
        assert normalized == "পঞ্চাশ শতাংশ"
        assert "%" not in normalized

    def test_symbol_map_matches_other_languages_keys(self):
        # The four maps must stay key-for-key parallel so no language
        # silently loses coverage for a symbol the others verbalize.
        from psdn_sonar.utils.symbols import (
            BENGALI_SYMBOL_MAP,
            ENGLISH_SYMBOL_MAP,
            HINDI_SYMBOL_MAP,
            KOREAN_SYMBOL_MAP,
        )

        assert set(BENGALI_SYMBOL_MAP) == set(ENGLISH_SYMBOL_MAP) == set(HINDI_SYMBOL_MAP) == set(KOREAN_SYMBOL_MAP)


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


class TestBengaliTokenizeBnlpBranch:
    """The ``tokenizer: bnlp`` branch after issue #223.

    Before the fix the branch imported ``bnlp.tokenize.Tokenizer`` — a module
    and class that do not exist in bnlp_toolkit — and a bare ``except: pass``
    discarded the ``ModuleNotFoundError`` on every call, so the branch always
    fell through to whitespace splitting in silence. It now uses the same
    ``BasicTokenizer`` the canonical WER pipeline loads, and an installed-but-
    broken bnlp is reported at WARNING instead of swallowed.
    """

    def test_uses_basic_tokenizer_when_bnlp_is_available(self, monkeypatch):
        # Stub bnlp with only the names the real package exposes: a
        # BasicTokenizer at the top level, no `bnlp.tokenize` submodule.
        # The old wrong import cannot succeed against this stub.
        from types import SimpleNamespace

        from psdn_sonar.utils import bnlp_compat

        marker = ["TOKENIZED", "BY", "BNLP"]

        class FakeBasicTokenizer:
            def tokenize(self, text):
                return list(marker)

        fake_bnlp = SimpleNamespace(BasicTokenizer=FakeBasicTokenizer)
        monkeypatch.setattr(bnlp_compat, "import_bnlp", lambda: fake_bnlp)

        proc = _make_processor()
        assert proc.tokenize("এটি একটি পরীক্ষা") == marker

    def test_broken_bnlp_warns_and_falls_back_to_whitespace(self, monkeypatch, caplog):
        import logging

        from psdn_sonar.utils import bnlp_compat

        class ExplodingTokenizer:
            def tokenize(self, text):
                raise RuntimeError("bnlp broke mid-call")

        from types import SimpleNamespace

        fake_bnlp = SimpleNamespace(BasicTokenizer=ExplodingTokenizer)
        monkeypatch.setattr(bnlp_compat, "import_bnlp", lambda: fake_bnlp)

        proc = _make_processor()
        with caplog.at_level(logging.WARNING, logger="psdn_sonar.language.bengali"):
            tokens = proc.tokenize("এটি একটি পরীক্ষা")

        assert tokens == ["এটি", "একটি", "পরীক্ষা"]
        assert any("falling back to whitespace splitting" in rec.message for rec in caplog.records)

    def test_missing_bnlp_stays_quiet_and_falls_back(self, monkeypatch, caplog):
        # Absent bnlp is the documented degradation without the [bengali]
        # extra: whitespace splitting with no WARNING noise per call.
        import logging

        from psdn_sonar.utils import bnlp_compat

        monkeypatch.setattr(bnlp_compat, "import_bnlp", lambda: None)

        proc = _make_processor()
        with caplog.at_level(logging.WARNING, logger="psdn_sonar.language.bengali"):
            tokens = proc.tokenize("এটি একটি পরীক্ষা")

        assert tokens == ["এটি", "একটি", "পরীক্ষা"]
        assert not caplog.records


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
