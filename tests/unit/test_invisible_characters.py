"""Regression tests for issue #140: characters that are invisible in
rendered text must not score as errors.

Two gaps made identical words score as substitutions: a zero-width space
(category Cf, untouched by the P*/S* punctuation strip) survived
normalization in every language, and English applied no Unicode
normalization at all, so the NFC and NFD encodings of the same word were
treated as different words. Both are invisible to a person reading the
transcript, and with byte-identical predictions moved WER 0.05 -> 0.15 in
the issue's end-to-end measurement.
"""

import unicodedata

import pytest

from psdn_sonar.utils.text_processing import (
    fold_invisible_characters,
    normalize_bengali_for_wer,
    normalize_text_unified,
)

ZWSP = "\u200b"
LANGS = ("en", "bn", "hi", "ko")


class TestFoldInvisibleCharacters:
    def test_zwsp_becomes_a_word_boundary(self):
        # A ZWSP used as an invisible separator must not glue words together.
        assert fold_invisible_characters(f"the{ZWSP}end") == "the end"

    def test_zwnbsp_becomes_a_word_boundary(self):
        assert fold_invisible_characters("\ufeffhello").strip() == "hello"

    def test_joiners_and_soft_hyphen_are_removed(self):
        # Intra-word format characters carry no pronunciation: remove, not space.
        assert fold_invisible_characters("ca\u00adfe\u200c\u200d") == "cafe"

    def test_directional_marks_are_removed(self):
        assert fold_invisible_characters("\u200ehello\u200f") == "hello"

    def test_nfc_composes_combining_sequences(self):
        assert fold_invisible_characters("cafe\u0301") == "caf\u00e9"

    def test_visible_text_is_untouched(self):
        # Emoji are category So — handled later by the P*/S* strip, not here.
        assert fold_invisible_characters("hello, world! 👋") == "hello, world! 👋"

    def test_idempotent(self):
        once = fold_invisible_characters(f"the{ZWSP} caf\u00e9")
        assert fold_invisible_characters(once) == once


class TestUnifiedNormalization:
    """The issue's two repro cases, across all four languages.

    Assertions compare poisoned input against clean input through the same
    pipeline (rather than pinning literal output), so they hold on both the
    processor and fallback paths regardless of loanword replacement.
    """

    @pytest.mark.parametrize("lang", LANGS)
    def test_zwsp_scores_identical_to_clean_text(self, lang):
        poisoned = normalize_text_unified(f"the{ZWSP} end", lang)
        assert poisoned == normalize_text_unified("the end", lang)
        assert ZWSP not in poisoned

    @pytest.mark.parametrize("lang", LANGS)
    def test_zwsp_inside_a_word_acts_as_boundary(self, lang):
        assert normalize_text_unified(f"the{ZWSP}end", lang) == normalize_text_unified("the end", lang)

    @pytest.mark.parametrize("lang", LANGS)
    def test_nfc_and_nfd_encodings_normalize_identically(self, lang):
        nfc = "caf\u00e9 open"
        nfd = unicodedata.normalize("NFD", nfc)
        assert nfc != nfd  # the two legal encodings really differ
        assert normalize_text_unified(nfc, lang) == normalize_text_unified(nfd, lang)

    def test_english_zwsp_repro_from_issue(self):
        assert normalize_text_unified(f"the{ZWSP} end", "en") == "the end"


class TestBengaliCanonicalDirectCall:
    """normalize_bengali_for_wer is callable directly; it must fold on its
    own, and fold BEFORE loanword replacement so a zero-width character
    inside a Latin token cannot defeat the cache lookup."""

    def test_zwsp_folded(self):
        assert normalize_bengali_for_wer(f"আমি{ZWSP}ভালো") == normalize_bengali_for_wer("আমি ভালো")

    def test_zwj_zwnj_still_removed(self):
        assert normalize_bengali_for_wer("র\u200d্যাব") == normalize_bengali_for_wer("র্যাব")

    def test_latin_token_with_zwsp_matches_clean_token(self):
        # Whatever the loanword cache does with "test", it must do the same
        # thing whether or not a ZWSP splits the token invisibly.
        assert normalize_bengali_for_wer(f"te{ZWSP}st শব্দ") == normalize_bengali_for_wer("te st শব্দ")


class TestScoringEndToEnd:
    def test_invisible_characters_do_not_move_wer(self):
        """The issue's measurement: ZWSP + combining acute in the reference,
        byte-identical prediction, WER moved 0.05 -> 0.15. Both must now
        score exactly as the clean reference does."""
        from psdn_sonar.evaluators.utterance import UtteranceEvaluator

        clean_ref = "the caf\u00e9 at the end of the street is open"
        poisoned_ref = unicodedata.normalize("NFD", f"the caf\u00e9 at the{ZWSP} end of the street is open")
        hyp = "the caf\u00e9 at the end of the street is open"

        _, wer_clean, _, _ = UtteranceEvaluator.score_single_variant(clean_ref, hyp, language="en")
        _, wer_poisoned, ref_norm, _ = UtteranceEvaluator.score_single_variant(poisoned_ref, hyp, language="en")

        assert wer_clean == 0.0
        assert wer_poisoned == 0.0
        assert ZWSP not in ref_norm
