"""Tests for shared UtteranceEvaluator scoring paths."""

from unittest.mock import MagicMock

from psdn_sonar.evaluators.utterance import UtteranceEvaluator
from psdn_sonar.utils.text_processing import is_devanagari


def test_score_single_variant_returns_metrics_and_normalized_text():
    cer, wer, ref_norm, hyp_norm = UtteranceEvaluator.score_single_variant(
        "hello world",
        "hello word",
        language="en",
    )
    assert ref_norm
    assert hyp_norm
    assert isinstance(cer, float)
    assert isinstance(wer, float)
    assert cer >= 0.0
    assert wer >= 0.0


def test_score_single_variant_normalizes_case_and_punctuation():
    """USAGE.md section 1 promises the evaluation-path score: WER/CER 0 for a
    transcription that differs only in case and punctuation (issue #100)."""
    cer, wer, ref_norm, hyp_norm = UtteranceEvaluator.score_single_variant(
        "Hello, World!",
        "hello world",
        language="en",
    )
    assert ref_norm == hyp_norm == "hello world"
    assert cer == 0.0
    assert wer == 0.0


def test_score_single_variant_preserves_none_cer_wer(monkeypatch):
    monkeypatch.setattr(
        "psdn_sonar.utils.metrics.calculate_cer_wer",
        lambda _ref, _hyp: (None, None),
    )
    cer, wer, _, _ = UtteranceEvaluator.score_single_variant("x", "y", language="en")
    assert cer is None
    assert wer is None


def test_convert_hypothesis_devanagari_for_bengali():
    devanagari_ka = "\u0915"  # क
    converted = UtteranceEvaluator.convert_hypothesis(devanagari_ka, language="bn")
    assert converted != devanagari_ka
    assert not is_devanagari(converted)


def test_score_dual_variant_bengali_splits_devanagari_path():
    devanagari_ka = "\u0915"
    scored = UtteranceEvaluator.score_dual_variant(
        "reference",
        devanagari_ka,
        language="bn",
        with_semantics=False,
    )
    assert scored.hyp_converted != devanagari_ka
    assert scored.hyp_norm_non != scored.hyp_norm_conv


def test_score_dual_variant_english_aligns_non_and_conv_paths():
    scored = UtteranceEvaluator.score_dual_variant(
        "reference",
        "hypothesis",
        language="en",
    )
    assert scored.ref_norm
    assert scored.hyp_norm_non == scored.hyp_norm_conv
    assert scored.non.cer == scored.conv.cer
    assert scored.non.wer == scored.conv.wer


def test_score_normalized_pair_skips_semantics(monkeypatch):
    sem_mock = MagicMock(return_value=0.9)
    monkeypatch.setattr("psdn_sonar.utils.metrics.compute_semantic_similarity", sem_mock)

    scored = UtteranceEvaluator.score_normalized_pair("hello", "hello", with_semantics=False)

    assert scored.cer is not None
    assert scored.wer is not None
    assert scored.semantic_similarity is None
    assert scored.poseidon_score is None
    sem_mock.assert_not_called()


def test_score_normalized_pair_computes_semantics_by_default(monkeypatch):
    sem_mock = MagicMock(return_value=0.8)
    poseidon_mock = MagicMock(return_value=0.7)
    monkeypatch.setattr("psdn_sonar.utils.metrics.compute_semantic_similarity", sem_mock)
    monkeypatch.setattr("psdn_sonar.utils.metrics.calculate_poseidon_score", poseidon_mock)

    scored = UtteranceEvaluator.score_normalized_pair("hello", "hello")

    sem_mock.assert_called_once()
    poseidon_mock.assert_called_once()
    assert scored.semantic_similarity == 0.8
    assert scored.poseidon_score == 0.7
