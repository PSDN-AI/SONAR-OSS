"""Tests for the public PoseidonScorer convenience API.

Issue #107: the scorer used to substitute worst-case values (WER/CER 1.0,
similarity 0.0) for pairs it could not score, while the evaluation pipelines
exclude missing metrics — so the same batch produced systematically shifted
aggregates depending on the entry point. The scorer now follows the single
project-wide convention: unmeasurable metrics are ``None``, POSEIDON is
``None`` when any component is, and similarity is cosine clamped to [0, 1].

jiwer is a test dependency, so CER/WER paths run everywhere; the semantic
model is always mocked (no downloads).
"""

from types import SimpleNamespace

import numpy as np
import pytest

from psdn_sonar.utils.scorer import PoseidonScorer


def _scorer_with_embeddings(vectors) -> PoseidonScorer:
    """Scorer whose semantic model returns the given embedding matrix."""
    pytest.importorskip("sentence_transformers")  # util.cos_sim used by calculate_similarity
    scorer = PoseidonScorer()
    scorer._model = SimpleNamespace(encode=lambda texts, **kwargs: np.array(vectors))
    return scorer


class TestMissingValueConvention:
    def test_empty_reference_is_unmeasurable_not_worst_case(self):
        scorer = PoseidonScorer()
        result = scorer.score("", "some hypothesis")
        assert result.wer is None
        assert result.cer is None
        assert result.similarity is None
        assert result.poseidon_score is None

    def test_blank_reference_is_unmeasurable(self):
        scorer = PoseidonScorer()
        assert scorer.calculate_wer("   ", "hypothesis") is None
        assert scorer.calculate_cer("   ", "hypothesis") is None
        assert scorer.calculate_similarity("   ", "hypothesis") is None

    def test_empty_hypothesis_is_measured_not_substituted(self):
        # A non-empty reference with an empty hypothesis IS measurable:
        # every word is wrong, so WER/CER are genuinely 1.0, not fallbacks.
        scorer = _scorer_with_embeddings([[1.0, 0.0], [0.0, 1.0]])
        result = scorer.score("hello world", "")
        assert result.wer == 1.0
        assert result.cer == 1.0
        assert result.similarity == 0.0
        assert result.poseidon_score == 0.0

    def test_similarity_backend_failure_yields_none_poseidon(self):
        scorer = PoseidonScorer()
        scorer._model = SimpleNamespace(encode=lambda texts, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
        result = scorer.score("hello world", "hello world")
        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.similarity is None
        assert result.poseidon_score is None


class TestPipelineConsistency:
    def test_wer_cer_match_canonical_helper(self):
        from psdn_sonar.utils.metrics import calculate_cer_wer

        scorer = PoseidonScorer()
        reference, hypothesis = "the quick brown fox", "the slow brown fox"
        cer, wer = calculate_cer_wer(reference, hypothesis)
        assert scorer.calculate_wer(reference, hypothesis) == wer
        assert scorer.calculate_cer(reference, hypothesis) == cer

    def test_similarity_clamped_to_unit_interval(self):
        scorer = _scorer_with_embeddings([[1.0, 0.0], [-1.0, 0.0]])  # raw cosine = -1.0
        assert scorer.calculate_similarity("reference text", "unrelated text") == 0.0

    def test_perfect_pair_scores_one(self):
        scorer = _scorer_with_embeddings([[1.0, 0.0], [1.0, 0.0]])
        result = scorer.score("hello world", "hello world")
        assert result.wer == 0.0
        assert result.cer == 0.0
        assert result.similarity == 1.0
        assert result.poseidon_score == 1.0

    def test_custom_weights_applied(self):
        scorer = _scorer_with_embeddings([[1.0, 0.0], [1.0, 0.0]])
        scorer.wer_weight, scorer.cer_weight, scorer.semantic_weight = 0.5, 0.3, 0.2
        # wer=cer=0, sim=1 -> 0.5*1 + 0.3*1 + 0.2*1 = 1.0
        assert scorer.score("hello world", "hello world").poseidon_score == 1.0

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            PoseidonScorer(wer_weight=0.5, cer_weight=0.5, semantic_weight=0.5)
