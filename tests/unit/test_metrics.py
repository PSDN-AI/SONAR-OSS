import numpy as np
import pytest

from psdn_sonar.utils.metrics import (
    calculate_cer_wer,
    calculate_poseidon_score,
    clamp_similarity,
    compute_semantic_similarity,
)
from psdn_sonar.utils.text_processing import normalize_text_unified


class TestMetrics:
    def test_calculate_cer_wer_identical(self):
        reference = "এটি একটি পরীক্ষা"
        hypothesis = "এটি একটি পরীক্ষা"
        cer, wer = calculate_cer_wer(reference, hypothesis)
        assert cer == 0.0
        assert wer == 0.0

    def test_calculate_cer_wer_different(self):
        reference = "এটি একটি পরীক্ষা"
        hypothesis = "এটি পরীক্ষা"
        cer, wer = calculate_cer_wer(reference, hypothesis)
        assert cer > 0.0
        assert wer > 0.0
        assert wer >= cer

    def test_calculate_cer_wer_empty_hypothesis(self):
        reference = "এটি একটি পরীক্ষা"
        hypothesis = ""
        cer, wer = calculate_cer_wer(reference, hypothesis)
        assert cer == 1.0
        assert wer == 1.0

    def test_semantic_similarity_similar(self):
        text1 = "The cat is sleeping"
        text2 = "A cat is asleep"
        similarity = compute_semantic_similarity(text1, text2)
        if similarity is None:
            return  # sentence-transformers not available in CI
        assert 0.5 <= similarity <= 1.0

    def test_semantic_similarity_different(self):
        text1 = "The weather is nice"
        text2 = "I like programming"
        similarity = compute_semantic_similarity(text1, text2)
        if similarity is None:
            return  # sentence-transformers not available in CI
        assert 0.0 <= similarity <= 1.0

    def test_poseidon_score_happy_path(self):
        score = calculate_poseidon_score(0.0, 0.0, 1.0)
        assert score == 1.0
        score = calculate_poseidon_score(1.0, 1.0, 0.0)
        assert score == 0.0

    def test_poseidon_score_none_similarity_names_ml_extra(self):
        """Issue #101: None similarity must raise an actionable error, not an

        opaque comparison TypeError. compute_semantic_similarity returns None
        when sentence-transformers ([ml] extra) is missing.
        """
        with pytest.raises(TypeError, match=r"psdn-sonar\[ml\]"):
            calculate_poseidon_score(0.1, 0.2, None)

    def test_poseidon_score_none_cer_wer_is_actionable(self):
        with pytest.raises(TypeError, match="calculate_cer_wer"):
            calculate_poseidon_score(None, 0.2, 0.9)
        with pytest.raises(TypeError, match="calculate_cer_wer"):
            calculate_poseidon_score(0.1, None, 0.9)

    def test_clamp_similarity_bounds(self):
        assert clamp_similarity(-0.5) == 0.0
        assert clamp_similarity(1.5) == 1.0


class TestPoseidonWeightValidation:
    """Weight validation after issue #238.

    Both validation sites checked only that the weights sum to 1.0, so a set
    with a negative entry that still summed to 1.0 (e.g. -0.5/0.75/0.75) was
    accepted silently and inverted the composite: a transcript with nine
    times the WER scored *higher* on identical CER and similarity. Negative
    weights are now rejected everywhere with an error naming the offending
    weight.
    """

    def test_negative_weight_summing_to_one_is_rejected_per_call(self):
        # The exact set from the issue's reproduction.
        with pytest.raises(ValueError, match=r"non-negative.*wer_weight=-0\.5"):
            calculate_poseidon_score(
                cer=0.1, wer=0.1, similarity=0.8, wer_weight=-0.5, cer_weight=0.75, semantic_weight=0.75
            )

    def test_the_inversion_can_no_longer_happen(self):
        """Pre-fix, wer=0.9 outscored wer=0.1 under -0.5/0.75/0.75."""
        for wer in (0.1, 0.9):
            with pytest.raises(ValueError, match="non-negative"):
                calculate_poseidon_score(
                    cer=0.1, wer=wer, similarity=0.8, wer_weight=-0.5, cer_weight=0.75, semantic_weight=0.75
                )

    def test_sum_check_still_fires(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            calculate_poseidon_score(
                cer=0.1, wer=0.1, similarity=0.8, wer_weight=0.5, cer_weight=0.5, semantic_weight=0.5
            )

    def test_all_negatives_are_named(self):
        with pytest.raises(ValueError, match=r"cer_weight=-0\.1.*semantic_weight=-0\.1"):
            calculate_poseidon_score(
                cer=0.1, wer=0.1, similarity=0.8, wer_weight=1.2, cer_weight=-0.1, semantic_weight=-0.1
            )

    def test_valid_custom_weights_still_accepted(self):
        score = calculate_poseidon_score(
            cer=0.0, wer=0.0, similarity=1.0, wer_weight=0.5, cer_weight=0.3, semantic_weight=0.2
        )
        assert score == 1.0

    def test_zero_weight_is_allowed(self):
        # Zero disables a component without inverting anything.
        score = calculate_poseidon_score(
            cer=0.0, wer=1.0, similarity=1.0, wer_weight=0.0, cer_weight=0.5, semantic_weight=0.5
        )
        assert score == 1.0

    def test_env_var_config_rejects_negative_weights(self, monkeypatch):
        """The env-var surface from the issue: POSEIDON_*_WEIGHT with a
        negative entry summing to 1.0 must fail at Config construction."""
        from psdn_sonar.config import Config

        monkeypatch.setenv("POSEIDON_WER_WEIGHT", "-0.5")
        monkeypatch.setenv("POSEIDON_CER_WEIGHT", "0.75")
        monkeypatch.setenv("POSEIDON_SEMANTIC_WEIGHT", "0.75")

        with pytest.raises(ValueError, match=r"non-negative.*wer_weight=-0\.5"):
            Config()

    def test_env_var_config_sum_check_unchanged(self, monkeypatch):
        from psdn_sonar.config import Config

        monkeypatch.setenv("POSEIDON_WER_WEIGHT", "0.5")
        monkeypatch.setenv("POSEIDON_CER_WEIGHT", "0.5")
        monkeypatch.setenv("POSEIDON_SEMANTIC_WEIGHT", "0.5")

        with pytest.raises(ValueError, match="sum to 1.0"):
            Config()

    def test_default_config_still_valid(self):
        from psdn_sonar.config import Config

        config = Config()
        assert config.wer_weight + config.cer_weight + config.semantic_weight == pytest.approx(1.0)
        assert clamp_similarity(0.37) == 0.37

    def test_semantic_similarity_clamped_at_source(self, monkeypatch):
        """Issue #107: similarity used to be clamped inside POSEIDON but
        stored and averaged raw, so semantic_similarity_mean could go
        negative while poseidon_score_mean could not. It is now clamped to
        [0, 1] once, where it is computed."""
        pytest.importorskip("sentence_transformers")

        class _OpposedEmbeddings:
            def encode(self, texts, **kwargs):
                return np.array([[1.0, 0.0], [-1.0, 0.0]])  # raw cosine = -1.0

        monkeypatch.setattr("psdn_sonar.utils.metrics._get_semantic_model", lambda: _OpposedEmbeddings())
        assert compute_semantic_similarity("some reference", "unrelated text") == 0.0

    def test_normalize_text_unified(self):
        text = "  এটি   একটি  পরীক্ষা  "
        normalized = normalize_text_unified(text)
        assert normalized == "এটি একটি পরীক্ষা"

    def test_normalize_text_unified_removes_punctuation(self):
        text = "এটি, একটি! পরীক্ষা?"
        normalized = normalize_text_unified(text)
        assert "," not in normalized
        assert "!" not in normalized
        assert "?" not in normalized
