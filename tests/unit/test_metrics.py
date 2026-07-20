from psdn_sonar.utils.metrics import calculate_cer_wer, compute_semantic_similarity
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
