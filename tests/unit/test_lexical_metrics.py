from psdn_sonar.reporting.metrics.lexical import (
    calculate_gini_coefficient,
    calculate_lexical_diversity_metrics,
    calculate_ngram_diversity,
    calculate_ngram_diversity_chunked,
)


class TestCalculateNgramDiversity:
    def test_normal_text(self):
        texts = ["the cat sat on the mat", "the dog sat on the rug"]
        result = calculate_ngram_diversity(texts, n=1)
        assert result["total_ngrams"] == 12
        assert result["unique_ngrams"] > 0
        assert 0.0 < result["diversity"] <= 1.0

    def test_identical_texts_lower_diversity(self):
        identical = ["hello world"] * 5
        varied = ["hello world", "foo bar", "baz qux", "one two", "red blue"]
        d_identical = calculate_ngram_diversity(identical, n=1)["diversity"]
        d_varied = calculate_ngram_diversity(varied, n=1)["diversity"]
        assert d_varied > d_identical

    def test_single_word_unigram(self):
        result = calculate_ngram_diversity(["hello"], n=1)
        assert result["total_ngrams"] == 1
        assert result["unique_ngrams"] == 1
        assert result["diversity"] == 1.0

    def test_single_word_bigram_returns_zero(self):
        result = calculate_ngram_diversity(["hello"], n=2)
        assert result["total_ngrams"] == 0
        assert result["diversity"] == 0.0

    def test_empty_list(self):
        result = calculate_ngram_diversity([], n=1)
        assert result["total_ngrams"] == 0
        assert result["diversity"] == 0.0

    def test_n_larger_than_text(self):
        result = calculate_ngram_diversity(["one two"], n=5)
        assert result["total_ngrams"] == 0
        assert result["diversity"] == 0.0

    def test_bigram_diversity(self):
        texts = ["a b c d", "a b e f"]
        result = calculate_ngram_diversity(texts, n=2)
        assert result["total_ngrams"] == 6
        assert result["unique_ngrams"] == 5


class TestCalculateNgramDiversityChunked:
    def test_small_corpus_falls_through_to_raw(self):
        texts = ["a b c", "d e f", "g h i"]
        raw = calculate_ngram_diversity(texts, n=1)
        chunked = calculate_ngram_diversity_chunked(texts, n=1, chunk_size=200)
        assert chunked["diversity"] == raw["diversity"]
        assert chunked["diversity_std"] == 0.0

    def test_large_corpus_returns_mean_and_std(self):
        texts = [f"word_{i} word_{i + 1}" for i in range(500)]
        result = calculate_ngram_diversity_chunked(texts, n=1, chunk_size=100)
        assert 0.0 < result["diversity"] <= 1.0
        assert result["diversity_std"] >= 0.0
        assert result["total_ngrams"] > 0

    def test_identical_chunks_have_zero_std(self):
        texts = ["alpha beta"] * 400
        result = calculate_ngram_diversity_chunked(texts, n=1, chunk_size=200)
        assert result["diversity_std"] == 0.0

    def test_chunked_higher_than_raw_for_large_corpus(self):
        texts = [f"word_{i % 50} extra_{i}" for i in range(600)]
        raw = calculate_ngram_diversity(texts, n=1)
        chunked = calculate_ngram_diversity_chunked(texts, n=1, chunk_size=200)
        assert chunked["diversity"] >= raw["diversity"]


class TestCalculateLexicalDiversityMetrics:
    def test_returns_all_expected_keys(self):
        texts = ["the cat sat on the mat", "a dog ran fast"]
        result = calculate_lexical_diversity_metrics(texts)
        expected_keys = {
            "unigram_diversity",
            "unigram_diversity_std",
            "unigram_total",
            "unigram_unique",
            "bigram_diversity",
            "bigram_diversity_std",
            "bigram_total",
            "bigram_unique",
            "trigram_diversity",
            "trigram_diversity_std",
            "trigram_total",
            "trigram_unique",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_non_negative(self):
        texts = ["hello world foo bar"]
        result = calculate_lexical_diversity_metrics(texts)
        for v in result.values():
            assert v >= 0.0

    def test_empty_input(self):
        result = calculate_lexical_diversity_metrics([])
        assert result["unigram_diversity"] == 0.0
        assert result["bigram_total"] == 0


class TestCalculateGiniCoefficient:
    def test_uniform_distribution(self):
        gini = calculate_gini_coefficient([10, 10, 10, 10])
        assert abs(gini) < 0.1

    def test_skewed_distribution(self):
        gini = calculate_gini_coefficient([1, 1, 1, 100])
        assert gini > 0.3

    def test_empty_list(self):
        assert calculate_gini_coefficient([]) == 0.0

    def test_single_element(self):
        gini = calculate_gini_coefficient([5])
        assert isinstance(gini, float)
