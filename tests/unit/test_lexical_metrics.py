from psdn_sonar.reporting.metrics.lexical import (
    calculate_gini_coefficient,
    calculate_lexical_diversity_metrics,
    calculate_ngram_diversity,
    calculate_ngram_diversity_chunked,
    compute_utterance_length_stats,
    compute_vocabulary_growth,
    compute_zipf_law,
)


class TestComputeUtteranceLengthStats:
    def test_quartiles_and_mean(self):
        transcripts = ["one", "one two", "one two three", "one two three four five six seven"]
        stats = compute_utterance_length_stats(transcripts)

        assert stats["total_utterances"] == 4
        assert stats["words_min"] == 1
        assert stats["words_median"] == 2.5
        assert stats["words_max"] == 7
        assert stats["words_mean"] == 3.25
        assert stats["pct_5_words_or_fewer"] == 0.75

    def test_char_stats(self):
        stats = compute_utterance_length_stats(["ab", "abcd"])
        assert stats["chars_median"] == 3.0
        assert stats["chars_mean"] == 3.0

    def test_short_utterance_corpus_profile_is_visible(self):
        # The issue-#119 scenario: a corpus dominated by <=5-word utterances
        # must be identifiable from the published stats alone.
        short_corpus = ["w1 w2 w3"] * 97 + ["w1 w2 w3 w4 w5 w6 w7 w8 w9 w10"] * 3
        stats = compute_utterance_length_stats(short_corpus)
        assert stats["words_median"] == 3.0
        assert stats["pct_5_words_or_fewer"] == 0.97

    def test_empty_returns_empty_dict(self):
        assert compute_utterance_length_stats([]) == {}


class TestComputeVocabularyGrowth:
    def test_growth_is_monotonic(self):
        texts = ["the cat sat", "the dog ran", "a bird flew"]
        growth = compute_vocabulary_growth(texts)

        assert growth[0]["tokens"] == 1
        vocab_sizes = [p["vocab_size"] for p in growth]
        assert vocab_sizes == sorted(vocab_sizes)
        assert growth[-1]["vocab_size"] == 8

    def test_repeated_words_do_not_grow_vocab(self):
        growth = compute_vocabulary_growth(["hello hello hello"])

        assert [p["vocab_size"] for p in growth] == [1, 1, 1]

    def test_sampling_limits_points(self):
        texts = ["word%d" % i for i in range(5000)]
        growth = compute_vocabulary_growth(texts, sample_points=100)

        assert len(growth) <= 101

    def test_empty_returns_empty(self):
        assert compute_vocabulary_growth([]) == []


class TestComputeZipfLaw:
    def test_frequencies_descend_by_rank(self):
        texts = ["the the the cat cat sat"]
        zipf = compute_zipf_law(texts)

        assert zipf[0] == {"rank": 1, "frequency": 3}
        freqs = [p["frequency"] for p in zipf]
        assert freqs == sorted(freqs, reverse=True)

    def test_sampling_limits_points(self):
        texts = ["word%d" % i for i in range(5000)]
        zipf = compute_zipf_law(texts, sample_points=100)

        assert len(zipf) <= 101

    def test_empty_returns_empty(self):
        assert compute_zipf_law([]) == []


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
