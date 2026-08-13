"""Lexical diversity metrics: n-gram diversity and frequency concentration."""

from typing import Dict, List

import numpy as np

DEFAULT_CHUNK_SIZE = 200


def calculate_ngram_diversity(texts: List[str], n: int) -> Dict:
    """Compute raw (un-chunked) n-gram diversity for *texts*."""
    all_ngrams = []

    for text in texts:
        words = text.split()
        if len(words) >= n:
            text_ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
            all_ngrams.extend(text_ngrams)

    if len(all_ngrams) == 0:
        return {"diversity": 0.0, "total_ngrams": 0, "unique_ngrams": 0}

    unique_ngrams = set(all_ngrams)

    return {
        "diversity": len(unique_ngrams) / len(all_ngrams),
        "total_ngrams": len(all_ngrams),
        "unique_ngrams": len(unique_ngrams),
    }


def _compute_ngram_diversity_from_indices(
    texts: List[str],
    indices,
    n: int,
    chunk_size: int,
) -> Dict:
    """Core chunked n-gram diversity computation with pre-computed indices."""
    if len(texts) <= chunk_size:
        raw = calculate_ngram_diversity(texts, n)
        raw["diversity_std"] = 0.0
        return raw

    chunk_diversities = []
    total_ngrams_sum = 0
    unique_ngrams_sum = 0

    for start in range(0, len(indices) - chunk_size + 1, chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        chunk_texts = [texts[i] for i in chunk_idx]
        result = calculate_ngram_diversity(chunk_texts, n)
        chunk_diversities.append(result["diversity"])
        total_ngrams_sum += result["total_ngrams"]
        unique_ngrams_sum += result["unique_ngrams"]

    if not chunk_diversities:
        raw = calculate_ngram_diversity(texts, n)
        raw["diversity_std"] = 0.0
        return raw

    return {
        "diversity": float(np.mean(chunk_diversities)),
        "diversity_std": float(np.std(chunk_diversities)),
        "total_ngrams": total_ngrams_sum,
        "unique_ngrams": unique_ngrams_sum,
    }


def calculate_ngram_diversity_chunked(
    texts: List[str],
    n: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    seed: int = 42,
) -> Dict:
    """Size-normalised n-gram diversity via chunking and averaging.

    Splits *texts* into non-overlapping chunks of *chunk_size* transcripts,
    computes diversity per chunk, and returns the mean (and std) across
    chunks.  This removes the corpus-size bias inherent in raw
    unique/total ratios so datasets of different sizes can be compared
    fairly.

    If the corpus is smaller than *chunk_size* the raw (un-chunked) score
    is returned instead, with ``diversity_std`` set to 0.
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(texts))
    return _compute_ngram_diversity_from_indices(texts, indices, n, chunk_size)


def calculate_lexical_diversity_metrics(
    transcripts: List[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict:
    """Compute unigram/bigram/trigram diversity with chunked averaging."""
    results = {}
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(transcripts))

    for n, name in [(1, "unigram"), (2, "bigram"), (3, "trigram")]:
        ngram_metrics = _compute_ngram_diversity_from_indices(transcripts, indices, n, chunk_size)
        results[f"{name}_diversity"] = ngram_metrics["diversity"]
        results[f"{name}_diversity_std"] = ngram_metrics.get("diversity_std", 0.0)
        results[f"{name}_total"] = ngram_metrics["total_ngrams"]
        results[f"{name}_unique"] = ngram_metrics["unique_ngrams"]

    return results


def compute_vocabulary_growth(
    texts: List[str],
    sample_points: int = 1000,
    max_tokens: int = 1_000_000,
) -> List[Dict]:
    """Vocabulary growth curve: cumulative unique words per token seen.

    Returns at most *sample_points* evenly spaced ``{"tokens", "vocab_size"}``
    rows, truncated at *max_tokens* for storage efficiency.
    """
    all_words = []
    vocab_sizes = []
    seen_vocab = set()

    for text in texts:
        for word in text.split():
            all_words.append(word)
            seen_vocab.add(word)
            vocab_sizes.append(len(seen_vocab))

    if not all_words:
        return []

    token_counts = range(1, len(all_words) + 1)
    sample_rate = max(1, len(all_words) // sample_points)
    return [
        {"tokens": tc, "vocab_size": vs}
        for tc, vs in zip(token_counts[::sample_rate], vocab_sizes[::sample_rate])
        if tc <= max_tokens
    ]


def compute_zipf_law(texts: List[str], sample_points: int = 1000) -> List[Dict]:
    """Zipf distribution: word frequencies by rank, descending.

    Returns at most *sample_points* evenly spaced ``{"rank", "frequency"}`` rows.
    """
    from collections import Counter

    all_words = []
    for text in texts:
        all_words.extend(text.split())

    if not all_words:
        return []

    frequencies = sorted(Counter(all_words).values(), reverse=True)
    ranks = range(1, len(frequencies) + 1)
    sample_rate = max(1, len(frequencies) // sample_points)
    return [{"rank": r, "frequency": f} for r, f in zip(ranks[::sample_rate], frequencies[::sample_rate])]


def calculate_gini_coefficient(frequencies: List[int]) -> float:
    """Gini coefficient of a frequency distribution (0 = uniform, →1 = concentrated)."""
    sorted_freq = np.array(sorted(frequencies))
    n = len(sorted_freq)
    if n == 0:
        return 0.0
    cumsum = np.cumsum(sorted_freq)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_freq))) / (n * cumsum[-1]) - (n + 1) / n
