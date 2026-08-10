"""Tests for lexical-diversity and hard-negatives plot generation."""

import pandas as pd

from psdn_sonar.reporting.plots.hard_negatives import (
    _calculate_user_stats,
    generate_hard_negatives_comparison,
    get_benchmark_stats,
    prepare_comparison_data,
)
from psdn_sonar.reporting.plots.lexical_diversity import (
    get_dataset_order,
    plot_ngram_diversity_comparison,
    plot_vocabulary_growth,
    plot_zipf_law,
    sorted_dataset_keys,
)

TRANSCRIPTS = {
    "Common Voice": ["the cat sat on the mat", "a dog ran fast", "hello world again"],
    "FLEURS": ["one two three four", "five six seven eight"],
}


class TestDatasetOrdering:
    def test_language_specific_order(self):
        assert get_dataset_order("ko") == ["Common Voice", "FLEURS"]
        assert "LibriSpeech" in get_dataset_order("english")

    def test_unknown_language_defaults_to_bengali(self):
        assert "OpenSLR53" in get_dataset_order("swahili")

    def test_sorted_keys_puts_unknown_last(self):
        keys = ["My Dataset", "FLEURS", "Common Voice"]
        assert sorted_dataset_keys(keys, "en") == ["Common Voice", "FLEURS", "My Dataset"]


class TestLexicalDiversityPlots:
    def test_ngram_comparison_renders(self, tmp_path):
        results = {
            "My Dataset": {
                "unigram_diversity": 0.5,
                "unigram_diversity_std": 0.05,
                "bigram_diversity": 0.7,
                "bigram_diversity_std": 0.0,
                "trigram_diversity": 0.9,
                "trigram_diversity_std": 0.0,
            }
        }
        out = tmp_path / "ngram.png"
        plot_ngram_diversity_comparison(results, str(out), include_benchmarks=False)
        assert out.exists()

    def test_vocabulary_growth_renders(self, tmp_path):
        out = tmp_path / "vocab.png"
        plot_vocabulary_growth(TRANSCRIPTS, str(out), include_public_benchmarks=False)
        assert out.exists()

    def test_zipf_renders(self, tmp_path):
        out = tmp_path / "zipf.png"
        plot_zipf_law(TRANSCRIPTS, str(out), include_public_benchmarks=False)
        assert out.exists()

    def test_empty_transcripts_no_output(self, tmp_path):
        out = tmp_path / "vocab.png"
        plot_vocabulary_growth({}, str(out), include_public_benchmarks=False)
        assert not out.exists()


class TestHardNegativesStats:
    def _csv(self, tmp_path, **cols):
        df = pd.DataFrame(cols)
        p = tmp_path / "results.csv"
        df.to_csv(p, index=False)
        return str(p)

    def test_stats_from_plain_columns(self, tmp_path):
        csv = self._csv(tmp_path, wer=[0.1, 0.2, 0.3, 0.9], cer=[0.05, 0.1, 0.15, 0.5])
        stats = _calculate_user_stats(csv)
        assert stats["wer"]["hard"] > stats["wer"]["overall"]
        assert set(stats["wer"]) == {"overall", "overall_std", "hard", "hard_std"}

    def test_conv_columns_preferred(self, tmp_path):
        csv = self._csv(tmp_path, wer_conv=[0.5, 0.5], wer=[0.0, 0.0], cer_conv=[0.2, 0.2])
        stats = _calculate_user_stats(csv)
        assert stats["wer"]["overall"] == 0.5

    def test_missing_metric_omitted(self, tmp_path):
        csv = self._csv(tmp_path, wer=[0.1, 0.2])
        stats = _calculate_user_stats(csv)
        assert "wer" in stats and "cer" not in stats

    def test_benchmark_stats_by_language(self):
        assert "Zeroth" in get_benchmark_stats("ko")["wer"]
        assert "OpenSLR53" in get_benchmark_stats("bn-unknown")["wer"]

    def test_prepare_comparison_puts_user_first(self):
        stats = {"wer": {"overall": 0.3, "overall_std": 0.1, "hard": 0.8, "hard_std": 0.2}}
        df = prepare_comparison_data(stats, "wer", language="en")
        assert df.iloc[0]["dataset"] == "User Dataset"
        assert set(df["condition"]) == {"Overall", "Hard Negatives"}


class TestGenerateHardNegativesComparison:
    def test_generates_both_plots(self, tmp_path):
        df = pd.DataFrame({"wer": [0.1, 0.2, 0.3, 0.9, 0.5], "cer": [0.05, 0.1, 0.2, 0.5, 0.3]})
        csv = tmp_path / "results.csv"
        df.to_csv(csv, index=False)
        out = tmp_path / "plots"
        generate_hard_negatives_comparison(str(csv), str(out), language="en")
        assert (out / "wer_overall_vs_hard_negatives.png").exists()
        assert (out / "cer_overall_vs_hard_negatives.png").exists()

    def test_no_metric_columns_no_plots(self, tmp_path):
        df = pd.DataFrame({"text": ["a", "b"]})
        csv = tmp_path / "results.csv"
        df.to_csv(csv, index=False)
        out = tmp_path / "plots"
        generate_hard_negatives_comparison(str(csv), str(out))
        assert not (out / "wer_overall_vs_hard_negatives.png").exists()
