import pandas as pd
import pytest

from psdn_sonar.reporting.metrics.hard_negatives import calculate_hard_negatives


class TestCalculateHardNegatives:
    def test_basic_calculation(self, tmp_path):
        df = pd.DataFrame(
            {
                "wer": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                "cer": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45],
            }
        )
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        stats = calculate_hard_negatives(str(csv_path))

        assert "wer" in stats
        assert "cer" in stats
        assert "overall" in stats["wer"]
        assert "hard" in stats["wer"]
        assert stats["wer"]["overall"] == pytest.approx(0.45, abs=0.01)
        assert stats["wer"]["hard"] > stats["wer"]["overall"]

    def test_hard_negatives_have_higher_error(self, tmp_path):
        df = pd.DataFrame(
            {
                "wer": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.9, 1.0],
                "cer": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.45, 0.5],
            }
        )
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        stats = calculate_hard_negatives(str(csv_path))
        assert stats["wer"]["hard"] >= stats["wer"]["overall"]
        assert stats["cer"]["hard"] >= stats["cer"]["overall"]

    def test_missing_columns_raises_error(self, tmp_path):
        df = pd.DataFrame({"audio_path": ["a.wav"], "text": ["hello"]})
        csv_path = tmp_path / "bad.csv"
        df.to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="Could not find WER/CER columns"):
            calculate_hard_negatives(str(csv_path))

    def test_custom_percentile_threshold(self, tmp_path):
        df = pd.DataFrame(
            {
                "wer": [0.1, 0.2, 0.3, 0.4, 0.5],
                "cer": [0.05, 0.1, 0.15, 0.2, 0.25],
            }
        )
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        stats_low = calculate_hard_negatives(str(csv_path), percentile_threshold=0.5)
        stats_high = calculate_hard_negatives(str(csv_path), percentile_threshold=0.9)
        assert stats_low["wer"]["hard"] <= stats_high["wer"]["hard"]

    def test_handles_nan_values(self, tmp_path):
        df = pd.DataFrame(
            {
                "wer": [0.1, float("nan"), 0.3, 0.4, 0.5],
                "cer": [0.05, 0.1, float("nan"), 0.2, 0.25],
            }
        )
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, index=False)

        stats = calculate_hard_negatives(str(csv_path))
        assert stats["wer"]["overall"] > 0
