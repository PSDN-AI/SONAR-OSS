"""Tests for cross-dataset and model-comparison plot generation."""

import numpy as np
import pandas as pd

from psdn_sonar.reporting.plots.cross_dataset import (
    generate_cross_dataset_plots,
    generate_model_comparison_plots,
    load_public_benchmark_data,
    load_user_dataset_results,
)
from psdn_sonar.utils.metrics import ensure_poseidon_score

rng = np.random.default_rng(11)


def _write_results_csv(path, n=15, prefix="conv"):
    suffix = f"_{prefix}" if prefix else ""
    df = pd.DataFrame(
        {
            f"cer{suffix}": rng.uniform(0.05, 0.4, n),
            f"wer{suffix}": rng.uniform(0.1, 0.6, n),
            f"semantic_similarity{suffix}": rng.uniform(0.5, 1.0, n),
        }
    )
    df.to_csv(path, index=False)
    return str(path)


class TestEnsurePoseidonScoreLayouts:
    def test_plain_columns(self):
        df = pd.DataFrame({"cer": [0.1], "wer": [0.2], "semantic_similarity": [0.9]})
        assert "poseidon_score" in ensure_poseidon_score(df).columns

    def test_non_conv_columns(self):
        df = pd.DataFrame({"cer_non": [0.1], "wer_non": [0.2]})
        out = ensure_poseidon_score(df)
        assert 0.0 <= out["poseidon_score"].iloc[0] <= 1.0

    def test_uppercase_columns(self):
        df = pd.DataFrame({"CER": [0.1], "WER": [0.2]})
        assert "poseidon_score" in ensure_poseidon_score(df).columns

    def test_conv_layout_preferred(self):
        df = pd.DataFrame({"cer_conv": [0.0], "wer_conv": [0.0], "cer": [1.0], "wer": [1.0]})
        out = ensure_poseidon_score(df)
        # Perfect conv metrics (with sim missing -> 0.0) score higher than the worst-case plain ones would.
        assert out["poseidon_score"].iloc[0] > 0.5


class TestLoadUserDatasetResults:
    def test_loads_conv_columns(self, tmp_path):
        csv = _write_results_csv(tmp_path / "r.csv")
        rows = load_user_dataset_results(csv, "my_model")
        assert len(rows) == 15
        assert rows[0]["dataset"] == "user_dataset"
        assert rows[0]["model"] == "my_model"
        assert rows[0]["poseidon"] is not None

    def test_nan_rows_skipped(self, tmp_path):
        df = pd.DataFrame({"cer": [0.1, np.nan, 0.2], "wer": [0.2, 0.3, np.nan]})
        path = tmp_path / "r.csv"
        df.to_csv(path, index=False)
        rows = load_user_dataset_results(str(path), "m")
        assert len(rows) == 1

    def test_missing_metric_columns(self, tmp_path):
        df = pd.DataFrame({"text": ["a", "b"]})
        path = tmp_path / "r.csv"
        df.to_csv(path, index=False)
        assert load_user_dataset_results(str(path), "m") == []

    def test_missing_file(self, tmp_path):
        assert load_user_dataset_results(str(tmp_path / "missing.csv"), "m") == []


class TestLoadPublicBenchmarkData:
    def test_no_shipped_benchmarks_returns_empty(self):
        assert load_public_benchmark_data("swahili") == []
        assert load_public_benchmark_data("bn") == []


class TestGenerateCrossDatasetPlots:
    def test_user_only_plots(self, tmp_path):
        csv = _write_results_csv(tmp_path / "results.csv")
        out = tmp_path / "plots"
        generate_cross_dataset_plots(csv, "test_model", str(out), language="en")
        for metric in ("cer", "wer", "sem", "poseidon"):
            assert (out / f"{metric}_by_dataset_model.png").exists()

    def test_no_data_no_plots(self, tmp_path):
        df = pd.DataFrame({"text": ["a"]})
        csv = tmp_path / "results.csv"
        df.to_csv(csv, index=False)
        out = tmp_path / "plots"
        generate_cross_dataset_plots(str(csv), "m", str(out), language="swahili")
        assert not any(out.glob("*.png"))


class TestGenerateModelComparisonPlots:
    def test_compares_two_models(self, tmp_path):
        csv_a = _write_results_csv(tmp_path / "a.csv")
        csv_b = _write_results_csv(tmp_path / "b.csv", prefix="")
        out = tmp_path / "plots"
        generate_model_comparison_plots([("model_a", csv_a), ("custom_org_model-b", csv_b)], str(out))
        for metric in ("cer", "wer", "sem", "poseidon"):
            assert (out / f"{metric}_model_comparison.png").exists()

    def test_missing_csv_skipped(self, tmp_path):
        csv_a = _write_results_csv(tmp_path / "a.csv")
        out = tmp_path / "plots"
        generate_model_comparison_plots([("a", csv_a), ("b", str(tmp_path / "nope.csv"))], str(out))
        assert (out / "wer_model_comparison.png").exists()
