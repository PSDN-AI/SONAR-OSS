"""Tests for inference latency tracking and latency plots."""

import os

import numpy as np
import pandas as pd


class TestLatencySummaryComputation:
    def test_avg_median_p95(self):
        from psdn_sonar.utils.metrics import compute_latency_summary

        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = compute_latency_summary(latencies)
        assert result["avg_latency_s"] == 0.55
        assert result["median_latency_s"] == 0.55
        assert result["p95_latency_s"] == round(float(np.percentile(latencies, 95)), 4)

    def test_single_value(self):
        from psdn_sonar.utils.metrics import compute_latency_summary

        result = compute_latency_summary([0.42])
        assert result["avg_latency_s"] == 0.42
        assert result["median_latency_s"] == 0.42
        assert result["p95_latency_s"] == 0.42

    def test_empty_list(self):
        from psdn_sonar.utils.metrics import compute_latency_summary

        result = compute_latency_summary([])
        assert result["avg_latency_s"] is None
        assert result["median_latency_s"] is None
        assert result["p95_latency_s"] is None

    def test_small_list_p95_uses_interpolation(self):
        from psdn_sonar.utils.metrics import compute_latency_summary

        latencies = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = compute_latency_summary(latencies)
        expected_p95 = round(float(np.percentile(latencies, 95)), 4)
        assert result["p95_latency_s"] == expected_p95


class TestLatencyPlots:
    def test_boxplot_creates_file(self, tmp_path):
        from psdn_sonar.reporting.plots.latency import plot_latency_boxplot

        df = pd.DataFrame(
            {
                "model": ["Model A"] * 20 + ["Model B"] * 20,
                "inference_latency_s": np.random.uniform(0.1, 1.0, 40),
            }
        )
        out = str(tmp_path / "boxplot.png")
        plot_latency_boxplot(df, out)
        assert os.path.exists(out)

    def test_boxplot_skips_empty(self, tmp_path):
        from psdn_sonar.reporting.plots.latency import plot_latency_boxplot

        df = pd.DataFrame({"model": [], "inference_latency_s": []})
        out = str(tmp_path / "boxplot.png")
        plot_latency_boxplot(df, out)
        assert not os.path.exists(out)

    def test_generate_latency_plots_creates_dir(self, tmp_path):
        from psdn_sonar.reporting.plots.latency import generate_latency_plots

        csv_path = tmp_path / "results.csv"
        df = pd.DataFrame(
            {
                "inference_latency_s": np.random.uniform(0.1, 1.0, 20),
                "wer": np.random.uniform(0.0, 0.5, 20),
            }
        )
        df.to_csv(csv_path, index=False)
        out_dir = str(tmp_path / "latency-out")
        generate_latency_plots([("test_model", str(csv_path))], out_dir)
        assert os.path.isdir(out_dir)
        assert os.path.exists(os.path.join(out_dir, "latency_boxplot.png"))

    def test_skips_when_no_latency_column(self, tmp_path):
        from psdn_sonar.reporting.plots.latency import generate_latency_plots

        csv_path = tmp_path / "results.csv"
        pd.DataFrame({"wer": [0.1, 0.2]}).to_csv(csv_path, index=False)
        out_dir = str(tmp_path / "latency-out")
        generate_latency_plots([("test_model", str(csv_path))], out_dir)
        assert not os.path.exists(os.path.join(out_dir, "latency_boxplot.png"))
        assert not os.path.isdir(out_dir)
