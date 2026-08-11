"""Tests for the demographic performance analyzer."""

import json

import pandas as pd
import pytest

from psdn_sonar.analysis import DemographicAnalyzer


def _write_metadata(dataset_dir, audio_id, speaker_a=None, speaker_b=None):
    audio_dir = dataset_dir / audio_id
    audio_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    if speaker_a:
        payload["speaker_a"] = speaker_a
    if speaker_b:
        payload["speaker_b"] = speaker_b
    (audio_dir / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_results_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def _results_rows(wer_values, audio_id="rec1", speaker="A"):
    return [{"audio_id": audio_id, "speaker": speaker, "wer_conv": wer} for wer in wer_values]


class TestLoadDataWithMetadata:
    def test_joins_speaker_metadata(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_metadata(
            dataset_dir,
            "rec1",
            speaker_a={"age": 30, "gender": "female", "region": "north"},
            speaker_b={"age": 45, "gender": "male", "region": "south"},
        )
        results_csv = tmp_path / "results_model_manifest.csv"
        _write_results_csv(
            results_csv,
            [
                {"audio_id": "rec1", "speaker": "A", "wer_conv": 0.1},
                {"audio_id": "rec1", "speaker": "B", "wer_conv": 0.2},
            ],
        )

        df = DemographicAnalyzer.load_data_with_metadata(results_csv, dataset_dir)

        assert df.loc[0, "gender"] == "female"
        assert df.loc[0, "region"] == "north"
        assert df.loc[1, "gender"] == "male"
        assert df.loc[1, "age"] == 45

    def test_missing_metadata_yields_na(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        results_csv = tmp_path / "results_model_manifest.csv"
        _write_results_csv(results_csv, [{"audio_id": "absent", "speaker": "A", "wer_conv": 0.1}])

        df = DemographicAnalyzer.load_data_with_metadata(results_csv, dataset_dir)

        assert pd.isna(df.loc[0, "age"])
        assert pd.isna(df.loc[0, "gender"])

    def test_derives_poseidon_score(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        results_csv = tmp_path / "results_model_manifest.csv"
        _write_results_csv(
            results_csv,
            [
                {
                    "audio_id": "rec1",
                    "speaker": "A",
                    "cer_conv": 0.0,
                    "wer_conv": 0.0,
                    "semantic_similarity_conv": 1.0,
                }
            ],
        )

        df = DemographicAnalyzer.load_data_with_metadata(results_csv, dataset_dir)

        assert df.loc[0, "poseidon_score"] == pytest.approx(1.0)


class TestGenerateSummaryStats:
    def test_per_group_aggregates_sorted_by_mean(self):
        df = pd.DataFrame(
            {
                "gender": ["female", "female", "male", "male"],
                "wer_conv": [0.1, 0.3, 0.5, 0.7],
            }
        )

        stats = DemographicAnalyzer.generate_summary_stats(df, "gender", "wer_conv")

        assert stats.iloc[0]["gender"] == "male"
        assert stats.iloc[0]["mean"] == pytest.approx(0.6)
        assert stats.iloc[0]["count"] == 2
        assert set(stats.columns) >= {"count", "mean", "std", "median", "min", "max"}

    def test_drops_na_rows(self):
        df = pd.DataFrame({"gender": ["female", None], "wer_conv": [0.1, 0.2]})

        stats = DemographicAnalyzer.generate_summary_stats(df, "gender", "wer_conv")

        assert len(stats) == 1
        assert stats.iloc[0]["count"] == 1


class TestCreateSummaryReport:
    def test_best_worst_and_gap(self, tmp_path):
        df = pd.DataFrame(
            {
                "gender": ["female", "female", "male", "male"],
                "wer_conv": [0.1, 0.1, 0.5, 0.5],
            }
        )

        report_path = tmp_path / "summary.txt"
        DemographicAnalyzer.create_summary_report(df, tmp_path, report_path)

        text = report_path.read_text(encoding="utf-8")
        assert "GENDER" in text
        # For error metrics the lowest mean is best.
        assert "Best:  female = 0.1000 (n=2)" in text
        assert "Worst: male = 0.5000 (n=2)" in text
        assert "Gap:   0.4000" in text

    def test_default_report_path(self, tmp_path):
        df = pd.DataFrame({"gender": ["female"], "wer_conv": [0.1]})

        DemographicAnalyzer.create_summary_report(df, tmp_path)

        assert (tmp_path / "demographic_analysis_summary.txt").exists()


class TestRunFullAnalysis:
    def test_generates_plots_stats_and_summary(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_metadata(
            dataset_dir,
            "rec1",
            speaker_a={"age": 30, "gender": "female", "region": "north"},
            speaker_b={"age": 45, "gender": "male", "region": "south"},
        )
        results_csv = tmp_path / "results_mymodel_manifest.csv"
        _write_results_csv(
            results_csv,
            _results_rows([0.1, 0.2, 0.3], speaker="A") + _results_rows([0.4, 0.5, 0.6], speaker="B"),
        )
        output_dir = tmp_path / "out"

        DemographicAnalyzer.run_full_analysis(results_csv, dataset_dir, output_dir)

        plots_dir = output_dir / "demographic_plots" / "mymodel"
        stats_dir = output_dir / "demographic_stats" / "mymodel"
        for demographic in ("age", "gender", "region"):
            assert (plots_dir / f"{demographic}_wer_conv.png").exists()
            assert (stats_dir / f"{demographic}_wer_conv.csv").exists()
        # Only wer_conv is present, so no other metric outputs.
        assert not (plots_dir / "gender_cer_conv.png").exists()
        assert (output_dir / "demographic_summary_mymodel.txt").exists()


class TestOverallBenchmark:
    def test_long_and_summary_across_models(self, tmp_path):
        for model, wers in [("alpha", [0.1, 0.2]), ("beta", [0.3, 0.5])]:
            _write_results_csv(tmp_path / f"results_{model}_manifest.csv", _results_rows(wers))
        csv_list = sorted(tmp_path.glob("results_*_manifest.csv"))

        long_dfs, summary_dfs = DemographicAnalyzer.build_overall_benchmark_long_and_summary(csv_list)

        assert set(long_dfs) == {"wer_conv"}
        assert len(long_dfs["wer_conv"]) == 4
        summary = summary_dfs["wer_conv"].set_index("model")
        assert summary.loc["alpha", "mean"] == pytest.approx(0.15)
        assert summary.loc["beta", "count"] == 2

    def test_unreadable_csv_skipped(self, tmp_path):
        _write_results_csv(tmp_path / "results_good_manifest.csv", _results_rows([0.1]))
        (tmp_path / "results_bad_manifest.csv").write_text("", encoding="utf-8")
        csv_list = sorted(tmp_path.glob("results_*_manifest.csv"))

        long_dfs, _ = DemographicAnalyzer.build_overall_benchmark_long_and_summary(csv_list)

        assert long_dfs["wer_conv"]["model"].unique().tolist() == ["good"]

    def test_generic_benchmark_plots_written(self, tmp_path):
        _write_results_csv(tmp_path / "results_alpha_manifest.csv", _results_rows([0.1, 0.2]))
        csv_list = [tmp_path / "results_alpha_manifest.csv"]
        long_dfs, summary_dfs = DemographicAnalyzer.build_overall_benchmark_long_and_summary(csv_list)
        output_dir = tmp_path / "out"

        DemographicAnalyzer.create_generic_benchmark_plots(long_dfs, summary_dfs, output_dir)

        assert (output_dir / "demographic_plots" / "overall_benchmark_wer_conv.png").exists()

    def test_empty_input_skips_plots(self, tmp_path):
        output_dir = tmp_path / "out"

        DemographicAnalyzer.create_generic_benchmark_plots({}, {}, output_dir)

        assert not output_dir.exists()


class TestWorstDemographic:
    def _dataset_with_gap(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_metadata(
            dataset_dir,
            "rec1",
            speaker_a={"age": 30, "gender": "female", "region": "north"},
            speaker_b={"age": 45, "gender": "male", "region": "south"},
        )
        results_csv = tmp_path / "results_alpha_manifest.csv"
        _write_results_csv(
            results_csv,
            [
                {"audio_id": "rec1", "speaker": "A", "wer_conv": 0.1},
                {"audio_id": "rec1", "speaker": "B", "wer_conv": 0.5},
            ],
        )
        return dataset_dir, [results_csv]

    def test_worst_group_is_highest_for_error_metrics(self, tmp_path):
        dataset_dir, csv_list = self._dataset_with_gap(tmp_path)

        worst_df, gap_df = DemographicAnalyzer.build_worst_demographic_table(csv_list, dataset_dir)

        gender_row = worst_df[worst_df["demographic"] == "gender"].iloc[0]
        assert gender_row["worst_mean"] == pytest.approx(0.5)
        assert gender_row["overall_mean"] == pytest.approx(0.3)
        gender_gap = gap_df[gap_df["demographic"] == "gender"].iloc[0]
        assert gender_gap["gap"] == pytest.approx(0.2)

    def test_single_group_demographic_skipped(self, tmp_path):
        dataset_dir = tmp_path / "dataset"
        _write_metadata(dataset_dir, "rec1", speaker_a={"age": 30, "gender": "female", "region": "north"})
        results_csv = tmp_path / "results_alpha_manifest.csv"
        _write_results_csv(results_csv, [{"audio_id": "rec1", "speaker": "A", "wer_conv": 0.1}])

        worst_df, gap_df = DemographicAnalyzer.build_worst_demographic_table([results_csv], dataset_dir)

        assert worst_df.empty
        assert gap_df.empty

    def test_worst_demographic_plots_written(self, tmp_path):
        dataset_dir, csv_list = self._dataset_with_gap(tmp_path)
        worst_df, gap_df = DemographicAnalyzer.build_worst_demographic_table(csv_list, dataset_dir)
        output_dir = tmp_path / "out"

        DemographicAnalyzer.create_worst_demographic_plots(worst_df, gap_df, output_dir)

        plots_dir = output_dir / "demographic_plots"
        assert (plots_dir / "worst_demographic_wer_conv.png").exists()
        assert (plots_dir / "worst_demographic_gap_wer_conv.png").exists()

    def test_empty_worst_table_skips_plots(self, tmp_path):
        output_dir = tmp_path / "out"

        DemographicAnalyzer.create_worst_demographic_plots(pd.DataFrame(), pd.DataFrame(), output_dir)

        assert not output_dir.exists()
