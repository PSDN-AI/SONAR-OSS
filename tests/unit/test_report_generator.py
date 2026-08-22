"""Tests for the markdown evaluation report generator."""

import json

import pytest

from psdn_sonar.reporting.generators.report_generator import generate_report, load_dataset_stats


def _write_tsv(path, transcripts, column="transcription"):
    lines = [f"id\t{column}"]
    lines.extend(f"{i}\t{text}" for i, text in enumerate(transcripts))
    path.write_text("\n".join(lines), encoding="utf-8")


class TestLoadDatasetStats:
    def test_tsv_stats(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        _write_tsv(tsv, ["hello world", "hello again world"])

        stats = load_dataset_stats(tsv)

        assert stats["total_transcripts"] == 2
        assert stats["total_tokens"] == 5
        assert stats["unique_tokens"] == 3
        assert stats["ttr"] == pytest.approx(60.0)
        assert stats["avg_length"] == pytest.approx(2.5)

    def test_csv_stats(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("id,transcript\n0,one two\n1,three\n", encoding="utf-8")

        stats = load_dataset_stats(csv)

        assert stats["total_transcripts"] == 2
        assert stats["total_tokens"] == 3

    def test_jsonl_stats(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        rows = [{"transcription": "a b c"}, {"transcript": "a b"}]
        jsonl.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        stats = load_dataset_stats(jsonl)

        assert stats["total_transcripts"] == 2
        assert stats["total_tokens"] == 5
        assert stats["unique_tokens"] == 3

    def test_jsonl_without_transcripts_counts_lines(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        jsonl.write_text('{"other": 1}\n{"other": 2}', encoding="utf-8")

        stats = load_dataset_stats(jsonl)

        assert stats["total_transcripts"] == 2
        assert stats["total_tokens"] == 0
        assert stats["ttr"] == 0

    def test_skips_nan_transcripts(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        tsv.write_text("id\ttranscription\n0\thello\n1\t\n", encoding="utf-8")

        stats = load_dataset_stats(tsv)

        assert stats["total_transcripts"] == 2
        assert stats["total_tokens"] == 1

    def test_missing_transcript_column_raises(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        tsv.write_text("id\tother\n0\tx\n", encoding="utf-8")

        with pytest.raises(ValueError, match="transcript column"):
            load_dataset_stats(tsv)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Expected a file"):
            load_dataset_stats(tmp_path / "absent.tsv")


class TestGenerateReport:
    def _dataset(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        _write_tsv(tsv, ["hello world", "another sample here"])
        return tsv

    def test_minimal_report(self, tmp_path):
        tsv = self._dataset(tmp_path)
        output = tmp_path / "report" / "EVAL_REPORT.md"

        result = generate_report("My Dataset", tsv, output, language="bengali")

        assert result == output.resolve()
        text = output.read_text(encoding="utf-8")
        assert "# SONAR - My Dataset Results" in text
        assert "## Dataset Statistics" in text
        # No benchmark data ships, so the setup table must say so (issue #113).
        assert "None included" in text
        assert "## Key Insights" in text
        # No plot directories exist, so plot sections are absent.
        assert "Cross-Dataset Comparison" not in text
        assert "## Audio Quality Analysis" not in text

    def test_unknown_language_setup(self, tmp_path):
        tsv = self._dataset(tmp_path)
        output = tmp_path / "EVAL_REPORT.md"

        generate_report("DS", tsv, output, language="swahili")

        text = output.read_text(encoding="utf-8")
        assert "| **Language** | Swahili |" in text
        assert "None included" in text

    def test_model_comparison_preferred_over_cross_dataset(self, tmp_path):
        tsv = self._dataset(tmp_path)
        report_dir = tmp_path / "report"
        for sub, name in [
            ("model-comparison", "cer_model_comparison.png"),
            ("cross-dataset-analysis", "cer_by_dataset_model.png"),
        ]:
            (report_dir / sub).mkdir(parents=True)
            (report_dir / sub / name).write_bytes(b"png")

        generate_report("DS", tsv, report_dir / "EVAL_REPORT.md")

        text = (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")
        assert "### Model Comparison" in text
        assert "![Character Error Rate (CER) by Model](model-comparison/cer_model_comparison.png)" in text
        assert "Cross-Dataset Comparison" not in text

    def test_cross_dataset_section(self, tmp_path):
        tsv = self._dataset(tmp_path)
        report_dir = tmp_path / "report"
        cross_dir = report_dir / "cross-dataset-analysis"
        cross_dir.mkdir(parents=True)
        (cross_dir / "wer_by_dataset_model.png").write_bytes(b"png")

        generate_report("DS", tsv, report_dir / "EVAL_REPORT.md")

        text = (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")
        # No benchmark data ships, so the section must not claim a comparison
        # the plot does not make (issue #113).
        assert "### Performance Distributions" in text
        assert "the plots contain your dataset only" in text
        assert "Cross-Dataset Comparison" not in text
        assert "![Word Error Rate (WER) by Dataset and Model](cross-dataset-analysis/wer_by_dataset_model.png)" in text
        # Only the existing plot is embedded.
        assert "cer_by_dataset_model.png" not in text

    def test_audio_quality_section_with_summary(self, tmp_path):
        tsv = self._dataset(tmp_path)
        report_dir = tmp_path / "report"
        quality_dir = report_dir / "audio-quality-analysis"
        quality_dir.mkdir(parents=True)
        (quality_dir / "snr_distribution.png").write_bytes(b"png")
        (quality_dir / "quality_summary.json").write_text(
            json.dumps(
                {
                    "total_utterances": 1200,
                    "mean_snr_db": 15.3,
                    "pct_clipped": 2.5,
                    "pct_excessive_silence": 1.0,
                    "pct_passing_all": 95.0,
                }
            ),
            encoding="utf-8",
        )

        generate_report("DS", tsv, report_dir / "EVAL_REPORT.md")

        text = (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")
        assert "## Audio Quality Analysis" in text
        assert "### SNR Distribution" in text
        assert "### Data Quality Summary" in text
        assert "| **Total Utterances** | 1,200 |" in text

    def test_demographic_and_diversity_sections(self, tmp_path):
        tsv = self._dataset(tmp_path)
        report_dir = tmp_path / "report"
        demo_dir = report_dir / "demographic-analysis" / "demographic_plots" / "model"
        demo_dir.mkdir(parents=True)
        (demo_dir / "gender_wer_conv.png").write_bytes(b"png")
        diversity_dir = report_dir / "diversity-analysis"
        diversity_dir.mkdir(parents=True)
        (diversity_dir / "diversity_gt_zipf_law.png").write_bytes(b"png")

        generate_report("DS", tsv, report_dir / "EVAL_REPORT.md", language="korean")

        text = (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")
        assert "## Demographic Analysis" in text
        assert "gender_wer_conv.png" in text
        assert "### Zipf's Law Curve" in text
        # No diversity benchmark JSONs ship, so no benchmark mention (issue #113).
        assert "public Korean benchmarks" not in text

    def test_latency_section(self, tmp_path):
        tsv = self._dataset(tmp_path)
        report_dir = tmp_path / "report"
        latency_dir = report_dir / "latency-analysis"
        latency_dir.mkdir(parents=True)
        (latency_dir / "latency_boxplot.png").write_bytes(b"png")

        generate_report("DS", tsv, report_dir / "EVAL_REPORT.md")

        text = (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")
        assert "## Inference Latency" in text
        assert "![Latency Distribution](latency-analysis/latency_boxplot.png)" in text

    def test_creates_output_directory(self, tmp_path):
        tsv = self._dataset(tmp_path)
        output = tmp_path / "deep" / "nested" / "EVAL_REPORT.md"

        generate_report("DS", tsv, output)

        assert output.exists()


class TestBenchmarkClaimGating:
    """Issue #113: every public-benchmark claim in EVAL_REPORT.md must be
    gated on benchmark data actually being present. In a stock install none
    ships, so the report may never say the user's numbers were compared
    against public benchmarks."""

    _FORBIDDEN_WITHOUT_DATA = [
        "against public benchmarks",
        "reference numbers shown for comparison",
        "public Bengali benchmarks",
        "public English benchmarks",
        "relative to established benchmarks",
        "compared to public",
        "consistent with public",
    ]

    def _dataset(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        _write_tsv(tsv, ["hello world", "another sample here"])
        return tsv

    def _full_report(self, tmp_path, language):
        report_dir = tmp_path / "report"
        for sub, name in [
            ("cross-dataset-analysis", "wer_by_dataset_model.png"),
            ("hard-negatives-analysis", "wer_overall_vs_hard_negatives.png"),
            ("diversity-analysis", "diversity_gt_zipf_law.png"),
        ]:
            (report_dir / sub).mkdir(parents=True, exist_ok=True)
            (report_dir / sub / name).write_bytes(b"png")
        generate_report("DS", self._dataset(tmp_path), report_dir / "EVAL_REPORT.md", language=language)
        return (report_dir / "EVAL_REPORT.md").read_text(encoding="utf-8")

    def test_stock_install_makes_no_benchmark_comparison_claim(self, tmp_path):
        """The exact report shape from the issue's repro (cross-dataset plots
        rendered, no benchmark data) must not claim any comparison."""
        text = self._full_report(tmp_path, "english")

        for forbidden in self._FORBIDDEN_WITHOUT_DATA:
            assert forbidden not in text, f"unsupported claim in report: {forbidden!r}"
        assert "None included" in text
        assert "describes your dataset only" in text
        assert "the plots contain your dataset only" in text

    def test_coverage_row_derived_from_data_actually_present(self, tmp_path, monkeypatch):
        """When raw-evaluation CSVs exist, the coverage row names exactly the
        datasets found on disk — not a hardcoded per-language constant."""
        benchmarks = tmp_path / "benchmarks"
        eval_dir = benchmarks / "bengali" / "raw-evaluations" / "some_model"
        eval_dir.mkdir(parents=True)
        (eval_dir / "commonvoice_results.csv").write_text("cer,wer\n0.1,0.2\n", encoding="utf-8")
        (eval_dir / "fleurs_results.csv").write_text("cer,wer\n0.1,0.2\n", encoding="utf-8")
        monkeypatch.setattr("psdn_sonar.reporting.loaders.benchmark_loader._BENCHMARKS_DIR", benchmarks)

        text = self._full_report(tmp_path, "bn")

        assert "benchmark evaluations (Common Voice, FLEURS)" in text
        assert "### Cross-Dataset Comparison" in text
        assert "against public benchmarks" in text
        assert "None included" not in text
        # OpenSLR ships no CSV here, so the old hardcoded claim must not return.
        assert "OpenSLR" not in text

    def test_diversity_captions_gated_on_diversity_data(self, tmp_path, monkeypatch):
        benchmarks = tmp_path / "benchmarks"
        benchmarks.mkdir()
        (benchmarks / "public_diversity_stats_korean.json").write_text('{"unigram": 0.5}', encoding="utf-8")
        monkeypatch.setattr("psdn_sonar.reporting.loaders.benchmark_loader._BENCHMARKS_DIR", benchmarks)

        text = self._full_report(tmp_path, "korean")

        assert "public Korean benchmark curves" in text
        # Cross-dataset evaluations are still absent, so that claim stays gated.
        assert "the plots contain your dataset only" in text
