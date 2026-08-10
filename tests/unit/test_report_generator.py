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
        assert "public Bengali benchmarks" in text
        assert "## Key Insights" in text
        # No plot directories exist, so plot sections are absent.
        assert "Cross-Dataset Comparison" not in text
        assert "## Audio Quality Analysis" not in text

    def test_unknown_language_setup(self, tmp_path):
        tsv = self._dataset(tmp_path)
        output = tmp_path / "EVAL_REPORT.md"

        generate_report("DS", tsv, output, language="swahili")

        text = output.read_text(encoding="utf-8")
        assert "Swahili (custom evaluation)" in text
        assert "Not available for this language" in text

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
        assert "### Cross-Dataset Comparison" in text
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
        assert "public Korean benchmarks" in text

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
