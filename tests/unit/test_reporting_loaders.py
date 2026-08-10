"""Tests for the reporting input loaders."""

import json

import pytest

from psdn_sonar.reporting.loaders.benchmark_loader import (
    load_public_benchmark_diversity,
    load_public_lexical_data,
)
from psdn_sonar.reporting.loaders.transcript_loader import (
    load_transcripts_from_file,
    load_transcripts_from_jsonl,
    load_transcripts_from_tsv,
)


class TestLoadTranscriptsFromTsv:
    def test_sentence_header(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("path\tsentence\na.wav\thello there\nb.wav\tworld\n")
        assert load_transcripts_from_tsv(p) == ["hello there", "world"]

    def test_transcription_header(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("audio_path\ttranscription\na.wav\thello\n")
        assert load_transcripts_from_tsv(p) == ["hello"]

    def test_headerless_two_column(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("clip_001\thello there\nclip_002\tanother line\n")
        assert load_transcripts_from_tsv(p) == ["hello there", "another line"]

    def test_headerless_three_column_falls_back(self, tmp_path):
        # Second column empty everywhere, so the two-column pass yields nothing.
        p = tmp_path / "t.tsv"
        p.write_text("utt_1\t\tthird col text\nutt_2\t\tmore text\n")
        assert load_transcripts_from_tsv(p) == ["third col text", "more text"]

    def test_unparseable_raises(self, tmp_path):
        p = tmp_path / "t.tsv"
        p.write_text("no tabs here\n")
        with pytest.raises(ValueError, match="Could not parse TSV"):
            load_transcripts_from_tsv(p)


class TestLoadTranscriptsFromJsonl:
    def test_reads_referenced_files_and_skips_missing(self, tmp_path):
        (tmp_path / "t1.txt").write_text("first transcript")
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(
            json.dumps({"transcript_path": "t1.txt"}) + "\n" + json.dumps({"transcript_path": "missing.txt"}) + "\n"
        )
        assert load_transcripts_from_jsonl(manifest) == ["first transcript"]

    def test_explicit_dataset_dir(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "t.txt").write_text("hello")
        manifest = tmp_path / "m.jsonl"
        manifest.write_text(json.dumps({"transcript_path": "t.txt"}) + "\n")
        assert load_transcripts_from_jsonl(manifest, dataset_dir=str(data_dir)) == ["hello"]


class TestLoadTranscriptsFromFile:
    def test_dispatches_by_extension(self, tmp_path):
        tsv = tmp_path / "t.tsv"
        tsv.write_text("path\tsentence\na.wav\thi\n")
        assert load_transcripts_from_file(str(tsv)) == ["hi"]

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text("x")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_transcripts_from_file(str(p))


class TestBenchmarkLoader:
    def test_missing_files_return_empty_dict(self):
        # No benchmarks/ data ships yet; loaders must degrade to {}.
        assert load_public_benchmark_diversity("korean") == {}
        assert load_public_lexical_data("korean") == {}

    def test_language_specific_file_loaded(self, monkeypatch, tmp_path):
        (tmp_path / "public_diversity_stats_korean.json").write_text('{"unigram": 0.5}')
        monkeypatch.setattr("psdn_sonar.reporting.loaders.benchmark_loader._BENCHMARKS_DIR", tmp_path)
        assert load_public_benchmark_diversity("korean") == {"unigram": 0.5}

    def test_legacy_fallback_only_for_bengali(self, monkeypatch, tmp_path):
        (tmp_path / "public_lexical_data.json").write_text('{"legacy": true}')
        monkeypatch.setattr("psdn_sonar.reporting.loaders.benchmark_loader._BENCHMARKS_DIR", tmp_path)
        assert load_public_lexical_data("bengali") == {"legacy": True}
        assert load_public_lexical_data("hindi") == {}
