"""Tests for the scripts/precompute_benchmarks.py helpers."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("precompute_benchmarks", SCRIPTS_DIR / "precompute_benchmarks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestParseTsvSpecs:
    def test_parses_name_path_pairs(self, script):
        result = script.parse_tsv_specs(["commonvoice=data/cv.tsv", "fleurs=data/fleurs.tsv"])

        assert result == {"commonvoice": Path("data/cv.tsv"), "fleurs": Path("data/fleurs.tsv")}

    def test_rejects_missing_separator(self, script):
        with pytest.raises(ValueError, match="NAME=PATH"):
            script.parse_tsv_specs(["justapath.tsv"])

    def test_rejects_empty_name(self, script):
        with pytest.raises(ValueError, match="NAME=PATH"):
            script.parse_tsv_specs(["=data/cv.tsv"])


class TestLoadTranscripts:
    def test_reads_transcription_column(self, script, tmp_path):
        tsv = tmp_path / "data.tsv"
        pd.DataFrame({"audio_path": ["a.wav", "b.wav"], "transcription": ["hello there", "  "]}).to_csv(
            tsv, sep="\t", index=False
        )

        assert script.load_transcripts(tsv) == ["hello there"]

    def test_falls_back_to_sentence_column(self, script, tmp_path):
        tsv = tmp_path / "data.tsv"
        pd.DataFrame({"sentence": ["one", "two"]}).to_csv(tsv, sep="\t", index=False)

        assert script.load_transcripts(tsv) == ["one", "two"]

    def test_no_transcript_column_returns_empty(self, script, tmp_path):
        tsv = tmp_path / "data.tsv"
        pd.DataFrame({"audio_path": ["a.wav"]}).to_csv(tsv, sep="\t", index=False)

        assert script.load_transcripts(tsv) == []

    def test_large_corpus_sampled_deterministically(self, script, tmp_path):
        tsv = tmp_path / "data.tsv"
        pd.DataFrame({"transcription": [f"line {i}" for i in range(6000)]}).to_csv(tsv, sep="\t", index=False)

        first = script.load_transcripts(tsv)
        second = script.load_transcripts(tsv)

        assert len(first) == script.MAX_TRANSCRIPTS_FOR_STATS
        assert first == second


class TestComputeLexicalStats:
    def test_writes_all_jsons(self, script, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "BENCHMARKS_ROOT", tmp_path)
        tsv = tmp_path / "cv.tsv"
        pd.DataFrame({"transcription": ["the cat sat", "the dog ran"]}).to_csv(tsv, sep="\t", index=False)

        script.compute_lexical_stats({"common_voice": tsv}, "english")

        lexical = json.loads((tmp_path / "public_lexical_data_english.json").read_text())
        diversity = json.loads((tmp_path / "public_diversity_stats_english.json").read_text())
        lengths = json.loads((tmp_path / "public_length_stats_english.json").read_text())

        assert lexical["Common Voice"]["total_transcripts"] == 2
        assert lexical["Common Voice"]["vocabulary_growth"][-1]["vocab_size"] == 5
        assert 0.0 < diversity["Common Voice"]["unigram_diversity"] <= 1.0
        assert diversity["Common Voice"]["ttr"] == diversity["Common Voice"]["unigram_diversity"]
        assert lengths["Common Voice"]["words_median"] == 3.0
        assert lengths["Common Voice"]["pct_5_words_or_fewer"] == 1.0
        assert "comparable within a dataset" in lengths["_note"]

    def test_skips_datasets_without_transcripts(self, script, tmp_path, monkeypatch):
        monkeypatch.setattr(script, "BENCHMARKS_ROOT", tmp_path)
        tsv = tmp_path / "bad.tsv"
        pd.DataFrame({"audio_path": ["a.wav"]}).to_csv(tsv, sep="\t", index=False)

        script.compute_lexical_stats({"bad": tsv}, "english")

        lexical = json.loads((tmp_path / "public_lexical_data_english.json").read_text())
        assert lexical == {}


class TestWriteDomainMarkers:
    def test_marks_declared_training_overlap(self, script, tmp_path):
        raw_eval = tmp_path / "raw-evaluations"
        out = script.write_domain_markers(
            ["kresnik_wav2vec2_large_xlsr_korean", "whisper_api"],
            {"zeroth": Path("z.tsv"), "fleurs": Path("f.tsv")},
            raw_eval,
        )

        payload = json.loads(out.read_text())
        markers = payload["markers"]
        assert markers["kresnik_wav2vec2_large_xlsr_korean"]["zeroth"] == "in-domain"
        assert markers["kresnik_wav2vec2_large_xlsr_korean"]["fleurs"] == "not-declared"
        assert markers["whisper_api"]["zeroth"] == "unknown"
        assert markers["whisper_api"]["fleurs"] == "unknown"
        assert "in-domain" in payload["note"]

    def test_covers_every_model_dataset_pair(self, script, tmp_path):
        out = script.write_domain_markers(
            ["khushids_bengali", "wav2vec2_bengali"],
            {"fleurs": Path("f.tsv"), "openslr53": Path("o.tsv")},
            tmp_path,
        )

        markers = json.loads(out.read_text())["markers"]
        assert markers["khushids_bengali"]["fleurs"] == "in-domain"
        assert markers["wav2vec2_bengali"]["openslr53"] == "in-domain"
        assert markers["khushids_bengali"]["openslr53"] == "not-declared"
        assert markers["wav2vec2_bengali"]["fleurs"] == "not-declared"


class TestEvaluateDataset:
    def test_existing_csv_skipped(self, script, tmp_path):
        raw_eval = tmp_path / "raw-evaluations"
        existing = raw_eval / "whisper_api" / "commonvoice.csv"
        existing.parent.mkdir(parents=True)
        existing.write_text("already here")

        assert script.evaluate_dataset(tmp_path / "cv.tsv", "commonvoice", "whisper_api", raw_eval) is True
        assert existing.read_text() == "already here"

    def test_result_csv_moved_into_place(self, script, tmp_path, monkeypatch):
        def fake_run_evaluation(**kwargs):
            out = Path(kwargs["output_dir"]) / "asr_detailed_whisper_api.csv"
            out.write_text("cer,wer\n0.1,0.2\n")

        import psdn_sonar.evaluators.single_speaker as ss

        monkeypatch.setattr(ss.SingleSpeakerEvaluator, "run_evaluation", staticmethod(fake_run_evaluation))
        raw_eval = tmp_path / "raw-evaluations"

        ok = script.evaluate_dataset(tmp_path / "cv.tsv", "commonvoice", "whisper_api", raw_eval, language="en")

        assert ok is True
        assert (raw_eval / "whisper_api" / "commonvoice.csv").read_text() == "cer,wer\n0.1,0.2\n"

    def test_failed_evaluation_returns_false(self, script, tmp_path, monkeypatch):
        def fake_run_evaluation(**kwargs):
            raise RuntimeError("model exploded")

        import psdn_sonar.evaluators.single_speaker as ss

        monkeypatch.setattr(ss.SingleSpeakerEvaluator, "run_evaluation", staticmethod(fake_run_evaluation))

        ok = script.evaluate_dataset(tmp_path / "cv.tsv", "commonvoice", "whisper_api", tmp_path / "raw")

        assert ok is False
