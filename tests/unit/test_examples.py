"""Tests that shipped examples import cleanly and their helpers work."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def load_example(name: str):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name",
    [
        "custom_dataset",
        "multispeaker_audio_dataset",
        "single_speaker_audio_dataset",
        "huggingface_dataset_loader",
        "huggingface_complete_workflow",
        "korean_language_smoke",
        "demographic_analysis",
        "visualization",
        "download_from_cloud",
    ],
)
def test_examples_import_cleanly(name):
    load_example(name)


class TestHuggingFaceDatasetLoader:
    @pytest.fixture
    def loader(self):
        return load_example("huggingface_dataset_loader")

    def test_resolve_column_prefers_requested(self, loader):
        dataset = type("D", (), {"column_names": ["audio", "sentence"]})()
        assert loader._resolve_column(dataset, "audio", loader._AUDIO_COLUMN_CANDIDATES, "audio") == "audio"

    def test_resolve_column_falls_back_to_candidate(self, loader):
        dataset = type("D", (), {"column_names": ["file", "sentence"]})()
        assert loader._resolve_column(dataset, "audio", loader._AUDIO_COLUMN_CANDIDATES, "audio") == "file"
        assert loader._resolve_column(dataset, "transcription", loader._TEXT_COLUMN_CANDIDATES, "text") == "sentence"

    def test_resolve_column_exits_when_unresolvable(self, loader):
        dataset = type("D", (), {"column_names": ["unrelated"]})()
        with pytest.raises(SystemExit):
            loader._resolve_column(dataset, "audio", loader._AUDIO_COLUMN_CANDIDATES, "audio")


class TestCustomDatasetHelpers:
    def test_validate_accepts_bundled_sample(self):
        custom_dataset = load_example("custom_dataset")
        assert custom_dataset.validate_dataset_format(EXAMPLES_DIR / "test_data.tsv") is True

    def test_validate_rejects_missing_columns(self, tmp_path):
        custom_dataset = load_example("custom_dataset")
        bad = tmp_path / "bad.tsv"
        bad.write_text("path\ttext\na.wav\thello\n")
        with pytest.raises(ValueError, match="Missing required columns"):
            custom_dataset.validate_dataset_format(bad)

    def test_prepare_renames_columns(self, tmp_path):
        custom_dataset = load_example("custom_dataset")
        source = tmp_path / "source.csv"
        source.write_text("clip,text\na.wav,hello\nb.wav,world\n")

        out = custom_dataset.prepare_custom_dataset(
            source, tmp_path / "prepared.tsv", audio_column="clip", text_column="text"
        )

        df = pd.read_csv(out, sep="\t")
        assert list(df.columns) == ["audio_path", "transcription"]
        assert df["audio_path"].tolist() == ["a.wav", "b.wav"]


class TestSampleData:
    def test_sample_tsv_has_required_columns(self):
        df = pd.read_csv(EXAMPLES_DIR / "test_data.tsv", sep="\t")
        assert {"audio_path", "transcription"}.issubset(df.columns)
        assert len(df) > 0

    def test_sample_manifest_entries_are_valid(self):
        from psdn_sonar.loaders.manifest import get_clip_files, load_manifest

        manifest_path = EXAMPLES_DIR / "test_manifest.jsonl"
        lines = manifest_path.read_text().strip().splitlines()
        assert lines
        for line in lines:
            entry = json.loads(line)
            assert {"audio_id", "audio_filepaths", "transcript_filepath", "num_speakers"}.issubset(entry)
            assert "speaker_a" in entry["audio_filepaths"]
            assert "speaker_b" in entry["audio_filepaths"]

        loaded = load_manifest(str(manifest_path))
        assert loaded
        audio_a, audio_b, transcript = get_clip_files(loaded[0])
        assert audio_a is not None and audio_a.is_file()
        assert audio_b is not None and audio_b.is_file()
        assert transcript.is_file()

    def test_portuguese_config_parses(self):
        from psdn_sonar.custom_eval import CustomEvalConfig

        config = CustomEvalConfig(str(EXAMPLES_DIR / "custom_eval_portuguese.yaml"))
        assert config.language_code == "pt"
        assert config.hf_dataset_id == "google/fleurs"
        assert config.models[0]["hf_model_id"] == "openai/whisper-small"
