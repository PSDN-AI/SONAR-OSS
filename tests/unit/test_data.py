"""Tests for dataset discovery, registry, and preparation."""

import csv
import json

import numpy as np
import pytest

from psdn_sonar.data import (
    DATASET_REGISTRY,
    AvailableDataset,
    DatasetDiscovery,
    DatasetPreparer,
    DatasetSpec,
    prepare_dataset,
)
from psdn_sonar.data.registry import FLEURS_CONFIG, resolve_config


class FakeDataset:
    """Minimal stand-in for a HuggingFace dataset split."""

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, i):
        return self._rows[i]

    def select(self, indices):
        return FakeDataset([self._rows[i] for i in indices])


class TestResolveConfig:
    def test_lang_template(self):
        spec = DATASET_REGISTRY["common_voice"]
        assert resolve_config(spec, "bn") == "bn"

    def test_fleurs_mapping(self):
        spec = DATASET_REGISTRY["fleurs"]
        assert resolve_config(spec, "ko") == "ko_kr"
        assert resolve_config(spec, "zz") is None

    def test_no_config_language_returns_empty_string(self):
        spec = DATASET_REGISTRY["zeroth"]
        assert resolve_config(spec, "ko") == ""

    def test_empty_template_without_no_config_lang_is_unsupported(self):
        spec = DatasetSpec(hf_id="x/y", config_template="")
        assert resolve_config(spec, "en") is None


class TestDiscovery:
    def test_korean_includes_zeroth_and_excludes_gated(self):
        names = {d.name for d in DatasetDiscovery.discover("ko")}
        assert {"common_voice", "fleurs", "zeroth"} <= names
        assert "voxpopuli" not in names
        assert "multilingual_librispeech" not in names

    def test_english_includes_gated_datasets(self):
        names = {d.name for d in DatasetDiscovery.discover("en")}
        assert {"voxpopuli", "multilingual_librispeech"} <= names

    def test_dataset_filter(self):
        results = DatasetDiscovery.discover("en", dataset_filter=["fleurs"])
        assert [d.name for d in results] == ["fleurs"]
        assert results[0].config == FLEURS_CONFIG["en"]

    def test_unknown_language_still_matches_templates(self):
        results = DatasetDiscovery.discover("xx")
        names = {d.name for d in results}
        assert "common_voice" in names  # template "{lang}" always resolves
        assert "fleurs" not in names  # no FLEURS mapping for "xx"

    def test_validate_remote_drops_missing_configs(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.data.discovery._remote_config_exists", lambda hf_id, config: False)
        assert DatasetDiscovery.discover("en", dataset_filter=["fleurs"], validate_remote=True) == []

    def test_validate_remote_attaches_split_sizes(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.data.discovery._remote_config_exists", lambda hf_id, config: True)
        monkeypatch.setattr(
            "psdn_sonar.data.discovery._get_split_sizes",
            lambda hf_id, config, splits: {"train": 10},
        )
        results = DatasetDiscovery.discover("en", dataset_filter=["fleurs"], validate_remote=True)
        assert results[0].num_examples == {"train": 10}

    def test_print_summary_smoke(self, capsys):
        DatasetDiscovery.print_summary(DatasetDiscovery.discover("ko"), "ko")
        out = capsys.readouterr().out
        assert "zeroth" in out
        DatasetDiscovery.print_summary([], "ko")
        assert "(none found)" in capsys.readouterr().out


class TestPrepareDataset:
    def test_unknown_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            prepare_dataset("nope", "en", "test", tmp_path)

    def test_unsupported_language(self, tmp_path):
        with pytest.raises(ValueError, match="does not support language"):
            prepare_dataset("fleurs", "zz", "test", tmp_path)

    def test_unknown_split(self, tmp_path):
        with pytest.raises(ValueError, match="has splits"):
            prepare_dataset("zeroth", "ko", "validation", tmp_path)

    def test_writes_tsv(self, tmp_path, monkeypatch):
        rows = [
            {"audio": {"path": "/data/a.wav"}, "sentence": "hello"},
            {"audio": "/data/b.wav", "sentence": "world"},
            {"audio": None, "sentence": "no audio"},
        ]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: FakeDataset(rows))
        tsv = prepare_dataset("common_voice", "en", "test", tmp_path)
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "audio_path\ttranscription"
        assert lines[1] == "/data/a.wav\thello"
        assert lines[2] == "/data/b.wav\tworld"
        assert lines[3] == "row_2\tno audio"

    def test_max_samples_and_array_audio(self, tmp_path, monkeypatch):
        arr = np.zeros(1600, dtype=np.float32)
        rows = [{"audio": {"path": "", "array": arr, "sampling_rate": 16000}, "sentence": f"s{i}"} for i in range(5)]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: FakeDataset(rows))
        tsv = prepare_dataset("common_voice", "en", "test", tmp_path, max_samples=2)
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # header + 2 rows
        assert (tmp_path / "audio" / "audio_0.wav").exists()


class TestDatasetPreparer:
    def _dataset(self, splits=("train", "validation", "test")):
        return AvailableDataset(
            name="fake",
            hf_id="org/fake",
            config="en",
            splits=list(splits),
            text_column="sentence",
            audio_column="audio",
        )

    def _rows(self, n, prefix="r"):
        return [{"audio": {"path": f"/data/{prefix}{i}.wav"}, "sentence": f"text {prefix}{i}"} for i in range(n)]

    def test_predefined_splits_are_mapped(self, tmp_path, monkeypatch):
        splits = {"train": self._rows(3, "tr"), "validation": self._rows(2, "va"), "test": self._rows(1, "te")}
        monkeypatch.setattr("datasets.load_dataset", lambda *a, split, **k: FakeDataset(splits[split]))
        prep = DatasetPreparer(self._dataset(), "en", tmp_path, skip_audio_validation=True, seed=0)
        out = prep.prepare()

        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["split_sizes"] == {"train": 3, "val": 2, "test": 1}
        with open(out / "val.tsv", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert [r["audio_path"] for r in rows] == ["/data/va0.wav", "/data/va1.wav"]
        assert rows[0]["transcription"] == "text va0"
        assert rows[0]["transcription_norm"]  # normalization ran

    def test_single_split_is_ratio_split_deterministically(self, tmp_path, monkeypatch):
        monkeypatch.setattr("datasets.load_dataset", lambda *a, split, **k: FakeDataset(self._rows(10)))
        kwargs = dict(language="en", max_samples=0, skip_audio_validation=True, seed=42)
        out1 = DatasetPreparer(self._dataset(splits=("train",)), output_dir=tmp_path / "a", **kwargs).prepare()
        out2 = DatasetPreparer(self._dataset(splits=("train",)), output_dir=tmp_path / "b", **kwargs).prepare()

        meta = json.loads((out1 / "metadata.json").read_text(encoding="utf-8"))
        assert meta["split_sizes"] == {"train": 8, "val": 1, "test": 1}
        assert (out1 / "train.tsv").read_text(encoding="utf-8") == (out2 / "train.tsv").read_text(encoding="utf-8")

    def test_array_audio_written_to_wav_with_duration(self, tmp_path, monkeypatch):
        arr = np.zeros(8000, dtype=np.float32)
        rows = [{"audio": {"array": arr, "sampling_rate": 16000}, "sentence": "hello"}]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, split, **k: FakeDataset(rows))
        out = DatasetPreparer(
            self._dataset(splits=("train",)), "en", tmp_path, skip_audio_validation=True, seed=0
        ).prepare()

        # With a single record, the 80/10/10 ratio split places it in test.
        with open(out / "test.tsv", encoding="utf-8") as f:
            row = next(csv.DictReader(f, delimiter="\t"))
        assert row["audio_path"].endswith("train_000000.wav")
        assert row["duration_s"] == "0.500"

    def test_blank_text_rows_are_dropped(self, tmp_path, monkeypatch):
        rows = self._rows(2) + [{"audio": {"path": "/data/x.wav"}, "sentence": "   "}]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, split, **k: FakeDataset(rows))
        out = DatasetPreparer(
            self._dataset(splits=("train",)), "en", tmp_path, skip_audio_validation=True, seed=0
        ).prepare()
        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert sum(meta["split_sizes"].values()) == 2

    def test_all_splits_failing_raises(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            raise OSError("offline")

        monkeypatch.setattr("datasets.load_dataset", boom)
        with pytest.raises(RuntimeError, match="No samples"):
            DatasetPreparer(self._dataset(), "en", tmp_path, skip_audio_validation=True, seed=0).prepare()
