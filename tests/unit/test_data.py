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
from psdn_sonar.data.registry import FLEURS_CONFIG, MLS_CONFIG, resolve_config

FAKE_REVISION = "a" * 40


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
    def test_retired_common_voice_is_disabled(self):
        spec = DATASET_REGISTRY["common_voice"]
        assert spec.enabled is False
        assert resolve_config(spec, "bn") == "bn"

    def test_fleurs_mapping(self):
        spec = DATASET_REGISTRY["fleurs"]
        assert resolve_config(spec, "ko") == "ko_kr"
        assert resolve_config(spec, "zz") is None

    def test_no_config_language_returns_empty_string(self):
        spec = DATASET_REGISTRY["zeroth"]
        assert resolve_config(spec, "ko") == ""

    def test_mls_uses_full_config_names_and_real_schema(self):
        spec = DATASET_REGISTRY["multilingual_librispeech"]
        assert resolve_config(spec, "de") == MLS_CONFIG["de"] == "german"
        assert resolve_config(spec, "en") is None
        assert spec.splits == ("train", "dev", "test")
        assert spec.text_column == "transcript"

    def test_empty_template_without_no_config_lang_is_unsupported(self):
        spec = DatasetSpec(hf_id="x/y", config_template="")
        assert resolve_config(spec, "en") is None


class TestDiscovery:
    def test_korean_includes_zeroth_and_excludes_gated(self):
        names = {d.name for d in DatasetDiscovery.discover("ko")}
        assert {"fleurs", "zeroth"} <= names
        assert "common_voice" not in names
        assert "voxpopuli" not in names
        assert "multilingual_librispeech" not in names

    def test_english_excludes_mls_but_includes_voxpopuli(self):
        names = {d.name for d in DatasetDiscovery.discover("en")}
        assert "voxpopuli" in names
        assert "multilingual_librispeech" not in names

    def test_dataset_filter(self):
        results = DatasetDiscovery.discover("en", dataset_filter=["fleurs"])
        assert [d.name for d in results] == ["fleurs"]
        assert results[0].config == FLEURS_CONFIG["en"]

    def test_unknown_language_still_matches_templates(self):
        results = DatasetDiscovery.discover("xx")
        assert results == []

    def test_validate_remote_drops_missing_configs(self, monkeypatch):
        seen = []

        def missing(hf_id, config, revision):
            seen.append((hf_id, config, revision))
            return False

        monkeypatch.setattr("psdn_sonar.data.discovery._remote_config_exists", missing)
        assert DatasetDiscovery.discover("en", dataset_filter=["fleurs"], validate_remote=True) == []
        assert len(seen) == 1
        _, _, revision = seen[0]
        assert revision == DATASET_REGISTRY["fleurs"].revision

    def test_validate_remote_attaches_split_sizes(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.data.discovery._remote_config_exists", lambda hf_id, config, revision: True)
        monkeypatch.setattr(
            "psdn_sonar.data.discovery._get_split_sizes",
            lambda hf_id, config, splits, revision: {"train": 10},
        )
        results = DatasetDiscovery.discover("en", dataset_filter=["fleurs"], validate_remote=True)
        assert results[0].num_examples == {"train": 10}
        assert results[0].revision == DATASET_REGISTRY["fleurs"].revision

    def test_huggingface_metadata_calls_are_revision_pinned(self, monkeypatch):
        from psdn_sonar.data.discovery import _get_split_sizes, _remote_config_exists

        calls = []

        def config_names(hf_id, *, revision):
            calls.append(("configs", hf_id, revision))
            return ["en_us"]

        class SplitInfo:
            num_examples = 12

        class Builder:
            info = type("Info", (), {"splits": {"test": SplitInfo()}})()

        def load_builder(hf_id, config, *, revision):
            calls.append(("builder", hf_id, config, revision))
            return Builder()

        monkeypatch.setattr("datasets.get_dataset_config_names", config_names)
        monkeypatch.setattr("datasets.load_dataset_builder", load_builder)

        spec = DATASET_REGISTRY["fleurs"]
        assert _remote_config_exists(spec.hf_id, "en_us", spec.revision)
        assert _get_split_sizes(spec.hf_id, "en_us", ["test"], spec.revision) == {"test": 12}
        assert len(calls) == 2
        assert all(call[-1] == spec.revision for call in calls)

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

    def test_retired_dataset_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="disabled by the benchmark catalog"):
            prepare_dataset("common_voice", "en", "test", tmp_path)

    def test_writes_tsv_with_pinned_revision(self, tmp_path, monkeypatch):
        rows = [
            {"audio": {"path": "/data/a.wav"}, "transcription": "hello"},
            {"audio": "/data/b.wav", "transcription": "world"},
            {"audio": None, "transcription": "no audio"},
        ]
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeDataset(rows)

        monkeypatch.setattr("datasets.load_dataset", load_dataset)
        tsv = prepare_dataset("fleurs", "en", "test", tmp_path)
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "audio_path\ttranscription"
        assert lines[1] == "/data/a.wav\thello"
        assert lines[2] == "/data/b.wav\tworld"
        assert lines[3] == "row_2\tno audio"
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["revision"] == DATASET_REGISTRY["fleurs"].revision

    def test_max_samples_and_array_audio(self, tmp_path, monkeypatch):
        arr = np.zeros(1600, dtype=np.float32)
        rows = [
            {"audio": {"path": "", "array": arr, "sampling_rate": 16000}, "transcription": f"s{i}"} for i in range(5)
        ]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, **k: FakeDataset(rows))
        tsv = prepare_dataset("fleurs", "en", "test", tmp_path, max_samples=2)
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3  # header + 2 rows
        assert (tmp_path / "audio" / "audio_0.wav").exists()

    def test_no_config_load_is_also_revision_pinned(self, tmp_path, monkeypatch):
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeDataset([{"audio": None, "text": "안녕하세요"}])

        monkeypatch.setattr("datasets.load_dataset", load_dataset)
        prepare_dataset("zeroth", "ko", "test", tmp_path)

        spec = DATASET_REGISTRY["zeroth"]
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["revision"] == spec.revision


class TestDatasetPreparer:
    def _dataset(self, splits=("train", "validation", "test"), revision=FAKE_REVISION):
        return AvailableDataset(
            name="fake",
            hf_id="org/fake",
            config="en",
            revision=revision,
            splits=list(splits),
            text_column="sentence",
            audio_column="audio",
        )

    def _rows(self, n, prefix="r"):
        return [{"audio": {"path": f"/data/{prefix}{i}.wav"}, "sentence": f"text {prefix}{i}"} for i in range(n)]

    def test_predefined_splits_are_mapped(self, tmp_path, monkeypatch):
        splits = {"train": self._rows(3, "tr"), "validation": self._rows(2, "va"), "test": self._rows(1, "te")}
        calls = []

        def load_dataset(*args, split, revision):
            calls.append((args, split, revision))
            return FakeDataset(splits[split])

        monkeypatch.setattr("datasets.load_dataset", load_dataset)
        prep = DatasetPreparer(self._dataset(), "en", tmp_path, skip_audio_validation=True, seed=0)
        out = prep.prepare()

        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["split_sizes"] == {"train": 3, "val": 2, "test": 1}
        assert meta["source_revision"] == FAKE_REVISION
        assert calls == [
            (("org/fake", "en"), "train", FAKE_REVISION),
            (("org/fake", "en"), "validation", FAKE_REVISION),
            (("org/fake", "en"), "test", FAKE_REVISION),
        ]
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

    def test_missing_revision_is_rejected_before_download(self, tmp_path):
        with pytest.raises(ValueError, match="immutable source revision"):
            DatasetPreparer(self._dataset(revision=""), "en", tmp_path, skip_audio_validation=True, seed=0)

    @pytest.mark.parametrize("revision", ["main", "latest", "v1.0", "a" * 39])
    def test_floating_or_short_revision_is_rejected(self, revision, tmp_path):
        with pytest.raises(ValueError, match="immutable source revision"):
            DatasetPreparer(self._dataset(revision=revision), "en", tmp_path, skip_audio_validation=True, seed=0)
