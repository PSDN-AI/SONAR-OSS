"""Tests for dataset discovery, registry, and preparation."""

import csv
import json
from types import SimpleNamespace

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
    def test_catalog_only_datasets_are_not_in_hf_registry(self):
        assert {"common_voice", "multilingual_librispeech"}.isdisjoint(DATASET_REGISTRY)

    def test_fleurs_mapping(self):
        spec = DATASET_REGISTRY["fleurs"]
        assert resolve_config(spec, "ko") == "ko_kr"
        assert resolve_config(spec, "zz") is None

    def test_no_config_language_returns_empty_string(self):
        spec = DATASET_REGISTRY["zeroth"]
        assert resolve_config(spec, "ko") == ""

    def test_empty_template_without_no_config_lang_is_unsupported(self):
        spec = DatasetSpec(hf_id="x/y", config_template="", revision=FAKE_REVISION)
        assert resolve_config(spec, "en") is None


class TestDiscovery:
    def test_korean_includes_zeroth_and_excludes_gated(self):
        names = {d.name for d in DatasetDiscovery.discover("ko")}
        assert {"fleurs", "zeroth"} <= names
        assert "common_voice" not in names
        assert "voxpopuli" not in names

    def test_english_includes_voxpopuli(self):
        names = {d.name for d in DatasetDiscovery.discover("en")}
        assert "voxpopuli" in names

    def test_dataset_filter(self):
        results = DatasetDiscovery.discover("en", dataset_filter=["fleurs"])
        assert [d.name for d in results] == ["fleurs"]
        assert results[0].config == FLEURS_CONFIG["en"]

    def test_unknown_language_still_matches_templates(self):
        results = DatasetDiscovery.discover("xx")
        assert results == []

    def test_validate_remote_drops_missing_configs(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.data.discovery._remote_config_exists", lambda hf_id, config, revision: False)
        assert DatasetDiscovery.discover("en", dataset_filter=["fleurs"], validate_remote=True) == []

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
        monkeypatch.setattr(
            "datasets.get_dataset_config_names",
            lambda hf_id, *, revision: calls.append((hf_id, revision)) or ["en_us"],
        )
        builder = SimpleNamespace(info=SimpleNamespace(splits={"test": SimpleNamespace(num_examples=12)}))
        monkeypatch.setattr(
            "datasets.load_dataset_builder",
            lambda hf_id, config, *, revision: calls.append((hf_id, revision)) or builder,
        )

        spec = DATASET_REGISTRY["fleurs"]
        assert _remote_config_exists(spec.hf_id, "en_us", spec.revision)
        assert _get_split_sizes(spec.hf_id, "en_us", ["test"], spec.revision) == {"test": 12}
        assert calls == [(spec.hf_id, spec.revision)] * 2

    def test_print_summary_smoke(self, capsys):
        DatasetDiscovery.print_summary(DatasetDiscovery.discover("ko"), "ko")
        out = capsys.readouterr().out
        assert "zeroth" in out
        DatasetDiscovery.print_summary([], "ko")
        assert "(none found)" in capsys.readouterr().out

    def test_print_summary_names_non_discoverable_catalog_entries(self, capsys):
        DatasetDiscovery.print_summary(DatasetDiscovery.discover("bn"), "bn")
        out = capsys.readouterr().out
        for name in ("openslr37_bd", "openslr37_in", "openslr53", "common_voice", "multilingual_librispeech"):
            assert name in out
        assert "library loaders" in out
        assert "disabled in the catalog" in out

    def test_print_summary_shows_license_and_review_state(self, capsys):
        """Issue #116: enabled-but-pending entries are downloadable, so the
        point of download must show the upstream license and the fact that
        redistribution review is not closed."""
        DatasetDiscovery.print_summary(DatasetDiscovery.discover("en"), "en")
        out = capsys.readouterr().out
        assert "user-initiated acquisition" in out
        assert "redistributes no dataset" in out
        assert "license: CC-BY-4.0" in out  # fleurs
        assert "redistribution review: pending" in out
        assert "docs/import-gate.md" in out

    def test_print_summary_no_rights_note_when_nothing_found(self, capsys):
        DatasetDiscovery.print_summary([], "ko")
        out = capsys.readouterr().out
        assert "user-initiated acquisition" not in out


class TestDatasetFilterValidation:
    def test_unknown_name_raises_and_lists_discoverable(self):
        with pytest.raises(ValueError, match="unknown dataset name") as exc_info:
            DatasetDiscovery.discover("en", dataset_filter=["definitely-not-a-dataset"])
        assert "fleurs" in str(exc_info.value)

    def test_disabled_benchmark_named_as_disabled(self):
        with pytest.raises(ValueError, match="catalogued but disabled"):
            DatasetDiscovery.discover("en", dataset_filter=["common_voice"])

    def test_non_hf_source_named_as_out_of_scope(self):
        with pytest.raises(ValueError, match="openslr source"):
            DatasetDiscovery.discover("bn", dataset_filter=["openslr37_bd"])

    def test_unwired_hf_benchmark_distinguished(self):
        with pytest.raises(ValueError, match="not wired into"):
            DatasetDiscovery.discover("de", dataset_filter=["multilingual_librispeech"])

    def test_mixed_valid_and_invalid_fails_fast(self):
        with pytest.raises(ValueError, match="commonvoice"):
            DatasetDiscovery.discover("en", dataset_filter=["fleurs", "commonvoice"])

    def test_valid_filter_wrong_language_returns_empty_without_raising(self):
        assert DatasetDiscovery.discover("en", dataset_filter=["zeroth"]) == []

    def test_language_support_hints(self):
        from psdn_sonar.data.discovery import dataset_language_support

        assert dataset_language_support("zeroth") == "zeroth supports: ko"
        assert "en" in dataset_language_support("fleurs")
        assert "bn" in dataset_language_support("fleurs")


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

    @pytest.mark.parametrize(
        ("dataset", "language", "text_column"),
        [("fleurs", "en", "transcription"), ("zeroth", "ko", "text")],
    )
    def test_writes_tsv_with_pinned_revision(self, dataset, language, text_column, tmp_path, monkeypatch):
        rows = [{"audio": {"path": "/data/a.wav"}, text_column: "hello"}]
        calls = []

        def load_dataset(*args, **kwargs):
            calls.append((args, kwargs))
            return FakeDataset(rows)

        monkeypatch.setattr("datasets.load_dataset", load_dataset)
        tsv = prepare_dataset(dataset, language, "test", tmp_path)
        lines = tsv.read_text(encoding="utf-8").splitlines()
        assert lines == ["audio_path\ttranscription", "/data/a.wav\thello"]
        assert calls[0][1]["revision"] == DATASET_REGISTRY[dataset].revision

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
        revisions = []

        def load_dataset(*args, split, revision):
            revisions.append(revision)
            return FakeDataset(splits[split])

        monkeypatch.setattr("datasets.load_dataset", load_dataset)
        prep = DatasetPreparer(self._dataset(), "en", tmp_path, skip_audio_validation=True, seed=0)
        out = prep.prepare()

        meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert meta["split_sizes"] == {"train": 3, "val": 2, "test": 1}
        assert meta["source_revision"] == FAKE_REVISION
        assert revisions == [FAKE_REVISION] * 3
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

    @pytest.mark.parametrize("revision", ["", "main", "latest", "v1.0", "a" * 39])
    def test_floating_or_short_revision_is_rejected(self, revision, tmp_path):
        with pytest.raises(ValueError, match="immutable source revision"):
            DatasetPreparer(self._dataset(revision=revision), "en", tmp_path, skip_audio_validation=True, seed=0)


class TestDiskFullAbort:
    """Issue #183: a full disk must abort the run with an actionable OSError,
    not be downgraded to a per-split warning and buried under an unchained
    'No samples could be loaded' RuntimeError."""

    def _dataset(self, splits=("train", "validation", "test")):
        return AvailableDataset(
            name="fake",
            hf_id="org/fake",
            config="en",
            revision=FAKE_REVISION,
            splits=list(splits),
            text_column="sentence",
            audio_column="audio",
        )

    def _preparer(self, tmp_path, splits=("train", "validation", "test")):
        return DatasetPreparer(self._dataset(splits), "en", tmp_path, skip_audio_validation=True, seed=0)

    def _assert_actionable(self, exc: OSError):
        import errno

        assert exc.errno == errno.ENOSPC
        message = str(exc)
        assert "org/fake" in message
        assert "HuggingFace cache" in message
        assert "--max-samples" in message

    def test_wrapped_enospc_aborts_instead_of_trying_remaining_splits(self, tmp_path, monkeypatch):
        calls = []

        def boom(*a, split, **k):
            calls.append(split)
            enospc = OSError(28, "No space left on device")
            raise RuntimeError("An error occurred while generating the dataset") from enospc

        monkeypatch.setattr("datasets.load_dataset", boom)
        with pytest.raises(OSError) as exc_info:
            self._preparer(tmp_path).prepare()

        self._assert_actionable(exc_info.value)
        # Continuing cannot succeed: the remaining splits must not be attempted.
        assert calls == ["train"]

    def test_rust_style_enospc_text_without_errno_is_recognized(self, tmp_path, monkeypatch):
        def boom(*a, **k):
            # hf_transfer/xet render ENOSPC without a Python errno.
            raise RuntimeError("No space left on device (os error 28)")

        monkeypatch.setattr("datasets.load_dataset", boom)
        with pytest.raises(OSError) as exc_info:
            self._preparer(tmp_path).prepare()
        self._assert_actionable(exc_info.value)

    def test_hub_disk_space_warning_stops_the_run_before_the_download(self, tmp_path, monkeypatch):
        import warnings

        calls = []

        def warn_then_return(*a, split, **k):
            calls.append(split)
            # huggingface_hub._check_disk_space warns and carries on downloading.
            warnings.warn(
                "Not enough free disk space to download the file. "
                "The expected file size is: 3254.87 MB. "
                "The target location /home/x/.cache only has 3208.73 MB free disk space."
            )
            return FakeDataset([{"audio": {"path": "/data/a.wav"}, "sentence": "hi"}])

        monkeypatch.setattr("datasets.load_dataset", warn_then_return)
        with pytest.raises(OSError) as exc_info:
            self._preparer(tmp_path).prepare()

        self._assert_actionable(exc_info.value)
        assert calls == ["train"]

    def test_enospc_while_writing_audio_gets_the_same_treatment(self, tmp_path, monkeypatch):
        arr = np.zeros(8000, dtype=np.float32)
        rows = [{"audio": {"array": arr, "sampling_rate": 16000}, "sentence": "hello"}]
        monkeypatch.setattr("datasets.load_dataset", lambda *a, split, **k: FakeDataset(rows))

        def full_disk(*a, **k):
            raise OSError(28, "No space left on device")

        monkeypatch.setattr("soundfile.write", full_disk)
        with pytest.raises(OSError) as exc_info:
            self._preparer(tmp_path, splits=("train",)).prepare()
        self._assert_actionable(exc_info.value)

    def test_benign_failures_still_continue_and_the_final_error_names_the_cause(self, tmp_path, monkeypatch):
        calls = []

        def boom(*a, split, **k):
            calls.append(split)
            raise ValueError(f"bad config for {split}")

        monkeypatch.setattr("datasets.load_dataset", boom)
        with pytest.raises(RuntimeError, match="No samples could be loaded") as exc_info:
            self._preparer(tmp_path).prepare()

        # All splits were still attempted, and the RuntimeError is no longer
        # unchained: it names and carries the last real failure.
        assert calls == ["train", "validation", "test"]
        assert "bad config for test" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)
