"""Tests for the dataset loaders package.

All dataset layouts are synthesised under ``tmp_path`` — no fixture files
and no network access.
"""

import pytest

from psdn_sonar.loaders import (
    CommonVoiceLoader,
    DatasetLoader,
    FleursLoader,
    OpenSLR37BDLoader,
    OpenSLR53Loader,
    OpenSLRLineIndexLoader,
    resolve_dataset_dir,
)

ALL_LOADERS = [CommonVoiceLoader, FleursLoader, OpenSLR37BDLoader, OpenSLR53Loader, OpenSLRLineIndexLoader]

# The evaluation output row contract: identical across datasets except for
# the leading ID column.
EXPECTED_METRIC_COLUMNS = [
    "path",
    "transcription",
    "transcription_norm",
    "asr_transcription",
    "asr_transcription_norm_non",
    "asr_transcription_norm_conv",
    "cer_non",
    "wer_non",
    "semantic_similarity_non",
    "poseidon_score_non",
    "cer_conv",
    "wer_conv",
    "semantic_similarity_conv",
    "poseidon_score_conv",
    "inference_latency_s",
]


class TestOutputRowContract:
    @pytest.mark.parametrize("loader", ALL_LOADERS)
    def test_conforms_to_protocol(self, loader):
        assert isinstance(loader, DatasetLoader)

    @pytest.mark.parametrize(
        ("loader", "id_field"),
        [
            (CommonVoiceLoader, "client_id"),
            (FleursLoader, "id"),
            (OpenSLR37BDLoader, "file_id"),
            (OpenSLR53Loader, "file_id"),
            (OpenSLRLineIndexLoader, "file_id"),
        ],
    )
    def test_fieldnames_shape(self, loader, id_field):
        assert loader.get_output_fieldnames() == [id_field, "path", *EXPECTED_METRIC_COLUMNS[1:]]

    @pytest.mark.parametrize("loader", ALL_LOADERS)
    def test_row_matches_fieldnames_and_formats_metrics(self, loader):
        row = loader.create_output_row(
            metadata={loader.ID_FIELD: "abc"},
            transcription="ref",
            transcription_norm="ref",
            asr_transcription="hyp",
            asr_norm_non="hyp",
            asr_norm_conv="hyp",
            cer_non=0.123456,
            wer_non=0.5,
            sem_non=None,
            poseidon_non=0.75,
            cer_conv=None,
            wer_conv=None,
            sem_conv=None,
            poseidon_conv=None,
            path_from_root="a/b.wav",
            inference_latency_s=1.5,
        )
        assert list(row.keys()) == loader.get_output_fieldnames()
        assert row[loader.ID_FIELD] == "abc"
        assert row["cer_non"] == "0.1235"  # 4-decimal formatting
        assert row["semantic_similarity_non"] == ""  # None -> empty string
        assert row["inference_latency_s"] == "1.5000"

    def test_transcription_extraction_keys(self):
        assert CommonVoiceLoader.get_transcription_from_metadata({"sentence": " hi "}) == "hi"
        assert FleursLoader.get_transcription_from_metadata({"transcription": " hi "}) == "hi"
        assert OpenSLRLineIndexLoader.get_transcription_from_metadata({"transcription": " hi "}) == "hi"


class TestCommonVoiceLoader:
    @pytest.fixture()
    def cv_dir(self, tmp_path):
        lang_dir = tmp_path / "bn"
        clips = lang_dir / "clips"
        clips.mkdir(parents=True)
        (lang_dir / "test.tsv").write_text(
            "client_id\tpath\tsentence\n"
            "spk1\tclip_001.mp3\tপ্রথম বাক্য\n"
            "spk2\tclip_002.mp3\tদ্বিতীয় বাক্য\n"
            "spk3\t\tmissing path skipped\n",
            encoding="utf-8",
        )
        (clips / "clip_001.mp3").write_bytes(b"\x00")
        (clips / "clip_002.mp3").write_bytes(b"\x00")
        (clips / "orphan.mp3").write_bytes(b"\x00")
        return tmp_path

    def test_load_metadata(self, cv_dir):
        meta = CommonVoiceLoader.load_metadata(str(cv_dir))
        assert set(meta) == {"clip_001", "clip_002"}  # blank-path row skipped
        assert meta["clip_001"]["sentence"] == "প্রথম বাক্য"
        assert meta["clip_001"]["client_id"] == "spk1"

    def test_find_audio_files(self, cv_dir):
        files = CommonVoiceLoader.find_audio_files(str(cv_dir))
        ids = {fid for _, fid, _ in files}
        assert ids == {"clip_001", "clip_002", "orphan"}
        for abs_path, _, rel in files:
            assert rel.startswith("bn/clips/")
            assert not rel.startswith("/")

    def test_missing_dir_returns_empty(self, tmp_path):
        assert CommonVoiceLoader.load_metadata(str(tmp_path)) == {}
        assert CommonVoiceLoader.find_audio_files(str(tmp_path)) == []


class TestFleursLoader:
    def _make(self, tmp_path, content):
        test_dir = tmp_path / "test"
        (test_dir / "audio").mkdir(parents=True)
        (test_dir / "test.tsv").write_text(content, encoding="utf-8")
        (test_dir / "audio" / "a.wav").write_bytes(b"\x00")
        return tmp_path

    def test_headerless_format(self, tmp_path):
        d = self._make(tmp_path, "1\ta.wav\tপ্রথম\n2\tb.wav\tদ্বিতীয়\n")
        meta = FleursLoader.load_metadata(str(d))
        assert meta["a.wav"] == {"id": "1", "audio_file": "a.wav", "transcription": "প্রথম"}
        assert set(meta) == {"a.wav", "b.wav"}

    def test_headered_format(self, tmp_path):
        d = self._make(tmp_path, "id\taudio_file\ttranscription\n1\ta.wav\tপ্রথম\n")
        meta = FleursLoader.load_metadata(str(d))
        assert meta["a.wav"]["transcription"] == "প্রথম"

    def test_find_audio_files(self, tmp_path):
        d = self._make(tmp_path, "1\ta.wav\tx\n")
        files = FleursLoader.find_audio_files(str(d))
        assert [(split, rel) for split, _, rel in files] == [("test", "audio/a.wav")]


class TestOpenSLRLineIndexLoader:
    @pytest.fixture()
    def slr_dir(self, tmp_path):
        wavs = tmp_path / "wavs"
        wavs.mkdir()
        (tmp_path / "line_index.tsv").write_text(
            "utt_001.wav\tপ্রথম বাক্য\nutt_002\tদ্বিতীয় বাক্য\n\nmalformed-line\n",
            encoding="utf-8",
        )
        (wavs / "utt_001.wav").write_bytes(b"\x00")
        (wavs / "utt_002.wav").write_bytes(b"\x00")
        (wavs / "unknown.wav").write_bytes(b"\x00")
        return tmp_path

    def test_load_metadata_keys_both_forms(self, slr_dir):
        meta = OpenSLRLineIndexLoader.load_metadata(str(slr_dir))
        # Filename entries also get a stem alias; malformed lines are skipped.
        assert "utt_001.wav" in meta and "utt_001" in meta
        assert meta["utt_001"]["file_id"] == "utt_001.wav"
        assert "malformed-line" not in meta

    def test_find_audio_only_with_metadata(self, slr_dir):
        files = OpenSLRLineIndexLoader.find_audio_files(str(slr_dir))
        ids = {fid for _, _, fid, _ in files}
        assert ids == {"utt_001.wav", "utt_002"}  # unknown.wav has no metadata


class TestOpenSLRUttSpkLoaders:
    @pytest.fixture()
    def slr_dir(self, tmp_path):
        sub = tmp_path / "asr_bengali_0" / "data"
        sub.mkdir(parents=True)
        (tmp_path / "utt_spk_text.tsv").write_text("utt_a\tspk1\tপ্রথম\n", encoding="utf-8")
        (tmp_path / "asr_bengali_0" / "utt_spk_text.tsv").write_text("utt_b\tspk2\tদ্বিতীয়\n", encoding="utf-8")
        (sub / "utt_b.flac").write_bytes(b"\x00")
        (tmp_path / "ignored_dir").mkdir()
        return tmp_path

    def test_load_metadata_merges_root_and_subdirs(self, slr_dir):
        meta = OpenSLR53Loader.load_metadata(str(slr_dir))
        assert set(meta) == {"utt_a", "utt_b"}
        assert meta["utt_b"]["transcription"] == "দ্বিতীয়"

    def test_find_audio_only_in_matching_subdirs(self, slr_dir):
        files = OpenSLR37BDLoader.find_audio_files(str(slr_dir))
        assert len(files) == 1
        subdir, _, fid, rel = files[0]
        assert (subdir, fid, rel) == ("asr_bengali_0", "utt_b", "asr_bengali_0/data/utt_b.flac")


class TestResolveDatasetDir:
    def test_resolves_common_voice(self, tmp_path):
        d = tmp_path / "Common_Voice" / "bn" / "clips"
        d.mkdir(parents=True)
        (tmp_path / "Common_Voice" / "bn" / "test.tsv").write_text("x", encoding="utf-8")
        assert resolve_dataset_dir(str(tmp_path), "commonvoice") == str(tmp_path / "Common_Voice")

    def test_resolves_cv_corpus_fallback(self, tmp_path):
        d = tmp_path / "cv-corpus-17.0" / "bn" / "clips"
        d.mkdir(parents=True)
        (tmp_path / "cv-corpus-17.0" / "bn" / "test.tsv").write_text("x", encoding="utf-8")
        assert resolve_dataset_dir(str(tmp_path), "commonvoice") == str(tmp_path / "cv-corpus-17.0")

    def test_resolves_fleurs(self, tmp_path):
        d = tmp_path / "fleurs" / "test" / "audio"
        d.mkdir(parents=True)
        (tmp_path / "fleurs" / "test" / "test.tsv").write_text("x", encoding="utf-8")
        assert resolve_dataset_dir(str(tmp_path), "fleurs") == str(tmp_path / "fleurs")

    def test_rejects_incomplete_layout(self, tmp_path):
        (tmp_path / "fleurs" / "test").mkdir(parents=True)  # no test.tsv / audio
        assert resolve_dataset_dir(str(tmp_path), "fleurs") is None

    def test_unknown_dataset_and_missing_base(self, tmp_path):
        assert resolve_dataset_dir(str(tmp_path), "not_a_dataset") is None
        assert resolve_dataset_dir(str(tmp_path / "nope"), "fleurs") is None
