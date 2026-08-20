"""Tests for TSV audio_path resolution and the dataset-root boundary (issue #127).

A relative ``audio_path`` that resolves outside the TSV's directory is always
rejected. Absolute paths are allowed by default (SONAR's own dataset preparer
writes them) and rejected only in strict mode, which also requires each path
to be an existing regular file.
"""

from pathlib import Path

import pytest

from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator, _resolve_audio_path


class TestResolveAudioPath:
    def test_relative_path_inside_root_resolves(self, tmp_path):
        wav = tmp_path / "audio" / "a.wav"
        wav.parent.mkdir()
        wav.write_bytes(b"")
        resolved = _resolve_audio_path("audio/a.wav", tmp_path, allow_absolute_audio_paths=True)
        assert resolved == str(wav.resolve())

    def test_traversal_rejected_by_default(self, tmp_path):
        with pytest.raises(ValueError, match="escapes dataset root"):
            _resolve_audio_path("../../../../etc/hosts", tmp_path, allow_absolute_audio_paths=True)

    def test_traversal_rejected_in_strict_mode(self, tmp_path):
        with pytest.raises(ValueError, match="escapes dataset root"):
            _resolve_audio_path("../outside.wav", tmp_path, allow_absolute_audio_paths=False)

    def test_internal_dotdot_staying_inside_root_is_fine(self, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")
        resolved = _resolve_audio_path("audio/../a.wav", tmp_path, allow_absolute_audio_paths=True)
        assert resolved == str(wav.resolve())

    def test_absolute_path_allowed_by_default(self, tmp_path):
        wav = tmp_path / "elsewhere.wav"
        wav.write_bytes(b"")
        other_root = tmp_path / "dataset"
        other_root.mkdir()
        resolved = _resolve_audio_path(str(wav), other_root, allow_absolute_audio_paths=True)
        assert resolved == str(wav.resolve())

    def test_absolute_path_rejected_in_strict_mode(self, tmp_path):
        wav = tmp_path / "elsewhere.wav"
        wav.write_bytes(b"")
        with pytest.raises(ValueError, match="must be relative"):
            _resolve_audio_path(str(wav), tmp_path, allow_absolute_audio_paths=False)

    def test_strict_mode_requires_existing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _resolve_audio_path("missing.wav", tmp_path, allow_absolute_audio_paths=False)

    def test_strict_mode_rejects_directories(self, tmp_path):
        (tmp_path / "adir").mkdir()
        with pytest.raises(ValueError, match="not a regular file"):
            _resolve_audio_path("adir", tmp_path, allow_absolute_audio_paths=False)


class TestLoadDataBoundary:
    def _tsv(self, tmp_path: Path, audio_path: str) -> str:
        path = tmp_path / "data.tsv"
        path.write_text(f"audio_path\ttranscription\n{audio_path}\thello\n")
        return str(path)

    def test_traversal_row_raises(self, tmp_path):
        tsv = self._tsv(tmp_path, "../../../../etc/hosts")
        with pytest.raises(ValueError, match="escapes dataset root"):
            SingleSpeakerEvaluator.load_data(tsv)

    def test_relative_row_resolves_against_tsv_directory(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"")
        tsv = self._tsv(tmp_path, "a.wav")
        data = SingleSpeakerEvaluator.load_data(tsv)
        assert data[0]["audio_path"] == str((tmp_path / "a.wav").resolve())

    def test_absolute_row_allowed_by_default(self, tmp_path):
        wav = tmp_path / "outside" / "a.wav"
        wav.parent.mkdir()
        wav.write_bytes(b"")
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        tsv = dataset_dir / "data.tsv"
        tsv.write_text(f"audio_path\ttranscription\n{wav}\thello\n")
        data = SingleSpeakerEvaluator.load_data(str(tsv))
        assert data[0]["audio_path"] == str(wav.resolve())

    def test_absolute_row_rejected_in_strict_mode(self, tmp_path):
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"")
        tsv = self._tsv(tmp_path, str(wav))
        with pytest.raises(ValueError, match="must be relative"):
            SingleSpeakerEvaluator.load_data(tsv, allow_absolute_audio_paths=False)
