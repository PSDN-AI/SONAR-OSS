"""Tests for the multi-speaker manifest loader."""

import json

import pytest

from psdn_sonar.loaders.manifest import (
    ManifestEntry,
    create_output_row,
    get_clip_files,
    get_output_fieldnames,
    load_manifest,
    load_transcript,
    load_transcript_with_segments,
)


@pytest.fixture()
def manifest_dir(tmp_path):
    entry = {
        "audio_id": "conv_001",
        "audio_filepaths": {"speaker_a": "audio/a.wav", "speaker_b": "audio/b.wav"},
        "transcript_filepath": "transcripts/conv_001.json",
        "num_speakers": 2,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n\n" + json.dumps({**entry, "audio_id": "conv_002"}) + "\n")
    return tmp_path


class TestLoadManifest:
    def test_parses_entries_and_skips_blank_lines(self, manifest_dir):
        entries = load_manifest(str(manifest_dir / "manifest.jsonl"))
        assert [e.audio_id for e in entries] == ["conv_001", "conv_002"]
        assert entries[0].num_speakers == 2
        assert entries[0].base_dir == manifest_dir

    def test_get_clip_files_resolves_against_manifest_dir(self, manifest_dir):
        entries = load_manifest(str(manifest_dir / "manifest.jsonl"))
        audio_a, audio_b, transcript = get_clip_files(entries[0])
        assert audio_a == (manifest_dir / "audio" / "a.wav").resolve()
        assert audio_b == (manifest_dir / "audio" / "b.wav").resolve()
        assert transcript == (manifest_dir / "transcripts" / "conv_001.json").resolve()

    def test_get_clip_files_missing_speaker_is_none(self, manifest_dir):
        entry = ManifestEntry(
            audio_id="x",
            audio_filepaths={"speaker_a": "a.wav"},
            transcript_filepath="t.json",
            num_speakers=1,
            base_dir=manifest_dir,
        )
        audio_a, audio_b, _ = get_clip_files(entry)
        assert audio_a is not None
        assert audio_b is None


class TestLoadTranscript:
    def _write(self, tmp_path, payload):
        p = tmp_path / "t.json"
        p.write_text(json.dumps(payload), encoding="utf-8")
        return p

    def test_segments_format(self, tmp_path):
        p = self._write(
            tmp_path,
            {
                "segments": [
                    {"speaker": "speaker_a", "text": "hello"},
                    {"speaker": "speaker_b", "text": "hi"},
                    {"speaker": "AGENT_A", "text": "again"},  # *_a suffix, case-insensitive
                    {"speaker": "narrator", "text": "ignored"},
                ]
            },
        )
        a, b = load_transcript(p)
        assert a == "hello again"
        assert b == "hi"

    def test_segments_format_returns_original_segments(self, tmp_path):
        segments = [{"speaker": "speaker_a", "text": "hello", "start": "00:00:01", "end": "00:00:02"}]
        p = self._write(tmp_path, {"segments": segments})
        a, b, segs = load_transcript_with_segments(p)
        assert (a, b) == ("hello", "")
        assert segs == segments

    def test_legacy_array_format(self, tmp_path):
        p = self._write(
            tmp_path,
            [
                {"speaker": "Speaker A", "text": "one"},
                {"speaker": "b", "text": "two"},
                {"speaker": "this is a very long free-text label", "text": "ignored"},
            ],
        )
        a, b = load_transcript(p)
        assert a == "one"
        assert b == "two"

    def test_legacy_array_synthesises_segments(self, tmp_path):
        p = self._write(tmp_path, [{"speaker": "a", "text": "one"}])
        _, _, segs = load_transcript_with_segments(p)
        assert segs == [{"speaker": "a", "text": "one", "start": "00:00:00", "end": "00:00:00"}]

    def test_transcript_and_with_segments_agree(self, tmp_path):
        payload = {"segments": [{"speaker": "speaker_a", "text": "x"}, {"speaker": "speaker_b", "text": "y"}]}
        p = self._write(tmp_path, payload)
        assert load_transcript(p) == load_transcript_with_segments(p)[:2]


class TestOutputRow:
    def _row(self, **overrides):
        kwargs = dict(
            audio_id="conv_001",
            speaker="speaker_a",
            best_method="pyannote",
            path="a.wav",
            transcription="ref",
            transcription_norm="ref",
            asr_transcription="hyp",
            asr_norm_non="hyp",
            asr_norm_conv="hyp",
            cer_non=0.1,
            wer_non=0.2,
            sem_non=None,
            poseidon_non=0.9,
            cer_conv=None,
            wer_conv=None,
            sem_conv=None,
            poseidon_conv=None,
            original_duration_s=12.345,
            trimmed_duration_s=0.0,
        )
        kwargs.update(overrides)
        return create_output_row(**kwargs)

    def test_row_matches_fieldnames(self):
        assert list(self._row().keys()) == get_output_fieldnames()

    def test_metric_and_duration_formatting(self):
        row = self._row(snr_db=25.5, clipping_ratio=0.0000123, inference_latency_s=2.0)
        assert row["cer_non"] == "0.1000"
        assert row["semantic_similarity_non"] == ""  # None -> empty
        assert row["original_duration_s"] == "12.35"
        assert row["trimmed_duration_s"] == ""  # zero-duration -> empty
        assert row["snr_db"] == "25.50"
        assert row["clipping_ratio"] == "0.000012"
        assert row["inference_latency_s"] == "2.0000"

    def test_all_method_scores_serialized_as_json(self):
        row = self._row(all_method_scores={"pyannote": 0.9})
        assert json.loads(row["all_method_scores"]) == {"pyannote": 0.9}
        assert self._row()["all_method_scores"] == ""
