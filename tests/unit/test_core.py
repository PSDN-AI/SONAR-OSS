"""Tests for the core dataset-evaluation loops."""

import csv
import json

import numpy as np
import pytest
import soundfile as sf

from psdn_sonar import core
from psdn_sonar.core import _mean_std, process_dataset_with_asr, process_manifest_with_asr

SR = 16_000


class _StubModel:
    """ASR stub that returns a fixed transcription for every audio file."""

    supports_diarization = False

    def __init__(self, text="hello world"):
        self.text = text
        self.calls = []

    def transcribe(self, audio_path):
        self.calls.append(audio_path)
        return self.text


class _StubLoader:
    """Dataset loader stub exercising the generic (non-CV/FLEURS) code path."""

    FIELDNAMES = ["id", "path", "transcription", "wer_conv", "latency"]

    def __init__(self, samples):
        # samples: {fid: transcription}
        _StubLoader.samples = samples

    @staticmethod
    def load_metadata(dataset_dir):
        return {fid: {"id": fid, "text": text} for fid, text in _StubLoader.samples.items()}

    @staticmethod
    def find_audio_files(dataset_dir):
        return [("train", f"{dataset_dir}/{fid}.wav", fid, f"{fid}.wav") for fid in _StubLoader.samples]

    @staticmethod
    def get_output_fieldnames():
        return list(_StubLoader.FIELDNAMES)

    @staticmethod
    def get_transcription_from_metadata(meta):
        return meta["text"]

    @staticmethod
    def create_output_row(
        meta,
        transcription,
        transcription_norm,
        asr_transcription,
        asr_norm_non,
        asr_norm_conv,
        cer_non,
        wer_non,
        sem_non,
        poseidon_non,
        cer_conv,
        wer_conv,
        sem_conv,
        poseidon_conv,
        path_from_root,
        inference_latency_s=None,
    ):
        return {
            "id": meta["id"],
            "path": path_from_root,
            "transcription": transcription,
            "wer_conv": wer_conv,
            "latency": inference_latency_s,
        }


class TestMeanStd:
    def test_empty(self):
        assert _mean_std([]) == (0.0, 0.0)

    def test_single_value_has_zero_std(self):
        assert _mean_std([0.4]) == (0.4, 0.0)

    def test_mean_and_std(self):
        m, s = _mean_std([0.0, 1.0])
        assert m == pytest.approx(0.5)
        assert s == pytest.approx(0.7071, abs=1e-4)


class TestProcessDatasetWithASR:
    def _run(self, tmp_path, monkeypatch, samples, **kwargs):
        monkeypatch.setitem(core._DATASET_LOADERS, "stub", _StubLoader(samples))
        output = tmp_path / "out" / "results.tsv"
        model = _StubModel("hello world")
        process_dataset_with_asr(
            "stub", str(tmp_path), model, str(output), asr_model_name="stub-model", language="en", **kwargs
        )
        return output, model

    def _read_rows(self, output):
        with open(output, encoding="utf-8") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def test_unknown_dataset_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown dataset"):
            process_dataset_with_asr("nope", str(tmp_path), _StubModel(), str(tmp_path / "o.tsv"))

    def test_writes_rows_and_stats(self, tmp_path, monkeypatch):
        output, model = self._run(tmp_path, monkeypatch, {"s1": "hello world", "s2": "other text"})

        rows = self._read_rows(output)
        assert len(rows) == 2
        by_id = {r["id"]: r for r in rows}
        # Perfect hypothesis for s1 gives WER 0.
        assert float(by_id["s1"]["wer_conv"]) == 0.0
        assert float(by_id["s1"]["latency"]) >= 0.0
        assert len(model.calls) == 2

        stats_text = (tmp_path / "out" / "results.txt").read_text(encoding="utf-8")
        assert "Model: stub-model" in stats_text
        assert "Samples: 2" in stats_text
        assert "Without Script Conversion" in stats_text

    def test_max_samples_limits_evaluation(self, tmp_path, monkeypatch):
        samples = {f"s{i}": "hello world" for i in range(5)}
        output, model = self._run(tmp_path, monkeypatch, samples, max_samples=2)

        assert len(self._read_rows(output)) == 2
        assert len(model.calls) == 2

    def test_empty_transcription_skipped(self, tmp_path, monkeypatch):
        output, _ = self._run(tmp_path, monkeypatch, {"s1": "hello world", "s2": ""})
        assert [r["id"] for r in self._read_rows(output)] == ["s1"]


def _write_wav(path, seconds=1.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    sf.write(str(path), (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), SR)
    return path


def _write_manifest_dataset(tmp_path, ref_a="hello world", ref_b="good morning"):
    _write_wav(tmp_path / "audio" / "a.wav")
    _write_wav(tmp_path / "audio" / "b.wav")
    transcript = {
        "segments": [
            {"speaker": "speaker_a", "text": ref_a, "start": 0.0, "end": 1.0},
            {"speaker": "speaker_b", "text": ref_b, "start": 1.0, "end": 2.0},
        ]
    }
    (tmp_path / "transcripts").mkdir(exist_ok=True)
    (tmp_path / "transcripts" / "conv_001.json").write_text(json.dumps(transcript), encoding="utf-8")
    entry = {
        "audio_id": "conv_001",
        "audio_filepaths": {"speaker_a": "audio/a.wav", "speaker_b": "audio/b.wav"},
        "transcript_filepath": "transcripts/conv_001.json",
        "num_speakers": 2,
    }
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return manifest


class TestProcessManifestWithASR:
    def test_no_valid_methods_raises(self, tmp_path):
        manifest = _write_manifest_dataset(tmp_path)
        with pytest.raises(ValueError, match="No valid preprocessing methods"):
            process_manifest_with_asr(
                str(manifest),
                _StubModel(),
                str(tmp_path / "out.csv"),
                methods=["scribe_diarize"],  # requires diarization support
            )

    def test_evaluates_both_speakers(self, tmp_path):
        manifest = _write_manifest_dataset(tmp_path, ref_a="hello world", ref_b="hello world")
        output = tmp_path / "results" / "out.csv"

        process_manifest_with_asr(
            str(manifest),
            _StubModel("hello world"),
            str(output),
            asr_model_name="stub-model",
            language="en",
            methods=["no_trim"],
        )

        with open(output, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert {r["speaker"] for r in rows} == {"A", "B"}
        assert all(r["best_method"] == "no_trim" for r in rows)
        assert all(float(r["wer_conv"]) == 0.0 for r in rows)

        stats_text = (tmp_path / "results" / "out.txt").read_text(encoding="utf-8")
        assert "Model: stub-model" in stats_text
        assert "Mode: fixed" not in stats_text  # auto-selection was used
        assert "Samples: 2" in stats_text

    def test_missing_transcript_skips_clip(self, tmp_path):
        manifest = _write_manifest_dataset(tmp_path)
        (tmp_path / "transcripts" / "conv_001.json").unlink()
        output = tmp_path / "out.csv"

        process_manifest_with_asr(str(manifest), _StubModel(), str(output), methods=["no_trim"], language="en")

        with open(output, encoding="utf-8") as f:
            assert list(csv.DictReader(f)) == []

    def test_explicit_method_recorded_in_stats(self, tmp_path):
        manifest = _write_manifest_dataset(tmp_path)
        output = tmp_path / "out.csv"

        process_manifest_with_asr(
            str(manifest),
            _StubModel("hello world"),
            str(output),
            methods=["no_trim"],
            method="no_trim",
            language="en",
        )

        stats_text = (tmp_path / "out.txt").read_text(encoding="utf-8")
        assert "Mode: fixed:no_trim" in stats_text
