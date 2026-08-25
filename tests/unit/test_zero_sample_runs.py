"""Regression tests for issue #102: abnormal runs must not look like clean ones.

Five conditions used to exit 0 with success-looking artifacts despite
evaluating zero samples: unknown model name, every sample failing, empty
reference transcriptions, TSV rows missing a field, and a multi run with
pyannote missing. These tests pin the corrected behavior: honest (null)
aggregates, error rows instead of silently dropped input, CSVs that always
carry a header, actionable dependency messages, and exceptions that make the
CLI exit non-zero.
"""

import csv
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from psdn_sonar.evaluators.single_speaker import (
    NoSamplesEvaluatedError,
    SingleSpeakerEvaluator,
    _model_factory,
)


@pytest.fixture
def patched_run_evaluation_env(monkeypatch):
    """Stub the heavy pieces of run_evaluation so tests exercise only the
    control flow (model construction, artifact writing, failure raising)."""
    monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
    monkeypatch.setattr(
        SingleSpeakerEvaluator,
        "load_data",
        lambda *args, **kwargs: [{"audio_path": "clip.wav", "ground_truth": "hello"}],
    )


def _fake_result(successful: int, failed: int, results=None):
    return {
        "model_name": "stub",
        "results": results or [],
        "summary": {
            "total_samples": successful + failed,
            "successful": successful,
            "failed": failed,
            "avg_wer": 0.0 if successful else None,
            "avg_cer": 0.0 if successful else None,
            "elapsed_time": 0.0,
            "avg_latency_s": None,
            "median_latency_s": None,
            "p95_latency_s": None,
        },
    }


class TestLoadDataKeepsBadRows:
    def test_empty_transcription_row_is_marked_not_dropped(self, tmp_path):
        tsv = tmp_path / "eval.tsv"
        tsv.write_text("audio_path\ttranscription\nclip.wav\t\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "empty transcription" in data[0]["load_error"]
        assert "line 2" in data[0]["load_error"]

    def test_row_missing_field_is_marked_not_dropped(self, tmp_path):
        tsv = tmp_path / "eval.tsv"
        tsv.write_text("audio_path\ttranscription\nclip.wav\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "transcription" in data[0]["load_error"]

    def test_valid_rows_have_no_load_error(self, tmp_path):
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"")
        tsv = tmp_path / "eval.tsv"
        tsv.write_text(f"audio_path\ttranscription\n{wav}\thello\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "load_error" not in data[0]

    def test_surplus_field_row_is_marked_not_truncated(self, tmp_path):
        # Issue #141: a literal tab inside the transcription used to silently
        # truncate the reference to the field before the tab (exit 0, bad
        # score). The row must be marked malformed instead.
        tsv = tmp_path / "eval.tsv"
        tsv.write_text("audio_path\ttranscription\nclip.wav\thello\tworld extra\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "line 2" in data[0]["load_error"]
        assert "3 fields" in data[0]["load_error"]
        assert "truncate" in data[0]["load_error"]

    def test_utf8_bom_is_stripped(self, tmp_path):
        # Issue #141: Excel prepends a BOM to exported TSVs; it used to
        # corrupt the first column name into \ufeffaudio_path and raise
        # "TSV missing required columns: audio_path" for a present column.
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"")
        tsv = tmp_path / "eval.tsv"
        tsv.write_bytes(b"\xef\xbb\xbf" + f"audio_path\ttranscription\n{wav}\thello world\n".encode())

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "load_error" not in data[0]
        assert data[0]["ground_truth"] == "hello world"

    def test_bom_free_file_unaffected_by_sig_decoding(self, tmp_path):
        # utf-8-sig must be a no-op for ordinary files.
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"")
        tsv = tmp_path / "eval.tsv"
        tsv.write_text(f"audio_path\ttranscription\n{wav}\tплитка বাংলা 테스트\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert data[0]["ground_truth"] == "плитка বাংলা 테스트"

    def test_extra_header_column_still_tolerated(self, tmp_path):
        # A header with surplus columns over fully-aligned rows is not a
        # surplus-field row; documented leniency stays.
        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"")
        tsv = tmp_path / "eval.tsv"
        tsv.write_text(f"audio_path\ttranscription\textra_col\n{wav}\thello\tignored\n", encoding="utf-8")

        data = SingleSpeakerEvaluator.load_data(str(tsv))

        assert len(data) == 1
        assert "load_error" not in data[0]
        assert data[0]["ground_truth"] == "hello"


class TestEvaluateOneHonestAggregates:
    def test_load_error_rows_counted_failed_and_avgs_null(self):
        data = [{"audio_path": "", "ground_truth": "", "load_error": "TSV line 2: missing or empty transcription"}]

        result = SingleSpeakerEvaluator.evaluate_one(MagicMock(), data, "stub-model")

        summary = result["summary"]
        assert summary["failed"] == 1
        assert summary["successful"] == 0
        assert summary["avg_wer"] is None
        assert summary["avg_cer"] is None
        assert result["results"][0]["error"] == "TSV line 2: missing or empty transcription"

    def test_all_empty_predictions_yield_null_avgs(self):
        # Nonexistent audio path -> prediction "" -> "Empty prediction" failure.
        data = [{"audio_path": "does-not-exist.wav", "ground_truth": "hello"}]

        result = SingleSpeakerEvaluator.evaluate_one(MagicMock(), data, "stub-model")

        summary = result["summary"]
        assert summary["successful"] == 0
        assert summary["failed"] == 1
        assert summary["avg_wer"] is None
        assert summary["avg_cer"] is None


class TestUncomputableMetricsExcluded:
    """Issue #107: a transcribed row whose CER/WER cannot be computed used to
    be scored as best case (0.0) and counted successful — deflating the run
    averages — while other paths excluded or worst-cased the same condition.
    Such rows are now failed and excluded from every aggregate, with the
    transcription preserved on the row."""

    @staticmethod
    def _tiny_wav(tmp_path) -> str:
        import struct
        import wave

        path = tmp_path / "clip.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(struct.pack("<" + "h" * 1600, *([1000] * 1600)))
        return str(path)

    def test_reference_normalizing_to_empty_fails_row(self, tmp_path):
        model = SimpleNamespace(transcribe=lambda path: "hello world")
        # Punctuation-only reference normalizes to "" -> CER/WER uncomputable.
        data = [{"audio_path": self._tiny_wav(tmp_path), "ground_truth": "...!?"}]

        result = SingleSpeakerEvaluator.evaluate_one(model, data, "stub-model", language="en")

        summary = result["summary"]
        assert summary["successful"] == 0
        assert summary["failed"] == 1
        assert summary["avg_wer"] is None
        assert summary["avg_cer"] is None
        row = result["results"][0]
        assert row["wer"] is None
        assert row["cer"] is None
        assert "CER/WER uncomputable" in row["error"]
        assert row["prediction"] == "hello world"

    def test_scorable_row_unaffected(self, tmp_path):
        model = SimpleNamespace(transcribe=lambda path: "hello world")
        data = [{"audio_path": self._tiny_wav(tmp_path), "ground_truth": "hello world"}]

        result = SingleSpeakerEvaluator.evaluate_one(model, data, "stub-model", language="en")

        assert result["summary"]["successful"] == 1
        assert result["summary"]["failed"] == 0
        assert result["summary"]["avg_wer"] == 0.0


class TestRunEvaluationFailsLoudly:
    def test_unknown_model_raises(self, tmp_path, patched_run_evaluation_env, monkeypatch):
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker._model_factory",
            lambda *args, **kwargs: None,
        )

        with pytest.raises(ValueError, match="could be constructed"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["whisper-hindi"],
                language="hi",
                write_scores=False,
            )

    def test_zero_successful_raises_after_writing_artifacts(self, tmp_path, patched_run_evaluation_env, monkeypatch):
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker._model_factory",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: _fake_result(successful=0, failed=1),
        )

        with pytest.raises(NoSamplesEvaluatedError, match="0 successful samples"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["whisper_base_en"],
                language="en",
                write_scores=True,
                compute_sem=False,
            )

        # Artifacts were still written, with an honest shape: a CSV that
        # carries its header even with zero rows, and null WER/CER means.
        csv_path = tmp_path / "asr_detailed_whisper_base_en.csv"
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert "wer" in reader.fieldnames and "error" in reader.fieldnames

        payload = json.loads((tmp_path / "scores_whisper_base_en.json").read_text(encoding="utf-8"))
        assert payload["aggregate"]["wer_mean"] is None
        assert payload["aggregate"]["cer_mean"] is None

    def test_some_successful_does_not_raise(self, tmp_path, patched_run_evaluation_env, monkeypatch):
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker._model_factory",
            lambda *args, **kwargs: object(),
        )
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: _fake_result(successful=1, failed=1),
        )

        results = SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["whisper_base_en"],
            language="en",
            write_scores=False,
        )
        assert "whisper_base_en" in results


class TestMissingApiKeyNotReportedAsUnknownModel:
    """Issue #168: an adapter's missing-API-key ValueError used to be swallowed
    by _model_factory's `except ValueError` and reported as 'not found in the
    registry' — a listing that contained the exact id the user passed. Only
    the registry's own UnknownModelError may be translated into that message."""

    def test_model_factory_returns_none_only_for_unknown_names(self):
        assert _model_factory("nonexistent_model_xyz") is None

    def test_model_factory_propagates_credential_valueerror(self, monkeypatch):
        def raising_create_model(*args, **kwargs):
            raise ValueError("ElevenLabs API key not found. Set ELEVENLABS_API_KEY ...")

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.create_model", raising_create_model)

        with pytest.raises(ValueError, match="API key not found"):
            _model_factory("elevenlabs_api")

    def test_missing_key_surfaces_adapter_message_not_registry_listing(
        self, tmp_path, patched_run_evaluation_env, monkeypatch, caplog
    ):
        # Real registry, real ElevenLabs adapter (needs only `requests`): with
        # no key anywhere, its __init__ raises the actionable ValueError.
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("XI_API_KEY", raising=False)

        with caplog.at_level("ERROR"), pytest.raises(ValueError, match="could be constructed"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["elevenlabs_api"],
                language="en",
                write_scores=False,
            )

        assert "could not be constructed" in caplog.text
        assert "ElevenLabs API key not found" in caplog.text
        assert "Set ELEVENLABS_API_KEY" in caplog.text
        assert "not found in the registry" not in caplog.text

    def test_truly_unknown_model_still_gets_the_registry_listing(
        self, tmp_path, patched_run_evaluation_env, monkeypatch, caplog
    ):
        with caplog.at_level("ERROR"), pytest.raises(ValueError, match="could be constructed"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["not_a_real_model"],
                language="en",
                write_scores=False,
            )

        assert "Model not_a_real_model not found in the registry" in caplog.text
        assert "Registered ids:" in caplog.text


class TestConstructorFailureIsolated:
    """Issue #108: a raising model constructor used to kill the entire
    multi-model loop, losing the output of models already evaluated in the
    run. It is now skipped with the reason logged; only a run where every
    model fails to construct errors out."""

    def test_one_failing_constructor_skips_and_run_continues(
        self, tmp_path, patched_run_evaluation_env, monkeypatch, caplog
    ):
        def factory(name, **kwargs):
            if name == "khushids_bengali":
                raise ModuleNotFoundError("No module named 'peft'")
            return object()

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", factory)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: _fake_result(successful=1, failed=0),
        )

        with caplog.at_level("ERROR"):
            results = SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["khushids_bengali", "wav2vec2_bengali"],
                language="bn",
                write_scores=False,
            )

        assert list(results) == ["wav2vec2_bengali"]
        assert "Skipping model khushids_bengali" in caplog.text
        assert "peft" in caplog.text
        # The surviving model's artifact was written.
        assert (tmp_path / "asr_detailed_wav2vec2_bengali.csv").exists()

    def test_earlier_results_survive_a_later_failure(self, tmp_path, patched_run_evaluation_env, monkeypatch):
        def factory(name, **kwargs):
            if name == "second_model":
                raise RuntimeError("constructor exploded")
            return object()

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", factory)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: _fake_result(successful=1, failed=0),
        )

        results = SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["first_model", "second_model"],
            language="bn",
            write_scores=False,
        )

        assert list(results) == ["first_model"]
        assert (tmp_path / "asr_detailed_first_model.csv").exists()

    def test_all_constructors_failing_raises(self, tmp_path, patched_run_evaluation_env, monkeypatch):
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker._model_factory",
            lambda *args, **kwargs: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'peft'")),
        )

        with pytest.raises(ValueError, match="could be constructed"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["khushids_bengali"],
                language="bn",
                write_scores=False,
            )


class TestScoresArtifactNullMeans:
    def test_null_avgs_stay_null_in_scores_json(self):
        from psdn_sonar.benchmark.scores import build_run_scores
        from psdn_sonar.benchmark.submission import SubmissionConfig

        submission = SubmissionConfig(
            provider="local",
            model_snapshot="stub",
            region="local",
            protocol="batch",
            inference_params={"language_code": "en"},
            seed=42,
            git_sha="abc",
            package_version="0.1.0",
            timestamp_utc="2026-08-19T12:00:00Z",
        )

        artifact = build_run_scores(submission, _fake_result(successful=0, failed=2))

        assert artifact.aggregate.wer_mean is None
        assert artifact.aggregate.cer_mean is None
        assert artifact.aggregate.successful == 0
        assert artifact.aggregate.failed == 2


class TestPyannoteMissingIsActionable:
    def test_multi_pipeline_fails_fast_with_install_hint(self, tmp_path, monkeypatch):
        from psdn_sonar.multispeaker_pipeline import run_multispeaker_evaluation

        monkeypatch.setattr("psdn_sonar.preprocessing.pyannote_utils.PYANNOTE_AVAILABLE", False)
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")

        with pytest.raises(RuntimeError, match=r"psdn-sonar\[pyannote\]"):
            run_multispeaker_evaluation(
                manifest_path=str(manifest),
                model_name="whisper_base_en",
                output_dir=str(tmp_path / "out"),
                method="pyannote_vad",
                language="en",
            )

    def test_run_single_method_records_install_hint_per_clip(self, monkeypatch):
        from psdn_sonar.preprocessing import preprocessing_selector

        monkeypatch.setattr(preprocessing_selector, "PYANNOTE_AVAILABLE", False)
        entry = SimpleNamespace(audio_id="TEST001")

        all_results, best_a, best_b = preprocessing_selector.run_single_method(
            entry=entry,
            asr_model=None,
            ref_a="hello",
            ref_b="world",
            segments=[],
            audio_a=None,
            audio_b=None,
            combined_audio=None,
            metric_fn=None,
            transcribe_fn=None,
            active_methods=["pyannote_vad"],
            method_name="pyannote_vad",
        )

        assert best_a is None and best_b is None
        for speaker in ("A", "B"):
            assert "psdn-sonar[pyannote]" in all_results[speaker][0]["error"]
