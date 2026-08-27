"""Tests for preprocessing selection and the pyannote-independent helpers."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from psdn_sonar.preprocessing import preprocessing_selector, pyannote_utils
from psdn_sonar.preprocessing.methods import run_pyannote_diarize
from psdn_sonar.preprocessing.preprocessing_selector import (
    _exc_reason,
    _first_valid,
    _score_preprocessed,
    _select_best_oracle,
    run_single_method,
    run_sweep,
    select_best_preprocessing,
)
from psdn_sonar.preprocessing.pyannote_utils import (
    assign_words_to_speakers,
    extract_and_concat_segments,
)

SR = 16_000


def _write_wav(path, pieces):
    """Write a wav built from ("tone"|"silence", seconds) pieces."""
    chunks = []
    for kind, seconds in pieces:
        n = int(SR * seconds)
        if kind == "tone":
            t = np.linspace(0, seconds, n, endpoint=False)
            chunks.append((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    sf.write(str(path), np.concatenate(chunks), SR)
    return path


def _metric_exact(ref, hyp):
    """(cer, wer, similarity, poseidon): perfect on match, worst otherwise."""
    if ref == hyp:
        return 0.0, 0.0, 1.0, 1.0
    return 1.0, 1.0, 0.0, 0.0


_ENTRY = SimpleNamespace(audio_id="conv_001")
_NO_DIARIZE_MODEL = SimpleNamespace(supports_diarization=False)


class TestScorePreprocessed:
    def test_zero_original_duration_scores_zero(self):
        assert _score_preprocessed("x.wav", 0.0, 0.0) == 0.0

    def test_overtrimmed_scores_zero(self):
        assert _score_preprocessed("x.wav", 10.0, 1.0) == 0.0

    def test_speech_scores_higher_than_silence(self, tmp_path):
        tone = _write_wav(tmp_path / "tone.wav", [("tone", 2.0)])
        silent = _write_wav(tmp_path / "silent.wav", [("silence", 2.0)])
        assert _score_preprocessed(tone, 2.0, 2.0) > _score_preprocessed(silent, 2.0, 2.0)


class TestSelectBestPreprocessing:
    def test_selects_a_valid_method(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0), ("silence", 2.0), ("tone", 1.0)])
        path, method, orig, trim = select_best_preprocessing(wav, "A", [], ["energy_trim", "no_trim"], {}, {}, {})
        assert method in {"energy_trim", "no_trim"}
        assert orig == pytest.approx(4.0, abs=0.05)
        assert trim > 0

    def test_no_per_channel_methods_falls_back_to_no_trim(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0)])
        path, method, orig, trim = select_best_preprocessing(wav, "A", [], ["scribe_diarize"], {}, {}, {})
        assert path == str(wav)
        assert method == "no_trim"
        assert orig == trim


class TestRunSingleMethod:
    def _call(self, tmp_path, **overrides):
        wav_a = _write_wav(tmp_path / "a.wav", [("tone", 1.0)])
        wav_b = _write_wav(tmp_path / "b.wav", [("tone", 1.0)])
        calls = []

        def transcribe(path):
            calls.append(path)
            return "hello"

        kwargs = dict(
            entry=_ENTRY,
            asr_model=_NO_DIARIZE_MODEL,
            ref_a="hello",
            ref_b="hello",
            segments=[],
            audio_a=wav_a,
            audio_b=wav_b,
            combined_audio=None,
            metric_fn=_metric_exact,
            transcribe_fn=transcribe,
            active_methods=["energy_trim", "no_trim"],
        )
        kwargs.update(overrides)
        return run_single_method(**kwargs), calls

    def test_auto_mode_one_asr_call_per_speaker(self, tmp_path):
        (all_results, best_a, best_b), calls = self._call(tmp_path)
        assert len(calls) == 2
        assert best_a["text"] == best_b["text"] == "hello"
        assert best_a["method"] in {"energy_trim", "no_trim"}
        # metrics are left to the caller in single-method mode
        assert best_a["cer"] is None

    def test_explicit_method(self, tmp_path):
        (all_results, best_a, _), _ = self._call(tmp_path, method_name="no_trim")
        assert best_a["method"] == "no_trim"
        assert best_a["original_duration_s"] == pytest.approx(1.0, abs=0.05)

    def test_per_clip_without_combined_audio_errors(self, tmp_path):
        # A model that *can* run the method, so the missing combined audio is
        # the reason reported rather than the capability precheck (issue #189).
        capable = SimpleNamespace(supports_diarization=True, transcribe_diarized=lambda path, num_speakers: {})
        (all_results, best_a, best_b), calls = self._call(tmp_path, method_name="scribe_diarize", asr_model=capable)
        assert calls == []
        assert best_a["error"] == "combined audio not found"
        assert best_b["error"] == "combined audio not found"

    def test_per_clip_with_diarizing_model(self, tmp_path):
        model = SimpleNamespace(
            supports_diarization=True,
            transcribe_diarized=lambda path, num_speakers: {"s0": "ref a text", "s1": "ref b text"},
        )
        wav = _write_wav(tmp_path / "combined.wav", [("tone", 1.0)])
        (all_results, best_a, best_b), _ = self._call(
            tmp_path,
            method_name="scribe_diarize",
            asr_model=model,
            combined_audio=wav,
            ref_a="ref a text",
            ref_b="ref b text",
        )
        assert best_a["text"] == "ref a text"
        assert best_b["text"] == "ref b text"
        assert best_a["method"] == "scribe_diarize"


class TestRunSweep:
    def test_oracle_selects_best_and_skips_unsupported(self, tmp_path):
        wav_a = _write_wav(tmp_path / "a.wav", [("tone", 1.0), ("silence", 1.5), ("tone", 0.5)])
        wav_b = _write_wav(tmp_path / "b.wav", [("tone", 1.0)])

        def transcribe(path):
            return "hello"

        all_results, best_a, best_b = run_sweep(
            entry=_ENTRY,
            asr_model=_NO_DIARIZE_MODEL,
            ref_a="hello",
            ref_b="hello",
            segments=[],
            audio_a=wav_a,
            audio_b=wav_b,
            combined_audio=None,
            metric_fn=_metric_exact,
            transcribe_fn=transcribe,
            methods=["no_trim", "energy_trim", "timestamp_trim", "scribe_diarize"],
        )
        methods_run = {r["method"] for r in all_results["A"]}
        # scribe_diarize skipped (no diarization support); timestamp_trim errors (no segments)
        assert "scribe_diarize" not in methods_run
        assert {"no_trim", "energy_trim", "timestamp_trim"} == methods_run
        ts_result = next(r for r in all_results["A"] if r["method"] == "timestamp_trim")
        assert ts_result["error"] == "No segments available"
        assert best_a["method"] in {"no_trim", "energy_trim"}
        assert best_a["wer"] == 0.0


class TestResultSelectionHelpers:
    def test_first_valid_skips_errors(self):
        results = [{"error": "boom"}, {"error": None, "text": "ok"}]
        assert _first_valid(results)["text"] == "ok"

    def test_first_valid_all_errors_returns_first(self):
        results = [{"error": "a"}, {"error": "b"}]
        assert _first_valid(results)["error"] == "a"

    def test_first_valid_empty(self):
        assert _first_valid([]) is None

    def test_select_best_oracle_picks_highest(self):
        results = [
            {"error": None, "cer": 0.5, "wer": 0.5, "similarity": 0.5},
            {"error": None, "cer": 0.0, "wer": 0.0, "similarity": 1.0},
        ]
        assert _select_best_oracle(results)["cer"] == 0.0

    def test_select_best_oracle_no_valid_returns_first(self):
        results = [{"error": "x", "cer": None}]
        assert _select_best_oracle(results)["error"] == "x"


class TestAssignWordsToSpeakers:
    SEGMENTS = [
        {"speaker": "S0", "start": 0.0, "end": 2.0},
        {"speaker": "S1", "start": 2.0, "end": 4.0},
    ]

    def test_midpoint_containment(self):
        words = [
            {"text": "hi", "start": 0.5, "end": 1.0},
            {"text": "there", "start": 2.5, "end": 3.0},
        ]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"S0": "hi", "S1": "there"}

    def test_overlap_fallback(self):
        # Midpoint (4.5) is outside every segment; overlap with S1 wins.
        words = [{"text": "late", "start": 3.5, "end": 5.5}]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"S1": "late"}

    def test_orphan_word_goes_to_the_nearest_speaker(self):
        # Was bucketed under "unknown", inventing a speaker that then competed
        # with the real ones for a reference (issue #189).
        words = [{"text": "orphan", "start": 10.0, "end": 11.0}]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"S1": "orphan"}

    def test_word_in_the_gap_between_turns_picks_the_closer_turn(self):
        segments = [
            {"speaker": "S0", "start": 0.0, "end": 2.0},
            {"speaker": "S1", "start": 6.0, "end": 8.0},
        ]
        words = [{"text": "early", "start": 2.4, "end": 2.6}, {"text": "late", "start": 5.4, "end": 5.6}]
        assert assign_words_to_speakers(words, segments) == {"S0": "early", "S1": "late"}

    def test_no_segments_yields_nothing_rather_than_one_speaker(self):
        # The whole transcript under a single phantom speaker is what dropped
        # speaker B from the evaluation in issue #189.
        words = [{"text": "hi", "start": 0.5, "end": 1.0}]
        assert assign_words_to_speakers(words, []) == {}

    def test_concatenates_in_order(self):
        words = [
            {"text": "a", "start": 0.1, "end": 0.2},
            {"text": "b", "start": 1.0, "end": 1.1},
        ]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"S0": "a b"}


class _FakeTurn:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    """Stand-in for ``pyannote.core.Annotation``."""

    def __init__(self, turns):
        self._turns = turns

    def itertracks(self, yield_label=False):
        for start, end, speaker in self._turns:
            yield _FakeTurn(start, end), "_", speaker


# The two non-overlapping turns of the issue's fixture: A 0–10.56, B 10.81–19.57.
_FIXTURE_TURNS = [(0.0, 10.56, "SPEAKER_00"), (10.81, 19.57, "SPEAKER_01")]


def _fake_pipeline(result, recorded=None):
    """A diarization pipeline returning *result*, recording its kwargs."""

    def pipeline(path, **kwargs):
        if recorded is not None:
            recorded.append(kwargs)
        return result

    return pipeline


class TestDiarizationOutputShapes:
    """Issue #189: pyannote.audio 4.x returns a ``DiarizeOutput`` dataclass, not
    an ``Annotation``. Reading only the 3.x shape silently produced zero speech
    turns, collapsing every word onto one speaker."""

    def test_pyannote_4x_diarize_output_is_read(self, monkeypatch):
        # DiarizeOutput has no itertracks of its own — only .speaker_diarization does.
        output = SimpleNamespace(
            speaker_diarization=_FakeAnnotation(_FIXTURE_TURNS),
            exclusive_speaker_diarization=_FakeAnnotation(_FIXTURE_TURNS),
            speaker_embeddings=None,
        )
        assert not hasattr(output, "itertracks")
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(output))

        segments = pyannote_utils.run_diarization(Path("combined.wav"))

        assert [s["speaker"] for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
        assert segments[1]["start"] == pytest.approx(10.81)

    def test_legacy_3x_annotation_still_works(self, monkeypatch):
        annotation = _FakeAnnotation(_FIXTURE_TURNS)
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(annotation))

        segments = pyannote_utils.run_diarization(Path("combined.wav"))

        assert len({s["speaker"] for s in segments}) == 2

    def test_num_speakers_is_passed_through(self, monkeypatch):
        recorded = []
        annotation = _FakeAnnotation(_FIXTURE_TURNS)
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(annotation, recorded))

        pyannote_utils.run_diarization(Path("combined.wav"), num_speakers=2)

        assert recorded == [{"num_speakers": 2}]

    def test_unreadable_shape_raises_instead_of_reporting_no_speakers(self, monkeypatch):
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(object()))

        with pytest.raises(RuntimeError) as excinfo:
            pyannote_utils.run_diarization(Path("combined.wav"))

        text = str(excinfo.value)
        assert "version mismatch" in text
        assert "not a problem with the audio" in text


class TestPyannoteDiarizeFailsLoudly:
    """Issue #189 expected behavior: two speakers in, two scored speakers out —
    or a failure that says so. Never one speaker holding both references'
    words while the other is dropped without an error."""

    WORDS = [
        {"text": "however", "start": 0.5, "end": 1.0},
        {"text": "sentence", "start": 12.0, "end": 12.5},
    ]

    def _model(self, words):
        return SimpleNamespace(
            supports_word_timestamps=True,
            transcribe_with_word_timestamps=lambda path: words,
        )

    def test_two_speakers_in_two_speakers_out(self, monkeypatch):
        output = SimpleNamespace(speaker_diarization=_FakeAnnotation(_FIXTURE_TURNS))
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(output))

        speaker_texts = run_pyannote_diarize("combined.wav", self._model(self.WORDS))

        assert speaker_texts == {"SPEAKER_00": "however", "SPEAKER_01": "sentence"}

    def test_diarization_finding_one_speaker_is_an_error_not_a_score(self, monkeypatch):
        one_speaker = SimpleNamespace(speaker_diarization=_FakeAnnotation([(0.0, 19.57, "SPEAKER_00")]))
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(one_speaker))

        with pytest.raises(RuntimeError) as excinfo:
            run_pyannote_diarize("combined.wav", self._model(self.WORDS))

        text = str(excinfo.value)
        assert "only 1 speaker(s)" in text
        assert "2 were requested" in text

    def test_no_speech_turns_is_an_error(self, monkeypatch):
        empty = SimpleNamespace(speaker_diarization=_FakeAnnotation([]))
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(empty))

        with pytest.raises(RuntimeError, match="no speech turns"):
            run_pyannote_diarize("combined.wav", self._model(self.WORDS))

    def test_missing_word_timestamps_names_the_recorded_cause(self):
        model = SimpleNamespace(
            supports_word_timestamps=True,
            transcribe_with_word_timestamps=lambda path: [],
            last_transcribe_error="401 invalid api key",
        )

        with pytest.raises(RuntimeError) as excinfo:
            run_pyannote_diarize("combined.wav", model)

        assert "401 invalid api key" in str(excinfo.value)


class TestPerClipCapabilityPrecheck:
    """Issue #189: ``supports_word_timestamps`` had no reader, so a model
    without it reached the strategy and failed on a bare NotImplementedError —
    whose str() is "", producing a warning with no reason at all."""

    @pytest.fixture(autouse=True)
    def _pyannote_installed(self, monkeypatch):
        # The capability precheck is about the ASR model, not the extra: these
        # must exercise it whether or not [pyannote] is installed (CI is not).
        monkeypatch.setattr(preprocessing_selector, "PYANNOTE_AVAILABLE", True)

    def _run(self, tmp_path, model, method_name):
        wav = _write_wav(tmp_path / "combined.wav", [("tone", 1.0)])
        return run_single_method(
            entry=_ENTRY,
            asr_model=model,
            ref_a="ref a",
            ref_b="ref b",
            segments=[],
            audio_a=None,
            audio_b=None,
            combined_audio=wav,
            metric_fn=_metric_exact,
            transcribe_fn=lambda path: "hello",
            active_methods=["energy_trim", "no_trim"],
            method_name=method_name,
        )

    def test_model_without_word_timestamps_is_skipped_with_a_reason(self, tmp_path):
        # whisper_base_en's adapter family: inherits the base NotImplementedError.
        model = SimpleNamespace(supports_diarization=False, supports_word_timestamps=False)

        all_results, best_a, best_b = self._run(tmp_path, model, "pyannote_diarize")

        for result in (best_a, best_b):
            assert "does not support pyannote_diarize" in result["error"]
            assert "supports_word_timestamps" in result["error"]
            assert "elevenlabs_api" in result["error"]

    def test_scribe_diarize_still_checks_its_own_capability(self, tmp_path):
        model = SimpleNamespace(supports_diarization=False, supports_word_timestamps=True)

        _, best_a, _ = self._run(tmp_path, model, "scribe_diarize")

        assert "does not support scribe_diarize" in best_a["error"]
        assert "supports_diarization" in best_a["error"]

    def test_capable_model_is_not_skipped(self, tmp_path, monkeypatch):
        output = SimpleNamespace(speaker_diarization=_FakeAnnotation(_FIXTURE_TURNS))
        monkeypatch.setattr(pyannote_utils, "get_diarization_pipeline", lambda: _fake_pipeline(output))
        model = SimpleNamespace(
            supports_word_timestamps=True,
            transcribe_with_word_timestamps=lambda path: [
                {"text": "ref a", "start": 1.0, "end": 2.0},
                {"text": "ref b", "start": 12.0, "end": 13.0},
            ],
        )

        _, best_a, best_b = self._run(tmp_path, model, "pyannote_diarize")

        assert best_a.get("error") is None
        assert best_b.get("error") is None
        assert {best_a["text"], best_b["text"]} == {"ref a", "ref b"}

    def test_sweep_checks_word_timestamps_too(self, tmp_path, caplog):
        # Checking only supports_diarization let this model through to a bare
        # NotImplementedError from the word-timestamp call (issue #189).
        model = SimpleNamespace(supports_diarization=True, supports_word_timestamps=False)
        wav = _write_wav(tmp_path / "combined.wav", [("tone", 1.0)])

        with caplog.at_level("WARNING"):
            all_results, _, _ = run_sweep(
                entry=_ENTRY,
                asr_model=model,
                ref_a="hello",
                ref_b="hello",
                segments=[],
                audio_a=_write_wav(tmp_path / "a.wav", [("tone", 1.0)]),
                audio_b=_write_wav(tmp_path / "b.wav", [("tone", 1.0)]),
                combined_audio=wav,
                metric_fn=_metric_exact,
                transcribe_fn=lambda path: "hello",
                methods=["no_trim", "pyannote_diarize"],
            )

        assert "pyannote_diarize" not in {r["method"] for r in all_results["A"]}
        assert "supports_word_timestamps" in caplog.text

    def test_empty_exception_message_still_names_a_reason(self):
        assert _exc_reason(NotImplementedError()) == "NotImplementedError"
        assert _exc_reason(RuntimeError("boom")) == "boom"


class TestExtractAndConcatSegments:
    def test_extracts_and_shortens(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 4.0)])
        out, orig, trimmed = extract_and_concat_segments(wav, [(0.0, 1.0), (3.0, 4.0)], gap_ms=200)
        assert orig == pytest.approx(4.0, abs=0.05)
        assert trimmed == pytest.approx(2.2, abs=0.1)  # 2s speech + one 0.2s gap
        assert str(out) != str(wav)

    def test_empty_segments_passthrough(self, tmp_path):
        wav = _write_wav(tmp_path / "a.wav", [("tone", 1.0)])
        out, orig, trimmed = extract_and_concat_segments(wav, [])
        assert out == wav
        assert orig == trimmed
