"""Tests for preprocessing selection and the pyannote-independent helpers."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from psdn_sonar.preprocessing.preprocessing_selector import (
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
        (all_results, best_a, best_b), calls = self._call(tmp_path, method_name="scribe_diarize")
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

    def test_unknown_when_no_overlap(self):
        words = [{"text": "orphan", "start": 10.0, "end": 11.0}]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"unknown": "orphan"}

    def test_concatenates_in_order(self):
        words = [
            {"text": "a", "start": 0.1, "end": 0.2},
            {"text": "b", "start": 1.0, "end": 1.1},
        ]
        assert assign_words_to_speakers(words, self.SEGMENTS) == {"S0": "a b"}


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
