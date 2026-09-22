"""Tests for the reference-script vs --language mismatch check (issue #148).

A supported --language applied to data in a different language used to run
with zero warnings and produce a complete, healthy-looking scorecard. The
check warns when a clear majority of script-bearing reference characters
belong to a script other than the one the selected language is written in,
and the warning is recorded in scores.json.
"""

import json

import pytest

from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator
from psdn_sonar.language.script_check import (
    hypothesis_row_mismatch_warning,
    hypothesis_row_script_warnings,
    hypothesis_script_mismatch_warning,
    script_mismatch_warning,
)

ENGLISH_REFS = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
]
BENGALI_REFS = ["আমি বাংলায় গান গাই", "এটি একটি পরীক্ষা বাক্য"]
HINDI_REFS = ["मैं हिंदी में बोलता हूँ", "यह एक परीक्षण वाक्य है"]
KOREAN_REFS = ["나는 한국어로 말합니다", "이것은 테스트 문장입니다"]


class TestScriptMismatchWarning:
    def test_english_data_with_korean_code_warns(self):
        warning = script_mismatch_warning(ENGLISH_REFS, "ko")
        assert warning is not None
        assert "Latin script" in warning
        assert "Hangul" in warning
        assert "--language 'ko'" in warning
        assert "en" in warning  # suggests the matching language

    def test_matching_language_is_silent(self):
        assert script_mismatch_warning(ENGLISH_REFS, "en") is None
        assert script_mismatch_warning(BENGALI_REFS, "bn") is None
        assert script_mismatch_warning(HINDI_REFS, "hi") is None
        assert script_mismatch_warning(KOREAN_REFS, "ko") is None

    @pytest.mark.parametrize(
        "refs,wrong_language,expected_hint",
        [
            (BENGALI_REFS, "en", "bn"),
            (HINDI_REFS, "ko", "hi"),
            (KOREAN_REFS, "hi", "ko"),
            (ENGLISH_REFS, "bn", "en"),
        ],
    )
    def test_cross_script_pairs_warn_with_hint(self, refs, wrong_language, expected_hint):
        warning = script_mismatch_warning(refs, wrong_language)
        assert warning is not None
        assert expected_hint in warning

    def test_code_switched_majority_script_is_silent(self):
        # Devanagari majority with Latin loanwords: legitimate Hindi
        # code-switching must not trip the warning as long as the expected
        # script keeps the majority.
        refs = ["मैंने कल अपने दोस्त के साथ बाजार जाकर एक नई app download की और उसका review लिखा"]
        assert script_mismatch_warning(refs, "hi") is None

    def test_too_little_text_is_silent(self):
        assert script_mismatch_warning(["hi"], "ko") is None
        assert script_mismatch_warning([], "ko") is None
        assert script_mismatch_warning([""], "bn") is None

    def test_language_without_implied_script_is_silent(self):
        # Recognized codes without a dedicated normalizer already get the
        # fallback warning; no script is implied for them.
        assert script_mismatch_warning(ENGLISH_REFS, "sw") is None

    def test_numbers_and_punctuation_are_neutral(self):
        assert script_mismatch_warning(["1234567890 !!! ... 42%"], "ko") is None

    def test_long_name_casing_handled(self):
        warning = script_mismatch_warning(ENGLISH_REFS, "KO")
        assert warning is not None


class TestHypothesisScriptMismatchWarning:
    """Issue #207: the reference-side check reads references only, so a
    service returning transcripts in a different writing system (observed:
    Devanagari for --language bn) floored every row at WER 1.0 with empty
    warnings — indistinguishable from a model that transcribed badly."""

    def test_devanagari_hypotheses_under_bn_warn(self):
        warning = hypothesis_script_mismatch_warning(HINDI_REFS, "bn", "assemblyai_api")
        assert warning is not None
        assert "Model 'assemblyai_api'" in warning
        assert "Devanagari" in warning
        assert "--language 'bn'" in warning
        assert "Bengali script" in warning
        assert "not transcription accuracy" in warning

    @pytest.mark.parametrize(
        "hyps,language",
        [
            (ENGLISH_REFS, "en"),
            (BENGALI_REFS, "bn"),
            (HINDI_REFS, "hi"),
            (KOREAN_REFS, "ko"),
        ],
    )
    def test_matching_script_is_silent(self, hyps, language):
        assert hypothesis_script_mismatch_warning(hyps, language, "m") is None

    def test_language_without_implied_script_is_silent(self):
        assert hypothesis_script_mismatch_warning(HINDI_REFS, "sw", "m") is None

    def test_all_failed_run_with_empty_predictions_is_silent(self):
        # An all-failed run has empty prediction cells; those carry no
        # script signal and already have their own error reporting.
        assert hypothesis_script_mismatch_warning(["", "", ""], "bn", "m") is None

    def test_too_little_text_is_silent(self):
        assert hypothesis_script_mismatch_warning(["नमस्ते"], "bn", "m") is None

    def test_code_switched_majority_expected_script_is_silent(self):
        hyps = ["আমি গতকাল আমার বন্ধুর সাথে বাজারে গিয়ে একটি নতুন app download করে তার review লিখেছি"]
        assert hypothesis_script_mismatch_warning(hyps, "bn", "m") is None


# Five Bengali predictions of which index 1 came back entirely in Latin
# script — the run shape from issue #292, where the batch-level majority
# (Bengali) hides the one foreign row.
ONE_FOREIGN_ROW = [
    "আমি বাংলায় গান গাই এবং কবিতা লিখি",
    "hello world this is a test transcript in english",
    "এটি একটি পরীক্ষা বাক্য যা যাচাই করা হবে",
    "আজ সকালে আবহাওয়া খুব সুন্দর ছিল",
    "আমরা আগামীকাল স্কুলে যাব এবং পড়াশোনা করব",
]


class TestPerRowScriptMarkers:
    """Issue #292: the run-level majority is computed across the batch while
    the condition happens per transcript — one row in another writing system
    left no trace. The per-row scan marks exactly that row."""

    def test_the_issues_run_shape_batch_silent_row_marked(self):
        # The gap the issue demonstrates, both halves: the batch-level check
        # stays silent, the per-row scan flags exactly the foreign row.
        assert hypothesis_script_mismatch_warning(ONE_FOREIGN_ROW, "bn", "elevenlabs_api") is None

        markers = hypothesis_row_script_warnings(ONE_FOREIGN_ROW, "bn")
        assert [m is not None for m in markers] == [False, True, False, False, False]
        assert "Latin script" in markers[1]
        assert "100%" in markers[1]
        assert "--language 'bn'" in markers[1]

    def test_summary_names_the_count_and_the_column(self):
        markers = hypothesis_row_script_warnings(ONE_FOREIGN_ROW, "bn")
        warning = hypothesis_row_mismatch_warning(markers, "bn", "elevenlabs_api", "asr_detailed_elevenlabs_api.csv")
        assert warning is not None
        assert "Model 'elevenlabs_api'" in warning
        assert "1 of 5" in warning
        assert "script_warning column of asr_detailed_elevenlabs_api.csv" in warning
        assert "not transcription accuracy" in warning

    def test_clean_run_marks_nothing(self):
        markers = hypothesis_row_script_warnings(BENGALI_REFS, "bn")
        assert markers == [None, None]
        assert hypothesis_row_mismatch_warning(markers, "bn", "m", "x.csv") is None

    def test_short_foreign_row_is_too_small_to_call(self):
        # Under _MIN_SCRIPT_CHARS script-bearing characters the row-level
        # gate stays silent, same as the run-level one.
        markers = hypothesis_row_script_warnings(["আমি বাংলায় গান গাই এবং কবিতা লিখি", "hi ok"], "bn")
        assert markers == [None, None]

    def test_code_switched_single_row_is_not_flagged(self):
        # Mixing inside one transcript is code-switching, not a foreign row.
        row = "আমি গতকাল আমার বন্ধুর সাথে বাজারে গিয়ে একটি নতুন app download করে তার review লিখেছি"
        assert hypothesis_row_script_warnings([row], "bn") == [None]

    def test_language_without_implied_script_marks_nothing(self):
        markers = hypothesis_row_script_warnings(ONE_FOREIGN_ROW, "sw")
        assert markers == [None] * len(ONE_FOREIGN_ROW)

    def test_empty_predictions_mark_nothing(self):
        assert hypothesis_row_script_warnings(["", "", ""], "bn") == [None, None, None]


class TestWarningRecordedInScores:
    """The warning must reach both the log and scores.json (issue #148: the
    artifact of a wrong-language run was indistinguishable from a correct
    one)."""

    @pytest.fixture
    def stubbed_run(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *args, **kwargs: [{"audio_path": "clip.wav", "ground_truth": ref} for ref in ENGLISH_REFS],
        )
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", lambda *a, **k: object())
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: {
                "model_name": "whisper_base_en",
                "results": [],
                "summary": {
                    "total_samples": 1,
                    "successful": 1,
                    "failed": 0,
                    "avg_wer": 0.1,
                    "avg_cer": 0.05,
                    "elapsed_time": 0.1,
                    "avg_latency_s": None,
                    "median_latency_s": None,
                    "p95_latency_s": None,
                },
            },
        )

    def _run(self, tmp_path, language):
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["whisper_base_en"],
            language=language,
            write_scores=True,
            compute_sem=False,
        )
        payload = json.loads((tmp_path / "scores_whisper_base_en.json").read_text(encoding="utf-8"))
        return payload

    def test_wrong_language_warns_and_marks_artifact(self, tmp_path, stubbed_run, caplog):
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path, "ko")
        assert "Hangul" in caplog.text
        assert len(payload["warnings"]) == 1
        assert "--language 'ko'" in payload["warnings"][0]

    def test_correct_language_leaves_artifact_clean(self, tmp_path, stubbed_run, caplog):
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path, "en")
        assert "script-bearing" not in caplog.text
        assert payload["warnings"] == []


class TestHypothesisWarningRecordedInScores:
    """End-to-end through run_evaluation: references in the right script,
    predictions in another — the run the issue #207 artifacts could not
    distinguish from a badly transcribing model."""

    def _stub(self, monkeypatch, predictions):
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *args, **kwargs: [
                {"audio_path": f"clip{i}.wav", "ground_truth": ref} for i, ref in enumerate(BENGALI_REFS)
            ],
        )
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", lambda *a, **k: object())
        results = [
            {"audio_path": f"clip{i}.wav", "ground_truth": ref, "prediction": pred, "wer": 1.0, "error": ""}
            for i, (ref, pred) in enumerate(zip(BENGALI_REFS, predictions))
        ]
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: {
                "model_name": "assemblyai_api",
                "results": results,
                "summary": {
                    "total_samples": len(results),
                    "successful": len(results),
                    "failed": 0,
                    "avg_wer": 1.0,
                    "avg_cer": 0.88,
                    "elapsed_time": 0.1,
                    "avg_latency_s": None,
                    "median_latency_s": None,
                    "p95_latency_s": None,
                },
            },
        )

    def _run(self, tmp_path):
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["assemblyai_api"],
            language="bn",
            write_scores=True,
            compute_sem=False,
        )
        return json.loads((tmp_path / "scores_assemblyai_api.json").read_text(encoding="utf-8"))

    def test_wrong_script_hypotheses_warn_and_mark_artifact(self, tmp_path, monkeypatch, caplog):
        self._stub(monkeypatch, HINDI_REFS)  # Devanagari predictions for Bengali references
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path)

        assert "Model 'assemblyai_api'" in caplog.text
        hyp_warnings = [w for w in payload["warnings"] if "returned transcripts" in w]
        assert len(hyp_warnings) == 1
        assert "Devanagari" in hyp_warnings[0]
        assert "--language 'bn'" in hyp_warnings[0]
        # The reference-side warning must not fire: the references are fine.
        assert not any(w.startswith("Reference transcriptions") for w in payload["warnings"])

    def test_matching_script_hypotheses_leave_artifact_clean(self, tmp_path, monkeypatch, caplog):
        self._stub(monkeypatch, BENGALI_REFS)
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path)
        assert "returned transcripts" not in caplog.text
        assert payload["warnings"] == []


class TestSingleForeignRowRecordedInArtifact:
    """End-to-end through run_evaluation for issue #292: one transcript of
    five comes back in Latin script under --language bn. The batch-level
    check is silent by design; the artifact must still mark the row and
    carry a run warning."""

    BENGALI_ONLY = [text for text in ONE_FOREIGN_ROW if "hello" not in text] + ["তিনি প্রতিদিন সকালে খবরের কাগজ পড়েন"]

    def _stub(self, monkeypatch, predictions):
        refs = self.BENGALI_ONLY
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *args, **kwargs: [{"audio_path": f"clip{i}.wav", "ground_truth": ref} for i, ref in enumerate(refs)],
        )
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", lambda *a, **k: object())
        results = [
            {"audio_path": f"clip{i}.wav", "ground_truth": ref, "prediction": pred, "wer": 0.1, "error": ""}
            for i, (ref, pred) in enumerate(zip(refs, predictions))
        ]
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: {
                "model_name": "elevenlabs_api",
                "results": results,
                "summary": {
                    "total_samples": len(results),
                    "successful": len(results),
                    "failed": 0,
                    "avg_wer": 0.1,
                    "avg_cer": 0.05,
                    "elapsed_time": 0.1,
                    "avg_latency_s": None,
                    "median_latency_s": None,
                    "p95_latency_s": None,
                },
            },
        )

    def _run(self, tmp_path):
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["elevenlabs_api"],
            language="bn",
            write_scores=True,
            compute_sem=False,
        )
        payload = json.loads((tmp_path / "scores_elevenlabs_api.json").read_text(encoding="utf-8"))
        import csv

        with open(tmp_path / "asr_detailed_elevenlabs_api.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return payload, rows

    def test_one_foreign_row_is_marked_and_the_run_warns(self, tmp_path, monkeypatch, caplog):
        self._stub(monkeypatch, ONE_FOREIGN_ROW)
        with caplog.at_level("WARNING"):
            payload, rows = self._run(tmp_path)

        row_warnings = [w for w in payload["warnings"] if "1 of 5" in w]
        assert len(row_warnings) == 1
        assert "script_warning column of asr_detailed_elevenlabs_api.csv" in row_warnings[0]
        assert "1 of 5" in caplog.text
        # The batch-level warning must not also fire: the run majority is fine.
        assert not any("returned transcripts that look like" in w for w in payload["warnings"])

        assert [bool(row["script_warning"]) for row in rows] == [False, True, False, False, False]
        assert "foreign_script: Latin script" in rows[1]["script_warning"]

    def test_whole_run_foreign_keeps_the_batch_warning_and_marks_every_row(self, tmp_path, monkeypatch, caplog):
        # The #207 case: every prediction foreign. The batch-level warning
        # fires (not the row summary), and every row carries its marker.
        latin_rows = ["hello world this is a test transcript in english"] * 5
        self._stub(monkeypatch, latin_rows)
        with caplog.at_level("WARNING"):
            payload, rows = self._run(tmp_path)

        assert any("returned transcripts that look like" in w for w in payload["warnings"])
        assert not any("5 of 5" in w for w in payload["warnings"])
        assert all("foreign_script: Latin script" in row["script_warning"] for row in rows)

    def test_clean_run_stays_clean(self, tmp_path, monkeypatch, caplog):
        self._stub(monkeypatch, self.BENGALI_ONLY)
        with caplog.at_level("WARNING"):
            payload, rows = self._run(tmp_path)

        assert payload["warnings"] == []
        assert all(row["script_warning"] == "" for row in rows)
