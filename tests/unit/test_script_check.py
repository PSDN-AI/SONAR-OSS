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
from psdn_sonar.language.script_check import script_mismatch_warning

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
