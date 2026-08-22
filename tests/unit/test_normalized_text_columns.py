"""Regression tests for issue #143: the per-utterance CSV must expose the
normalized text WER/CER were computed over.

Before the fix, the single-speaker path wrote no normalized text at all —
a reference poisoned by an invisible character (e.g. a zero-width space)
was undiagnosable from the artifact — while the docstring on
``normalize_bengali_for_wer`` promised ``normalized_reference`` /
``normalized_hypothesis`` columns that no path produced.
"""

from unittest.mock import MagicMock

import pytest

from psdn_sonar.evaluators.single_speaker import _EMPTY_AUDIO_QUALITY, SingleSpeakerEvaluator
from psdn_sonar.utils.text_processing import normalize_text_unified


@pytest.fixture
def hermetic_audio_quality(monkeypatch):
    """Skip real audio decoding: return empty audio-quality metrics."""
    monkeypatch.setattr(
        SingleSpeakerEvaluator,
        "_compute_audio_quality",
        lambda item: (item["audio_path"], dict(_EMPTY_AUDIO_QUALITY)),
    )


def _evaluate(tmp_path, reference: str, hypothesis: str, language: str = "en"):
    wav = tmp_path / "clip.wav"
    wav.write_bytes(b"")
    model = MagicMock()
    model.transcribe.return_value = hypothesis
    result = SingleSpeakerEvaluator.evaluate_one(
        model=model,
        data=[{"audio_path": str(wav), "ground_truth": reference}],
        model_name="stub",
        language=language,
    )
    assert len(result["results"]) == 1
    return result["results"][0]


class TestCsvSchema:
    def test_normalized_columns_follow_prediction(self):
        fields = SingleSpeakerEvaluator._csv_fieldnames()
        i = fields.index("prediction")
        assert fields[i + 1 : i + 3] == ["normalized_reference", "normalized_hypothesis"]

    def test_documented_column_names_exist_on_the_single_path(self):
        """The names the normalize_bengali_for_wer docstring commits to."""
        fields = SingleSpeakerEvaluator._csv_fieldnames()
        assert "normalized_reference" in fields
        assert "normalized_hypothesis" in fields


class TestScoredRows:
    def test_row_carries_exactly_what_was_scored(self, tmp_path, hermetic_audio_quality):
        row = _evaluate(tmp_path, "Hello, World!", "hello world")
        assert row["wer"] == 0.0
        assert row["normalized_reference"] == normalize_text_unified("Hello, World!", language="en")
        assert row["normalized_hypothesis"] == normalize_text_unified("hello world", language="en")

    def test_invisible_character_is_diagnosable_from_the_artifact(self, tmp_path, hermetic_audio_quality):
        """The issue's motivating case: a zero-width space makes two visually
        identical references score differently. The artifact must expose the
        scored string so the difference is findable without re-running
        normalization by hand."""
        clean = "hello world"
        poisoned = "hello\u200b world"
        row = _evaluate(tmp_path, poisoned, "hello world")
        assert row["normalized_reference"] == normalize_text_unified(poisoned, language="en")
        # Whatever the normalizer does with the ZWSP, the artifact shows it:
        # the scored string is on the row, not reconstructable-only.
        assert row["normalized_hypothesis"] == normalize_text_unified(clean, language="en")

    def test_uncomputable_row_still_carries_the_pair(self, tmp_path, hermetic_audio_quality):
        """A reference that normalizes to empty makes CER/WER None (issue
        #107). The normalized pair must still be written — the empty
        normalized_reference IS the diagnosis."""
        row = _evaluate(tmp_path, "!!! ...", "hello world")
        assert row["wer"] is None and row["cer"] is None
        assert row["error"] is not None
        assert row["normalized_reference"] == ""
        assert row["normalized_hypothesis"] == normalize_text_unified("hello world", language="en")

    def test_unscored_rows_have_empty_normalized_columns(self, tmp_path, hermetic_audio_quality):
        """Rows where scoring never ran (empty prediction) leave the columns
        empty rather than fabricating text."""
        row = _evaluate(tmp_path, "hello world", "")
        assert row["error"] == "Empty prediction"
        assert row["normalized_reference"] == ""
        assert row["normalized_hypothesis"] == ""
