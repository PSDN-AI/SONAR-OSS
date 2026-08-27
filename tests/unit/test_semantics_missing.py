"""Issue #191: without the [ml] extra, a run must not report clean success
while semantic_similarity and poseidon_score are silently null.

The missing extra is named in one actionable line (like the other dependency
paths, #169/#177) before any transcription spend, and the reason is recorded
in the scores.json ``warnings`` array so a reader of the artifact alone can
tell the headline metrics are absent and why. A genuine runtime failure in
the semantics block keeps its traceback and is recorded too.
"""

import json
import sys
import types

import numpy as np

from psdn_sonar.evaluators.single_speaker import (
    _SEMANTICS_MISSING_EXTRA_WARNING,
    SingleSpeakerEvaluator,
    _semantics_dependency_missing,
)

ENGLISH_REFS = [
    "the quick brown fox jumps over the lazy dog",
    "speech recognition evaluation needs honest artifacts",
    "a missing metric must say that it is missing",
]


def _hide_sentence_transformers(monkeypatch):
    """Make ``import sentence_transformers`` raise ImportError."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)


def _fake_sentence_transformers(monkeypatch):
    """Install a stand-in module so the import inside the try succeeds."""
    fake = types.ModuleType("sentence_transformers")
    fake.util = types.SimpleNamespace(cos_sim=lambda a, b: np.array([[0.5]]))
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return fake


def _rows_and_pairs():
    row = {
        "semantic_similarity": None,
        "poseidon_score": None,
        "cer": 0.1,
        "wer": 0.2,
    }
    return [dict(row)], [(0, "reference text", "predicted text")]


class TestApplyBatchSemantics:
    def test_missing_extra_returns_the_one_liner_without_traceback(self, monkeypatch, caplog):
        _hide_sentence_transformers(monkeypatch)
        results, pairs = _rows_and_pairs()

        with caplog.at_level("WARNING"):
            warning = SingleSpeakerEvaluator._apply_batch_semantics(results, pairs)

        assert warning == _SEMANTICS_MISSING_EXTRA_WARNING
        assert 'pip install "psdn-sonar[ml]"' in warning
        assert results[0]["semantic_similarity"] is None
        assert results[0]["poseidon_score"] is None
        records = [r for r in caplog.records if "sentence-transformers is not installed" in r.getMessage()]
        assert len(records) == 1
        # Known remedy: one clean line, no traceback (the old behavior was a
        # raw ModuleNotFoundError traceback at WARNING level).
        assert records[0].exc_info is None

    def test_genuine_runtime_failure_keeps_traceback_and_names_the_cause(self, monkeypatch, caplog):
        _fake_sentence_transformers(monkeypatch)

        def explode():
            raise RuntimeError("CUDA out of memory")

        monkeypatch.setattr("psdn_sonar.utils.metrics._get_semantic_model", explode)
        results, pairs = _rows_and_pairs()

        with caplog.at_level("WARNING"):
            warning = SingleSpeakerEvaluator._apply_batch_semantics(results, pairs)

        assert warning is not None
        assert "CUDA out of memory" in warning
        assert warning != _SEMANTICS_MISSING_EXTRA_WARNING
        records = [r for r in caplog.records if "Batch semantic similarity failed" in r.getMessage()]
        assert len(records) == 1
        assert records[0].exc_info is not None

    def test_success_returns_none_and_fills_the_rows(self, monkeypatch):
        _fake_sentence_transformers(monkeypatch)

        fake_model = types.SimpleNamespace(encode=lambda texts, **kwargs: [np.ones(4, dtype=np.float32) for _ in texts])
        monkeypatch.setattr("psdn_sonar.utils.metrics._get_semantic_model", lambda: fake_model)
        results, pairs = _rows_and_pairs()

        warning = SingleSpeakerEvaluator._apply_batch_semantics(results, pairs)

        assert warning is None
        assert results[0]["semantic_similarity"] == 0.5
        assert results[0]["poseidon_score"] is not None

    def test_nothing_to_do_returns_none(self):
        assert SingleSpeakerEvaluator._apply_batch_semantics([], []) is None


class TestDependencyProbe:
    def test_missing_module_is_detected(self, monkeypatch):
        _hide_sentence_transformers(monkeypatch)
        assert _semantics_dependency_missing() is True

    def test_present_module_is_detected(self, monkeypatch):
        _fake_sentence_transformers(monkeypatch)
        assert _semantics_dependency_missing() is False


class TestWarningRecordedInScores:
    """A reader of scores.json alone must be able to tell the headline
    metrics are absent and why (issue #191)."""

    def _stub_run(self, monkeypatch, semantics_warning=None):
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
                "model_name": "elevenlabs_api",
                "results": [],
                "semantics_warning": semantics_warning,
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

    def _run(self, tmp_path):
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["elevenlabs_api"],
            language="en",
            write_scores=True,
            compute_sem=True,
        )
        return json.loads((tmp_path / "scores_elevenlabs_api.json").read_text(encoding="utf-8"))

    def test_missing_extra_is_warned_up_front_and_recorded_once(self, tmp_path, monkeypatch, caplog):
        _hide_sentence_transformers(monkeypatch)
        # evaluate_one reports the same one-liner: it must not be duplicated
        # on top of the preflight entry.
        self._stub_run(monkeypatch, semantics_warning=_SEMANTICS_MISSING_EXTRA_WARNING)

        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path)

        assert payload["warnings"] == [_SEMANTICS_MISSING_EXTRA_WARNING]
        preflight = [r for r in caplog.records if "sentence-transformers is not installed" in r.getMessage()]
        assert len(preflight) == 1

    def test_runtime_semantics_failure_is_recorded_in_the_artifact(self, tmp_path, monkeypatch):
        _fake_sentence_transformers(monkeypatch)
        failure = (
            "semantic_similarity and poseidon_score were not computed: "
            "batch semantic similarity failed (CUDA out of memory). WER/CER are "
            "unaffected; the traceback is in the run log."
        )
        self._stub_run(monkeypatch, semantics_warning=failure)

        payload = self._run(tmp_path)

        assert payload["warnings"] == [failure]

    def test_healthy_run_keeps_the_artifact_clean(self, tmp_path, monkeypatch):
        _fake_sentence_transformers(monkeypatch)
        self._stub_run(monkeypatch, semantics_warning=None)

        payload = self._run(tmp_path)

        assert payload["warnings"] == []

    def test_without_compute_sem_no_preflight_warning(self, tmp_path, monkeypatch, caplog):
        _hide_sentence_transformers(monkeypatch)
        self._stub_run(monkeypatch, semantics_warning=None)

        with caplog.at_level("WARNING"):
            SingleSpeakerEvaluator.run_evaluation(
                tsv_path="eval.tsv",
                output_dir=str(tmp_path),
                models=["elevenlabs_api"],
                language="en",
                write_scores=True,
                compute_sem=False,
            )

        assert "sentence-transformers is not installed" not in caplog.text
        payload = json.loads((tmp_path / "scores_elevenlabs_api.json").read_text(encoding="utf-8"))
        assert payload["warnings"] == []
