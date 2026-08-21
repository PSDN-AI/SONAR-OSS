"""Tests for the run lineage recorded in scores.json (issue #120).

Published numbers could not be compared against a reproduction because the
checkpoint revision and normalization rule set behind them were recorded
nowhere. These tests anchor the lineage block that closes that gap.
"""

import json
from types import SimpleNamespace

from psdn_sonar.benchmark.scores import RunLineage, build_run_scores, write_scores_json
from psdn_sonar.benchmark.submission import SubmissionConfig
from psdn_sonar.evaluators.single_speaker import _run_lineage
from psdn_sonar.models.base import ASRModel
from psdn_sonar.utils import text_processing
from psdn_sonar.utils.text_processing import wer_normalization_contract

FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"


def _fake_config(model_id="org/model", sha=FAKE_SHA):
    return SimpleNamespace(_name_or_path=model_id, _commit_hash=sha)


def _submission():
    return SubmissionConfig.from_env(
        provider="local",
        model_snapshot="fake_model",
        region="local",
        inference_params={"language_code": "bn"},
    )


class TestGetHfLineage:
    def test_direct_model_attribute(self):
        model = ASRModel()
        model.model = SimpleNamespace(config=_fake_config())
        assert model.get_hf_lineage() == ("org/model", FAKE_SHA)

    def test_pipeline_attribute(self):
        model = ASRModel()
        model.pipe = SimpleNamespace(model=SimpleNamespace(config=_fake_config()))
        assert model.get_hf_lineage() == ("org/model", FAKE_SHA)

    def test_no_checkpoint_returns_none_pair(self):
        assert ASRModel().get_hf_lineage() == (None, None)

    def test_partial_config_returns_what_exists(self):
        model = ASRModel()
        model.model = SimpleNamespace(config=SimpleNamespace(_name_or_path="org/model", _commit_hash=None))
        assert model.get_hf_lineage() == ("org/model", None)

    def test_non_string_values_ignored(self):
        model = ASRModel()
        model.model = SimpleNamespace(config=SimpleNamespace(_name_or_path=123, _commit_hash=b"raw"))
        assert model.get_hf_lineage() == (None, None)


class TestWerNormalizationContract:
    def test_versioned_languages(self):
        assert wer_normalization_contract("en") == "en:v1"
        assert wer_normalization_contract("hi") == "hi:v1"
        assert wer_normalization_contract("ko") == "ko:v1"

    def test_case_insensitive(self):
        assert wer_normalization_contract("EN") == "en:v1"

    def test_unversioned_language(self):
        assert wer_normalization_contract("pt") == "pt:unversioned"
        assert wer_normalization_contract("") == ":unversioned"

    def test_bengali_marks_bnlp_availability(self, monkeypatch):
        # bn is at v2 since symbol verbalization was added (issue #136).
        monkeypatch.setattr(text_processing, "_BNLP_TOKENIZER_AVAILABLE", True)
        assert wer_normalization_contract("bn") == "bn:v2+bnlp"
        monkeypatch.setattr(text_processing, "_BNLP_TOKENIZER_AVAILABLE", False)
        assert wer_normalization_contract("bn") == "bn:v2-bnlp"


class TestRunLineageHelper:
    def test_populates_from_model(self):
        model = ASRModel()
        model.model = SimpleNamespace(config=_fake_config())
        lineage = _run_lineage(model, "en")
        assert lineage.hf_model_id == "org/model"
        assert lineage.hf_revision == FAKE_SHA
        assert lineage.normalization == "en:v1"

    def test_never_raises_for_hostile_doubles(self):
        class Broken:
            def get_hf_lineage(self):
                raise RuntimeError("boom")

        class WrongShape:
            def get_hf_lineage(self):
                return "not-a-tuple"

        class NoMethod:
            pass

        for double in (Broken(), WrongShape(), NoMethod()):
            lineage = _run_lineage(double, "ko")
            assert lineage.hf_model_id is None
            assert lineage.hf_revision is None
            assert lineage.normalization == "ko:v1"


class TestScoresArtifactLineage:
    def test_lineage_serialized_into_scores_json(self, tmp_path):
        lineage = RunLineage(hf_model_id="org/model", hf_revision=FAKE_SHA, normalization="bn:v2+bnlp")
        artifact = build_run_scores(
            _submission(),
            {"summary": {"successful": 1, "total_samples": 1, "failed": 0, "elapsed_time": 1.0}, "results": []},
            lineage=lineage,
        )
        out = write_scores_json(tmp_path / "scores_fake.json", artifact)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["lineage"] == {
            "hf_model_id": "org/model",
            "hf_revision": FAKE_SHA,
            "normalization": "bn:v2+bnlp",
        }

    def test_lineage_defaults_to_null(self, tmp_path):
        artifact = build_run_scores(
            _submission(),
            {"summary": {"successful": 1, "total_samples": 1, "failed": 0, "elapsed_time": 1.0}, "results": []},
        )
        out = write_scores_json(tmp_path / "scores_fake.json", artifact)
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["lineage"] is None
