"""Issue #184: scores.json must describe the run that actually happened.

Three legs, same defect class — the artifact asserting things the run
didn't do:

1. A hosted-API run recorded ``provider: local`` / ``region: local`` because
   the submission block was filled from undocumented env vars with a
   ``local`` default, never from the model that ran.
2. The fallback-normalizer caveat was printed at load time but never written
   to the ``warnings`` array, while the script-mismatch warning on the same
   subcommand was — the same claim about metric comparability, auditable in
   one case and invisible in the other.
3. ``prompt_version`` (the LLM-judge rubric hash) was stamped onto every run
   with semantic similarity enabled, though ``compute_sem`` is local
   sentence-transformers similarity and no LLM judge ever ran; and
   ``judge_model`` echoed env vars that were never wired to the judge.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from psdn_sonar.evaluators.single_speaker import (
    SingleSpeakerEvaluator,
    _default_submission_for_model,
)
from psdn_sonar.utils.text_processing import (
    WER_NORMALIZATION_CONTRACTS,
    fallback_normalizer_warning,
)


def _hosted_api_model():
    return SimpleNamespace(provider="openai", provider_model_id="whisper-1")


class _LocalModel:
    """Adapter-shaped stub that, like real local adapters, inherits nothing
    provider-specific — the base-class defaults must apply."""


class TestDefaultSubmissionProvider:
    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch):
        for var in ("SONAR_PROVIDER", "SONAR_REGION", "SONAR_PROTOCOL", "SONAR_JUDGE_MODEL", "GEMINI_MODEL"):
            monkeypatch.delenv(var, raising=False)

    def test_hosted_api_model_records_its_provider(self):
        cfg = _default_submission_for_model(_hosted_api_model(), "whisper_api", language="en")
        assert cfg.provider == "openai"
        assert cfg.model_snapshot == "whisper-1"

    def test_local_model_records_local(self):
        cfg = _default_submission_for_model(_LocalModel(), "whisper_base_en", language="en")
        assert cfg.provider == "local"
        assert cfg.model_snapshot == "whisper_base_en"

    def test_region_is_null_not_local(self):
        """Hosted providers disclose no region and a local run has none; the
        old 'local' default asserted a region nobody measured."""
        for model in (_hosted_api_model(), _LocalModel()):
            assert _default_submission_for_model(model, "m", language="en").region is None

    def test_env_overrides_still_win_and_are_now_documented(self, monkeypatch):
        monkeypatch.setenv("SONAR_PROVIDER", "self-hosted")
        monkeypatch.setenv("SONAR_REGION", "eu-west-1")
        cfg = _default_submission_for_model(_hosted_api_model(), "whisper_api", language="en")
        assert cfg.provider == "self-hosted"
        assert cfg.region == "eu-west-1"

    def test_judge_fields_stay_null_even_with_judge_env_vars_set(self, monkeypatch):
        """SONAR_JUDGE_MODEL/GEMINI_MODEL were read into the record but never
        passed to the judge, so setting them labeled the artifact with a
        judge model that was not used."""
        monkeypatch.setenv("SONAR_JUDGE_MODEL", "gemini-2.5-pro")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
        cfg = _default_submission_for_model(_hosted_api_model(), "whisper_api", language="en")
        assert cfg.judge_model is None
        assert cfg.prompt_version is None


class TestAdapterProviderContract:
    """Class-level attributes, checkable without SDKs or API keys."""

    def test_base_default_is_local(self):
        from psdn_sonar.models.base import ASRModel

        assert ASRModel.provider == "local"
        assert ASRModel.provider_model_id is None

    def test_hosted_adapters_name_their_provider(self):
        from psdn_sonar.models import apis

        assert apis.WhisperAPIModel.provider == "openai"
        assert apis.ElevenLabsAPIModel.provider == "elevenlabs"
        assert apis.AssemblyAIAPIModel.provider == "assemblyai"

    def test_elevenlabs_records_the_requested_model_id(self, monkeypatch):
        from psdn_sonar.models.apis import ElevenLabsAPIModel

        monkeypatch.setenv("ELEVENLABS_API_KEY", "fake-key")
        model = ElevenLabsAPIModel()
        assert model.provider_model_id == "scribe_v2"


class TestFallbackNormalizerWarningText:
    def test_dedicated_languages_are_silent(self):
        for lang in WER_NORMALIZATION_CONTRACTS:
            assert fallback_normalizer_warning(lang) is None

    def test_unsupported_language_names_the_fallback_and_the_alternatives(self):
        warning = fallback_normalizer_warning("sw")
        assert warning is not None
        assert "'sw'" in warning
        assert "generic fallback normalization" in warning
        assert "bn, en, hi, ko" in warning
        assert "psdn-sonar custom" in warning

    def test_case_insensitive(self):
        assert fallback_normalizer_warning("KO") is None
        assert fallback_normalizer_warning("SW") is not None


class TestArtifactDescribesTheRun:
    """End-to-end through run_evaluation to the scores.json payload."""

    @pytest.fixture
    def stubbed_run(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._semantics_dependency_missing", lambda: False)
        for var in ("SONAR_PROVIDER", "SONAR_REGION", "SONAR_JUDGE_MODEL", "GEMINI_MODEL"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "load_data",
            lambda *args, **kwargs: [
                {"audio_path": "clip.wav", "ground_truth": "habari za asubuhi rafiki yangu"},
            ],
        )
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker._model_factory",
            lambda *a, **k: _hosted_api_model(),
        )
        monkeypatch.setattr(
            SingleSpeakerEvaluator,
            "evaluate_one",
            lambda *args, **kwargs: {
                "model_name": "whisper_api",
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

    def _run(self, tmp_path, language, compute_sem=False):
        SingleSpeakerEvaluator.run_evaluation(
            tsv_path="eval.tsv",
            output_dir=str(tmp_path),
            models=["whisper_api"],
            language=language,
            write_scores=True,
            compute_sem=compute_sem,
        )
        return json.loads((tmp_path / "scores_whisper_api.json").read_text(encoding="utf-8"))

    def test_hosted_api_artifact_names_the_provider(self, tmp_path, stubbed_run):
        payload = self._run(tmp_path, "en")
        assert payload["submission"]["provider"] == "openai"
        assert payload["submission"]["model_snapshot"] == "whisper-1"
        assert payload["submission"]["region"] is None
        assert payload["model_name"] == "whisper_api"

    def test_poseidon_run_does_not_claim_llm_judge_metrics(self, tmp_path, stubbed_run, monkeypatch):
        monkeypatch.setenv("SONAR_JUDGE_MODEL", "gemini-2.5-pro")
        payload = self._run(tmp_path, "en", compute_sem=True)
        assert payload["submission"]["prompt_version"] is None
        assert payload["submission"]["judge_model"] is None

    def test_fallback_normalizer_caveat_reaches_the_artifact(self, tmp_path, stubbed_run, caplog):
        with caplog.at_level("WARNING"):
            payload = self._run(tmp_path, "sw")
        expected = fallback_normalizer_warning("sw")
        assert expected in payload["warnings"]
        assert payload["lineage"]["normalization"] == "sw:unversioned"
        # Same wording in the log and the artifact — a reader of either sees
        # the same caveat.
        assert expected in caplog.text

    def test_dedicated_normalizer_leaves_warnings_clean(self, tmp_path, stubbed_run):
        payload = self._run(tmp_path, "en")
        assert payload["warnings"] == []
