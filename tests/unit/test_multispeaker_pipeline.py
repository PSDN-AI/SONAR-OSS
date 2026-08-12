"""Tests for the multi-speaker evaluation pipeline."""

import pytest

from psdn_sonar.multispeaker_pipeline import run_multispeaker_evaluation


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text('{"audio_id": "a1"}\n')
    return path


class TestRunMultispeakerEvaluation:
    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Manifest file not found"):
            run_multispeaker_evaluation(str(tmp_path / "missing.jsonl"), "whisper_api")

    def test_unknown_model_raises(self, manifest):
        with pytest.raises(ValueError, match="Unknown model"):
            run_multispeaker_evaluation(str(manifest), "not_a_model")

    def test_runs_evaluation_and_returns_csv_path(self, manifest, tmp_path, monkeypatch):
        captured = {}

        def fake_process(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", fake_process)
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name: f"model:{name}")

        output_dir = tmp_path / "out"
        result = run_multispeaker_evaluation(
            str(manifest),
            "whisper_api",
            output_dir=str(output_dir),
            max_samples=3,
            sweep=True,
            method="energy_trim",
        )

        assert result == output_dir / "asr_eval_results_whisper_api_manifest.csv"
        assert output_dir.is_dir()
        assert captured["manifest_path"] == str(manifest)
        assert captured["asr_model"] == "model:whisper_api"
        assert captured["asr_model_name"] == "whisper_api"
        assert captured["max_samples"] == 3
        assert captured["sweep"] is True
        assert captured["method"] == "energy_trim"
        assert captured["methods"] == captured["config_settings"]["methods"]

    def test_methods_override_config(self, manifest, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: captured.update(kw))
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name: object())

        run_multispeaker_evaluation(
            str(manifest),
            "whisper_api",
            output_dir=str(tmp_path / "out"),
            methods=["energy_trim", "no_trim"],
        )

        assert captured["methods"] == ["energy_trim", "no_trim"]
        assert captured["config_settings"]["methods"] == ["energy_trim", "no_trim"]
