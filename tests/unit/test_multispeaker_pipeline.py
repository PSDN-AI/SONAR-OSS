"""Tests for the multi-speaker evaluation pipeline."""

import os

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
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: f"model:{name}")

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
        assert captured["language"] == "bn"
        assert captured["methods"] == captured["config_settings"]["methods"]

    def test_methods_override_config(self, manifest, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: captured.update(kw))
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        run_multispeaker_evaluation(
            str(manifest),
            "whisper_api",
            output_dir=str(tmp_path / "out"),
            methods=["energy_trim", "no_trim"],
        )

        assert captured["methods"] == ["energy_trim", "no_trim"]
        assert captured["config_settings"]["methods"] == ["energy_trim", "no_trim"]

    def test_config_path_reaches_the_loader(self, manifest, tmp_path, monkeypatch):
        """Issue #210: ``load_multi_speaker_config`` always took a path, but
        nothing passed one, so editing the file inside the installed package
        was the only way to change the method list or the trim settings."""
        captured = {}
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: captured.update(kw))
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        cfg = tmp_path / "preprocessing.yaml"
        cfg.write_text("methods:\n  - energy_trim\n  - no_trim\ntimestamp:\n  padding_ms: 250\n")

        run_multispeaker_evaluation(
            str(manifest),
            "whisper_api",
            output_dir=str(tmp_path / "out"),
            config_path=str(cfg),
        )

        assert captured["methods"] == ["energy_trim", "no_trim"]
        assert captured["config_settings"]["timestamp"]["padding_ms"] == 250

    @pytest.mark.parametrize(
        "override",
        [
            {"methods": ["no_trim"], "method": "not_a_method"},
            {"methods": ["energy_trim"], "method": "no_trim"},
        ],
    )
    def test_method_and_methods_together_raise(self, manifest, tmp_path, monkeypatch, override):
        """The entry point resolves the override; ``process_manifest_with_asr``
        resolves a pin as the active set. Accepting both let the two layers
        disagree — a list passed beside a pin was dropped without a word, and
        an unknown pin passed validation because the list beside it was fine.
        """
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: None)
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        with pytest.raises(ValueError, match="not both"):
            run_multispeaker_evaluation(str(manifest), "whisper_api", output_dir=str(tmp_path / "out"), **override)

    def test_empty_methods_list_raises(self, manifest, tmp_path, monkeypatch):
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: None)
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        with pytest.raises(ValueError, match="must not be empty"):
            run_multispeaker_evaluation(str(manifest), "whisper_api", output_dir=str(tmp_path / "out"), methods=[])

    @pytest.mark.parametrize("override", [{"methods": ["no_trim"]}, {"method": "no_trim"}])
    def test_override_survives_a_config_whose_method_list_is_unusable(self, manifest, tmp_path, monkeypatch, override):
        """The help says --methods/--method override the config's list, so the
        list being unusable must not stop the run before the override applies.
        The config's settings still take effect."""
        captured = {}
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: captured.update(kw))
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        cfg = tmp_path / "c.yaml"
        cfg.write_text("methods:\n  - bogus\nsilence:\n  silence_thresh: -35\n")

        run_multispeaker_evaluation(
            str(manifest),
            "whisper_api",
            output_dir=str(tmp_path / "out"),
            config_path=str(cfg),
            **override,
        )

        assert captured["config_settings"]["silence"]["silence_thresh"] == -35
        if "methods" in override:
            assert captured["methods"] == ["no_trim"]
        else:
            assert captured["method"] == "no_trim"

    def test_unusable_config_method_list_without_an_override_still_raises(self, manifest, tmp_path, monkeypatch):
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: None)
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        cfg = tmp_path / "c.yaml"
        cfg.write_text("methods:\n  - bogus\n")

        with pytest.raises(ValueError, match="no known methods"):
            run_multispeaker_evaluation(
                str(manifest), "whisper_api", output_dir=str(tmp_path / "out"), config_path=str(cfg)
            )

    @pytest.mark.parametrize(
        "override",
        [
            {"methods": ["not_a_method"]},
            {"methods": ["no_trim", "nope"]},
            {"method": "not_a_method"},
            # "" is falsy but not None, so it used to skip validation here and
            # still count as an explicit method downstream.
            {"method": ""},
        ],
    )
    def test_unknown_method_in_an_override_raises(self, manifest, tmp_path, monkeypatch, override):
        """``--method`` and ``--methods`` both replace the config's list, so
        both bypass the loader's ``KNOWN_METHODS`` check. Only the list was
        validated: an unknown single name became the sole active method, ran
        the whole evaluation, and failed every row with "No per-channel methods
        available" — which names the wrong problem — before the generic "no
        clips were successfully processed" at the end.
        """
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: None)
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", lambda name, **kwargs: object())

        with pytest.raises(ValueError, match="Unknown preprocessing method"):
            run_multispeaker_evaluation(
                str(manifest),
                "whisper_api",
                output_dir=str(tmp_path / "out"),
                **override,
            )

    def test_loads_env_before_model_creation(self, manifest, tmp_path, monkeypatch):
        """load_env() must run before create_model so API adapters see .env keys (issue #167)."""
        calls = []
        monkeypatch.setattr("psdn_sonar.multispeaker_pipeline.load_env", lambda: calls.append("load_env"))
        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: calls.append("process"))
        monkeypatch.setattr(
            "psdn_sonar.models.registry.create_model",
            lambda name, **kw: (calls.append("create_model"), "model")[1],
        )

        run_multispeaker_evaluation(str(manifest), "whisper_api", output_dir=str(tmp_path / "out"))

        assert "load_env" in calls
        assert calls.index("load_env") < calls.index("create_model")
        assert calls.index("create_model") < calls.index("process")

    def test_dotenv_credential_reaches_api_adapter(self, manifest, tmp_path, monkeypatch):
        """End-to-end repro of issue #167: a key set only in .env (not the shell)
        must reach the real ElevenLabs adapter instead of raising 'API key not found'."""
        monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
        monkeypatch.delenv("XI_API_KEY", raising=False)
        monkeypatch.delenv("HF_TOKEN", raising=False)
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".env").write_text("ELEVENLABS_API_KEY=fake-key-from-dotenv\nHF_TOKEN=fake-hf-token-from-dotenv\n")

        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: None)

        # Pre-fix this raised ValueError("ElevenLabs API key not found...") from
        # the adapter's __init__ despite .env sitting in the working directory.
        run_multispeaker_evaluation(str(manifest), "elevenlabs_api", output_dir=str(tmp_path / "out"))

        assert os.environ["ELEVENLABS_API_KEY"] == "fake-key-from-dotenv"
        # HF_TOKEN is read later by pyannote VAD/diarization via os.getenv, so
        # loading .env here also fixes the 401-with-valid-token failure mode.
        assert os.environ["HF_TOKEN"] == "fake-hf-token-from-dotenv"

    def test_custom_hf_model_and_language_forwarded(self, manifest, tmp_path, monkeypatch):
        captured = {}
        created = {}

        def fake_create(name, **kwargs):
            created["name"] = name
            created.update(kwargs)
            return "custom-model"

        monkeypatch.setattr("psdn_sonar.core.process_manifest_with_asr", lambda **kw: captured.update(kw))
        monkeypatch.setattr("psdn_sonar.models.registry.create_model", fake_create)

        run_multispeaker_evaluation(
            str(manifest),
            "custom_openai_whisper_tiny",
            output_dir=str(tmp_path / "out"),
            language="en",
            custom_hf_model="openai/whisper-tiny",
        )

        assert created["name"] == "custom_openai_whisper_tiny"
        assert created["custom_hf_model"] == "openai/whisper-tiny"
        assert created["language"] == "en"
        assert captured["language"] == "en"
        assert captured["asr_model"] == "custom-model"
