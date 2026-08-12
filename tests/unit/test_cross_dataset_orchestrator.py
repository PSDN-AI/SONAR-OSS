"""Tests for the cross-dataset comparison orchestrator."""

import os

from psdn_sonar.orchestrators.cross_dataset import (
    CrossDatasetComparison,
    default_models_config,
)

MODULE = "psdn_sonar.orchestrators.cross_dataset"


class TestParseSampleSize:
    def test_positive_arg(self):
        assert CrossDatasetComparison.parse_sample_size(samples_arg=42) == 42

    def test_prompt_input(self):
        assert CrossDatasetComparison.parse_sample_size(prompt_input=" 300 ") == 300

    def test_invalid_prompt_falls_back(self):
        assert CrossDatasetComparison.parse_sample_size(prompt_input="abc") == 100

    def test_non_positive_prompt_falls_back(self):
        assert CrossDatasetComparison.parse_sample_size(prompt_input="0") == 100

    def test_no_input_defaults(self):
        assert CrossDatasetComparison.parse_sample_size() == 100


class TestDefaultModelsConfig:
    def test_bengali_roster_from_registry(self):
        config = default_models_config("bengali")
        keys = [key for key, _, _ in config]
        assert "banglaspeech2text" in keys
        assert "whisper_api" in keys

    def test_api_models_carry_env_keys(self):
        env_keys = {key: env for key, _, env in default_models_config("english")}
        assert env_keys["whisper_api"] == "OPENAI_API_KEY"
        assert env_keys["elevenlabs_api"] == "ELEVENLABS_API_KEY"
        assert env_keys["whisper_base_en"] is None

    def test_unknown_language_empty(self):
        assert default_models_config("klingon") == []


class TestBuildModelsToRun:
    def test_skips_models_with_missing_env(self):
        config = [("needs_key", lambda: "instance", "SOME_KEY")]
        assert CrossDatasetComparison.build_models_to_run(config, env_getter=lambda k: None) == []

    def test_includes_models_with_env_set(self):
        config = [("needs_key", lambda: "instance", "SOME_KEY")]
        result = CrossDatasetComparison.build_models_to_run(config, env_getter=lambda k: "set")
        assert result == [("needs_key", "instance")]

    def test_skips_failing_factory(self):
        def boom():
            raise RuntimeError("no ML extras")

        config = [("broken", boom, None), ("ok", lambda: "instance", None)]
        result = CrossDatasetComparison.build_models_to_run(config, env_getter=lambda k: None)
        assert result == [("ok", "instance")]


class TestGetAvailableDatasets:
    def test_filters_unresolvable(self, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.resolve_dataset_dir", lambda base, name: "/d" if name == "fleurs" else None)
        available = CrossDatasetComparison.get_available_datasets("/base", ["fleurs", "commonvoice"])
        assert available == ["fleurs"]


class TestRunSingle:
    def test_success_writes_expected_path(self, tmp_path, monkeypatch):
        captured = {}
        monkeypatch.setattr(f"{MODULE}.process_dataset_with_asr", lambda **kw: captured.update(kw))

        ok = CrossDatasetComparison.run_single("m1", "model", "fleurs", "/data", str(tmp_path), 50)

        assert ok is True
        assert captured["output_tsv"] == os.path.join(str(tmp_path), "m1_fleurs_n50.csv")
        assert captured["max_samples"] == 50
        assert captured["asr_model_name"] == "m1"

    def test_failure_returns_false(self, tmp_path, monkeypatch):
        def boom(**kwargs):
            raise RuntimeError("eval failed")

        monkeypatch.setattr(f"{MODULE}.process_dataset_with_asr", boom)
        assert CrossDatasetComparison.run_single("m1", "model", "fleurs", "/data", str(tmp_path), 50) is False


class TestRunAll:
    def test_records_completed_and_failed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.resolve_dataset_dir", lambda base, name: None if name == "missing" else "/d")

        def process(**kwargs):
            if kwargs["dataset_name"] == "bad":
                raise RuntimeError("eval failed")

        monkeypatch.setattr(f"{MODULE}.process_dataset_with_asr", process)

        completed, failed = CrossDatasetComparison.run_all(
            [("m1", "model")], ["fleurs", "bad", "missing"], "/base", str(tmp_path / "res"), 10
        )

        assert completed == ["m1/fleurs"]
        assert failed == ["m1/bad", "m1/missing"]
        assert (tmp_path / "res").is_dir()


class TestRun:
    def test_no_datasets_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.resolve_dataset_dir", lambda base, name: None)
        completed, failed, results_dir = CrossDatasetComparison.run(
            10, base_dir=str(tmp_path), results_dir=str(tmp_path / "res")
        )
        assert completed == []
        assert failed == []
        assert results_dir == str(tmp_path / "res")

    def test_no_models_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.resolve_dataset_dir", lambda base, name: "/d")
        completed, failed, _ = CrossDatasetComparison.run(
            10, base_dir=str(tmp_path), results_dir=str(tmp_path / "res"), models_config=[]
        )
        assert completed == []
        assert failed == []

    def test_full_run(self, tmp_path, monkeypatch):
        monkeypatch.setattr(f"{MODULE}.resolve_dataset_dir", lambda base, name: "/d")
        monkeypatch.setattr(f"{MODULE}.process_dataset_with_asr", lambda **kwargs: None)

        completed, failed, results_dir = CrossDatasetComparison.run(
            10,
            base_dir=str(tmp_path),
            results_dir=str(tmp_path / "res"),
            dataset_names=["fleurs"],
            models_config=[("m1", lambda: "model", None)],
        )

        assert completed == ["m1/fleurs"]
        assert failed == []
        assert results_dir == str(tmp_path / "res")
