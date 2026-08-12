"""Tests for the custom evaluation pipeline."""

import csv
import sys
import types

import pytest
import yaml

from psdn_sonar.custom_eval import (
    CustomEvalConfig,
    _create_api_model,
    _evaluate_model_on_dataset,
    prepare_dataset,
    run_custom_evaluation,
)


def write_config(tmp_path, **overrides):
    raw = {
        "language": {"code": "pt", "name": "Portuguese"},
        "models": ["org/model-one"],
        "dataset": {"tsv_path": str(tmp_path / "data.tsv")},
    }
    raw.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return str(path)


class TestCustomEvalConfig:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            CustomEvalConfig(str(tmp_path / "missing.yaml"))

    def test_no_models_raises(self, tmp_path):
        path = write_config(tmp_path, models=[])
        with pytest.raises(ValueError, match="at least one model"):
            CustomEvalConfig(path)

    def test_no_dataset_raises(self, tmp_path):
        path = write_config(tmp_path, dataset={})
        with pytest.raises(ValueError, match="tsv_path or dataset.hf_dataset_id"):
            CustomEvalConfig(path)

    def test_string_model_shorthand(self, tmp_path):
        config = CustomEvalConfig(write_config(tmp_path))
        assert config.models == [{"hf_model_id": "org/model-one"}]

    def test_dict_model_preserved(self, tmp_path):
        path = write_config(tmp_path, models=[{"hf_model_id": "org/m", "whisper_language": "pt"}])
        config = CustomEvalConfig(path)
        assert config.models[0]["whisper_language"] == "pt"

    def test_defaults(self, tmp_path):
        path = write_config(tmp_path, dataset={"hf_dataset_id": "org/ds"})
        config = CustomEvalConfig(path)
        assert config.hf_split == "test"
        assert config.text_column == "sentence"
        assert config.audio_column == "audio"
        assert config.include_api_models is True
        assert config.api_models_list == ["whisper_api", "elevenlabs_api", "assemblyai_api"]
        assert "HF" in repr(config)

    def test_api_models_disabled(self, tmp_path):
        path = write_config(tmp_path, api_models={"enabled": False})
        config = CustomEvalConfig(path)
        assert config.include_api_models is False


class TestPrepareDataset:
    def test_local_tsv_used_as_is(self, tmp_path):
        tsv = tmp_path / "data.tsv"
        tsv.write_text("audio_path\ttranscription\na.wav\thello\n")
        config = CustomEvalConfig(write_config(tmp_path))
        assert prepare_dataset(config, str(tmp_path / "out")) == str(tsv)

    def test_missing_local_tsv_raises(self, tmp_path):
        config = CustomEvalConfig(write_config(tmp_path))
        with pytest.raises(FileNotFoundError, match="Dataset TSV not found"):
            prepare_dataset(config, str(tmp_path / "out"))

    def test_hf_dataset_converted_to_tsv(self, tmp_path, monkeypatch):
        items = [
            {"sentence": "first", "audio": {"path": "/audio/a.wav"}},
            {"sentence": "  ", "audio": {"path": "/audio/b.wav"}},
            {"sentence": "third", "audio": None},
        ]
        fake_datasets = types.SimpleNamespace(load_dataset=lambda *a, **kw: items)
        monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

        config = CustomEvalConfig(write_config(tmp_path, dataset={"hf_dataset_id": "org/ds"}))
        tsv_path = prepare_dataset(config, str(tmp_path / "out"))

        with open(tsv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert rows == [
            {"audio_path": "/audio/a.wav", "transcription": "first"},
            {"audio_path": "", "transcription": "third"},
        ]

    def test_hf_dataset_max_samples(self, tmp_path, monkeypatch):
        items = [{"sentence": f"s{i}", "audio": None} for i in range(5)]
        monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *a, **kw: items))

        config = CustomEvalConfig(write_config(tmp_path, dataset={"hf_dataset_id": "org/ds"}))
        tsv_path = prepare_dataset(config, str(tmp_path / "out"), max_samples=2)

        with open(tsv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        assert [r["transcription"] for r in rows] == ["s0", "s1"]

    def test_hf_load_failure_raises_runtime_error(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise OSError("gated dataset")

        monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=boom))
        config = CustomEvalConfig(write_config(tmp_path, dataset={"hf_dataset_id": "org/ds"}))
        with pytest.raises(RuntimeError, match="Could not load HuggingFace dataset"):
            prepare_dataset(config, str(tmp_path / "out"))


class _StubModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class TestCreateApiModel:
    def test_whisper_gets_language(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.models.apis.WhisperAPIModel", _StubModel)
        model = _create_api_model("whisper_api", "pt")
        assert model.kwargs == {"language": "pt"}

    def test_elevenlabs_maps_language_code(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.models.apis.ElevenLabsAPIModel", _StubModel)
        model = _create_api_model("elevenlabs_api", "pt")
        assert model.kwargs == {"language_code": "por"}

    def test_assemblyai_gets_language_code(self, monkeypatch):
        monkeypatch.setattr("psdn_sonar.models.apis.AssemblyAIAPIModel", _StubModel)
        model = _create_api_model("assemblyai_api", "pt")
        assert model.kwargs == {"language_code": "pt"}

    def test_unknown_api_returns_none(self):
        assert _create_api_model("unknown_api", "pt") is None


class TestEvaluateModelOnDataset:
    def test_writes_results_csv(self, tmp_path, monkeypatch):
        evaluator = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator"
        monkeypatch.setattr(f"{evaluator}.load_data", staticmethod(lambda path: [{"audio_path": "a"}]))
        monkeypatch.setattr(
            f"{evaluator}.evaluate_one",
            staticmethod(lambda **kw: {"results": [{"wer": 0.1, "cer": 0.05}]}),
        )

        csv_path = _evaluate_model_on_dataset(
            model=object(),
            tsv_path="in.tsv",
            model_name="m1",
            output_dir=str(tmp_path),
            max_samples=0,
            language_code="pt",
        )

        assert csv_path == str(tmp_path / "asr_detailed_m1.csv")
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == [{"wer": "0.1", "cer": "0.05"}]

    def test_failure_returns_none(self, tmp_path, monkeypatch):
        def boom(path):
            raise RuntimeError("bad tsv")

        monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.load_data", staticmethod(boom))
        result = _evaluate_model_on_dataset(
            model=object(),
            tsv_path="in.tsv",
            model_name="m1",
            output_dir=str(tmp_path),
            max_samples=0,
            language_code="pt",
        )
        assert result is None


class TestRunCustomEvaluation:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        tsv = tmp_path / "data.tsv"
        tsv.write_text("audio_path\ttranscription\na.wav\thello\n")

        monkeypatch.setattr("psdn_sonar.config.load_env", lambda: None)
        monkeypatch.setattr("psdn_sonar.models.huggingface.CustomHuggingFaceModel", _StubModel)
        evaluator = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator"
        monkeypatch.setattr(f"{evaluator}.load_data", staticmethod(lambda path: [{"audio_path": "a"}]))
        monkeypatch.setattr(f"{evaluator}.evaluate_one", staticmethod(lambda **kw: {"results": [{"wer": 0.1}]}))
        for key in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        return tmp_path

    def test_evaluates_hf_models_without_api_keys(self, env):
        config = CustomEvalConfig(write_config(env))
        evaluated = run_custom_evaluation(config, str(env / "out"), generate_report=False)

        assert evaluated == [("custom_org_model_one", str(env / "out" / "asr_detailed_custom_org_model_one.csv"))]

    def test_includes_api_models_with_keys(self, env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        monkeypatch.setattr("psdn_sonar.custom_eval._create_api_model", lambda name, code: _StubModel())

        config = CustomEvalConfig(write_config(env))
        evaluated = run_custom_evaluation(config, str(env / "out"), generate_report=False)

        assert [name for name, _ in evaluated] == ["custom_org_model_one", "whisper_api"]

    def test_skips_failed_models(self, env, monkeypatch):
        monkeypatch.setattr(
            "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.load_data",
            staticmethod(lambda path: (_ for _ in ()).throw(RuntimeError("bad"))),
        )
        config = CustomEvalConfig(write_config(env))
        evaluated = run_custom_evaluation(config, str(env / "out"), generate_report=False)
        assert evaluated == []
