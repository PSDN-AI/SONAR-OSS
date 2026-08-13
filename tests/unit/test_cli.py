"""Tests for the psdn-sonar command-line interface."""

from unittest.mock import MagicMock, patch

import pytest

from psdn_sonar.cli import main


@pytest.fixture
def tsv(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("audio_path\ttranscription\ntest.wav\ttest text\n")
    return str(path)


def run_cli(*argv):
    with patch("sys.argv", ["psdn-sonar", *argv]):
        main()


class TestArgumentParsing:
    def test_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            run_cli("--help")
        assert exc_info.value.code == 0

    def test_version_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            run_cli("--version")
        assert exc_info.value.code == 0

    def test_mode_is_required(self):
        with pytest.raises(SystemExit):
            run_cli()

    def test_single_requires_input(self):
        with pytest.raises(SystemExit):
            run_cli("single", "--models", "wav2vec2_bengali")

    def test_missing_input_file_rejected(self):
        with pytest.raises(SystemExit):
            run_cli("single", "--input", "/nonexistent/data.tsv", "--models", "wav2vec2_bengali")

    def test_models_and_hf_model_conflict(self, tsv):
        with pytest.raises(SystemExit):
            run_cli("single", "--input", tsv, "--models", "m1", "--hf-model", "org/m2")

    def test_missing_config_file_rejected(self):
        with pytest.raises(SystemExit):
            run_cli("custom", "--config", "/nonexistent/config.yaml")

    def test_multi_demographics_requires_dataset_dir(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        with pytest.raises(SystemExit):
            run_cli("multi", "--input", str(manifest), "--models", "whisper_api", "--demographics")

    def test_multi_report_requires_dataset_dir(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        with pytest.raises(SystemExit):
            run_cli("multi", "--input", str(manifest), "--models", "whisper_api", "--report")

    def test_worker_subcommands_removed(self):
        for legacy in ("evaluate-single", "prep-from-source"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli(legacy)
            assert exc_info.value.code != 0


class TestSingleSpeakerDispatch:
    def test_language_and_models_forwarded(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--models", "wav2vec2_bengali", "--language", "ko", "--max-samples", "1")

        kwargs = mock_eval.call_args[1]
        assert kwargs["language"] == "ko"
        assert kwargs["models"] == ["wav2vec2_bengali"]
        assert kwargs["max_samples"] == 1
        assert kwargs["custom_hf_model"] is None

    def test_hf_model_sanitized_name(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--hf-model", "openai/whisper-small")

        kwargs = mock_eval.call_args[1]
        assert kwargs["custom_hf_model"] == "openai/whisper-small"
        assert kwargs["models"] == ["custom_openai_whisper_small"]

    def test_language_defaults_when_no_models(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--language", "ko")

        kwargs = mock_eval.call_args[1]
        assert kwargs["models"]
        assert all(isinstance(m, str) for m in kwargs["models"])

    def test_unsupported_language_without_models_exits(self, tsv):
        with pytest.raises(SystemExit) as exc_info:
            run_cli("single", "--input", tsv, "--language", "xx")
        assert exc_info.value.code == 1


class TestDiscoverDispatch:
    @staticmethod
    def _fake_dataset():
        fake = MagicMock()
        fake.name = "fleurs"
        fake.hf_id = "google/fleurs"
        fake.config = "en_us"
        return fake

    def test_all_preparers_failing_exits_nonzero(self):
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[self._fake_dataset()]),
            patch("psdn_sonar.data.preparer.DatasetPreparer.prepare", side_effect=RuntimeError("decode failed")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("discover", "--language", "en")
        assert exc_info.value.code == 1

    def test_partial_failure_still_succeeds(self, tmp_path):
        ok, bad = self._fake_dataset(), self._fake_dataset()
        bad.name = "voxpopuli"
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[ok, bad]),
            patch(
                "psdn_sonar.data.preparer.DatasetPreparer.prepare",
                side_effect=[tmp_path, RuntimeError("decode failed")],
            ),
        ):
            run_cli("discover", "--language", "en", "--output", str(tmp_path))


class TestMultiSpeakerDispatch:
    def test_requires_models_or_hf_model(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        with pytest.raises(SystemExit) as exc_info:
            run_cli("multi", "--input", str(manifest))
        assert exc_info.value.code == 1

    def test_forwards_sweep_and_method(self, tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        out_csv = tmp_path / "out.csv"
        out_csv.write_text("cer_conv\n0.1\n")

        with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
            mock_run.return_value = str(out_csv)
            run_cli(
                "multi",
                "--input",
                str(manifest),
                "--models",
                "whisper_api",
                "--sweep",
                "--method",
                "energy_trim",
            )

        kwargs = mock_run.call_args[1]
        assert kwargs["model_name"] == "whisper_api"
        assert kwargs["sweep"] is True
        assert kwargs["method"] == "energy_trim"


class TestCustomDispatch:
    def test_forwards_config_and_report_flag(self, tmp_path):
        config_file = tmp_path / "eval.yaml"
        config_file.write_text("language:\n  code: pt\nmodels:\n  - org/m\ndataset:\n  tsv_path: d.tsv\n")

        with (
            patch("psdn_sonar.custom_eval.CustomEvalConfig") as mock_config,
            patch("psdn_sonar.custom_eval.run_custom_evaluation") as mock_run,
        ):
            mock_run.return_value = []
            run_cli("custom", "--config", str(config_file), "--report")

        mock_config.assert_called_once_with(str(config_file))
        assert mock_run.call_args[1]["generate_report"] is True


PRIVATE_CONTROL_PLANE_MODULE = ".".join(["psdn_sonar", "service"])


class TestServiceDecoupling:
    def test_no_service_imports(self):
        import psdn_sonar.cli as cli_module

        source = open(cli_module.__file__, encoding="utf-8").read()
        assert PRIVATE_CONTROL_PLANE_MODULE not in source
        assert "typer" not in source

    def test_no_service_package_shipped(self):
        import importlib

        with pytest.raises(ImportError):
            importlib.import_module(PRIVATE_CONTROL_PLANE_MODULE)
