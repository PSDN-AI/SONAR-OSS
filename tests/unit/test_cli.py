"""Tests for the psdn-sonar command-line interface."""

import subprocess
import sys
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

    def test_audio_path_strictness_default_allows_absolute(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--models", "wav2vec2_bengali", "--language", "bn")

        assert mock_eval.call_args[1]["allow_absolute_audio_paths"] is True

    def test_strict_audio_paths_flag_forwarded(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli(
                "single",
                "--input",
                tsv,
                "--models",
                "wav2vec2_bengali",
                "--language",
                "bn",
                "--strict-audio-paths",
            )

        assert mock_eval.call_args[1]["allow_absolute_audio_paths"] is False

    def test_traversal_audio_path_exits_nonzero(self, tmp_path, caplog):
        # Regression for issue #127: a TSV audio_path with ../ escaping the
        # dataset directory used to be opened; now the run refuses it before
        # any model loads (load_data raises, the CLI exits 1 cleanly).
        path = tmp_path / "traversal.tsv"
        path.write_text("audio_path\ttranscription\n../../../../etc/hosts\tignored\n")

        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli(
                    "single",
                    "--input",
                    str(path),
                    "--models",
                    "wav2vec2_bengali",
                    "--language",
                    "bn",
                )

        assert exc_info.value.code == 1
        assert "escapes dataset root" in caplog.text

    def test_missing_ffmpeg_exits_cleanly(self, tsv, caplog):
        # Issue #109: pipeline adapters preflight for ffmpeg at load time;
        # the CLI must surface that as a clean actionable error and exit 1,
        # not one "Transcription failed" per utterance or a raw traceback.
        from psdn_sonar.models.base import MissingFfmpegError

        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        error = MissingFfmpegError("StandardHuggingFaceASR requires the ffmpeg binary — including WAV")
        with caplog.at_level("ERROR"):
            with patch(target, side_effect=error):
                with pytest.raises(SystemExit) as exc_info:
                    run_cli("single", "--input", tsv, "--models", "whisper_base_en", "--language", "en")

        assert exc_info.value.code == 1
        assert "ffmpeg" in caplog.text
        assert "Traceback" not in caplog.text

    def test_hf_model_sanitized_name(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--hf-model", "openai/whisper-small")

        kwargs = mock_eval.call_args[1]
        assert kwargs["custom_hf_model"] == "openai/whisper-small"
        assert kwargs["models"] == ["custom_openai_whisper_small"]

    def test_language_defaults_when_no_models(self, tsv, monkeypatch):
        for var in ("OPENAI_API_KEY", "ELEVENLABS_API_KEY", "XI_API_KEY", "ASSEMBLYAI_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--language", "ko")

        kwargs = mock_eval.call_args[1]
        assert kwargs["models"]
        assert all(isinstance(m, str) for m in kwargs["models"])
        assert not any(name.endswith("_api") for name in kwargs["models"])

    def test_implicit_language_defaults_to_bn(self, tsv, caplog):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with caplog.at_level("WARNING"):
            with patch(target) as mock_eval:
                mock_eval.return_value = {"results": []}
                run_cli("single", "--input", tsv, "--models", "whisper_base_en")

        assert mock_eval.call_args[1]["language"] == "bn"
        assert "No --language specified" in caplog.text

    def test_unsupported_language_without_models_exits(self, tsv, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("single", "--input", tsv, "--language", "xx")
        assert exc_info.value.code == 1
        assert "Unknown --language 'xx'" in caplog.text

    def test_unknown_language_with_explicit_models_exits_before_scoring(self, tsv, caplog):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with caplog.at_level("ERROR"):
            with patch(target) as mock_eval:
                with pytest.raises(SystemExit) as exc_info:
                    run_cli("single", "--input", tsv, "--models", "whisper_base_en", "--language", "xx")
        assert exc_info.value.code == 1
        assert "Unknown --language 'xx'" in caplog.text
        mock_eval.assert_not_called()

    def test_language_without_dedicated_normalizer_warns_and_proceeds(self, tsv, caplog):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with caplog.at_level("WARNING"):
            with patch(target) as mock_eval:
                mock_eval.return_value = {"results": []}
                run_cli("single", "--input", tsv, "--models", "whisper_base_en", "--language", "pt")
        assert "no dedicated normalizer" in caplog.text
        assert mock_eval.call_args[1]["language"] == "pt"

    def test_language_long_name_and_case_canonicalized(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--models", "wav2vec2_bengali", "--language", "Bengali")
        assert mock_eval.call_args[1]["language"] == "bn"

    def test_streaming_flag_reaches_the_evaluator(self, tsv):
        # Issue #186: AssemblyAI's streaming mode (the only way to measure
        # ttft_s) previously had no entry point anywhere in the CLI.
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--models", "assemblyai_api", "--language", "en", "--streaming")
        assert mock_eval.call_args[1]["streaming"] is True

    def test_streaming_defaults_off(self, tsv):
        target = "psdn_sonar.evaluators.single_speaker.SingleSpeakerEvaluator.run_evaluation"
        with patch(target) as mock_eval:
            mock_eval.return_value = {"results": []}
            run_cli("single", "--input", tsv, "--models", "whisper_base_en", "--language", "en")
        assert mock_eval.call_args[1]["streaming"] is False


class TestDiscoverDispatch:
    @staticmethod
    def _fake_dataset():
        fake = MagicMock()
        fake.name = "fleurs"
        fake.hf_id = "google/fleurs"
        fake.config = "en_us"
        fake.revision = "a" * 40
        return fake

    def test_all_preparers_failing_exits_nonzero(self):
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[self._fake_dataset()]),
            patch("psdn_sonar.data.preparer.DatasetPreparer.prepare", side_effect=RuntimeError("decode failed")),
        ):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("discover", "--language", "en")
        assert exc_info.value.code == 1

    def test_unknown_datasets_filter_exits_nonzero(self, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("discover", "--language", "en", "--datasets", "commonvoice", "--dry-run")
        assert exc_info.value.code == 1
        assert "unknown dataset name" in caplog.text

    def test_disabled_dataset_named_as_disabled(self, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("discover", "--language", "en", "--datasets", "common_voice", "--dry-run")
        assert exc_info.value.code == 1
        assert "catalogued but disabled" in caplog.text

    def test_valid_filter_matching_nothing_blames_filter(self, caplog):
        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit) as exc_info:
                run_cli("discover", "--language", "en", "--datasets", "zeroth", "--dry-run")
        assert exc_info.value.code == 1
        assert "The --datasets filter, not the language" in caplog.text
        assert "zeroth supports: ko" in caplog.text

    def test_dry_run_with_valid_filter_exits_zero(self):
        with pytest.raises(SystemExit) as exc_info:
            run_cli("discover", "--language", "en", "--datasets", "fleurs", "--dry-run")
        assert exc_info.value.code == 0

    def test_unwritable_output_logs_clean_error_without_traceback(self, caplog):
        """Issue #149: an unwritable --output must produce the actionable
        one-line ERROR only — no traceback whose exception chain names a
        FileNotFoundError before the real PermissionError."""
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[self._fake_dataset()]),
            patch(
                "psdn_sonar.data.preparer.DatasetPreparer.prepare",
                side_effect=PermissionError(13, "Permission denied", "/tmp/readonly-out/x"),
            ),
        ):
            with caplog.at_level("ERROR"):
                with pytest.raises(SystemExit) as exc_info:
                    run_cli("discover", "--language", "en", "--output", "/tmp/readonly-out/x")
        assert exc_info.value.code == 1
        prepare_records = [r for r in caplog.records if "Failed to prepare fleurs" in r.getMessage()]
        assert len(prepare_records) == 1
        assert "Permission denied" in prepare_records[0].getMessage()
        assert prepare_records[0].exc_info is None
        assert "All 1 dataset(s) failed to prepare: fleurs" in caplog.text

    def test_disk_full_logs_clean_actionable_error_without_traceback(self, caplog):
        """Issue #183: a full disk must reach the OSError handler as one clean
        actionable line naming the cache, not the catch-all with a traceback
        about 'no samples could be loaded'."""
        disk_full = OSError(
            28,
            "the disk filled up while preparing org/fake (No space left on device). "
            "The partial download is kept in the HuggingFace cache at /home/x/.cache/huggingface.",
        )
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[self._fake_dataset()]),
            patch("psdn_sonar.data.preparer.DatasetPreparer.prepare", side_effect=disk_full),
        ):
            with caplog.at_level("ERROR"):
                with pytest.raises(SystemExit) as exc_info:
                    run_cli("discover", "--language", "en")
        assert exc_info.value.code == 1
        prepare_records = [r for r in caplog.records if "Failed to prepare fleurs" in r.getMessage()]
        assert len(prepare_records) == 1
        assert "HuggingFace cache" in prepare_records[0].getMessage()
        assert prepare_records[0].exc_info is None

    def test_unexpected_preparer_error_keeps_traceback(self, caplog):
        """Genuine bugs (non-OSError) must stay loud with their traceback."""
        with (
            patch("psdn_sonar.data.discovery.DatasetDiscovery.discover", return_value=[self._fake_dataset()]),
            patch("psdn_sonar.data.preparer.DatasetPreparer.prepare", side_effect=RuntimeError("decode failed")),
        ):
            with caplog.at_level("ERROR"):
                with pytest.raises(SystemExit):
                    run_cli("discover", "--language", "en")
        prepare_records = [r for r in caplog.records if "Failed to prepare fleurs" in r.getMessage()]
        assert len(prepare_records) == 1
        assert prepare_records[0].exc_info is not None

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
        assert kwargs["custom_hf_model"] is None
        assert kwargs["language"] == "bn"
        # Neither new option given: the config file still decides (issue #210).
        assert kwargs["methods"] is None
        assert kwargs["config_path"] is None

    def test_forwards_methods_and_preprocessing_config(self, tmp_path):
        """Issue #210: --sweep could only ever reach the packaged config's
        method list, because the subcommand had no option for either the list
        or the file."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        out_csv = tmp_path / "out.csv"
        out_csv.write_text("cer_conv\n0.1\n")
        cfg = tmp_path / "preprocessing.yaml"
        cfg.write_text("methods:\n  - no_trim\n")

        with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
            mock_run.return_value = str(out_csv)
            run_cli(
                "multi",
                "--input",
                str(manifest),
                "--models",
                "whisper_api",
                "--sweep",
                "--methods",
                "energy_trim",
                "timestamp_trim",
                "no_trim",
                "--preprocessing-config",
                str(cfg),
            )

        kwargs = mock_run.call_args[1]
        assert kwargs["methods"] == ["energy_trim", "timestamp_trim", "no_trim"]
        assert kwargs["config_path"] == str(cfg)
        assert kwargs["sweep"] is True

    def test_method_and_methods_are_mutually_exclusive(self, tmp_path):
        """Passing both states two different intents. The pinned one used to
        lose silently, or abort the run when the pool filtered down to empty."""
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")

        with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
            with pytest.raises(SystemExit) as exc_info:
                run_cli(
                    "multi",
                    "--input",
                    str(manifest),
                    "--models",
                    "whisper_api",
                    "--method",
                    "no_trim",
                    "--methods",
                    "scribe_diarize",
                )
        assert exc_info.value.code == 2
        mock_run.assert_not_called()

    def test_unknown_language_code_exits_before_pipeline(self, tmp_path, caplog):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        with caplog.at_level("ERROR"):
            with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
                with pytest.raises(SystemExit) as exc_info:
                    run_cli("multi", "--input", str(manifest), "--models", "whisper_api", "--language", "xx")
        assert exc_info.value.code == 1
        assert "Unknown --language 'xx'" in caplog.text
        mock_run.assert_not_called()

    def test_hf_model_forwards_custom_id(self, tmp_path):
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
                "--hf-model",
                "openai/whisper-tiny",
                "--language",
                "en",
            )

        kwargs = mock_run.call_args[1]
        assert kwargs["model_name"] == "custom_openai_whisper_tiny"
        assert kwargs["custom_hf_model"] == "openai/whisper-tiny"
        assert kwargs["language"] == "en"


class TestMultiDemographicsStage:
    """The optional --demographics stage after issue #234.

    On the shipped example (which carries no metadata.json) the stage used to
    crash inside plotnine, log the same traceback twice, and report
    ``Evaluation failed`` with exit 1 — although the evaluation had completed
    and written its artifacts. Missing metadata is now a warned skip with
    exit 0; a genuine failure in the stage exits non-zero under a message
    that names the stage and states that the evaluation itself succeeded.
    """

    @staticmethod
    def _eval_outputs(tmp_path):
        manifest = tmp_path / "manifest.jsonl"
        manifest.write_text("{}\n")
        out_csv = tmp_path / "asr_eval_results_mymodel_manifest.csv"
        out_csv.write_text("audio_id,speaker,wer_conv\nrec1,A,0.1\nrec1,B,0.2\n")
        return manifest, out_csv

    def test_missing_metadata_skips_and_exits_zero(self, tmp_path, caplog):
        manifest, out_csv = self._eval_outputs(tmp_path)
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()
        output_dir = tmp_path / "out"

        with caplog.at_level("WARNING"):
            with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
                mock_run.return_value = str(out_csv)
                run_cli(
                    "multi",
                    "--input",
                    str(manifest),
                    "--models",
                    "whisper_api",
                    "--demographics",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output",
                    str(output_dir),
                )  # completes without SystemExit

        assert "metadata.json" in caplog.text
        assert "Evaluation failed" not in caplog.text
        assert "Demographic analysis complete" not in caplog.text
        # No empty demographic directory skeletons.
        assert not (output_dir / "demographic-analysis").exists()

    def test_stage_failure_exits_one_without_claiming_the_evaluation_failed(self, tmp_path, caplog):
        manifest, out_csv = self._eval_outputs(tmp_path)
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        with caplog.at_level("ERROR"):
            with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
                mock_run.return_value = str(out_csv)
                with patch(
                    "psdn_sonar.analysis.demographic_analyzer.DemographicAnalyzer.run_full_analysis",
                    side_effect=RuntimeError("plotting exploded"),
                ):
                    with pytest.raises(SystemExit) as exc_info:
                        run_cli(
                            "multi",
                            "--input",
                            str(manifest),
                            "--models",
                            "whisper_api",
                            "--demographics",
                            "--dataset-dir",
                            str(dataset_dir),
                            "--output",
                            str(tmp_path / "out"),
                        )

        assert exc_info.value.code == 1
        assert "Evaluation failed" not in caplog.text
        assert "Demographic analysis failed for" in caplog.text
        assert "completed" in caplog.text  # ...and says the evaluation succeeded
        # The traceback is logged exactly once, not once per handler layer.
        assert sum(1 for r in caplog.records if r.exc_info) == 1

    def test_successful_stage_logs_completion_and_exits_zero(self, tmp_path, caplog):
        manifest, out_csv = self._eval_outputs(tmp_path)
        dataset_dir = tmp_path / "dataset"
        meta_dir = dataset_dir / "rec1"
        meta_dir.mkdir(parents=True)
        (meta_dir / "metadata.json").write_text(
            '{"speaker_a": {"age": 30, "gender": "female", "region": "north"},'
            ' "speaker_b": {"age": 45, "gender": "male", "region": "south"}}'
        )
        output_dir = tmp_path / "out"

        with caplog.at_level("INFO"):
            with patch("psdn_sonar.multispeaker_pipeline.run_multispeaker_evaluation") as mock_run:
                mock_run.return_value = str(out_csv)
                run_cli(
                    "multi",
                    "--input",
                    str(manifest),
                    "--models",
                    "whisper_api",
                    "--demographics",
                    "--dataset-dir",
                    str(dataset_dir),
                    "--output",
                    str(output_dir),
                )

        assert "Demographic analysis complete" in caplog.text
        assert "Evaluation failed" not in caplog.text
        plots = output_dir / "demographic-analysis" / "demographic_plots" / "mymodel"
        assert (plots / "gender_wer_conv.png").exists()


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


class TestEntrypointExitStability:
    """Issue #139: the console script must always deliver the exit code the
    run decided on. A torch-family native extension intermittently SIGABRTs
    (exit 134) in interpreter teardown after a data-reason failure had
    already been reported as exit 1, so ``entrypoint`` leaves via
    ``os._exit`` once logging and the std streams are flushed."""

    def _exit_code(self, main_behavior):
        from psdn_sonar import cli

        recorded = {}
        with (
            patch.object(cli, "main", side_effect=main_behavior),
            patch.object(cli.os, "_exit", side_effect=lambda code: recorded.setdefault("code", code)),
        ):
            cli.entrypoint()
        return recorded["code"]

    def test_clean_return_exits_zero(self):
        assert self._exit_code(lambda: None) == 0

    def test_systemexit_code_is_preserved(self):
        assert self._exit_code(lambda: sys.exit(1)) == 1
        assert self._exit_code(lambda: sys.exit(2)) == 2

    def test_systemexit_none_means_zero(self):
        assert self._exit_code(lambda: sys.exit(None)) == 0

    def test_systemexit_message_prints_and_exits_one(self, capsys):
        assert self._exit_code(lambda: sys.exit("boom message")) == 1
        assert "boom message" in capsys.readouterr().err

    def test_keyboard_interrupt_exits_130(self):
        assert self._exit_code(KeyboardInterrupt()) == 130

    def test_unexpected_exception_stays_loud(self, capsys):
        assert self._exit_code(RuntimeError("genuine bug")) == 1
        err = capsys.readouterr().err
        assert "Traceback" in err
        assert "genuine bug" in err

    def test_console_script_targets_entrypoint(self):
        # The stable-exit guarantee only holds if the installed script goes
        # through entrypoint(), not main() directly.
        from importlib.metadata import entry_points

        (script,) = entry_points(group="console_scripts", name="psdn-sonar")
        assert script.value == "psdn_sonar.cli:entrypoint"

    def test_subprocess_exit_codes_are_stable(self, tmp_path):
        # Real-process check of the wiring: a success and a data-reason
        # failure must come back as exactly 0 and 1, never a signal code.
        snippet = (
            "import sys; sys.argv = ['psdn-sonar'] + sys.argv[1:]; from psdn_sonar.cli import entrypoint; entrypoint()"
        )

        ok = subprocess.run([sys.executable, "-c", snippet, "--version"], capture_output=True, text=True)
        assert ok.returncode == 0

        data = tmp_path / "data.tsv"
        data.write_text("audio_path\ttranscription\ntest.wav\ttest text\n")
        bad_language = subprocess.run(
            [sys.executable, "-c", snippet, "single", "--input", str(data), "--models", "m", "--language", "xx"],
            capture_output=True,
            text=True,
        )
        assert bad_language.returncode == 1


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
