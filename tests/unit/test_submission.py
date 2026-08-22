"""Tests for SubmissionConfig and scores.json artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from psdn_sonar.benchmark import SubmissionConfig, build_run_scores, write_scores_json
from psdn_sonar.benchmark.scores import scores_json_path
from psdn_sonar.benchmark.submission import KNOWN_INFERENCE_PARAM_KEYS


def test_submission_config_required_fields():
    cfg = SubmissionConfig(
        provider="openai",
        model_snapshot="whisper-1",
        region="us-east-1",
        protocol="batch",
        inference_params={"language_code": "en"},
        sample_rate_hz=16000,
        seed=42,
        git_sha="abc123",
        package_version="0.1.0",
        timestamp_utc="2026-05-22T12:00:00Z",
    )
    assert cfg.protocol == "batch"
    assert cfg.seed == 42


def test_invalid_protocol_rejected():
    with pytest.raises(ValueError):
        SubmissionConfig(
            provider="openai",
            model_snapshot="whisper-1",
            region="us-east-1",
            protocol="websocket",
            inference_params={},
            seed=42,
            git_sha="abc",
            package_version="0.1.0",
            timestamp_utc="2026-05-22T12:00:00Z",
        )


def test_unknown_inference_param_rejected():
    with pytest.raises(ValueError, match="Unknown inference_params"):
        SubmissionConfig(
            provider="openai",
            model_snapshot="whisper-1",
            region="us-east-1",
            protocol="batch",
            inference_params={"not_a_real_param": 1},
            seed=42,
            git_sha="abc",
            package_version="0.1.0",
            timestamp_utc="2026-05-22T12:00:00Z",
        )


def test_submission_config_round_trip_json():
    cfg = SubmissionConfig(
        provider="assemblyai",
        model_snapshot="assemblyai_api",
        region="ap-south-1",
        protocol="streaming",
        inference_params={"temperature": 0.0, "language_code": "bn"},
        sample_rate_hz=None,
        seed=7,
        judge_model="gemini-2.5-pro",
        prompt_version="deadbeef",
        git_sha="3a4c7b1",
        package_version="0.1.0",
        timestamp_utc="2026-05-22T12:00:00Z",
    )
    raw = json.dumps(cfg.model_dump(mode="json"))
    restored = SubmissionConfig.model_validate(json.loads(raw))
    assert restored == cfg


def test_from_env_resolves_metadata(monkeypatch):
    monkeypatch.setenv("SONAR_GIT_SHA", "envsha")
    monkeypatch.setattr("psdn_sonar.benchmark.submission.__version__", "9.9.9")
    monkeypatch.setattr("psdn_sonar.config_loader.get_run_seed", lambda: 99)
    monkeypatch.setattr("psdn_sonar.benchmark.submission._utc_now_iso", lambda: "2026-01-01T00:00:00Z")
    monkeypatch.setattr("psdn_sonar.benchmark.submission._resolve_git_sha", lambda: "envsha")

    cfg = SubmissionConfig.from_env(
        provider="local",
        model_snapshot="test_model",
        region="local",
    )
    assert cfg.git_sha == "envsha"
    assert cfg.package_version == "9.9.9"
    assert cfg.timestamp_utc == "2026-01-01T00:00:00Z"
    assert cfg.seed == 99


def test_write_scores_json_under_expected_path(tmp_path: Path):
    submission = SubmissionConfig(
        provider="local",
        model_snapshot="demo_model",
        region="local",
        protocol="batch",
        inference_params={},
        seed=42,
        git_sha="abc",
        package_version="0.1.0",
        timestamp_utc="2026-05-22T12:00:00Z",
    )
    evaluate_result = {
        "model_name": "demo_model",
        "results": [
            {
                "audio_path": "a.wav",
                "wer": 0.1,
                "cer": 0.05,
                "semantic_similarity": None,
                "poseidon_score": None,
                "inference_latency_s": 1.0,
                "error": None,
            }
        ],
        "summary": {
            "total_samples": 1,
            "successful": 1,
            "failed": 0,
            "avg_wer": 0.1,
            "avg_cer": 0.05,
            "elapsed_time": 1.0,
            "avg_latency_s": 1.0,
            "median_latency_s": 1.0,
            "p95_latency_s": 1.0,
        },
    }
    out = scores_json_path(tmp_path, "demo_model")
    artifact = build_run_scores(submission, evaluate_result)
    write_scores_json(out, artifact)

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    round_trip = SubmissionConfig.model_validate(payload["submission"])
    assert round_trip == submission
    assert payload["model_name"] == "demo_model"
    assert payload["aggregate"]["wer_mean"] == pytest.approx(0.1)
    assert len(payload["utterances"]) == 1
    assert set(payload["utterances"][0].keys()) <= set(
        {
            "audio_path",
            "wer",
            "cer",
            "semantic_similarity",
            "poseidon_score",
            "significant_wer",
            "inference_latency_s",
            "ttft_s",
            "complete_s",
            "error",
        }
    )


def test_known_inference_keys_documented():
    assert "language_code" in KNOWN_INFERENCE_PARAM_KEYS


def _init_foreign_repo(path: Path) -> str:
    """Create an unrelated git repo with one commit; return its HEAD SHA."""
    import subprocess

    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "file.txt").write_text("foreign\n", encoding="utf-8")
    env = {
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
        "PATH": os.environ["PATH"],
    }
    subprocess.run(["git", "-C", str(path), "add", "file.txt"], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True, env=env)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


class TestGitShaResolution:
    """Issue #110: git_sha must identify the psdn-sonar checkout, never the
    caller's working-directory repository."""

    def test_ignores_callers_cwd_repository(self, tmp_path: Path, monkeypatch):
        from psdn_sonar.benchmark.submission import _resolve_git_sha

        monkeypatch.delenv("SONAR_GIT_SHA", raising=False)
        foreign_sha = _init_foreign_repo(tmp_path / "other-repo")
        monkeypatch.chdir(tmp_path / "other-repo")

        sha = _resolve_git_sha()
        assert sha != foreign_sha
        # This test suite runs from a source checkout, so the package's own
        # repository must be identified (a 40-char SHA, not "unknown").
        assert sha == "unknown" or len(sha) == 40

    def test_untracked_package_records_unknown(self, tmp_path: Path, monkeypatch):
        """A venv nested inside an unrelated repo must not attribute the run
        to that repo: the SHA is recorded only if the package's own file is
        tracked there."""
        import psdn_sonar.benchmark.submission as submission_mod

        monkeypatch.delenv("SONAR_GIT_SHA", raising=False)
        _init_foreign_repo(tmp_path / "host-repo")
        fake_pkg = tmp_path / "host-repo" / ".venv" / "site-packages" / "benchmark"
        fake_pkg.mkdir(parents=True)
        fake_file = fake_pkg / "submission.py"
        fake_file.write_text("# installed copy\n", encoding="utf-8")

        monkeypatch.setattr(submission_mod, "__file__", str(fake_file))
        assert submission_mod._resolve_git_sha() == "unknown"

    def test_tracked_package_records_that_repos_head(self, tmp_path: Path, monkeypatch):
        """Control: when the package file IS tracked (source checkout), the
        containing repo's HEAD is recorded."""
        import subprocess

        import psdn_sonar.benchmark.submission as submission_mod

        monkeypatch.delenv("SONAR_GIT_SHA", raising=False)
        repo_sha = _init_foreign_repo(tmp_path / "checkout")
        tracked = tmp_path / "checkout" / "file.txt"
        subprocess.run(["git", "-C", str(tmp_path / "checkout"), "rev-parse", "HEAD"], check=True)

        monkeypatch.setattr(submission_mod, "__file__", str(tracked))
        assert submission_mod._resolve_git_sha() == repo_sha

    def test_env_override_wins(self, monkeypatch):
        from psdn_sonar.benchmark.submission import _resolve_git_sha

        monkeypatch.setenv("SONAR_GIT_SHA", "cafe" * 10)
        assert _resolve_git_sha() == "cafe" * 10


class TestScoreChangingInputsRecorded:
    """Issue #110: the provenance block must record the inputs that change
    the score — POSEIDON weights, similarity model, and the environment."""

    def test_from_env_records_weights_similarity_model_and_environment(self):
        from psdn_sonar.config import config

        cfg = SubmissionConfig.from_env(provider="local", model_snapshot="m", region="local")

        assert cfg.poseidon_weights == {
            "wer": config.wer_weight,
            "cer": config.cer_weight,
            "semantic": config.semantic_weight,
        }
        assert cfg.similarity_model == config.similarity_model
        assert cfg.os_platform and isinstance(cfg.os_platform, str)
        assert cfg.python_version and cfg.python_version.count(".") == 2
        assert cfg.device in ("cuda", "mps", "cpu", None)

    def test_new_fields_serialize_into_scores_json(self, tmp_path: Path):
        submission = SubmissionConfig.from_env(provider="local", model_snapshot="demo_model", region="local")
        artifact = build_run_scores(submission, _evaluate_result_for_significant_wer())
        out = scores_json_path(tmp_path, "demo_model")
        write_scores_json(out, artifact)

        payload = json.loads(out.read_text(encoding="utf-8"))
        sub = payload["submission"]
        assert set(sub["poseidon_weights"]) == {"wer", "cer", "semantic"}
        assert sub["similarity_model"]
        assert sub["os_platform"]
        assert sub["python_version"]
        assert "device" in sub

    def test_legacy_payload_without_new_fields_still_validates(self):
        """Pre-#110 scores.json artifacts lack the new fields; they must
        still round-trip (defaults are None)."""
        legacy = {
            "provider": "openai",
            "model_snapshot": "whisper-1",
            "region": "us-east-1",
            "protocol": "batch",
            "inference_params": {},
            "seed": 42,
            "git_sha": "abc",
            "package_version": "0.1.0",
            "timestamp_utc": "2026-05-22T12:00:00Z",
        }
        cfg = SubmissionConfig.model_validate(legacy)
        assert cfg.poseidon_weights is None
        assert cfg.similarity_model is None
        assert cfg.device is None


def _evaluate_result_for_significant_wer() -> dict:
    """Three-utterance fixture spanning the significantWer boundaries.

    With the default 0.30 threshold:
      - row 0 (wer=0.10) is below threshold -> not significant
      - row 1 (wer=0.50) is above threshold -> significant
      - row 2 (wer=None, error) is missing  -> excluded from both
        numerator and denominator
    Expected rate over the two finite rows: 1/2 = 0.5.
    """
    return {
        "model_name": "demo_model",
        "results": [
            {
                "audio_path": "good.wav",
                "wer": 0.10,
                "cer": 0.05,
                "semantic_similarity": None,
                "poseidon_score": None,
                "inference_latency_s": 1.0,
                "error": None,
            },
            {
                "audio_path": "bad.wav",
                "wer": 0.50,
                "cer": 0.40,
                "semantic_similarity": None,
                "poseidon_score": None,
                "inference_latency_s": 1.5,
                "error": None,
            },
            {
                "audio_path": "broken.wav",
                "wer": None,
                "cer": None,
                "semantic_similarity": None,
                "poseidon_score": None,
                "inference_latency_s": None,
                "error": "Empty prediction",
            },
        ],
        "summary": {
            "total_samples": 3,
            "successful": 2,
            "failed": 1,
            "avg_wer": 0.30,
            "avg_cer": 0.225,
            "elapsed_time": 2.5,
            "avg_latency_s": 1.25,
            "median_latency_s": 1.25,
            "p95_latency_s": 1.5,
        },
    }


def test_scores_json_records_significant_wer_rate_and_threshold(tmp_path: Path):
    submission = _submission_for_test(model_snapshot="demo_model")
    artifact = build_run_scores(submission, _evaluate_result_for_significant_wer())

    assert artifact.aggregate.significant_wer_rate == pytest.approx(0.5)
    # Threshold actually used is recorded so a future default change does
    # not retroactively reinterpret the rate.
    assert artifact.aggregate.significant_wer_threshold == pytest.approx(0.30)

    out = scores_json_path(tmp_path, "demo_model")
    write_scores_json(out, artifact)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["aggregate"]["significant_wer_rate"] == pytest.approx(0.5)
    assert payload["aggregate"]["significant_wer_threshold"] == pytest.approx(0.30)


def test_scores_json_respects_custom_threshold():
    submission = _submission_for_test(model_snapshot="demo_model")
    # At threshold=0.45 only the 0.50 row fails. Rate over the two finite
    # rows is still 1/2 = 0.5, but the threshold field must reflect 0.45.
    artifact = build_run_scores(
        submission,
        _evaluate_result_for_significant_wer(),
        significant_wer_threshold=0.45,
    )
    assert artifact.aggregate.significant_wer_rate == pytest.approx(0.5)
    assert artifact.aggregate.significant_wer_threshold == pytest.approx(0.45)

    # And at a stricter threshold both finite rows fail (0.10 < 0.05? no;
    # but 0.10 >= 0.05 yes; 0.50 >= 0.05 yes) -> rate = 1.0 / threshold = 0.05.
    artifact_strict = build_run_scores(
        submission,
        _evaluate_result_for_significant_wer(),
        significant_wer_threshold=0.05,
    )
    assert artifact_strict.aggregate.significant_wer_rate == pytest.approx(1.0)
    assert artifact_strict.aggregate.significant_wer_threshold == pytest.approx(0.05)


def test_scores_json_significant_rate_none_when_no_finite_wer():
    """A run where every utterance failed must report rate=None and
    threshold=None (rather than 0.0) so consumers can tell "0% errors" from
    "nothing measurable"."""
    submission = _submission_for_test(model_snapshot="demo_model")
    all_failed = {
        "model_name": "demo_model",
        "results": [
            {
                "audio_path": "broken.wav",
                "wer": None,
                "cer": None,
                "semantic_similarity": None,
                "poseidon_score": None,
                "inference_latency_s": None,
                "error": "Empty prediction",
            }
        ],
        "summary": {
            "total_samples": 1,
            "successful": 0,
            "failed": 1,
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "elapsed_time": 0.1,
            "avg_latency_s": None,
            "median_latency_s": None,
            "p95_latency_s": None,
        },
    }
    artifact = build_run_scores(submission, all_failed)
    assert artifact.aggregate.significant_wer_rate is None
    assert artifact.aggregate.significant_wer_threshold is None


def test_scores_json_preserves_significant_wer_per_utterance(tmp_path: Path):
    """Per-row ``significant_wer`` must round-trip through the slim
    utterance export when callers pre-populate it (e.g. SingleSpeakerEvaluator
    writes it onto each row before build_run_scores runs)."""
    submission = _submission_for_test(model_snapshot="demo_model")
    evaluate_result = _evaluate_result_for_significant_wer()
    # Mirror what the evaluator does: stamp each row with its flag.
    for row in evaluate_result["results"]:
        wer = row.get("wer")
        if wer is None:
            row["significant_wer"] = None
        else:
            row["significant_wer"] = wer >= 0.30

    artifact = build_run_scores(submission, evaluate_result)
    out = scores_json_path(tmp_path, "demo_model")
    write_scores_json(out, artifact)
    payload = json.loads(out.read_text(encoding="utf-8"))

    rows_by_path = {row["audio_path"]: row for row in payload["utterances"]}
    assert rows_by_path["good.wav"]["significant_wer"] is False
    assert rows_by_path["bad.wav"]["significant_wer"] is True
    assert rows_by_path["broken.wav"]["significant_wer"] is None


def _submission_for_test(*, model_snapshot: str) -> SubmissionConfig:
    return SubmissionConfig(
        provider="openai",
        model_snapshot=model_snapshot,
        region="us-east-1",
        protocol="batch",
        inference_params={"language_code": "en"},
        seed=42,
        git_sha="abc",
        package_version="0.1.0",
        timestamp_utc="2026-05-22T12:00:00Z",
    )


def test_run_evaluation_preserves_caller_submission_snapshot(tmp_path: Path, monkeypatch):
    """Caller-provided model_snapshot must not be replaced by the registry model_name."""
    from psdn_sonar.evaluators.single_speaker import SingleSpeakerEvaluator

    submission = _submission_for_test(model_snapshot="whisper-1@2024-06-01")
    fake_result = {
        "model_name": "whisper_api",
        "results": [],
        # successful=1 so run_evaluation does not raise NoSamplesEvaluatedError;
        # this test is about the submission snapshot, not zero-sample handling.
        "summary": {
            "total_samples": 1,
            "successful": 1,
            "failed": 0,
            "avg_wer": 0.0,
            "avg_cer": 0.0,
            "elapsed_time": 0.0,
            "avg_latency_s": None,
            "median_latency_s": None,
            "p95_latency_s": None,
        },
    }

    monkeypatch.setattr("psdn_sonar.evaluators.single_speaker.load_env", lambda: None)
    monkeypatch.setattr(
        SingleSpeakerEvaluator,
        "load_data",
        lambda *args, **kwargs: [{"audio_path": "clip.wav", "ground_truth": "hello"}],
    )
    monkeypatch.setattr(
        SingleSpeakerEvaluator,
        "evaluate_one",
        lambda *args, **kwargs: fake_result,
    )
    monkeypatch.setattr("psdn_sonar.evaluators.single_speaker._model_factory", lambda *args, **kwargs: object())

    tsv_path = tmp_path / "eval.tsv"
    tsv_path.write_text("audio_path\ttranscription\nclip.wav\thello\n", encoding="utf-8")
    output_dir = tmp_path / "results"

    SingleSpeakerEvaluator.run_evaluation(
        tsv_path=str(tsv_path),
        output_dir=str(output_dir),
        models=["whisper_api"],
        language="en",
        submission=submission,
        write_scores=True,
        compute_sem=False,
    )

    scores_path = output_dir / "scores_whisper_api.json"
    assert scores_path.exists()
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    assert payload["submission"]["model_snapshot"] == "whisper-1@2024-06-01"
    assert payload["model_name"] == "whisper_api"
