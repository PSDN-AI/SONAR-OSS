"""Tests for scripts/verify_release_commit.py."""

import importlib.util
import json
import pathlib
import subprocess
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "verify_release_commit.py"
_spec = importlib.util.spec_from_file_location("verify_release_commit", _SCRIPT)
vrc = importlib.util.module_from_spec(_spec)
sys.modules["verify_release_commit"] = vrc
_spec.loader.exec_module(vrc)

_MAIN = "a" * 40
_OLD = "b" * 40


def _run(**overrides):
    run = {
        "id": 1,
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "head_repository": {"full_name": vrc.REPO},
        "status": "completed",
    }
    run.update(overrides)
    return run


def _job(name, conclusion="success", completed_at="2026-08-12T18:00:00Z", job_id=10):
    return {
        "name": name,
        "conclusion": conclusion,
        "completed_at": completed_at,
        "id": job_id,
        "html_url": "https://example.test/job",
    }


def _all_success():
    return {
        name: {
            "name": name,
            "conclusion": "success",
            "check_run_id": 10,
            "url": "https://example.test/job",
        }
        for name in vrc.REQUIRED_CHECKS
    }


def test_github_requests_and_returns_every_page(monkeypatch):
    commands = []

    def fake_run(command, **_):
        commands.append(command)
        output = json.dumps([{"items": [1]}, {"items": [2]}])
        return subprocess.CompletedProcess(command, 0, stdout=output, stderr="")

    monkeypatch.setattr(vrc.subprocess, "run", fake_run)
    assert vrc.github("/example") == [{"items": [1]}, {"items": [2]}]
    assert "--paginate" in commands[0] and "--slurp" in commands[0]


def test_collect_checks_ignores_untrusted_runs(monkeypatch):
    runs = [
        _run(id=1),
        _run(id=2, head_branch="feature"),
        _run(id=3, head_repository={"full_name": "someone/fork"}),
        _run(id=4, event="pull_request"),
        _run(id=5, path=".github/workflows/release.yml"),
        _run(id=6, status="in_progress"),
    ]

    def fake_github(path):
        if path.startswith("/actions/runs?"):
            return [{"workflow_runs": runs}]
        assert "/runs/1/jobs" in path
        return [{"jobs": [_job("Secret scan")]}]

    monkeypatch.setattr(vrc, "github", fake_github)
    assert list(vrc.collect_checks(_MAIN)) == ["Secret scan"]


def test_collect_checks_handles_partial_reruns(monkeypatch):
    jobs = [
        _job("Secret scan", completed_at="2026-08-12T10:00:00Z", job_id=10),
        _job("Dependency audit", conclusion="failure", completed_at="2026-08-12T10:00:00Z", job_id=11),
        _job("Dependency audit", completed_at="2026-08-12T12:00:00Z", job_id=12),
    ]

    def fake_github(path):
        if path.startswith("/actions/runs?"):
            return [{"workflow_runs": [_run()]}]
        assert "filter=all" in path
        return [{"jobs": jobs[:2]}, {"jobs": jobs[2:]}]

    monkeypatch.setattr(vrc, "github", fake_github)
    checks = vrc.collect_checks(_MAIN)
    assert checks["Secret scan"]["check_run_id"] == 10
    assert checks["Dependency audit"]["check_run_id"] == 12


def test_main_passes_for_current_main(monkeypatch, tmp_path):
    evidence_path = tmp_path / "evidence.json"
    monkeypatch.setattr(vrc, "github", lambda _: [{"object": {"sha": _MAIN}}])
    monkeypatch.setattr(vrc, "collect_checks", lambda _: _all_success())
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), "--tag", "v1.0.0", "--evidence", str(evidence_path)])

    assert vrc.main() == 0
    evidence = json.loads(evidence_path.read_text())
    assert evidence["verified"] is True
    assert evidence["tag"] == "v1.0.0"


def test_main_rejects_old_commit_before_checking_ci(monkeypatch):
    def unexpected_check(_):
        raise AssertionError("CI should not be queried for an old commit")

    monkeypatch.setattr(vrc, "github", lambda _: [{"object": {"sha": _MAIN}}])
    monkeypatch.setattr(vrc, "collect_checks", unexpected_check)
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), _OLD])

    assert vrc.main() == 1


def test_main_rejects_failed_and_missing_checks_with_red_output(monkeypatch, capsys):
    monkeypatch.setattr(vrc, "github", lambda _: [{"object": {"sha": _MAIN}}])
    failed = _all_success()["Secret scan"] | {"conclusion": "failure"}
    monkeypatch.setattr(vrc, "collect_checks", lambda _: {"Secret scan": failed})
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT)])

    assert vrc.main() == 1
    output = capsys.readouterr().out
    assert "❌ Secret scan: failure" in output
    assert "✅ Secret scan" not in output
    assert "missing required check" in output
