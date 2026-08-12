"""Unit tests for the pure logic in scripts/verify_release_commit.py."""

import importlib.util
import json
import pathlib
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "verify_release_commit.py"
_spec = importlib.util.spec_from_file_location("verify_release_commit", _SCRIPT)
vrc = importlib.util.module_from_spec(_spec)
sys.modules["verify_release_commit"] = vrc
_spec.loader.exec_module(vrc)

_BASELINE = pathlib.Path(__file__).resolve().parents[2] / "security" / "repo-settings-baseline.json"
_COMMIT = "e" * 40
_TAG_OBJECT = "d" * 40


def _fake_api(responses):
    """gh_api stand-in serving canned responses keyed by path prefix."""

    def fake(path, **_):
        for prefix, value in responses.items():
            if path.startswith(prefix):
                return value
        raise AssertionError(f"unexpected gh_api call: {path}")

    return fake


def test_gh_api_items_requests_and_flattens_every_page(monkeypatch):
    commands = []

    def fake_run(cmd, **_):
        commands.append(cmd)
        pages = [{"workflow_runs": [{"id": 1}]}, {"workflow_runs": [{"id": 2}]}]
        return vrc.subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(pages).encode(), stderr=b"")

    monkeypatch.setattr(vrc.subprocess, "run", fake_run)
    assert vrc.gh_api_items("/actions/runs?per_page=100", "workflow_runs") == [{"id": 1}, {"id": 2}]
    assert "--paginate" in commands[0]
    assert "--slurp" in commands[0]


# --- resolve_ref -----------------------------------------------------------


def test_annotated_tag_dereferences_to_commit(monkeypatch):
    monkeypatch.setattr(
        vrc,
        "gh_api",
        _fake_api(
            {
                "/git/ref/tags/v1": {"ref": "refs/tags/v1", "object": {"type": "tag", "sha": _TAG_OBJECT}},
                f"/git/tags/{_TAG_OBJECT}": {"object": {"type": "commit", "sha": _COMMIT}},
            }
        ),
    )
    assert vrc.resolve_ref("v1") == (_COMMIT, [])


def test_lightweight_tag_needs_no_dereference(monkeypatch):
    monkeypatch.setattr(
        vrc,
        "gh_api",
        _fake_api({"/git/ref/tags/v1": {"ref": "refs/tags/v1", "object": {"type": "commit", "sha": _COMMIT}}}),
    )
    assert vrc.resolve_ref("v1") == (_COMMIT, [])


def test_branch_name_is_rejected_structurally(monkeypatch):
    monkeypatch.setattr(vrc, "gh_api", _fake_api({"/git/ref/tags/main": None}))
    sha, failures = vrc.resolve_ref("main")
    assert sha == "" and "neither a 40-hex commit SHA nor an existing tag" in failures[0]


def test_prefix_matched_ref_list_is_rejected(monkeypatch):
    # /git/ref prefix-matches: querying tag "v1" can return a list for "v1.0" etc.
    monkeypatch.setattr(vrc, "gh_api", _fake_api({"/git/ref/tags/v1": [{"ref": "refs/tags/v1.0"}]}))
    sha, failures = vrc.resolve_ref("v1")
    assert sha == "" and failures


def test_abbreviated_sha_is_rejected(monkeypatch):
    monkeypatch.setattr(vrc, "gh_api", _fake_api({"/git/ref/tags/e7db574": None}))
    sha, failures = vrc.resolve_ref("e7db574")
    assert sha == "" and failures


def test_tag_object_sha_is_a_clean_rejection(monkeypatch):
    monkeypatch.setattr(vrc, "gh_api", _fake_api({f"/git/commits/{_TAG_OBJECT}": None}))
    sha, failures = vrc.resolve_ref(_TAG_OBJECT)
    assert sha == "" and "not a commit in this repository" in failures[0]


# --- classify_position -----------------------------------------------------


@pytest.mark.parametrize(
    ("compare", "ok"),
    [
        ({"status": "identical"}, True),
        ({"status": "behind", "behind_by": 1}, False),
        ({"status": "ahead", "ahead_by": 2}, False),
        ({"status": "diverged", "ahead_by": 20, "behind_by": 32}, False),
        (None, False),
    ],
)
def test_classify_position(compare, ok):
    _, failures = vrc.classify_position(compare)
    assert (not failures) is ok


# --- accept_run ------------------------------------------------------------


def _run(**overrides):
    base = {
        "id": 1,
        "path": ".github/workflows/ci.yml",
        "event": "push",
        "head_branch": "main",
        "head_repository": {"full_name": vrc.REPO},
        "status": "completed",
        "run_attempt": 1,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("overrides", "reason_fragment"),
    [
        ({}, None),
        ({"event": "workflow_dispatch"}, None),
        ({"event": "pull_request"}, "PR-head"),
        ({"event": "pull_request_target"}, "PR-head"),
        ({"event": "dynamic", "path": "dynamic/dependabot/update-graph"}, "not an evidence source"),
        ({"path": ".github/workflows/pr-title.yml"}, "not an evidence source"),
        ({"path": ".github/workflows/release.yml"}, "not an evidence source"),
        ({"head_branch": "feat/core-and-recipe"}, "not main"),
        ({"head_repository": {"full_name": "someone/fork"}}, "fork"),
        ({"status": "in_progress"}, "not completed"),
    ],
)
def test_accept_run(overrides, reason_fragment):
    reason = vrc.accept_run(_run(**overrides))
    if reason_fragment is None:
        assert reason is None
    else:
        assert reason is not None and reason_fragment in reason


# --- select_checks / evaluate_checks --------------------------------------


def _job(name, conclusion="success", completed_at="2026-08-12T18:00:00Z", job_id=10):
    return {"name": name, "conclusion": conclusion, "completed_at": completed_at, "id": job_id, "html_url": "u"}


def test_latest_wins_across_two_accepted_runs():
    runs = [_run(id=1), _run(id=2, event="workflow_dispatch")]
    jobs = {
        1: [_job("Dependency audit", conclusion="failure", completed_at="2026-08-12T10:00:00Z")],
        2: [_job("Dependency audit", conclusion="success", completed_at="2026-08-12T12:00:00Z")],
    }
    selected = vrc.select_checks(runs, jobs)
    assert selected["Dependency audit"]["conclusion"] == "success"
    assert selected["Dependency audit"]["workflow_run_id"] == 2


def test_latest_job_wins_without_losing_jobs_from_an_earlier_attempt():
    run = _run(id=1, run_attempt=2)
    jobs = {
        1: [
            _job("Secret scan", completed_at="2026-08-12T10:00:00Z", job_id=10),
            _job("Dependency audit", conclusion="failure", completed_at="2026-08-12T10:00:00Z", job_id=11),
            _job("Dependency audit", completed_at="2026-08-12T12:00:00Z", job_id=12),
        ]
    }
    selected = vrc.select_checks([run], jobs)
    assert selected["Secret scan"]["check_run_id"] == 10
    assert selected["Dependency audit"]["check_run_id"] == 12


def test_unsuccessful_and_missing_checks_all_reported():
    runs = [_run(id=1)]
    jobs = {1: [_job("Secret scan", conclusion="cancelled")]}
    failures = vrc.evaluate_checks(vrc.select_checks(runs, jobs))
    assert "required check 'Secret scan' concluded 'cancelled'" in failures
    assert len([f for f in failures if f.startswith("missing required check")]) == len(vrc.REQUIRED_CHECKS) - 1


def test_unknown_job_names_are_ignored():
    selected = vrc.select_checks([_run(id=1)], {1: [_job("update-uv-graph")]})
    assert selected == {}


def test_render_markdown_marks_failed_checks_red():
    evidence = {
        "verified": False,
        "commit": _COMMIT,
        "tag": "v1",
        "main_position": "identical",
        "main_head": _COMMIT,
        "failures": ["required check failed"],
        "checks": [
            {"name": "Secret scan", "conclusion": "failure", "check_run_id": 10, "url": "u"},
            {"name": "Dependency audit", "conclusion": "success", "check_run_id": 11, "url": "u"},
        ],
    }
    rendered = vrc.render_markdown(evidence)
    assert "❌ Secret scan: failure" in rendered
    assert "✅ Dependency audit: success" in rendered


# --- coupling with the settings baseline (#25) -----------------------------


@pytest.mark.skipif(not _BASELINE.exists(), reason="settings baseline not merged yet (PR #71)")
def test_branch_protection_names_match_settings_baseline():
    baseline = json.loads(_BASELINE.read_text())
    required = baseline["branch_protection_main"]["required_status_checks"]["checks"]
    assert all(entry.endswith("@15368") for entry in required)
    baseline_names = {entry.rsplit("@", 1)[0] for entry in required}
    verifier_names = set(vrc.BRANCH_PROTECTION_MAIN_CHECKS) | {"Validate PR title"}
    assert baseline_names == verifier_names
