"""Unit tests for scripts/check_repo_settings.py (comparison logic, API
failure handling, and live-state shaping against the committed baseline)."""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_repo_settings.py"
_spec = importlib.util.spec_from_file_location("check_repo_settings", _SCRIPT)
check_repo_settings = importlib.util.module_from_spec(_spec)
sys.modules["check_repo_settings"] = check_repo_settings
_spec.loader.exec_module(check_repo_settings)

diff_states = check_repo_settings.diff_states
flatten = check_repo_settings.flatten
shape_environment = check_repo_settings.shape_environment
BASELINE_PATH = check_repo_settings.BASELINE_PATH


def test_flatten_nests_dicts_and_serializes_lists():
    flat = flatten({"a": {"b": 1, "c": [2, 1]}, "d": True})
    assert flat == {"a.b": 1, "a.c": "[2, 1]", "d": True}


def test_flatten_skips_underscore_keys_at_top_level_only():
    assert flatten({"_comment": "ignored", "a": 1}) == {"a": 1}
    # Nested underscore keys are data, not metadata: an environment named
    # `_publish` must not be invisible to the drift check.
    assert flatten({"environments": {"_publish": {"x": 1}}}) == {"environments._publish.x": 1}


def test_underscore_named_environment_surfaces_as_drift():
    drift = diff_states({"environments": {}}, {"environments": {"_publish": {"can_admins_bypass": True}}})
    assert drift == [("environments._publish.can_admins_bypass", "<not in baseline>", True)]


def test_identical_states_produce_no_drift():
    state = {"x": {"y": False, "z": ["v*"]}}
    assert diff_states(state, json.loads(json.dumps(state))) == []


def test_changed_value_is_reported():
    drift = diff_states({"enforce_admins": False}, {"enforce_admins": True})
    assert drift == [("enforce_admins", False, True)]


def test_list_order_is_not_drift_after_sorting_but_content_is():
    # Callers sort list fields before diffing; unequal content must surface.
    base = {"checks": ["A@1", "B@1"]}
    assert diff_states(base, {"checks": ["A@1", "B@1"]}) == []
    drift = diff_states(base, {"checks": ["A@1"]})
    assert len(drift) == 1 and drift[0][0] == "checks"


def test_key_missing_from_live_reads_as_not_readable():
    drift = diff_states({"a": {"b": 1}}, {"a": {}})
    assert drift == [("a.b", 1, "<not readable>")]


def test_unexpected_live_key_reads_as_not_in_baseline():
    drift = diff_states({}, {"surprise": 1})
    assert drift == [("surprise", "<not in baseline>", 1)]


def test_committed_baseline_parses_and_flattens():
    baseline = json.loads(BASELINE_PATH.read_text())
    flat = flatten(baseline)
    # Spot-check governed keys the release evidence (issue #12) depends on.
    assert flat["branch_protection_main.enforce_admins"] is False
    assert flat["actions_permissions.allowed_actions"] == "selected"
    assert "Dependency audit@15368" in flat["branch_protection_main.required_status_checks.checks"]
    assert flat["actions_access_level"] == "none"
    assert flat["environments.pypi.deployment_policies"] == '["tag:v*"]'
    assert flat["environments.pypi.reviewers"] == "[]"
    assert flat["repo_flags.allow_forking"] is False
    assert flat["security_and_analysis.secret_scanning"] == "disabled"


def _completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=["gh"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_gh_api_aborts_on_read_failure(monkeypatch):
    monkeypatch.setattr(check_repo_settings.subprocess, "run", lambda *a, **k: _completed(1, stderr="gh: HTTP 403"))
    with pytest.raises(SystemExit) as exc:
        check_repo_settings.gh_api("/actions/permissions")
    assert "Refusing" in str(exc.value)


def test_gh_api_maps_404_to_none_only_with_missing_ok(monkeypatch):
    monkeypatch.setattr(
        check_repo_settings.subprocess, "run", lambda *a, **k: _completed(1, stderr="gh: Not Found (HTTP 404)")
    )
    assert check_repo_settings.gh_api("/vulnerability-alerts", missing_ok=True) is None
    with pytest.raises(SystemExit):
        check_repo_settings.gh_api("/vulnerability-alerts")


def test_gh_api_returns_empty_dict_for_204_style_empty_body(monkeypatch):
    monkeypatch.setattr(check_repo_settings.subprocess, "run", lambda *a, **k: _completed(0, stdout=""))
    assert check_repo_settings.gh_api("/vulnerability-alerts", missing_ok=True) == {}


def test_gh_api_missing_ok_still_aborts_on_non_404_failure(monkeypatch):
    monkeypatch.setattr(check_repo_settings.subprocess, "run", lambda *a, **k: _completed(1, stderr="gh: HTTP 500"))
    with pytest.raises(SystemExit):
        check_repo_settings.gh_api("/vulnerability-alerts", missing_ok=True)


def test_truncated_policy_listing_aborts():
    policies = {"total_count": 2, "branch_policies": [{"name": "v*", "type": "tag"}]}
    with pytest.raises(SystemExit):
        shape_environment({}, policies)


def test_shape_environment_records_branch_and_tag_policies_with_type():
    env = {"can_admins_bypass": False, "deployment_branch_policy": {"custom_branch_policies": True}}
    policies = {"branch_policies": [{"name": "v*", "type": "tag"}, {"name": "main", "type": "branch"}]}
    shaped = shape_environment(env, policies)
    # A branch policy loosening the deployment source must surface, not vanish.
    assert shaped["deployment_policies"] == ["branch:main", "tag:v*"]


def test_shape_environment_records_required_reviewers():
    env = {
        "protection_rules": [
            {"type": "branch_policy"},
            {
                "type": "required_reviewers",
                "reviewers": [
                    {"type": "User", "reviewer": {"login": "octocat"}},
                    {"type": "Team", "reviewer": {"slug": "release-approvers"}},
                ],
            },
        ]
    }
    shaped = shape_environment(env, None)
    assert shaped["reviewers"] == ["Team:release-approvers", "User:octocat"]
    assert shaped["deployment_policies"] == []


def _canned_responses():
    baseline = json.loads(BASELINE_PATH.read_text())
    checks = [
        {"context": entry.rsplit("@", 1)[0], "app_id": int(entry.rsplit("@", 1)[1])}
        for entry in baseline["branch_protection_main"]["required_status_checks"]["checks"]
    ]
    env_payload = {
        "deployment_branch_policy": {"custom_branch_policies": True},
        "protection_rules": [{"type": "branch_policy"}],
    }
    return {
        "/branches/main/protection": {
            "required_status_checks": {"strict": True, "checks": checks},
            "required_pull_request_reviews": {
                "required_approving_review_count": 0,
                "require_code_owner_reviews": False,
                "dismiss_stale_reviews": False,
            },
            "enforce_admins": {"enabled": False},
            "required_conversation_resolution": {"enabled": True},
            "allow_force_pushes": {"enabled": False},
            "allow_deletions": {"enabled": False},
            "lock_branch": {"enabled": False},
        },
        "/actions/permissions": {"enabled": True, "allowed_actions": "selected", "sha_pinning_required": True},
        "/actions/permissions/selected-actions": {
            "github_owned_allowed": True,
            "verified_allowed": False,
            "patterns_allowed": baseline["selected_actions"]["patterns_allowed"],
        },
        "/actions/permissions/workflow": {
            "default_workflow_permissions": "read",
            "can_approve_pull_request_reviews": False,
        },
        "/actions/permissions/access": {"access_level": "none"},
        "": {
            "visibility": "private",
            "has_wiki": False,
            "allow_forking": False,
            "allow_squash_merge": True,
            "allow_merge_commit": False,
            "allow_rebase_merge": False,
            "allow_auto_merge": False,
            "delete_branch_on_merge": True,
            "security_and_analysis": {
                "secret_scanning": {"status": "disabled"},
                "secret_scanning_push_protection": {"status": "disabled"},
            },
        },
        "/vulnerability-alerts": {},
        "/automated-security-fixes": {"enabled": False, "paused": False},
        "/environments?per_page=100": {
            "total_count": 2,
            "environments": [
                {"name": "pypi", "can_admins_bypass": False, **env_payload},
                {"name": "testpypi", "can_admins_bypass": True, **env_payload},
            ],
        },
        "/environments/pypi/deployment-branch-policies?per_page=100": {
            "total_count": 1,
            "branch_policies": [{"name": "v*", "type": "tag"}],
        },
        "/environments/testpypi/deployment-branch-policies?per_page=100": {
            "total_count": 1,
            "branch_policies": [{"name": "main", "type": "branch"}],
        },
    }


def test_collect_live_state_matches_committed_baseline_shape(monkeypatch):
    # Locks the shaping code and the committed baseline to the same schema:
    # a healthy API readback must produce zero drift against the baseline.
    canned = _canned_responses()

    def fake_gh_api(path, *, missing_ok=False):
        assert path in canned, f"unexpected API call: {path}"
        return canned[path]

    monkeypatch.setattr(check_repo_settings, "gh_api", fake_gh_api)
    live = check_repo_settings.collect_live_state()
    baseline = json.loads(BASELINE_PATH.read_text())
    assert diff_states(baseline, live) == []


def _collect_with(monkeypatch, canned):
    def fake_gh_api(path, *, missing_ok=False):
        assert path in canned, f"unexpected API call: {path}"
        return canned[path]

    monkeypatch.setattr(check_repo_settings, "gh_api", fake_gh_api)
    return check_repo_settings.collect_live_state()


def test_internal_visibility_still_reads_access_level(monkeypatch):
    canned = _canned_responses()
    canned[""] = {**canned[""], "visibility": "internal"}
    live = _collect_with(monkeypatch, canned)
    assert live["actions_access_level"] == "none"


def test_public_visibility_uses_sentinel_and_skips_access_endpoint(monkeypatch):
    canned = _canned_responses()
    canned[""] = {**canned[""], "visibility": "public"}
    del canned["/actions/permissions/access"]  # calling it would fail the test
    live = _collect_with(monkeypatch, canned)
    assert live["actions_access_level"] == "not_applicable_public"


def test_unreadable_automated_security_fixes_never_reads_as_baseline_false(monkeypatch):
    canned = _canned_responses()
    canned["/automated-security-fixes"] = None  # gh_api's missing_ok 404 result
    live = _collect_with(monkeypatch, canned)
    assert live["automated_security_fixes"] == "dependabot_disabled"
    baseline = json.loads(BASELINE_PATH.read_text())
    assert any(key == "automated_security_fixes" for key, _, _ in diff_states(baseline, live))
