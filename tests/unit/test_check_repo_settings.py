"""Unit tests for the pure comparison logic in scripts/check_repo_settings.py."""

import importlib.util
import json
import pathlib
import sys

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "check_repo_settings.py"
_spec = importlib.util.spec_from_file_location("check_repo_settings", _SCRIPT)
check_repo_settings = importlib.util.module_from_spec(_spec)
sys.modules["check_repo_settings"] = check_repo_settings
_spec.loader.exec_module(check_repo_settings)

diff_states = check_repo_settings.diff_states
flatten = check_repo_settings.flatten
BASELINE_PATH = check_repo_settings.BASELINE_PATH


def test_flatten_nests_dicts_and_serializes_lists():
    flat = flatten({"a": {"b": 1, "c": [2, 1]}, "d": True})
    assert flat == {"a.b": 1, "a.c": "[2, 1]", "d": True}


def test_flatten_skips_underscore_keys():
    assert flatten({"_comment": "ignored", "a": 1}) == {"a": 1}


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
    # Spot-check governed keys the release checklist depends on.
    assert flat["branch_protection_main.enforce_admins"] is False
    assert flat["actions_permissions.allowed_actions"] == "selected"
    assert "Dependency audit@15368" in flat["branch_protection_main.required_status_checks.checks"]
