"""Tests for scripts/dependency_audit.py alias-aware exception matching.

Issue #248: the gate matched reviewed exceptions on the one advisory id
pip-audit reported as primary and ignored the ``aliases`` list in the same
JSON. Different pip-audit versions promote different ids of the same
advisory, so an unchanged lockfile and exceptions file passed under one
scanner version and reported four false failures — two of them instructions
to delete the exceptions keeping CI green — under another.
"""

import importlib.util
import io
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

# The audited script hard-requires tomllib (stdlib from 3.11; CI runs the
# gate on 3.12) and exits at import time on 3.10 — skip the module there
# rather than dying at collection.
pytest.importorskip("tomllib", reason="dependency_audit requires Python >= 3.11")

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location("dependency_audit", REPO_ROOT / "scripts" / "dependency_audit.py")
dependency_audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spec and dependency_audit)

FUTURE = (date.today() + timedelta(days=365)).isoformat()

EXCEPTIONS_TOML = f"""
[[exception]]
id = "GHSA-8mgp-746c-j5xp"
package = "nltk"
owner = "RN0311"
rationale = "reviewed"
review_by = {FUTURE}
removal_condition = "fixed upstream"

[[exception]]
id = "GHSA-xrqw-3rrv-vx5w"
package = "transformers"
owner = "RN0311"
rationale = "reviewed"
review_by = {FUTURE}
removal_condition = "fixed upstream"
"""


def _report(primary_swapped: bool) -> str:
    """pip-audit JSON with the same two advisories; the primary/alias roles
    swap between scanner versions (the issue's observed table)."""
    if not primary_swapped:  # pip-audit 2.9.0 shape
        nltk_vuln = {"id": "GHSA-8mgp-746c-j5xp", "aliases": ["PYSEC-2026-3740", "CVE-2026-81726"]}
        tf_vuln = {"id": "GHSA-xrqw-3rrv-vx5w", "aliases": ["CVE-2026-9856"]}
    else:  # pip-audit 2.10.1 shape
        nltk_vuln = {"id": "PYSEC-2026-3740", "aliases": ["GHSA-8mgp-746c-j5xp", "CVE-2026-81726"]}
        tf_vuln = {"id": "CVE-2026-9856", "aliases": ["GHSA-xrqw-3rrv-vx5w"]}
    return json.dumps(
        {
            "dependencies": [
                {"name": "nltk", "version": "3.10.3", "vulns": [nltk_vuln]},
                {"name": "transformers", "version": "4.57.6", "vulns": [tf_vuln]},
            ]
        }
    )


def _run_gate(monkeypatch, tmp_path, report_json: str, exceptions_toml: str = EXCEPTIONS_TOML) -> int:
    exc_file = tmp_path / "exceptions.toml"
    exc_file.write_text(exceptions_toml, encoding="utf-8")
    monkeypatch.setattr(dependency_audit, "EXCEPTIONS_PATH", exc_file)
    monkeypatch.setattr(sys, "stdin", io.StringIO(report_json))
    return dependency_audit.main()


class TestAliasAwareMatching:
    @pytest.mark.parametrize("primary_swapped", [False, True], ids=["pip-audit-2.9.0", "pip-audit-2.10.1"])
    def test_same_exceptions_pass_under_both_scanner_id_choices(self, monkeypatch, tmp_path, primary_swapped, capsys):
        exit_code = _run_gate(monkeypatch, tmp_path, _report(primary_swapped))
        out = capsys.readouterr()
        assert exit_code == 0, out.err
        assert "reviewed exceptions applied: 2" in out.out
        assert "OBSOLETE" not in out.err
        assert "NEW" not in out.err

    def test_unreviewed_finding_still_fails(self, monkeypatch, tmp_path, capsys):
        report = json.dumps(
            {"dependencies": [{"name": "leftpad", "version": "1.0", "vulns": [{"id": "GHSA-zzzz", "aliases": []}]}]}
        )
        exit_code = _run_gate(monkeypatch, tmp_path, report)
        out = capsys.readouterr()
        assert exit_code == 1
        assert "NEW: GHSA-zzzz in leftpad 1.0" in out.err

    def test_truly_obsolete_exception_still_flagged(self, monkeypatch, tmp_path, capsys):
        report = json.dumps({"dependencies": []})
        exit_code = _run_gate(monkeypatch, tmp_path, report)
        out = capsys.readouterr()
        assert exit_code == 1
        assert out.err.count("OBSOLETE") == 2

    def test_package_mismatch_still_flagged_via_alias(self, monkeypatch, tmp_path, capsys):
        # The exception names nltk but the aliased finding is in another package.
        report = json.dumps(
            {
                "dependencies": [
                    {
                        "name": "requests",
                        "version": "2.0",
                        "vulns": [{"id": "PYSEC-2026-3740", "aliases": ["GHSA-8mgp-746c-j5xp"]}],
                    }
                ]
            }
        )
        exceptions = EXCEPTIONS_TOML.replace(
            'id = "GHSA-xrqw-3rrv-vx5w"\npackage = "transformers"',
            'id = "GHSA-xrqw-3rrv-vx5w"\npackage = "requests"',
        )
        # Keep only the nltk entry relevant: the transformers entry now names
        # requests but matches nothing, so expect MISMATCH for the nltk one.
        exit_code = _run_gate(monkeypatch, tmp_path, report, exceptions_toml=exceptions)
        out = capsys.readouterr()
        assert exit_code == 1
        assert "MISMATCH: exception GHSA-8mgp-746c-j5xp names package 'nltk'" in out.err

    def test_missing_aliases_field_tolerated(self, monkeypatch, tmp_path):
        report = json.dumps(
            {
                "dependencies": [
                    {"name": "nltk", "version": "3.10.3", "vulns": [{"id": "GHSA-8mgp-746c-j5xp"}]},
                    {"name": "transformers", "version": "4.57.6", "vulns": [{"id": "GHSA-xrqw-3rrv-vx5w"}]},
                ]
            }
        )
        assert _run_gate(monkeypatch, tmp_path, report) == 0
