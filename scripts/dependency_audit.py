#!/usr/bin/env python3
"""Blocking dependency audit of the committed lockfile (M1-SEC-01).

Reads pip-audit JSON findings for the exported ``uv.lock`` and compares them
against the reviewed exceptions in ``security/dependency-audit-exceptions.toml``.

Exit is non-zero when any of the following holds, so the check fails closed:

- a reported vulnerability has no exception entry (new finding);
- an exception entry matches no reported vulnerability (obsolete entry);
- an exception entry's ``review_by`` date has passed (stale entry).

Usage: pip-audit -r <requirements> --format json | python scripts/dependency_audit.py
"""

import datetime
import json
import pathlib
import sys

try:
    import tomllib
except ImportError:  # Python 3.10: tomllib landed in 3.11
    sys.exit("ERROR: scripts/dependency_audit.py requires Python >= 3.11 (CI runs it on 3.12)")

EXCEPTIONS_PATH = pathlib.Path(__file__).resolve().parent.parent / "security" / "dependency-audit-exceptions.toml"


def load_findings(stream) -> dict:
    """Map advisory ID -> {package, version, fix_versions} from pip-audit JSON."""
    report = json.load(stream)
    findings = {}
    for dep in report.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            findings[vuln["id"]] = {
                "package": dep["name"],
                "version": dep["version"],
                "fix_versions": vuln.get("fix_versions", []),
            }
    return findings


def load_exceptions() -> dict:
    """Map advisory ID -> exception entry, validating required fields."""
    data = tomllib.loads(EXCEPTIONS_PATH.read_text(encoding="utf-8"))
    required = {"id", "package", "owner", "rationale", "review_by", "removal_condition"}
    exceptions = {}
    for entry in data.get("exception", []):
        missing = required - entry.keys()
        if missing:
            sys.exit(f"ERROR: exception entry {entry.get('id', '<no id>')} is missing fields: {sorted(missing)}")
        exceptions[entry["id"]] = entry
    return exceptions


def main() -> int:
    findings = load_findings(sys.stdin)
    exceptions = load_exceptions()
    today = datetime.date.today()
    failures = []

    for advisory_id, finding in sorted(findings.items()):
        entry = exceptions.get(advisory_id)
        if entry is None:
            fixes = ", ".join(finding["fix_versions"]) or "none published"
            failures.append(
                f"NEW: {advisory_id} in {finding['package']} {finding['version']} "
                f"(fix: {fixes}) has no reviewed exception"
            )
        elif entry["package"].lower() != finding["package"].lower():
            failures.append(
                f"MISMATCH: exception {advisory_id} names package {entry['package']!r} "
                f"but the finding is in {finding['package']!r}"
            )

    for advisory_id, entry in sorted(exceptions.items()):
        if advisory_id not in findings:
            failures.append(
                f"OBSOLETE: exception {advisory_id} ({entry['package']}) matches no current finding — remove it"
            )
            continue
        review_by = datetime.date.fromisoformat(str(entry["review_by"]))
        if review_by < today:
            failures.append(
                f"STALE: exception {advisory_id} ({entry['package']}) passed its review date "
                f"{review_by} — re-review or remove (owner: {entry['owner']})"
            )

    excepted = sorted(set(findings) & set(exceptions))
    print(f"Findings: {len(findings)} | reviewed exceptions applied: {len(excepted)} | failures: {len(failures)}")
    for advisory_id in excepted:
        entry = exceptions[advisory_id]
        print(f"  excepted: {advisory_id} ({entry['package']}) review by {entry['review_by']} owner {entry['owner']}")

    if failures:
        print("\nDependency audit FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("Dependency audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
