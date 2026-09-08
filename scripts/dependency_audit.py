#!/usr/bin/env python3
"""Blocking dependency audit of the committed lockfile (M1-SEC-01).

Reads pip-audit JSON findings for the exported ``uv.lock`` and compares them
against the reviewed exceptions in ``security/dependency-audit-exceptions.toml``.

Exit is non-zero when any of the following holds, so the check fails closed:

- a reported vulnerability has no exception entry (new finding);
- an exception entry matches no reported vulnerability (obsolete entry);
- an exception entry's ``review_by`` date has passed (stale entry).

An exception matches a finding when its id equals the finding's reported id
**or any of the finding's aliases**. Different pip-audit versions promote
different ids of the same advisory to primary (e.g. GHSA-8mgp-746c-j5xp
vs. its alias PYSEC-2026-3740), so an exact-id match made the gate's verdict
depend on the scanner version — one unchanged lockfile and exceptions file
passed under the CI pin and reported four false failures under a newer
pip-audit, two of them instructions to delete the exceptions keeping CI
green (issue #248).

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


def load_findings(stream) -> list:
    """List of findings from pip-audit JSON.

    Each finding carries ``ids`` — the reported id plus every alias — so a
    reviewed exception keyed on any of the advisory's ids keeps matching
    regardless of which id the scanner version reports as primary.
    """
    report = json.load(stream)
    findings = []
    for dep in report.get("dependencies", []):
        for vuln in dep.get("vulns", []):
            aliases = [alias for alias in (vuln.get("aliases") or []) if alias]
            findings.append(
                {
                    "id": vuln["id"],
                    "ids": {vuln["id"], *aliases},
                    "package": dep["name"],
                    "version": dep["version"],
                    "fix_versions": vuln.get("fix_versions", []),
                }
            )
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
    matched_exception_ids = set()

    for finding in sorted(findings, key=lambda f: (f["package"], f["id"])):
        # An exception counts if it names the reported id or any alias.
        matches = [eid for eid in sorted(finding["ids"]) if eid in exceptions]
        matched_exception_ids.update(matches)
        if not matches:
            fixes = ", ".join(finding["fix_versions"]) or "none published"
            failures.append(
                f"NEW: {finding['id']} in {finding['package']} {finding['version']} "
                f"(fix: {fixes}) has no reviewed exception"
            )
            continue
        for eid in matches:
            entry = exceptions[eid]
            if entry["package"].lower() != finding["package"].lower():
                failures.append(
                    f"MISMATCH: exception {eid} names package {entry['package']!r} "
                    f"but the finding is in {finding['package']!r}"
                )

    for advisory_id, entry in sorted(exceptions.items()):
        if advisory_id not in matched_exception_ids:
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

    excepted = sorted(matched_exception_ids)
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
