#!/usr/bin/env python3
"""Verify that a release commit is the current main tip with green CI."""

import argparse
import json
import subprocess
import sys

REPO = "PSDN-AI/SONAR-OSS"
REQUIRED_CHECKS = (
    "Dependency audit",
    "Internal-reference gate",
    "Lint and type check",
    "Pre-commit baseline",
    "Secret scan",
    "Tests (Python 3.10)",
    "Tests (Python 3.11)",
    "Tests (Python 3.12)",
    "Package artifacts",
)
ALLOWED_WORKFLOWS = {
    ".github/workflows/ci.yml",
    ".github/workflows/dependency-audit.yml",
    ".github/workflows/gitleaks.yml",
    ".github/workflows/package.yml",
    ".github/workflows/pre-commit.yml",
}


def github(path: str) -> list[dict]:
    """Fetch every page from a GitHub API endpoint."""
    result = subprocess.run(
        ["gh", "api", f"repos/{REPO}{path}", "--paginate", "--slurp"],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        sys.exit(f"GitHub API error: {result.stderr.strip()}")
    return json.loads(result.stdout)


def collect_checks(commit: str) -> dict[str, dict]:
    """Return the latest required job from trusted main workflow runs."""
    pages = github(f"/actions/runs?head_sha={commit}&per_page=100")
    runs = [run for page in pages for run in page.get("workflow_runs", [])]
    selected: dict[str, dict] = {}
    latest: dict[str, str] = {}

    for run in runs:
        if not (
            run.get("path") in ALLOWED_WORKFLOWS
            and run.get("event") in {"push", "workflow_dispatch"}
            and run.get("head_branch") == "main"
            and (run.get("head_repository") or {}).get("full_name") == REPO
            and run.get("status") == "completed"
        ):
            continue

        pages = github(f"/actions/runs/{run['id']}/jobs?filter=all&per_page=100")
        for page in pages:
            for job in page.get("jobs", []):
                name = job.get("name")
                completed_at = job.get("completed_at") or ""
                if name in REQUIRED_CHECKS and completed_at > latest.get(name, ""):
                    latest[name] = completed_at
                    selected[name] = {
                        "name": name,
                        "conclusion": job.get("conclusion"),
                        "check_run_id": job.get("id"),
                        "url": job.get("html_url"),
                    }

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("commit", nargs="?", help="40-hex commit SHA (default: current main)")
    parser.add_argument("--tag", help="tag to record in release evidence")
    parser.add_argument("--evidence", help="write release evidence JSON to this path")
    args = parser.parse_args()

    main_commit = github("/git/ref/heads/main")[0]["object"]["sha"]
    commit = args.commit or main_commit
    failures = []

    if commit != main_commit:
        failures.append(f"candidate is not the current main commit ({main_commit})")
        selected = {}
    else:
        selected = collect_checks(commit)
        for name in REQUIRED_CHECKS:
            check = selected.get(name)
            if check is None:
                failures.append(f"missing required check: {name}")
            elif check["conclusion"] != "success":
                failures.append(f"required check {name!r} concluded {check['conclusion']!r}")

    checks = [selected[name] for name in REQUIRED_CHECKS if name in selected]
    evidence = {
        "verified": not failures,
        "tag": args.tag,
        "commit": commit,
        "main_commit": main_commit,
        "checks": checks,
        "failures": failures,
    }
    if args.evidence:
        with open(args.evidence, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2)
            handle.write("\n")

    print(f"## Release verification: {'PASS' if not failures else 'REJECTED'}\n")
    print(f"- commit: `{commit}`")
    if args.tag:
        print(f"- tag: `{args.tag}`")
    for failure in failures:
        print(f"- ❌ {failure}")
    for check in checks:
        icon = "✅" if check["conclusion"] == "success" else "❌"
        print(f"- {icon} {check['name']}: {check['conclusion']} ([{check['check_run_id']}]({check['url']}))")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
