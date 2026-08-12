#!/usr/bin/env python3
"""Exact release-commit verification (M3-CI-01).

Proves that a proposed release candidate — a tag name or a 40-hex commit
SHA — is a commit on ``main`` whose post-merge CI evidence is complete and
green, by consuming the check runs GitHub Actions already recorded for that
exact SHA. It never re-runs jobs: re-running would prove "the code passes
now", not "the merged commit everyone reviewed passed".

Rejected, with all reasons collected rather than the first one found:
branch names, abbreviated SHAs, tag objects, commits not on ``main``,
commits behind ``main`` (unless ``--allow-behind``, used on the tag-push
release path where the tagged commit may be a verified ancestor), evidence
from PR heads / forks / unrelated workflows, and missing, cancelled, or
unsuccessful required checks.

The verifier runs from a checkout of the commit under test, so a hostile
commit could weaken it — the same trust model as every other CI job here;
that commit sits behind branch protection and review.

Usage:
    python scripts/verify_release_commit.py [REF] [--allow-behind]
        [--tag NAME] [--evidence PATH]

REF is a tag name or 40-hex commit SHA; omitted means the current tip of
``main``. Exit codes: 0 verified, 1 rejected, 2 usage or API error.
Requires ``gh`` authenticated with contents:read + actions:read.
"""

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys

REPO = "PSDN-AI/SONAR-OSS"

# Canonical job names (docs/public-release-checklist.md §1.3). Branch
# protection requires the first group on every PR and main push; "Package
# artifacts" runs on every main push but is deliberately not a PR merge
# gate, so it is required here — for release — instead.
BRANCH_PROTECTION_MAIN_CHECKS = (
    "Dependency audit",
    "Internal-reference gate",
    "Lint and type check",
    "Pre-commit baseline",
    "Secret scan",
    "Tests (Python 3.10)",
    "Tests (Python 3.11)",
    "Tests (Python 3.12)",
)
RELEASE_ONLY_CHECKS = ("Package artifacts",)
# Extension point for issue #23: when runtime manifest/schema validation
# lands as a push-to-main job, add its exact job name here and a row to
# checklist §1.3. Nothing else changes.
SCHEMA_CHECKS: tuple[str, ...] = ()
REQUIRED_CHECKS = BRANCH_PROTECTION_MAIN_CHECKS + RELEASE_ONLY_CHECKS + SCHEMA_CHECKS

# Only runs from these workflow files are acceptable evidence. Filtering by
# the posting app alone is insufficient: Dependabot's update-graph posts
# check runs under the same GitHub Actions app id (15368).
ALLOWED_WORKFLOWS = frozenset(
    f".github/workflows/{name}"
    for name in ("ci.yml", "gitleaks.yml", "package.yml", "dependency-audit.yml", "pre-commit.yml")
)
ALLOWED_EVENTS = frozenset({"push", "workflow_dispatch"})


def gh_api(path: str, *, accept: str | None = None, binary: bool = False):
    """Run ``gh api`` and return parsed JSON (or raw bytes). None on 404."""
    cmd = ["gh", "api", f"repos/{REPO}{path}"]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        if b"HTTP 404" in proc.stderr or b"Not Found" in proc.stderr:
            return None
        sys.exit(f"ERROR: gh api {path} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout if binary else json.loads(proc.stdout.decode())


def resolve_ref(ref: str | None) -> tuple[str, list[str]]:
    """Resolve None (main HEAD), a 40-hex SHA, or a tag name to a commit SHA.

    Never passes user input to an endpoint that accepts branch names, so
    branch names are rejected structurally. Returns (sha, failures).
    """
    if ref is None:
        head = gh_api("/git/ref/heads/main")
        return head["object"]["sha"], []
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        commit = gh_api(f"/git/commits/{ref}")  # commit objects only; 404 on tag/tree/blob SHAs
        if commit is None:
            return "", [f"{ref} is not a commit in this repository"]
        return commit["sha"], []
    name = ref.removeprefix("refs/tags/")
    obj = gh_api(f"/git/ref/tags/{name}")
    if obj is None or not isinstance(obj, dict) or obj.get("ref") != f"refs/tags/{name}":
        return "", [f"{ref!r} is neither a 40-hex commit SHA nor an existing tag"]
    target = obj["object"]
    if target["type"] == "tag":  # annotated tag: dereference the tag object
        target = gh_api(f"/git/tags/{target['sha']}")["object"]
    if target["type"] != "commit":
        return "", [f"tag {name} resolves to a {target['type']}, not a commit"]
    return target["sha"], []


def classify_position(compare: dict | None, allow_behind: bool) -> tuple[str, list[str]]:
    """Map a compare/main...{sha} response to (position label, failures)."""
    if compare is None:
        return "unknown", ["commit not comparable to main (unknown to the repository)"]
    status = compare.get("status")
    if status == "identical":
        return "identical", []
    if status == "behind":
        behind = compare.get("behind_by", "?")
        label = f"behind:{behind}"
        if allow_behind:
            return label, []
        return label, [f"candidate is {behind} commit(s) behind main — re-verify the current tip"]
    return str(status), [f"commit is not on main (compare status: {status})"]


def accept_run(run: dict) -> str | None:
    """Return a rejection reason for a workflow run, or None if acceptable."""
    if run.get("path") not in ALLOWED_WORKFLOWS:
        return f"workflow {run.get('path')!r} is not an evidence source"
    if run.get("event") not in ALLOWED_EVENTS:
        return f"event {run.get('event')!r} (PR-head or scheduled evidence is not acceptable)"
    if run.get("head_branch") != "main":
        return f"head_branch {run.get('head_branch')!r} is not main"
    if (run.get("head_repository") or {}).get("full_name") != REPO:
        return "run originated from a fork"
    if run.get("status") != "completed":
        return f"run is {run.get('status')!r}, not completed"
    return None


def select_checks(accepted_runs: list[dict], jobs_by_run: dict[int, list[dict]]) -> dict[str, dict]:
    """Pick the latest completed job per required check name (latest-wins by completed_at,
    matching branch-protection re-run semantics)."""
    selected: dict[str, dict] = {}
    for run in accepted_runs:
        for job in jobs_by_run.get(run["id"], []):
            name = job.get("name")
            if name not in REQUIRED_CHECKS:
                continue
            record = {
                "name": name,
                "conclusion": job.get("conclusion"),
                "check_run_id": job.get("id"),
                "url": job.get("html_url"),
                "workflow_path": run.get("path"),
                "workflow_run_id": run["id"],
                "run_attempt": run.get("run_attempt"),
                "event": run.get("event"),
                "completed_at": job.get("completed_at") or "",
            }
            current = selected.get(name)
            if current is None or record["completed_at"] > current["completed_at"]:
                selected[name] = record
    return selected


def evaluate_checks(selected: dict[str, dict]) -> list[str]:
    """One failure per required check that is missing or not successful."""
    failures = []
    for name in REQUIRED_CHECKS:
        job = selected.get(name)
        if job is None:
            failures.append(f"missing required check: {name}")
        elif job["conclusion"] != "success":
            failures.append(f"required check {name!r} concluded {job['conclusion']!r}")
    return failures


def lockfile_digest(sha: str) -> str | None:
    raw = gh_api(f"/contents/uv.lock?ref={sha}", accept="application/vnd.github.raw", binary=True)
    return hashlib.sha256(raw).hexdigest() if raw else None


def render_markdown(evidence: dict) -> str:
    lines = [
        f"## Release verification: {'PASS' if evidence['verified'] else 'REJECTED'}",
        "",
        f"- commit: `{evidence['commit'] or '(unresolved)'}`"
        + (f" (tag `{evidence['tag']}`)" if evidence["tag"] else ""),
        f"- main position: {evidence['main_position']} (main HEAD `{evidence['main_head'] or '?'}`)",
    ]
    for failure in evidence["failures"]:
        lines.append(f"- ❌ {failure}")
    for check in evidence["checks"]:
        lines.append(f"- ✅ {check['name']} ([{check['check_run_id']}]({check['url']}))")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ref", nargs="?", default=None, help="tag name or 40-hex commit SHA (default: main HEAD)")
    parser.add_argument("--allow-behind", action="store_true", help="accept a verified ancestor of main (tag path)")
    parser.add_argument("--tag", default=None, help="tag name to record in the evidence (release path)")
    parser.add_argument("--evidence", default=None, help="write the evidence JSON to this path")
    args = parser.parse_args()

    failures: list[str] = []
    sha, resolve_failures = resolve_ref(args.ref)
    failures += resolve_failures

    compare = gh_api(f"/compare/main...{sha}") if sha else None
    position, position_failures = classify_position(compare, args.allow_behind)
    failures += position_failures
    main_head = (compare or {}).get("base_commit", {}).get("sha")

    selected: dict[str, dict] = {}
    if sha:
        runs = (gh_api(f"/actions/runs?head_sha={sha}&per_page=100") or {}).get("workflow_runs", [])
        accepted, rejected = [], []
        for run in runs:
            reason = accept_run(run)
            (accepted if reason is None else rejected).append((run, reason))
        jobs_by_run = {
            run["id"]: (gh_api(f"/actions/runs/{run['id']}/jobs?filter=latest&per_page=100") or {}).get("jobs", [])
            for run, _ in accepted
        }
        selected = select_checks([run for run, _ in accepted], jobs_by_run)
        check_failures = evaluate_checks(selected)
        if check_failures and rejected:
            summary = ", ".join(sorted({reason for _, reason in rejected}))
            check_failures.append(f"({len(rejected)} run(s) rejected as evidence: {summary})")
        failures += check_failures

    evidence = {
        "schema": "sonar-release-evidence/v1",
        "repository": REPO,
        "verified": not failures,
        "verified_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "tag": args.tag,
        "commit": sha or None,
        "commit_url": f"https://github.com/{REPO}/commit/{sha}" if sha else None,
        "main_head": main_head,
        "main_position": position,
        "uv_lock_sha256": lockfile_digest(sha) if sha and not failures else None,
        "checks": [selected[name] for name in REQUIRED_CHECKS if name in selected],
        "artifacts": [],  # merged in from the release build's SHA256SUMS.txt by release.yml
        "failures": failures,
    }
    if args.evidence:
        with open(args.evidence, "w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2)
            handle.write("\n")

    print(render_markdown(evidence))
    return 0 if evidence["verified"] else 1


if __name__ == "__main__":
    sys.exit(main())
