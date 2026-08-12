#!/usr/bin/env python3
"""Repository-settings drift check (M3-SEC-01).

Compares the live GitHub settings of this repository against the approved
baseline in ``security/repo-settings-baseline.json`` and exits non-zero when
any governed setting has drifted. The baseline is the reviewed source of
truth: a deliberate settings change updates the baseline in the same PR, so
anything this script reports is unapproved drift.

Only governed fields are read and printed — no URLs, node ids, or tokens —
so the output is safe to attach to release evidence (issue #12).

Usage:
    python scripts/check_repo_settings.py          # human-readable diff
    python scripts/check_repo_settings.py --json   # sanitized live snapshot

Requires an authenticated ``gh`` CLI with admin read access to the repo.
"""

import argparse
import json
import pathlib
import subprocess
import sys

REPO = "PSDN-AI/SONAR-OSS"
BASELINE_PATH = pathlib.Path(__file__).resolve().parent.parent / "security" / "repo-settings-baseline.json"


def gh_api(path: str) -> tuple[int, dict | list | None]:
    """Run ``gh api <path>`` and return (exit_code, parsed_json_or_None)."""
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}{path}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return proc.returncode, None
    body = proc.stdout.strip()
    return 0, json.loads(body) if body else None


def collect_live_state() -> dict:
    """Read live settings and shape them exactly like the baseline file."""
    code, prot = gh_api("/branches/main/protection")
    if code != 0 or prot is None:
        sys.exit("ERROR: could not read branch protection (is gh authenticated with admin access?)")
    reviews = prot.get("required_pull_request_reviews", {})
    checks = prot.get("required_status_checks", {})

    _, actions = gh_api("/actions/permissions")
    _, selected = gh_api("/actions/permissions/selected-actions")
    _, wf_perms = gh_api("/actions/permissions/workflow")
    _, repo = gh_api("")
    alerts_code, _ = gh_api("/vulnerability-alerts")
    _, autofix = gh_api("/automated-security-fixes")
    _, envs = gh_api("/environments")

    environments = {}
    for env in (envs or {}).get("environments", []):
        name = env["name"]
        _, policies = gh_api(f"/environments/{name}/deployment-branch-policies")
        environments[name] = {
            "can_admins_bypass": env.get("can_admins_bypass"),
            "custom_branch_policies": bool((env.get("deployment_branch_policy") or {}).get("custom_branch_policies")),
            "tag_policies": sorted(
                p["name"] for p in (policies or {}).get("branch_policies", []) if p.get("type") == "tag"
            ),
        }

    return {
        "branch_protection_main": {
            "required_status_checks": {
                "strict": checks.get("strict"),
                "checks": sorted(f"{c['context']}@{c['app_id']}" for c in checks.get("checks", [])),
            },
            "required_pull_request_reviews": {
                "required_approving_review_count": reviews.get("required_approving_review_count"),
                "require_code_owner_reviews": reviews.get("require_code_owner_reviews"),
                "dismiss_stale_reviews": reviews.get("dismiss_stale_reviews"),
            },
            "enforce_admins": prot.get("enforce_admins", {}).get("enabled"),
            "required_conversation_resolution": prot.get("required_conversation_resolution", {}).get("enabled"),
            "allow_force_pushes": prot.get("allow_force_pushes", {}).get("enabled"),
            "allow_deletions": prot.get("allow_deletions", {}).get("enabled"),
            "lock_branch": prot.get("lock_branch", {}).get("enabled"),
        },
        "actions_permissions": {
            "enabled": (actions or {}).get("enabled"),
            "allowed_actions": (actions or {}).get("allowed_actions"),
            "sha_pinning_required": (actions or {}).get("sha_pinning_required"),
        },
        "selected_actions": {
            "github_owned_allowed": (selected or {}).get("github_owned_allowed"),
            "verified_allowed": (selected or {}).get("verified_allowed"),
            "patterns_allowed": sorted((selected or {}).get("patterns_allowed", [])),
        },
        "workflow_permissions": {
            "default_workflow_permissions": (wf_perms or {}).get("default_workflow_permissions"),
            "can_approve_pull_request_reviews": (wf_perms or {}).get("can_approve_pull_request_reviews"),
        },
        "repo_flags": {
            "visibility": (repo or {}).get("visibility"),
            "has_wiki": (repo or {}).get("has_wiki"),
            "allow_squash_merge": (repo or {}).get("allow_squash_merge"),
            "allow_merge_commit": (repo or {}).get("allow_merge_commit"),
            "allow_rebase_merge": (repo or {}).get("allow_rebase_merge"),
            "allow_auto_merge": (repo or {}).get("allow_auto_merge"),
            "delete_branch_on_merge": (repo or {}).get("delete_branch_on_merge"),
        },
        "vulnerability_alerts": alerts_code == 0,
        "automated_security_fixes": bool((autofix or {}).get("enabled")),
        "environments": environments,
    }


def flatten(value, prefix: str = "") -> dict:
    """Flatten nested dicts/lists into {dotted.path: scalar}."""
    flat: dict = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            if key.startswith("_"):
                continue
            flat.update(flatten(sub, f"{prefix}{key}." if prefix else f"{key}."))
    elif isinstance(value, list):
        flat[prefix.rstrip(".")] = json.dumps(value, sort_keys=True)
    else:
        flat[prefix.rstrip(".")] = value
    return flat


def diff_states(baseline: dict, live: dict) -> list[tuple[str, object, object]]:
    """Return [(setting, expected, actual)] for every mismatched or missing key."""
    flat_base = flatten(baseline)
    flat_live = flatten(live)
    drift = []
    for key in sorted(set(flat_base) | set(flat_live)):
        expected = flat_base.get(key, "<not in baseline>")
        actual = flat_live.get(key, "<not readable>")
        if expected != actual:
            drift.append((key, expected, actual))
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the sanitized live snapshot as JSON and exit (no diff)",
    )
    args = parser.parse_args()

    live = collect_live_state()
    if args.json:
        print(json.dumps(live, indent=2, sort_keys=True))
        return 0

    baseline = json.loads(BASELINE_PATH.read_text())
    drift = diff_states(baseline, live)
    if not drift:
        print(f"OK: live settings for {REPO} match {BASELINE_PATH.name}")
        return 0

    print(f"DRIFT: {len(drift)} setting(s) differ from {BASELINE_PATH.name}\n")
    width = max(len(key) for key, _, _ in drift)
    for key, expected, actual in drift:
        print(f"  {key.ljust(width)}  expected={expected!r}  actual={actual!r}")
    print("\nEither revert the setting or update the baseline via a reviewed PR.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
