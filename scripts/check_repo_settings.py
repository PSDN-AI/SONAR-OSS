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


def gh_api(path: str, *, missing_ok: bool = False) -> dict | list | None:
    """Run ``gh api <path>`` and return the parsed JSON body ({} when empty).

    Any API failure aborts the run: a snapshot built from partial reads
    would silently match the baseline's false-y values, or pass as ``--json``
    release evidence while missing governed settings. ``missing_ok=True``
    maps an HTTP 404 to ``None`` so endpoints that report state through 404
    (e.g. ``/vulnerability-alerts``) can distinguish "off" from "unreadable".
    """
    proc = subprocess.run(
        ["gh", "api", f"repos/{REPO}{path}"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        if missing_ok and "HTTP 404" in proc.stderr:
            return None
        detail = proc.stderr.strip() or f"exit code {proc.returncode}"
        sys.exit(
            f"ERROR: gh api repos/{REPO}{path} failed: {detail}\n"
            "Refusing to emit a partial snapshot (is gh authenticated with admin access?)"
        )
    body = proc.stdout.strip()
    return json.loads(body) if body else {}


def complete_list(payload: dict | None, key: str) -> list:
    """Return ``payload[key]`` after checking ``total_count`` for truncation.

    A silently truncated page would read as a complete, drift-free listing —
    entries past the page boundary would vanish instead of surfacing.
    """
    items = (payload or {}).get(key, [])
    total = (payload or {}).get("total_count", len(items))
    if total != len(items):
        sys.exit(f"ERROR: {key} listing truncated ({len(items)} of {total}); refusing a partial snapshot")
    return items


def shape_environment(env: dict, policies: dict | None) -> dict:
    """Shape one environment (plus its deployment policies) like the baseline."""
    reviewers = sorted(
        f"{entry.get('type')}:{(entry.get('reviewer') or {}).get('login') or (entry.get('reviewer') or {}).get('slug')}"
        for rule in env.get("protection_rules", [])
        if rule.get("type") == "required_reviewers"
        for entry in rule.get("reviewers", [])
    )
    return {
        "can_admins_bypass": env.get("can_admins_bypass"),
        "custom_branch_policies": bool((env.get("deployment_branch_policy") or {}).get("custom_branch_policies")),
        "deployment_policies": sorted(
            f"{p.get('type') or 'unknown'}:{p['name']}" for p in complete_list(policies, "branch_policies")
        ),
        "reviewers": reviewers,
    }


def collect_live_state() -> dict:
    """Read live settings and shape them exactly like the baseline file."""
    prot = gh_api("/branches/main/protection")
    reviews = prot.get("required_pull_request_reviews", {})
    checks = prot.get("required_status_checks", {})

    actions = gh_api("/actions/permissions")
    selected = gh_api("/actions/permissions/selected-actions")
    wf_perms = gh_api("/actions/permissions/workflow")
    repo = gh_api("")
    alerts = gh_api("/vulnerability-alerts", missing_ok=True)
    autofix = gh_api("/automated-security-fixes", missing_ok=True)
    envs = gh_api("/environments?per_page=100")

    # /actions/permissions/access applies to private and internal repositories;
    # the public-flip PR updates the baseline to the sentinel in the same change.
    if repo.get("visibility") in ("private", "internal"):
        access_level = gh_api("/actions/permissions/access").get("access_level")
    else:
        access_level = "not_applicable_public"

    environments = {}
    for env in complete_list(envs, "environments"):
        name = env["name"]
        policies = gh_api(f"/environments/{name}/deployment-branch-policies?per_page=100", missing_ok=True)
        environments[name] = shape_environment(env, policies)

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
            "enabled": actions.get("enabled"),
            "allowed_actions": actions.get("allowed_actions"),
            "sha_pinning_required": actions.get("sha_pinning_required"),
        },
        "selected_actions": {
            "github_owned_allowed": selected.get("github_owned_allowed"),
            "verified_allowed": selected.get("verified_allowed"),
            "patterns_allowed": sorted(selected.get("patterns_allowed", [])),
        },
        "workflow_permissions": {
            "default_workflow_permissions": wf_perms.get("default_workflow_permissions"),
            "can_approve_pull_request_reviews": wf_perms.get("can_approve_pull_request_reviews"),
        },
        "actions_access_level": access_level,
        "repo_flags": {
            "visibility": repo.get("visibility"),
            "has_wiki": repo.get("has_wiki"),
            "allow_forking": repo.get("allow_forking"),
            "allow_squash_merge": repo.get("allow_squash_merge"),
            "allow_merge_commit": repo.get("allow_merge_commit"),
            "allow_rebase_merge": repo.get("allow_rebase_merge"),
            "allow_auto_merge": repo.get("allow_auto_merge"),
            "delete_branch_on_merge": repo.get("delete_branch_on_merge"),
        },
        "security_and_analysis": {
            "secret_scanning": ((repo.get("security_and_analysis") or {}).get("secret_scanning") or {}).get("status"),
            "secret_scanning_push_protection": (
                (repo.get("security_and_analysis") or {}).get("secret_scanning_push_protection") or {}
            ).get("status"),
        },
        "vulnerability_alerts": alerts is not None,
        # 404 means Dependabot itself is off (or the toggle is unreadable);
        # never collapse that to False, which would equal the baseline.
        "automated_security_fixes": "dependabot_disabled" if autofix is None else bool(autofix.get("enabled")),
        "environments": environments,
    }


def flatten(value, prefix: str = "") -> dict:
    """Flatten nested dicts/lists into {dotted.path: scalar}.

    Only top-level ``_``-prefixed keys are metadata (the baseline's
    ``_comment``); nested ones are data — e.g. an environment named
    ``_publish`` must still surface in the diff.
    """
    flat: dict = {}
    if isinstance(value, dict):
        for key, sub in value.items():
            if not prefix and key.startswith("_"):
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
