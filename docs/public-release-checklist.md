# Public release checklist

Coordinates the evidence required to make this repository public (M3-REL-01,
issue #12). Section 1 records the repository-security state required by
M3-SEC-01 (issue #25); its canonical check names are shared with the
release-commit validation work in M3-CI-01 (issue #24). Sections 2+ are
placeholders owned by issue #12 and are filled as their source issues close.

Settings changes below were applied and verified via the GitHub API on
2026-08-12; sanitized before/after readbacks are attached to issue #25.

## 1. Repository security settings (M3-SEC-01, #25)

### 1.1 Branch protection — `main`

| Setting | Approved value (2026-08-12) | Target at public flip |
| --- | --- | --- |
| Required status checks | 9 checks (§1.3), `strict: true` | same |
| `required_approving_review_count` | 0 — waiver W1 | ≥ 1 |
| `require_code_owner_reviews` | false — waiver W1 | true |
| `dismiss_stale_reviews` | false — waiver W1 | true |
| `enforce_admins` | false — exception E1 | true |
| `required_conversation_resolution` | true | true |
| `allow_force_pushes` / `allow_deletions` | false / false | false / false |

### 1.2 Documented waivers and exceptions

- **W1 — zero required approving reviews.** Approved by @AndyBoWu,
  2026-08-12. Rationale: two-maintainer review capacity during the launch
  window would serialize every in-flight launch PR. Expires at the public
  visibility flip (target 2026-08-15), when the review requirements in §1.1
  move to their target values. Compensating controls: 9 required checks
  including secret scan and dependency audit, required conversation
  resolution, and a private repository with an org-managed collaborator set.
- **E1 — administrators not bound by branch protection.** Approved by
  @AndyBoWu, 2026-08-12, as a break-glass path during the launch window.
  Note: because admin access is inherited from organization ownership, the
  bypass population is managed outside this repository and is not a fixed
  list — this is why E1 must not outlive the flip. Expires at the public
  visibility flip (`enforce_admins` → true, §1.7). Compensating controls:
  settings drift check (§1.9) plus readback snapshots attached to issue #25.

### 1.3 Canonical required status checks

The names below are the exact job `name:` values; branch protection binds
each to the GitHub Actions app (`app_id: 15368`).

| Check | Source workflow | Runs on |
| --- | --- | --- |
| Pre-commit baseline | pre-commit.yml | PR + push to main |
| Secret scan | gitleaks.yml | PR + push to main |
| Validate PR title | pr-title.yml | PR only (by design) |
| Internal-reference gate | ci.yml | PR + push to main |
| Lint and type check | ci.yml | PR + push to main |
| Tests (Python 3.10 / 3.11 / 3.12) | ci.yml | PR + push to main |
| Dependency audit | dependency-audit.yml | PR + push to main |

Operational notes:

- **API mechanics.** Update required checks only via
  `PATCH /repos/{owner}/{repo}/branches/main/protection/required_status_checks`
  with the full `checks` array including `app_id` on every entry. Writing the
  legacy `contexts` field resets every entry to `app_id: null`, which lets any
  status-posting integration satisfy the check — a silent downgrade.
- **PR-only contexts.** "Validate PR title" never appears on a main-push SHA
  (its workflow has no push trigger; title validation is meaningless
  post-merge). Release tooling built for issue #24 must not expect it on the
  merge commit. The same applies to the labeler checks ("Apply path labels",
  "Require area label") if they are ever added to the required set.
- **Dependency audit freeze date.** Every exception in
  `security/dependency-audit-exceptions.toml` currently carries
  `review_by = 2026-10-15`. Because the audit fails closed on stale entries
  and is now a required check, letting that date pass unreviewed freezes all
  merges. Owner: @RN0311 — re-review the five entries before 2026-10-08.
- **Package artifacts** is deliberately not required during the launch
  window: it is a build-quality gate, and under `strict: true` every added
  check lengthens the update-and-rerun cycle for all open PRs. Revisit after
  the flip.

### 1.4 Actions policy

| Setting | Value |
| --- | --- |
| `default_workflow_permissions` | `read` |
| `can_approve_pull_request_reviews` | false |
| `allowed_actions` | `selected`: GitHub-owned + `astral-sh/setup-uv@*`, `amannn/action-semantic-pull-request@*`, `pypa/gh-action-pypi-publish@*` |
| `sha_pinning_required` | true |
| Workflow access from other repos | `none` |

Adding a new third-party action requires updating the allowlist pattern and
`security/repo-settings-baseline.json` together, via a reviewed PR.

### 1.5 Dependency security features

- **Dependabot alerts + dependency graph: enabled** (2026-08-12). Free on
  private repositories; not plan-gated.
- **Dependabot automated security fixes: deliberately off.** Dependency
  updates flow through the locked-lockfile deliberate-update policy
  (M1-SEC-01, issue #19) with reviewed, time-bound exceptions; automatic
  version-bump PRs would bypass that review. For the same reason there is
  no `.github/dependabot.yml`: version updates stay disabled for every
  ecosystem, including GitHub Actions, whose SHA pins maintainers update
  manually (CONTRIBUTING.md). Dependabot alerts are a repository setting
  and do not require that file. Re-introducing it is an ordinary PR owned
  by the default CODEOWNERS rule (both maintainers) — merge-blocking once
  `require_code_owner_reviews` turns on at the flip (§1.7); until then the
  guard is the update policy itself.

### 1.6 Deployment environments

| Environment | Deployments restricted to | Required reviewers | Admins can bypass |
| --- | --- | --- | --- |
| `pypi` | `v*` tags | plan-gated while private — added at flip (§1.7) | no |
| `testpypi` | `v*` tags | none (rehearsals stay unattended) | yes |

Required reviewers on `pypi` were rejected by the API on the current plan for
a private repository (verified 2026-08-12); they become available at the
public flip and must be in place before issue #51 wires the production
publish job.

The drift check (§1.9) records each environment's deployment policies with
their type (`tag:v*`, and any `branch:*` entry that appears), its required
reviewers, and the admin-bypass flag, so loosening any of them — adding a
branch policy, or the `pypi` reviewer going missing after the flip — fails
the check.

### 1.7 Post-public flip steps

Owner for all steps: @AndyBoWu. Execute on flip day, then re-run the drift
check and update `security/repo-settings-baseline.json` in the same change;
attach the fresh readback to issue #12.

1. `enforce_admins` → true (closes E1).
2. `required_approving_review_count` → 1, `require_code_owner_reviews` →
   true, `dismiss_stale_reviews` → true (closes W1).
3. Enable secret scanning and push protection (free once public).
4. Enable private vulnerability reporting; drop the "if it is enabled" hedge
   in SECURITY.md.
5. Add required reviewer(s) to the `pypi` environment — before #51 lands.
6. Verify `allow_forking` = true (fork-based contributions must work).
7. UI-only check: fork pull-request workflow approval set to "require
   approval for all outside collaborators" (no API readback exists).
8. Create a tag ruleset protecting `v*` tags (rulesets are free on public
   repositories; replaces compensating control C2).

### 1.8 Compensating controls register

| ID | Gap | Compensating control | Owner | Review |
| --- | --- | --- | --- | --- |
| C1 | GitHub-native secret scanning unavailable while private | Pinned, checksum-verified gitleaks binary runs a full-history scan in CI (`Secret scan`, see docs/import-gate.md) | @AndyBoWu | at flip (§1.7 step 3) |
| C2 | No tag-immutability ruleset on the current plan | release.yml re-resolves the tag at runtime and aborts if it moved off the built commit | @AndyBoWu | at flip (§1.7 step 8) |
| C3 | `uvx`-installed tools (pip-audit, twine) have no explicit in-workflow checksum | uv verifies downloaded wheels against the package index's recorded hashes during resolution | @RN0311 | post-launch |

### 1.9 Drift detection and evidence

Run `python scripts/check_repo_settings.py` (requires an authenticated `gh`
CLI with admin read access). It compares live settings against
`security/repo-settings-baseline.json` and exits non-zero on drift;
`--json` emits a sanitized snapshot for attaching as release evidence.
The baseline changes only via reviewed PR.

Known evidence gap: organization-level Actions policy and org rulesets are
not readable without `admin:org` scope, so the readback cannot prove the
absence of conflicting org-level policy. Owner: @AndyBoWu — capture an
org-level readback (or record that none exists) before issue #12 closes.

## 2. Release-commit evidence (M3-CI-01, #24) — pending

Exact-SHA validation of the release commit on `main`, using the canonical
check names in §1.3. Filled by issue #24.

## 3. Import, data rights, and synchronization (#8, #20, #21) — pending

## 4. Install and dependency evidence (#9, #19) — pending

## 5. Reproducibility evidence (#10, #22, #23) — pending

Requires an explicit scope decision (implement minimal versions vs documented
waiver with owner and expiry) before issue #12 can close.

## 6. Package distribution (#11: #49, #50, #51) — pending

## 7. External scanner disposition (#14) — pending

## 8. Clean-environment quickstart — pending

## 9. Final human sign-off — pending

Named human approval, first production publish, and the repository
visibility flip are human-only actions (per issue #12) and are never
performed by automation.
