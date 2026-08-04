# Releasing

`psdn-sonar` is published through `.github/workflows/release.yml`, triggered by
pushing a `v*` tag. Publishing uses PyPI Trusted Publishing (OIDC): GitHub
Actions exchanges its identity for a short-lived upload credential, so **no
API token exists anywhere** — not in the repo, not in GitHub secrets, not in
1Password. If you ever find yourself creating a PyPI token, stop; something is
wrong with the process.

Current scope: releases go to **TestPyPI** (M3-REL-02b). The production PyPI
job lands with M3-REL-02c.

## When to cut a release

A change needs a new package version when delivering it changes what users
receive from `pip install psdn-sonar`: installed behavior or results, public
interfaces, runtime dependencies, install metadata, or packaged resources.

Needing a new version does not mean every qualifying merge must be published
immediately. Qualifying changes may be grouped into the next planned release.
Security fixes and urgent correctness fixes should be published promptly.

A new package version is required to deliver:

- runtime behavior changes, including bug fixes and new features;
- public API, CLI, configuration-schema, or output-format changes;
- runtime dependency, optional-extra, or supported-Python-version changes;
- packaged runtime configuration, defaults, or other resources; and
- packaged loanword cache changes, because they can change normalization and
  WER/CER results.

A standalone production package release is normally not required for
documentation-only, test-only, CI/GitHub Actions, repository-settings, or
behavior-preserving internal-refactor changes. They may ride with the next
package release.

Use the following version guidance:

- **Patch:** compatible bug fixes, limited configuration corrections, and small
  loanword cache corrections or additions.
- **Minor:** new compatible features, languages, metrics, or substantial
  resource changes expected to change results materially.
- **Breaking:** incompatible public API, configuration, output, or installation
  changes use a major version after `1.0`; while the project is on `0.x`, bump
  the minor version and call out the break explicitly.
- **Development or release candidate:** use `.devN` for development/TestPyPI
  rehearsals and `rcN` for release candidates.

Maintainers choose patch or minor based on the expected user and result impact.

For a loanword cache change, the version-bump PR and production release notes
must identify the affected languages, describe whether entries were corrected,
added, removed, or broadly refreshed, and state the expected normalization or
scoring impact.

Long-term supported-version, release-cadence, SBOM, extras-matrix, and rollback
policy remains tracked in [M3-REL-02d](https://github.com/PSDN-AI/SONAR-OSS/issues/52).

## Prerequisites

- `scripts/check_installed_package.py` must exist on the tagged commit — the
  pre-publish verification job runs it.
- The version lives in **three** files that must agree: `pyproject.toml`
  (`[project].version`), `uv.lock` (it records the project's own version),
  and `psdn_sonar/__init__.py` (`__version__`). The workflow refuses to
  build on drift — `uv lock --check` catches a stale lockfile, the release
  gate catches `__init__.py`.
- The person pushing the tag needs push access to `v*` refs.

## Cutting a release

1. Open a version-bump PR and merge it. Let `uv` keep the first two files
   in sync, then mirror the string into `__init__.py` by hand:

   ```bash
   uv version 0.1.0.dev1 --no-sync   # updates pyproject.toml AND uv.lock
   # then set __version__ = "0.1.0.dev1" in psdn_sonar/__init__.py
   ```

   Editing `pyproject.toml` by hand without regenerating the lockfile makes
   `uv lock --check` fail the release build.
2. Tag the merge commit with an annotated tag and push it:

   ```bash
   git switch main && git pull
   git tag -a v0.1.0.dev1 -m "psdn-sonar 0.1.0.dev1"
   git push origin v0.1.0.dev1
   ```

   The tag must be `v` + a PEP 440 version (`v0.1.0`, `v0.1.0.dev1`,
   `v1.2.0rc1`). Anything else fails the release gate before building.
3. Watch the `Release` workflow run. On success there is a GitHub Release for
   the tag carrying both distributions, their SHA-256 checksums, the source
   commit, and the smoke-test result.

A `workflow_dispatch` run of the same workflow is a **dry run**: it builds and
verifies the wheel but never publishes, regardless of the ref it targets.

## What the pipeline guarantees

| Job | Guarantee |
| --- | --- |
| Build distributions | Tag, `pyproject.toml`, and `__init__.py` agree on one version; exactly one wheel + one sdist; `twine check --strict` passes; checksums recorded |
| Verify wheel before publish | The exact wheel about to be published installs cleanly and passes `scripts/check_installed_package.py` — catches broken artifacts **before** the version number is burned |
| Publish to TestPyPI | Upload via Trusted Publishing with PEP 740 attestations; only this job holds `id-token: write` |
| Smoke test from TestPyPI | The index serves files byte-identical to the build output (digest comparison); a clean environment installs `psdn-sonar==<version>` from TestPyPI (dependencies strictly from real PyPI) and exercises a real subsystem |
| Create GitHub Release | Tag ↔ version ↔ source commit ↔ artifacts bound in one place; created whenever publish succeeded, recording the smoke result |

## Version hygiene

- An index **never accepts the same filename twice**, even after deletion. A
  published version number is spent forever on that index — this is why the
  gate and the pre-publish verification exist.
- Use `.devN` versions (`0.1.0.dev1`) for rehearsals so the plain version
  stays available: `pip` ignores dev releases unless explicitly requested.
- dev/alpha/beta/rc versions are created as GitHub **prereleases** and never
  become the repository's "Latest release"; plain and `.postN` versions do.
- The first tag push also **creates** the `psdn-sonar` project on TestPyPI and
  converts the pending Trusted Publisher registered there. Registration alone
  does not reserve the name — only the first successful publish claims it.

## Troubleshooting

- **Smoke test times out waiting for the release:** TestPyPI index propagation
  is usually seconds, occasionally minutes. Re-run failed jobs — publish is
  already done and `skip-existing: true` makes a full re-run harmless. The
  GitHub Release job is create-or-update: a re-run refreshes the existing
  release's notes (including the smoke result) and re-uploads assets in
  place, so re-running never collides with the release it already made.
- **Partial upload (one file failed):** re-run the workflow from the same tag;
  already-uploaded files are skipped, missing ones are uploaded. (This is why
  the TestPyPI job sets `skip-existing: true`. The production job must not.)
- **Gate failure on version drift:** bump the file you forgot, merge, delete
  nothing — cut the next patch/dev version instead of force-moving the tag.

## Notes and follow-ups

- Artifact attestations are signed through the Sigstore public-good instance:
  the repository name, workflow ref, and commit land in a public transparency
  log at publish time. Accepted deliberately — these names are public at
  launch anyway.
- When the repository goes public (before launch): add required reviewers to
  the `pypi` environment and a tag ruleset restricting `v*` creation to
  maintainers (tracked on M3-REL-02a/M3-REL-02c). Free-plan private repos
  cannot enforce either.
