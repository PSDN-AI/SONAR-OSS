# Releasing

`psdn-sonar` is published through `.github/workflows/release.yml`, triggered by
pushing a `v*` tag. Publishing uses PyPI Trusted Publishing (OIDC): GitHub
Actions exchanges its identity for a short-lived upload credential, so **no
API token exists anywhere** — not in the repo, not in GitHub secrets, not in
1Password. If you ever find yourself creating a PyPI token, stop; something is
wrong with the process.

Current scope: releases go to **TestPyPI** (M3-REL-02b). The production PyPI
job lands with M3-REL-02c.

Broader release policy remains tracked in
[M3-REL-02d](https://github.com/PSDN-AI/SONAR-OSS/issues/52).

## Prerequisites

- `scripts/check_installed_package.py` must exist on the tagged commit — the
  build verifies the wheel with it before publishing.
- The version lives in **three** files that must agree: `pyproject.toml`
  (`[project].version`), `uv.lock` (it records the project's own version),
  and `psdn_sonar/__init__.py` (`__version__`). The workflow refuses to
  build on drift — `uv lock --check` catches a stale lockfile, the release
  gate catches `__init__.py`. `tests/test_version.py` is not a version
  source, but its pinned expectation must change in the same version-bump PR.
- The person pushing the tag needs push access to `v*` refs.

## Cutting a release

1. Open a version-bump PR and merge it. Update exactly four paths:
   `pyproject.toml`, generated `uv.lock`, `psdn_sonar/__init__.py`, and the
   expected value in `tests/test_version.py`:

   ```bash
   uv version 0.1.0.dev1 --no-sync   # updates pyproject.toml AND uv.lock
   # then update __version__ in psdn_sonar/__init__.py
   # and the expected value in tests/test_version.py
   ```

   Editing `pyproject.toml` by hand without regenerating the lockfile makes
   `uv lock --check` fail the release build.
2. Verify the release candidate (M3-CI-01). Dispatch the
   **Release verification** workflow (Actions → Release verification → Run
   workflow; leave `ref` blank to verify the tip of `main`, or pass the merge
   commit's 40-hex SHA). It consumes the post-merge check runs already
   recorded for that exact SHA — it re-runs nothing — and passes only when
   the commit is the current tip of `main` with every required check green.
   If `main` has moved since your merge, verify the new tip instead. Tag only
   after a PASS.

   ```bash
   gh workflow run "Release verification" --ref main
   ```

3. Tag the verified commit with an annotated tag and push it:

   ```bash
   git switch main && git pull
   git tag -a v0.1.0.dev1 -m "psdn-sonar 0.1.0.dev1"
   git push origin v0.1.0.dev1
   ```

   The tag must be `v` + the exact declared package version.
4. Watch the `Release` workflow run. Its `Verify release commit` job repeats
   the verification against the tagged commit (accepting a verified ancestor
   of `main`, since another PR may have merged after your tag) and blocks
   publishing on any failure. On success there is a GitHub Release for the
   tag carrying both distributions, their SHA-256 checksums, the source
   commit, the smoke-test result, and `release-evidence.json` — the durable
   verification record (commit, per-check run links, `uv.lock` digest,
   artifact digests) consumed by the public-release gate.

## What the pipeline guarantees

| Job | Guarantee |
| --- | --- |
| Build distributions | Tag, `pyproject.toml`, and `__init__.py` agree; the wheel and sdist pass metadata and clean-install checks |
| Verify release commit | The exact built commit is on `main` and every required post-merge check succeeded on that SHA — evidence from PR heads, forks, or unrelated workflows is rejected; nothing publishes without it |
| Publish to TestPyPI | Upload via Trusted Publishing with PEP 740 attestations, then wait until the index serves files byte-identical to the build output; only this job holds `id-token: write` |
| Smoke test from TestPyPI | A clean environment installs and imports `psdn-sonar==<version>` from TestPyPI, with dependencies from real PyPI |
| Create GitHub Release | Tag ↔ version ↔ source commit ↔ artifacts bound in one place; created whenever publish succeeded, recording the smoke result |

## Troubleshooting

- **Publish verification times out or an upload is partial:** re-run failed
  jobs in the same workflow run. The publish job reuses the original build
  artifacts, skips files already uploaded, uploads missing files, and succeeds
  only after TestPyPI serves the exact expected files. The production job must
  not use `skip-existing`.
- **Gate failure on version drift:** bump the file you forgot, merge, delete
  nothing — cut the next patch/dev version instead of force-moving the tag.
- **Verification: "candidate is N commit(s) behind main":** `main` moved
  after your merge. Re-dispatch Release verification for the new tip (or, on
  the tag path, this is accepted automatically for a verified ancestor).
- **Verification: "missing required check":** the named workflow never
  completed on that SHA — dispatch it on `main` and re-verify. Note that
  workflow-run history is finite and commits from before a history rewrite
  carry no evidence; such commits cannot be released.
- **Verification: "commit is not on main":** the tag points at a commit
  outside `main`'s history. Cut a new tag from the current `main`.
