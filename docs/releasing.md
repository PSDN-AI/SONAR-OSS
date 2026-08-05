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

   The tag must be `v` + the exact declared package version.
3. Watch the `Release` workflow run. On success there is a GitHub Release for
   the tag carrying both distributions, their SHA-256 checksums, the source
   commit, and the smoke-test result.

## What the pipeline guarantees

| Job | Guarantee |
| --- | --- |
| Build distributions | Tag, `pyproject.toml`, and `__init__.py` agree; the wheel and sdist pass metadata and clean-install checks |
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
