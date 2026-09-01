# Releasing

## Purpose

This is the maintainer runbook for publishing `psdn-sonar`. Contributors do not need it; cutting a
release requires push access to tags, membership in the reviewer group on the `pypi` environment, and
owner rights on the PyPI project.

Releases are published through Trusted Publishing (OIDC). No API token for either index is stored in
this repository, in an organization secret, or on a maintainer's machine.

## Release targets

| Target | Dispatch type | GitHub environment | `skip-existing` | Used for |
| --- | --- | --- | --- | --- |
| TestPyPI | `release-testpypi` | `testpypi` | `true` | Rehearsals, `.devN` builds, pipeline changes |
| PyPI | `release-pypi` | `pypi` | `false` | Production releases, reviewer-approved |

Both targets run the same `.github/workflows/release.yml`. The dispatch type is the only thing that
selects the index; the workflow refuses a `.devN` version on the production target.

Three strings are registered in the index-side Trusted Publisher and must match the workflow exactly:
the repository `PSDN-AI/SONAR-OSS`, the workflow filename `release.yml`, and the environment name
(`testpypi` or `pypi`). Renaming any of them breaks publishing until the publisher is updated on the
index side.

## Before cutting a release

- The candidate must be the current tip of `main` with all nine required checks green. The release
  workflow enforces this and any newer `main` commit invalidates the candidate, so announce a merge
  freeze before tagging.
- `pyproject.toml` and `psdn_sonar/__init__.py` must declare the same version, and the tag must be that
  version prefixed with `v`. The workflow's release gate checks all three; every mismatch it catches is
  a version number saved from being burned on an index.
- `uv.lock` must be in sync (`uv lock --check`). Regenerate it with `uv lock`, never by hand.
- `CHANGELOG.md` must have a dated section for the version being released.
- For a production release, complete the release cut-off in
  [the upstream synchronization policy](upstream-sync-policy.md): after it, only a named release blocker
  or security fix may merge.

Dry-run the gate at any time without publishing anything:

```bash
python scripts/verify_release_commit.py            # defaults to the current main tip
```

The same check is available as the `Release verification` workflow (`workflow_dispatch`, optional
`ref` input) when you want the evidence recorded as a workflow artifact.

## Cutting a release

1. Merge the version-bump pull request. It must be the last merge before the tag.
2. Confirm the gate passes on the new tip:

   ```bash
   python scripts/verify_release_commit.py
   ```

3. Tag the exact commit and push the tag:

   ```bash
   git tag -a v0.1.0 <sha> -m "psdn-sonar 0.1.0"
   git push origin v0.1.0
   ```

4. Dispatch the release. The tag travels as `client_payload`; it is untrusted input and never the source
   of the workflow or the verifier:

   ```bash
   # rehearsal
   gh api --method POST repos/PSDN-AI/SONAR-OSS/dispatches \
     -f event_type=release-testpypi -f 'client_payload[tag]=v0.1.0.dev6'

   # production
   gh api --method POST repos/PSDN-AI/SONAR-OSS/dispatches \
     -f event_type=release-pypi -f 'client_payload[tag]=v0.1.0'
   ```

5. For a production release, a reviewer approves the `pypi` environment in the Actions run. This is the
   human approval on the release; it cannot be delegated to automation.
6. Watch `build → verify-release-commit → publish → smoke-test → github-release` finish green. The
   publish job does not report success until the index serves the exact files the build produced.

## After publishing

The workflow attaches the wheel, the sdist, `SHA256SUMS.txt`, and `release-evidence.json` (source
commit, tag, check-run evidence, artifact digests) to the GitHub Release, and the index carries PEP 740
attestations for the published files.

Verify the published package outside CI:

```bash
python -m venv /tmp/verify
/tmp/verify/bin/pip install "psdn-sonar==<version>"
/tmp/verify/bin/psdn-sonar --version

# Run from outside the checkout so the import cannot resolve to the source tree.
# SONAR_SOURCE_ROOT names the checkout the installed package is compared against.
cd /tmp && SONAR_SOURCE_ROOT=<path-to-repo> \
  /tmp/verify/bin/python <path-to-repo>/scripts/check_installed_package.py
```

For a production release, also install the advertised extras and import one module from each, and
inspect the published wheel and sdist metadata for anything that should not be public.

## If a release goes wrong

**A job failed before the publish step.** Nothing was uploaded. Fix the cause and re-dispatch the same
tag, or move the tag and re-dispatch; the workflow refuses to create a Release whose tag no longer
points at the commit the artifacts were built from.

**The publish step failed partway.** On TestPyPI, `skip-existing: true` makes a retry of the same tag
safe. On PyPI it is `false` by design — a silent skip would hide a real problem — so a partial upload
leaves the version consumed. Recovery is a new patch version, not a retry.

**A published release is broken.** Yank it, then ship a fixed patch version.

## Yanking a release

Yanking ([PEP 592](https://peps.python.org/pep-0592/)) is the only way to withdraw a release from PyPI
short of deleting it, and it is the correct one.

- **How.** On <https://pypi.org/project/psdn-sonar/>, open *Manage project* → *Releases* → the affected
  version → *Options* → *Yank*. Give a short public reason; it is shown to anyone who installs the
  version. A PyPI project owner or maintainer must do this, which is why the project keeps more than one
  owner.
- **What it does.** Resolvers skip a yanked version, so `pip install psdn-sonar` and any range
  requirement stop selecting it. An exact pin (`psdn-sonar==<version>`) still installs it, with a
  warning, so pinned builds and lockfiles do not break.
- **What it does not do.** It does not delete the files, and it does not free the version number. That
  number can never be reused on PyPI, whether yanked or deleted. Prefer yanking over deleting: deleting
  breaks every pinned install and buys nothing.
- **Afterwards.** Release a fixed patch version, record the yank and its reason in `CHANGELOG.md`, mark
  the corresponding GitHub Release so the two do not disagree, and note the cause on the tracking issue.
