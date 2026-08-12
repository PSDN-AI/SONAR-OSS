# Contributing

Thanks for your interest in contributing to `PSDN-AI/SONAR-OSS`, the multi-language ASR evaluation
toolkit (`psdn-sonar`).

## Before You Start

- Check existing issues and open pull requests before starting overlapping work.
- Keep each change focused on one problem or one small set of closely related changes.
- For larger proposals (new metrics, new language support, new benchmark integrations), start with an
  issue so the scope and direction can be discussed early.
- For a curated change from the canonical development source, follow the tracked-files-only
  [upstream synchronization policy](docs/upstream-sync-policy.md) before copying anything.

## Good Contributions

Useful contributions usually include one of the following:

- Bug reports with clear reproduction steps, expected behavior, and actual behavior.
- Focused fixes with a short explanation of the problem being solved.
- Documentation improvements that make the project easier to understand or contribute to.
- Well-scoped feature proposals that explain the user need and tradeoffs (for example, a new metric,
  a new language normalizer, or a new evaluation harness).

## Local Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Contributor
environments install the committed lockfile exactly (this is what CI runs):

```bash
uv sync --frozen --extra dev
```

This installs the core toolkit plus dev tooling. Add extras as needed for the parts you are
working on:

| Extra | Purpose |
| --- | --- |
| `[ml-common]` | Transformers, sentence-transformers, and MOS scoring without a pinned torch |
| `[ml]` | Full local-model stack: torch, torchaudio, transformers, MOS scorers |
| `[pyannote]` | Speaker diarization via pyannote.audio |
| `[bengali]` | Bengali-language text normalization and tokenization support |
| `[korean]` | Korean-language text normalization and tokenization support (needs a Java runtime) |
| `[hindi]` | Hindi unicode normalization and tokenization support |
| `[apis]` | Optional integrations with third-party ASR/model APIs |
| `[cloud]` | S3-compatible dataset download support |
| `[dev]` | Linting, formatting, pre-commit, and test tooling |

For example, to work on Korean-language evaluation with the full dev toolchain:

```bash
uv sync --frozen --extra korean --extra dev
```

Or install everything at once with `[all]`:

```bash
uv sync --frozen --extra all --extra dev
```

If `uv sync --frozen` fails because `uv.lock` is out of date, do not resolve around it: update the
lockfile deliberately (see "Dependency Updates" below) so the change is reviewed.

### Running Checks Locally

If a `Makefile` is present, use the provided targets:

```bash
make pre-commit-install
make pre-commit-run
```

Equivalently, using `pre-commit` directly:

```bash
pre-commit install
pre-commit run --all-files
```

Run these before opening a pull request. CI runs on pull requests and every resulting push to
`main`: `Pre-commit baseline`, full-history `Secret scan`, `Internal-reference gate`,
`Lint and type check` (ruff + ty after a frozen install), `Tests (Python 3.10/3.11/3.12)`, and
`Package artifacts` (wheel/sdist build, metadata validation, clean-install smoke test, and
runtime-extras verification).

## Pull Request Titles

Pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/):
`type(optional-scope): short description`. A CI check ("Validate PR title") enforces this on every
pull request.

- **Allowed types:** `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, `revert`.
- **Scope** is optional and free-form; add it when it clarifies the change (for example: `metrics`,
  `language`, `benchmarks`, `ci`, `docs`, `deps`).
- Write the description in the imperative mood and keep it concise.

Examples:

- `feat(metrics): add semantic similarity scorer`
- `fix(language): handle empty transcript in Bengali normalizer`
- `docs: add benchmark reproduction guide`

Avoid generic titles such as `Update code` or `Fix bug` that don't convey the type or scope of the change.

## Pull Request Labels

Every pull request carries at least one `track:*` area label. The `PR auto-label` workflow first
reconciles path labels and then runs its `Require area label` check. The check becomes merge-blocking
only when branch protection requires it.

- Managed `track:*` labels are applied automatically from the current changed paths using
  `.github/labeler.yml`. A PR that touches several mapped areas gets several labels.
- Managed labels are reconciled on every update, so labels for paths no longer in the PR are removed.
- Priority and effort labels are not copied from linked issues; they remain issue-planning metadata.

If every changed path is intentionally unmapped, ask a repository maintainer to apply the
`track: manual` fallback label. Repository labels require maintainer permissions, so external fork
contributors cannot apply this label themselves.

`track: manual` is not managed by the path labeler, so reconciliation preserves it. It also signals
that maintainers should consider a follow-up update to the path map.

## Pull Request Expectations

- Link the relevant issue when one exists.
- Keep the diff narrow and avoid unrelated cleanup.
- Match existing project style and naming.
- Update documentation when behavior, usage, or contributor workflow changes.
- If the PR imports code, data, model artifacts, generated output, or sample assets, complete the import
  gate in `docs/import-gate.md`.
- If the PR synchronizes shared code from the canonical development source, include the public sync record,
  exact tracked-file allowlist, dry-run result, and named maintainer review required by
  `docs/upstream-sync-policy.md`.
- Be ready to address review feedback or split oversized work into follow-up PRs.

## Dependency Updates

Dependency changes are deliberate and reviewed; there is no automated version-bump bot.

- **Python dependencies** follow an update → audit → relock → review cycle:
  1. Change the constraint in `pyproject.toml` (or upgrade a transitive pin with
     `uv lock --upgrade-package <name>`).
  2. Run `uv lock` and commit the resulting `uv.lock` in the same PR; CI's `uv lock --check`
     rejects manifests and lockfile drifting apart. Never hand-edit `uv.lock`, and avoid a
     whole-environment `uv lock --upgrade` unless the PR intentionally reviews that full change.
  3. The blocking `Dependency audit` check audits the exported lockfile (never a fresh
     resolution) with `pip-audit` on every PR, push to `main`, weekly schedule, and manual
     dispatch.
  4. Review happens in the PR like any other change.
- **Vulnerability exceptions** live in `security/dependency-audit-exceptions.toml`. Every entry
  names one advisory ID with an owner, rationale, review date, and removal condition. The audit
  fails closed on any new unexcepted advisory, on entries whose advisory has disappeared
  (obsolete), and on entries past their review date (stale) — see `scripts/dependency_audit.py`.
- **Cadence and ownership:** the weekly scheduled audit surfaces new advisories without waiting
  for a PR; repository maintainers triage failures and either relock to a fixed version or add a
  reviewed, time-bounded exception. Exceptions are re-reviewed at their `review_by` date at the
  latest.
- **GitHub Actions** are pinned to full commit SHAs. Maintainers review and update those
  references manually; automated Dependabot version-update pull requests are intentionally
  disabled so dependency changes remain deliberate during release-readiness work.

## Public Repository Safety

This project is public. Please keep contributions safe to publish.

- Do not include secrets, credentials, tokens, or private keys.
- Do not add references to private datasets, internal infrastructure, internal service names, or
  non-public systems.
- Use generic, publicly reproducible examples (public benchmark datasets, synthetic samples) rather than
  internal or proprietary audio/transcript data.

## Security Reports

For suspected vulnerabilities, do not open a public issue. Follow the process in `SECURITY.md`.
