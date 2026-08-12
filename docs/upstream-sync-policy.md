# Upstream Synchronization Policy

## Purpose

SONAR-OSS accepts selected changes from the canonical development source, but it is not a repository
mirror. Every transfer is a curated, reviewed pull request containing only an explicit allowlist of
Git-tracked, public-safe files. The [import gate](import-gate.md) applies to the initial import and every
later synchronization.

## Ownership

| Material | Source of truth |
| --- | --- |
| Shared evaluation code, tests, and reusable public configuration | Canonical development source until the project owners announce a transfer of ownership |
| Public package metadata and lockfile | SONAR-OSS |
| Public documentation, governance, and contributor workflow | SONAR-OSS |
| GitHub Actions, repository security configuration, and release metadata | SONAR-OSS |
| Public tags, releases, and published package provenance | SONAR-OSS |

Security fixes may originate in either source. Maintainers coordinate a public-safe patch in both places,
but never copy private commits, issue text, paths, or repository history into SONAR-OSS. A synchronization
must not overwrite a SONAR-OSS-owned file; a deliberate change to public packaging, documentation,
security configuration, or release metadata is a normal SONAR-OSS pull request.

## Responsibilities

- The initial-import coordinator owns the curated inventory and evidence required by #8.
- The synchronization author prepares the exact tracked-file allowlist, public sync record, and validation
  evidence for each recurring transfer.
- A named SONAR-OSS maintainer other than the author verifies the private source revision in the private
  audit record and reviews the public allowlist, diff, import gate, and validation evidence.
- The release coordinator owns release cut-offs and any documented exception during a release freeze.

## Required Process

1. Start from the current `main` commit and record its 40-character SHA as `public_base_sha`.
2. In the canonical source, generate a temporary tracked-file inventory with `git ls-files`. Do not copy a
   working tree, untracked files, ignored files, or private Git history.
3. Select the exact files to transfer. Incoming synchronization is limited to these public component
   roots:
   - `psdn_sonar/` (`product-code`)
   - `tests/` (`tests`)
   - `config/` (`public-config`)
4. Copy only the allowlisted files into a dedicated branch, then stage exactly those files for review.
5. Verify that the staged diff matches the allowlist exactly. This plain-`git` comparison is the dry-run
   verification for the synchronization:

   ```bash
   git diff --cached --name-only --no-ext-diff --no-renames <public-base-sha> | sort > /tmp/sonar-sync-changed.txt
   jq -r '.files[]' /tmp/sonar-sync-record.json | sort | diff - /tmp/sonar-sync-changed.txt
   ```

   Empty diff output means the allowlist matches the staged diff. Deletions, file-type changes, symlinks,
   submodules, unstaged changes, and unrelated changes remain stop conditions.
6. Apply every section of `docs/import-gate.md`: the public lint, tests, internal-reference check,
   pre-commit hooks, secret scan, and any component-specific validation.
7. Write the public sync record once, using the format below, with final values including
   `"import_gate": "PASS"`. `files` is the exact allowlist, not a glob or directory. Use only the public
   component names below; keep source revisions, source paths, private issue links, and internal
   identifiers in the private system of record.
8. Open a pull request containing the public sync record and the dry-run and import-gate evidence. A named
   maintainer other than the author re-runs the allowlist-vs-diff comparison from step 5 and reviews the
   import-gate evidence. No automated or unreviewed full-repository synchronization may merge.

Dedicated validation tooling for this process was considered and deliberately deferred; revisit automation
after the public launch if manual verification proves error-prone.

Temporary inventories and records may be attached to the pull request only after confirming that they
contain public paths and the record fields below. Never commit or attach the full canonical-source
inventory.

## Public Sync Record

The pull request records only public-safe provenance:

```json
{
  "sync_type": "recurring-sync",
  "sync_date": "YYYY-MM-DD",
  "public_base_sha": "40-character-lowercase-SHA",
  "included_components": ["product-code", "tests"],
  "files": ["psdn_sonar/example.py", "tests/unit/test_example.py"],
  "reviewer": "@github-handle",
  "validation": ["Pre-commit baseline", "Secret scan", "Internal-reference gate", "Lint and type check", "Tests"],
  "import_gate": "PASS"
}
```

Replace every example value. The reviewer field records the maintainer who approved the synchronization,
not the author or an automation identity. The exact canonical source revision remains in the private audit
record and is verified there by the reviewer.

## Public Contributions and Carry-Back

Public contributions land in SONAR-OSS through the normal contribution process. A maintainer then decides
whether the behavior is also appropriate for the canonical source. Carry-back is a separate reviewed change
in that source; it preserves public authorship and license attribution and does not rewrite or replace the
public commit. If the implementations have diverged, port the behavior rather than copying a conflicting
patch blindly.

## Conflicts, Security Fixes, and Release Cut-Offs

- SONAR-OSS-owned packaging, documentation, workflows, security policy, and release metadata win any
  ownership conflict.
- For shared code conflicts, preserve public API compatibility and public-safety controls. Resolve the
  conflict in a focused pull request and rerun the full import gate.
- Urgent security fixes follow `SECURITY.md`. Coordinate disclosure privately, create independently
  reviewable public-safe patches, and do not expose embargoed details through sync records or commit
  metadata.
- After a release cut-off announced by the release coordinator, defer normal synchronization. Only a named
  release blocker or security fix may enter, with explicit release-coordinator approval in the pull request.

Stop the synchronization without copying or publishing material when any selected file is untracked, a
license or data-rights decision is unknown, the source cannot be mapped to the exact public allowlist, a
private reference cannot be removed confidently, validations disagree, or the reviewer cannot reconstruct
the public diff from the record. A deletion, file-type change, symlink, submodule, unstaged change, stale
public base, or unrelated public change is also a stop condition. Record only the public reason for stopping;
keep private details in the private system of record.

## Pull Request Checklist

- [ ] The public base SHA and exact tracked-file allowlist are recorded.
- [ ] The recorded allowlist matches the final staged diff exactly (author verified with the diff commands
      above; reviewer re-verified).
- [ ] No SONAR-OSS-owned path is overwritten by synchronization.
- [ ] The import gate is repeated and records `Import gate: PASS`.
- [ ] Public lint, tests, secret scan, internal-reference check, and component checks pass.
- [ ] A named maintainer other than the author reviewed the allowlist and evidence.
- [ ] Carry-back, conflicts, security handling, and any release-cut-off exception are recorded without
      private details.
