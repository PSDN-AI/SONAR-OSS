# Import Gate Checklist

## Purpose

SONAR-OSS is the public mirror of a mature private ASR evaluation codebase (WER/CER/semantic
metrics, MOS audio quality, benchmarks, leaderboard, and report generation) plus a set of
non-proprietary public datasets and cookbooks. Moving that private code, its configuration, and
supporting data into a soon-to-be-public repository is the single highest-risk step in this
project. This document is the one auditable gate every import must pass, regardless of type
(code, config, dataset, feature, cookbook, or artifact). An import passes only when every
applicable item below is checked. If any item fails or is unknown, do not merge the material.

Every code import that lands in an already-public repo (see `M1-IMP-01`, #8) must reference a
passing run of this gate in its PR description.

## Checklist

### 1. Secrets

- [ ] A full-history secret scan has been run over the import set using the pinned gitleaks
      binary (v8.30.1), the same tool used by the "Secret scan" CI check (`.github/workflows/gitleaks.yml`,
      `gitleaks dir . --redact --exit-code 1`), and it reports zero findings.
- [ ] No real API keys, tokens, passwords, private keys, session cookies, or credentials of any
      kind are present in file contents, filenames, commit messages, or metadata.
- [ ] No real `.env` file or real environment values are included. `.env.example` with placeholder
      values only (e.g. `API_KEY=your-key-here`) is acceptable.

### 2. Private infrastructure

- [ ] No internal-only CI/CD workflows are present. Deploy, promote, and GPU/package build
      workflows from the private repository must be stripped, along with any steps that reference
      them, before import.
- [ ] No internal bucket, cluster, or container-registry names are present.
- [ ] No internal or private network endpoints, hostnames, or URLs are present.
- [ ] Internal dataset paths (private buckets, private mounts, internal file shares) are replaced
      with public dataset discovery only (e.g. HuggingFace Hub, OpenSLR, or other public mirrors).
      No internal path strings remain, even as comments or defaults.
- [ ] No internal-only file names or archives are referenced (for example internal `.env.*`
      variants or internal dataset tarball names).

### 3. Data rights

- [ ] Every dataset shipped, referenced, or used for benchmarking (e.g. Common Voice, FLEURS,
      Zeroth, VoxPopuli, MLS, OpenSLR) has a documented license and confirmed redistribution
      rights recorded alongside the import.
- [ ] Only non-proprietary datasets are imported. No proprietary, internally licensed, or
      customer-supplied data is included.
- [ ] Required attribution or citation for each dataset is included in the docs or dataset card
      that ships with the import.

### 4. PII / private references

- [ ] No raw audio recordings are included.
- [ ] No raw transcripts of real speech are included.
- [ ] No demographic records, speaker identity data, or other personally identifiable
      information is included.
- [ ] No internal leaderboard data, internal benchmark results, or internal run logs are included.

### 5. License

- [ ] Every source file carries or permits the repository's MIT license (see `LICENSE`,
      "Copyright (c) 2026 PSDN AI contributors").
- [ ] Third-party code, models, or datasets retained in the import carry the required
      attribution and are compatible with redistribution under MIT.

## Recording a passing run

Record the outcome of this checklist in the pull request that performs the import:

- [ ] Check every applicable box above in the PR description (copy this checklist into the PR,
      or link to this file and confirm each section).
- [ ] Note which sections do not apply (e.g. "No datasets in this PR") rather than omitting them.
- [ ] Link the "Secret scan", "Pre-commit baseline", and "Validate PR title" CI check runs that
      passed for the PR.
- [ ] State the reviewer(s) who confirmed the gate passed.

Use this sign-off format in the PR description or a PR comment:

```md
Import gate: PASS
Import type(s): <code | config | dataset | feature | cookbook | artifact>
Items: <files or directories covered>
Reviewed by: <name or handle>
Date: YYYY-MM-DD
Sections: secrets - pass; private infra - pass; data rights - pass/N/A; PII - pass; license - pass
Notes: <required attribution, license names, or "none">
```

`M1-IMP-01` (#8), which imports the first code, must cite a passing run of this gate using the
format above. If an import changes after sign-off, repeat the checklist and post a new sign-off.
