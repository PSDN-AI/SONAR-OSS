# Benchmark data governance

SONAR treats a dataset license as evidence, not as an approval decision. A
benchmark is publishable only after a named human has reviewed the exact
source revision, access method, attribution, redistribution scope, and
prepared-artifact policy recorded in the benchmark catalog.

## Current review queue

All entries below remain pending until the catalog records a named approver
and approval date through a reviewed pull request. Pending entries may be used
for exploratory local runs, but must be rejected by publishable or leaderboard
workflows.

| Dataset | Immutable source evidence | Upstream terms or license | Decision still required |
| --- | --- | --- | --- |
| Common Voice 17 | [Mozilla release manifest commit](https://github.com/common-voice/cv-dataset/commit/f99d8239d2796131b73ac99f92ee7cb4443bf3ba) and manifest SHA-256 `d1c77b3672a3af20e06fa184e30b1d1dc0f9d613b4dcab35fd6a9b3101ac20d6` | [Common Voice datasets and terms](https://commonvoice.mozilla.org/en/datasets) | The old Hugging Face repository is now a tombstone. Approve an MDC access method and decide whether any prepared artifacts may be redistributed under the current terms. |
| FLEURS | [Hugging Face dataset commit](https://huggingface.co/datasets/google/fleurs/commit/70bb2e84b976b7e960aa89f1c648e09c59f894dd) | [FLEURS dataset card](https://huggingface.co/datasets/google/fleurs) lists CC-BY-4.0 | Confirm attribution text and whether SONAR may redistribute prepared subsets or only reference the upstream source. |
| Zeroth Korean | [Hugging Face mirror commit](https://huggingface.co/datasets/Bingsu/zeroth-korean/commit/bd173fe2c8ed0dccd47acb4eda77542593651622) | [OpenSLR 40](https://www.openslr.org/40/) lists CC-BY-4.0 and archive checksums | Approve the community mirror provenance, attribution, and prepared-artifact policy against the official archive. |
| VoxPopuli | [Hugging Face dataset commit](https://huggingface.co/datasets/facebook/voxpopuli/commit/42f01879c780b4a2e90ec0b4f616c2ece526e4f1) | [VoxPopuli source repository](https://github.com/facebookresearch/voxpopuli) documents the ASR data terms | Separate the ASR-data terms from code/model terms, review the European Parliament notice, and approve attribution and redistribution scope. |
| Multilingual LibriSpeech | [Hugging Face dataset commit](https://huggingface.co/datasets/facebook/multilingual_librispeech/commit/2e83e61823b4c47dcbcb1980bb88601274127609) | [OpenSLR 94](https://www.openslr.org/94/) lists CC-BY-4.0 | Confirm attribution and whether prepared subsets may be redistributed. |
| OpenSLR 37 | [OpenSLR 37](https://www.openslr.org/37/) publishes archive checksums | OpenSLR lists CC-BY-SA-4.0 | Confirm which Bengali locale archives are in scope and the share-alike obligations for prepared artifacts. |
| OpenSLR 53 | [OpenSLR 53](https://www.openslr.org/53/) publishes shard and transcript checksums | OpenSLR lists CC-BY-SA-4.0 | Confirm whether the benchmark consumes all 16 shards or a reviewed subset, plus the share-alike obligations for prepared artifacts. |

## Approval record

An approval changes the matching catalog entry in the same reviewed pull
request and records:

- exact dataset, source, revision, config, split, and selection rules;
- license or terms URL and required attribution;
- redistribution decision for raw data and prepared artifacts;
- named approver, approval date, and evidence URL;
- expected content fingerprint for every approved reference selection.

Approval is not inferred from an SPDX identifier. A license change, source
revision, config, split, preprocessing version, or selection-rule change
invalidates the prior identity and requires review.

## Catalog update process

1. Pin the upstream source to an immutable commit or archive digest.
2. Update the catalog; when the model changes, update its generated JSON Schema in the same pull request.
3. Run `make check-benchmark-catalog` and the identity tests offline.
4. Resolve a representative selection and record its content fingerprint.
5. Obtain the named human approval in the catalog and link its evidence.
6. Merge only after the catalog diff, attribution, and fingerprint are
   independently reviewable.

Raw audio, transcripts, credentials, private paths, and private source
locations must never be committed as approval evidence.
