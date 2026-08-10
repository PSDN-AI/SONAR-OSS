"""Versioned public benchmark catalog and immutable dataset identities.

The catalog is deliberately validated without contacting an upstream service.
An evaluation identity is built only from resolved, immutable inputs; callers
must provide a digest for the exact ordered data they consumed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from argparse import ArgumentParser
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Optional
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_serializer, model_validator
from yaml.resolver import BaseResolver

IDENTITY_SCHEMA_VERSION = 1
_DEFAULT_CATALOG_RESOURCE = "benchmark_catalog.yaml"
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^md5:[0-9a-f]{32}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class CatalogValidationError(ValueError):
    """Raised when a benchmark catalog is incomplete or unsafe."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _nonempty(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("must not be empty")
    return value


def _require_sha256(value: str) -> str:
    value = value.strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("must be a sha256:<64 lowercase hex characters> digest")
    return value


def validate_huggingface_revision(value: str) -> str:
    """Return a normalized full Hugging Face commit SHA or reject it."""
    revision = value.strip().lower()
    if not _GIT_SHA_RE.fullmatch(revision):
        raise ValueError("Hugging Face revisions must be full 40-character commit SHAs")
    return revision


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


class SourceSpec(_StrictModel):
    """Immutable public or private source coordinate."""

    kind: Literal["huggingface", "openslr", "archive"]
    identifier: str
    canonical_url: str
    revision: str
    visibility: Literal["public", "private"]
    artifacts: dict[str, str]

    @field_validator("identifier", "canonical_url", "revision")
    @classmethod
    def _required_strings(cls, value: str) -> str:
        return _nonempty(value)

    @field_validator("canonical_url")
    @classmethod
    def _public_url_shape(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("must use an https:// canonical URL")
        return value

    @model_validator(mode="after")
    def _immutable_revision(self) -> SourceSpec:
        revision = self.revision.lower()
        floating = {"main", "master", "head", "latest", "default", "stable", "current"}
        if revision in floating or revision.startswith(("refs/heads/", "branch:", "tag:")):
            raise ValueError(f"floating source revision is not allowed: {self.revision!r}")
        if self.kind == "huggingface":
            validate_huggingface_revision(revision)
        if self.kind == "archive" and not _SHA256_RE.fullmatch(revision):
            raise ValueError("archive revisions must be sha256 content digests")
        if self.kind == "openslr" and not (_SHA256_RE.fullmatch(revision) or _MD5_RE.fullmatch(revision)):
            raise ValueError("OpenSLR revisions must be content digests")
        for name, digest in self.artifacts.items():
            if not name.strip():
                raise ValueError("source artifact names must not be empty")
            if not (_SHA256_RE.fullmatch(digest) or _MD5_RE.fullmatch(digest)):
                raise ValueError("source artifact identities must be sha256 or md5 digests")
        return self


class LicenseSpec(_StrictModel):
    """License evidence, including an explicit unverified state."""

    status: Literal["verified", "unverified"]
    identifier: Optional[str]
    url: Optional[str]

    @model_validator(mode="after")
    def _verified_has_evidence(self) -> LicenseSpec:
        if self.status == "verified":
            if not self.identifier or not self.identifier.strip():
                raise ValueError("verified licenses require an identifier")
            if not self.url or not self.url.startswith("https://"):
                raise ValueError("verified licenses require an https:// evidence URL")
        return self


class RedistributionSpec(_StrictModel):
    """Reviewed decision about copying dataset contents."""

    decision: Literal["reference_only", "redistributable", "prohibited", "pending_human_approval"]
    rationale: str

    @field_validator("rationale")
    @classmethod
    def _rationale_required(cls, value: str) -> str:
        return _nonempty(value)


class RightsApproval(_StrictModel):
    """Named human approval; pending is intentionally not publishable."""

    status: Literal["approved", "pending", "rejected"]
    approved_by: Optional[str]
    approved_at: Optional[str]
    evidence_url: Optional[str]

    @model_validator(mode="after")
    def _approval_is_named(self) -> RightsApproval:
        if self.status == "approved" and (not self.approved_by or not self.approved_at or not self.evidence_url):
            raise ValueError("approved data rights require approved_by, approved_at, and evidence_url")
        if self.status == "approved":
            if not self.approved_by or not self.approved_by.strip():
                raise ValueError("approved data rights require a non-empty approver")
            try:
                date.fromisoformat(self.approved_at or "")
            except ValueError as exc:
                raise ValueError("approved_at must be an ISO date (YYYY-MM-DD)") from exc
            parsed_evidence = urlparse(self.evidence_url or "")
            if parsed_evidence.scheme != "https" or not parsed_evidence.netloc:
                raise ValueError("approved data-rights evidence must use an https:// URL with a host")
        if self.status != "approved" and (self.approved_by or self.approved_at or self.evidence_url):
            raise ValueError("only approved data rights may include approval evidence")
        return self


class SelectionSpec(_StrictModel):
    """Versioned rule that determines the ordered sample set."""

    name: str
    version: str

    @field_validator("name", "version")
    @classmethod
    def _selection_strings_required(cls, value: str) -> str:
        return _nonempty(value)


class PreprocessingSpec(_StrictModel):
    """Versioned preprocessing identity."""

    name: str
    version: str

    @field_validator("name", "version")
    @classmethod
    def _preprocessing_strings_required(cls, value: str) -> str:
        return _nonempty(value)


class DatasetIdentity(_StrictModel):
    """Immutable identity for the exact data consumed by an evaluation."""

    identity_schema_version: Literal[1] = IDENTITY_SCHEMA_VERSION
    catalog_version: int
    benchmark_id: str
    source_identifier: str
    source_revision: str
    source_artifacts_json: str
    resolved_config: Optional[str]
    split: str
    data_fingerprint: str
    expected_schema_json: str
    selection_json: str
    preprocessing_json: str

    @model_validator(mode="before")
    @classmethod
    def _accept_canonical_wire_shape(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        wire_keys = {
            "identity_schema_version",
            "catalog_version",
            "benchmark_id",
            "source",
            "config",
            "split",
            "data_fingerprint",
            "expected_schema",
            "selection",
            "preprocessing",
        }
        source = value.get("source")
        if set(value) != wire_keys or not isinstance(source, Mapping):
            return value
        if set(source) != {"identifier", "revision", "artifacts"}:
            return value
        return {
            "identity_schema_version": value["identity_schema_version"],
            "catalog_version": value["catalog_version"],
            "benchmark_id": value["benchmark_id"],
            "source_identifier": source["identifier"],
            "source_revision": source["revision"],
            "source_artifacts_json": _canonical_json(source["artifacts"]),
            "resolved_config": value["config"],
            "split": value["split"],
            "data_fingerprint": value["data_fingerprint"],
            "expected_schema_json": _canonical_json(value["expected_schema"]),
            "selection_json": _canonical_json(value["selection"]),
            "preprocessing_json": _canonical_json(value["preprocessing"]),
        }

    @field_validator("data_fingerprint")
    @classmethod
    def _data_digest_is_strong(cls, value: str) -> str:
        return _require_sha256(value)

    def as_dict(self) -> dict[str, Any]:
        """Return the canonical, JSON-compatible identity payload."""
        return {
            "identity_schema_version": self.identity_schema_version,
            "catalog_version": self.catalog_version,
            "benchmark_id": self.benchmark_id,
            "source": {
                "identifier": self.source_identifier,
                "revision": self.source_revision,
                "artifacts": json.loads(self.source_artifacts_json),
            },
            "config": self.resolved_config,
            "split": self.split,
            "data_fingerprint": self.data_fingerprint,
            "expected_schema": json.loads(self.expected_schema_json),
            "selection": json.loads(self.selection_json),
            "preprocessing": json.loads(self.preprocessing_json),
        }

    @model_serializer
    def _serialize_wire_shape(self) -> dict[str, Any]:
        return self.as_dict()

    @property
    def fingerprint(self) -> str:
        """SHA-256 digest of the canonical identity payload."""
        digest = hashlib.sha256(_canonical_json(self.as_dict()).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


class BenchmarkSpec(_StrictModel):
    """One cataloged benchmark integration."""

    id: str
    display_name: str
    availability: Literal["active", "retired"]
    runtime: Literal["enabled", "disabled"]
    source: SourceSpec
    config_template: str
    allowed_configs: tuple[str, ...]
    splits: tuple[str, ...]
    license: LicenseSpec
    redistribution: RedistributionSpec
    attribution: str
    text_column: str
    audio_column: str
    selection: SelectionSpec
    preprocessing: PreprocessingSpec
    expected_fingerprints: dict[str, str]
    public_default: bool
    rights_approval: RightsApproval

    @field_validator("id", "display_name", "attribution", "text_column", "audio_column")
    @classmethod
    def _benchmark_strings_required(cls, value: str) -> str:
        return _nonempty(value)

    @field_validator("splits")
    @classmethod
    def _splits_are_unique_and_named(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not split.strip() for split in value):
            raise ValueError("splits must not contain empty names")
        if len(value) != len(set(value)):
            raise ValueError("splits must be unique")
        return value

    @field_validator("allowed_configs")
    @classmethod
    def _configs_are_unique_and_named(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not config.strip() for config in value):
            raise ValueError("allowed_configs must not contain empty names")
        if len(value) != len(set(value)):
            raise ValueError("allowed_configs must be unique")
        return value

    @field_validator("expected_fingerprints")
    @classmethod
    def _expected_digests_are_strong(cls, value: dict[str, str]) -> dict[str, str]:
        return {key: _require_sha256(digest) for key, digest in value.items()}

    @model_validator(mode="after")
    def _public_default_is_fully_approved(self) -> BenchmarkSpec:
        if self.availability == "active" and not self.splits:
            raise ValueError("active benchmarks must define at least one split")
        if self.config_template and self.runtime == "enabled" and not self.allowed_configs:
            raise ValueError("enabled config-based benchmarks must define allowed_configs")
        if not self.config_template and self.allowed_configs:
            raise ValueError("benchmarks without a config template cannot define allowed_configs")
        for key in self.expected_fingerprints:
            parts = key.split("::")
            if len(parts) != 2 or not all(parts):
                raise ValueError(f"invalid expected fingerprint key: {key!r}")
            config, split = parts
            if split not in self.splits:
                raise ValueError(f"fingerprint key uses unknown split: {key!r}")
            if self.config_template and config not in self.allowed_configs:
                raise ValueError(f"fingerprint key uses unknown config: {key!r}")
            if not self.config_template and config != "_default":
                raise ValueError(f"no-config benchmark fingerprint must use _default: {key!r}")
        if not self.public_default:
            return self
        blockers = []
        if self.availability != "active":
            blockers.append("source is not active")
        if self.runtime != "enabled":
            blockers.append("runtime integration is disabled")
        if self.source.visibility != "public":
            blockers.append("source is private")
        if self.license.status != "verified":
            blockers.append("license is not verified")
        if self.redistribution.decision not in {"reference_only", "redistributable"}:
            blockers.append("redistribution decision is not approved")
        if self.rights_approval.status != "approved":
            blockers.append("human data-rights approval is missing")
        if not self.expected_fingerprints:
            blockers.append("expected data fingerprints are missing")
        if blockers:
            raise ValueError("public_default requires: " + "; ".join(blockers))
        return self

    @staticmethod
    def fingerprint_key(resolved_config: Optional[str], split: str) -> str:
        """Return the catalog key for one resolved config/split pair."""
        return f"{resolved_config or '_default'}::{split}"

    def identity(
        self,
        *,
        catalog_version: int,
        resolved_config: Optional[str],
        split: str,
        data_fingerprint: str,
        publishable: bool = False,
    ) -> DatasetIdentity:
        """Build an immutable identity, optionally enforcing leaderboard policy."""
        if self.runtime != "enabled" or self.availability != "active":
            raise ValueError(f"benchmark {self.id!r} is not enabled for runtime use")
        if split not in self.splits:
            raise ValueError(f"benchmark {self.id!r} does not define split {split!r}")
        if self.config_template and not resolved_config:
            raise ValueError(f"benchmark {self.id!r} requires a resolved config")
        if self.config_template and resolved_config not in self.allowed_configs:
            raise ValueError(f"benchmark {self.id!r} does not define config {resolved_config!r}")
        if not self.config_template and resolved_config:
            raise ValueError(f"benchmark {self.id!r} does not accept a config")

        data_fingerprint = _require_sha256(data_fingerprint)
        if publishable:
            self._assert_publishable(resolved_config, split, data_fingerprint)
        return DatasetIdentity(
            catalog_version=catalog_version,
            benchmark_id=self.id,
            source_identifier=self.source.identifier,
            source_revision=self.source.revision,
            source_artifacts_json=_canonical_json(self.source.artifacts),
            resolved_config=resolved_config,
            split=split,
            data_fingerprint=data_fingerprint,
            expected_schema_json=_canonical_json({"audio_column": self.audio_column, "text_column": self.text_column}),
            selection_json=_canonical_json(self.selection.model_dump(mode="json")),
            preprocessing_json=_canonical_json(self.preprocessing.model_dump(mode="json")),
        )

    def _assert_publishable(self, resolved_config: Optional[str], split: str, data_fingerprint: str) -> None:
        if not self.public_default:
            raise ValueError(f"benchmark {self.id!r} is not an approved public default")
        expected = self.expected_fingerprints.get(self.fingerprint_key(resolved_config, split))
        if expected is None:
            raise ValueError(f"benchmark {self.id!r} has no approved fingerprint for config/split")
        if expected != data_fingerprint:
            raise ValueError(f"benchmark {self.id!r} data fingerprint does not match the catalog")


class BenchmarkCatalog(_StrictModel):
    """Validated catalog document."""

    schema_version: Literal[1]
    catalog_version: int = Field(ge=1)
    benchmarks: dict[str, BenchmarkSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _mapping_ids_match(self) -> BenchmarkCatalog:
        mismatches = [key for key, benchmark in self.benchmarks.items() if key != benchmark.id]
        if mismatches:
            raise ValueError(f"benchmark mapping keys must equal their id fields: {mismatches}")
        return self

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        """Return a benchmark or raise a useful error listing valid ids."""
        try:
            return self.benchmarks[benchmark_id]
        except KeyError as exc:
            raise KeyError(f"unknown benchmark {benchmark_id!r}; known: {sorted(self.benchmarks)}") from exc

    def identity(self, benchmark_id: str, **kwargs: Any) -> DatasetIdentity:
        """Build an identity using this catalog's version."""
        return self.get(benchmark_id).identity(catalog_version=self.catalog_version, **kwargs)

    def verify_publishable_identity(self, identity: DatasetIdentity) -> None:
        """Rebuild and verify an identity against this catalog's public policy."""
        if identity.catalog_version != self.catalog_version:
            raise ValueError(
                f"dataset identity uses catalog v{identity.catalog_version}, "
                f"but the bundled catalog is v{self.catalog_version}"
            )
        expected = self.identity(
            identity.benchmark_id,
            resolved_config=identity.resolved_config,
            split=identity.split,
            data_fingerprint=identity.data_fingerprint,
            publishable=True,
        )
        if identity != expected:
            raise ValueError("dataset identity does not match the approved catalog identity")


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise CatalogValidationError(f"duplicate catalog key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def validate_catalog_document(document: Mapping[str, Any]) -> BenchmarkCatalog:
    """Validate an already parsed catalog document without network access."""
    try:
        return BenchmarkCatalog.model_validate(document)
    except ValidationError as exc:
        raise CatalogValidationError(str(exc)) from exc


def load_catalog(path: Optional[str | Path] = None) -> BenchmarkCatalog:
    """Load and validate the bundled catalog or a caller-provided YAML file."""
    try:
        if path is None:
            raw = resources.files("psdn_sonar.data").joinpath(_DEFAULT_CATALOG_RESOURCE).read_text(encoding="utf-8")
        else:
            raw = Path(path).read_text(encoding="utf-8")
        document = yaml.load(raw, Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise CatalogValidationError(f"could not read benchmark catalog: {exc}") from exc
    if not isinstance(document, Mapping):
        raise CatalogValidationError("benchmark catalog root must be a mapping")
    return validate_catalog_document(document)


def fingerprint_records(records: Iterable[Mapping[str, Any]]) -> str:
    """Digest an ordered sequence of JSON records deterministically."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(record).encode("utf-8"))
        digest.update(b"\n")
    return f"sha256:{digest.hexdigest()}"


def main(argv: Optional[list[str]] = None) -> int:
    """Offline ``python -m psdn_sonar.data.catalog`` validator."""
    parser = ArgumentParser(description="Validate the SONAR benchmark catalog offline")
    parser.add_argument("path", nargs="?", help="Catalog YAML path (defaults to the bundled catalog)")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.path)
    except CatalogValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(
        f"validated {len(catalog.benchmarks)} benchmarks "
        f"(schema v{catalog.schema_version}, catalog v{catalog.catalog_version})"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
