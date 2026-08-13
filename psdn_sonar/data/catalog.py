"""Offline benchmark-catalog validation and dataset identities."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Mapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError, field_validator, model_validator
from yaml.resolver import BaseResolver

IDENTITY_SCHEMA_VERSION = 1
_DEFAULT_CATALOG_RESOURCE = "benchmark_catalog.yaml"
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class CatalogValidationError(ValueError):
    """Raised when a benchmark catalog is incomplete or unsafe."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


RequiredStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _require_sha256(value: str) -> str:
    value = value.strip().lower()
    if not _SHA256_RE.fullmatch(value):
        raise ValueError("must be a sha256:<64 lowercase hex characters> digest")
    return value


def validate_huggingface_revision(value: str) -> str:
    """Return a normalized Hugging Face commit SHA or reject it."""
    revision = value.strip().lower()
    if not _GIT_SHA_RE.fullmatch(revision):
        raise ValueError("Hugging Face revisions must be full 40-character commit SHAs")
    return revision


class SourceSpec(_StrictModel):
    kind: Literal["huggingface", "openslr", "archive"]
    identifier: RequiredStr
    url: RequiredStr
    revision: RequiredStr

    @field_validator("revision")
    @classmethod
    def _normalize_revision(cls, value: str) -> str:
        return value.lower()

    @field_validator("url")
    @classmethod
    def _https_url(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("source urls must use https://")
        return value

    @model_validator(mode="after")
    def _immutable_revision(self) -> SourceSpec:
        if self.kind == "huggingface":
            validate_huggingface_revision(self.revision)
        else:
            _require_sha256(self.revision)
        return self


class ReviewSpec(_StrictModel):
    decision: Literal["pending", "reference_only", "redistributable", "prohibited"]
    rationale: RequiredStr
    approved_by: Optional[RequiredStr] = None
    approved_at: Optional[str] = None
    evidence_url: Optional[str] = None
    fingerprints: dict[str, str] = Field(default_factory=dict)

    @field_validator("fingerprints")
    @classmethod
    def _fingerprints_are_sha256(cls, value: dict[str, str]) -> dict[str, str]:
        return {key: _require_sha256(digest) for key, digest in value.items()}

    @model_validator(mode="after")
    def _approval_is_reviewable(self) -> ReviewSpec:
        approved = self.decision in {"reference_only", "redistributable"}
        evidence = (self.approved_by, self.approved_at, self.evidence_url)
        if approved and (not all(evidence) or not self.fingerprints):
            raise ValueError("approved data rights require named evidence and fingerprints")
        if not approved and any(evidence):
            raise ValueError("only approved data rights may include approval evidence")
        if self.approved_at is not None:
            try:
                date.fromisoformat(self.approved_at)
            except ValueError as exc:
                raise ValueError("approved_at must be an ISO date (YYYY-MM-DD)") from exc
        if self.evidence_url is not None and not self.evidence_url.startswith("https://"):
            raise ValueError("approval evidence must use https://")
        return self

    @property
    def approved(self) -> bool:
        return self.decision in {"reference_only", "redistributable"}


class DatasetIdentity(_StrictModel):
    schema_version: Literal[1] = IDENTITY_SCHEMA_VERSION
    benchmark: RequiredStr
    source_identifier: RequiredStr
    source_revision: RequiredStr
    config: Optional[str]
    split: RequiredStr
    data_fingerprint: str
    selection: RequiredStr
    preprocessing: RequiredStr
    text_column: RequiredStr
    audio_column: RequiredStr

    @field_validator("data_fingerprint")
    @classmethod
    def _fingerprint_is_sha256(cls, value: str) -> str:
        return _require_sha256(value)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(_canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()
        return f"sha256:{digest}"


class BenchmarkSpec(_StrictModel):
    display_name: RequiredStr
    enabled: bool = True
    source: SourceSpec
    config_template: str = ""
    allowed_configs: tuple[str, ...] = ()
    splits: tuple[str, ...] = ()
    license: RequiredStr
    license_url: RequiredStr
    attribution: RequiredStr
    text_column: RequiredStr
    audio_column: RequiredStr = "audio"
    selection: RequiredStr = "ordered_upstream_split@1"
    preprocessing: RequiredStr = "psdn_sonar.export_tsv@1"
    review: ReviewSpec

    @field_validator("allowed_configs", "splits")
    @classmethod
    def _unique_nonempty_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("names must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("names must be unique")
        return value

    @model_validator(mode="after")
    def _validate_shape(self) -> BenchmarkSpec:
        if self.enabled and not self.splits:
            raise ValueError("enabled benchmarks must define at least one split")
        if self.config_template and self.enabled and not self.allowed_configs:
            raise ValueError("enabled config-based benchmarks must define allowed_configs")
        if not self.config_template and self.allowed_configs:
            raise ValueError("benchmarks without a config template cannot define allowed_configs")
        for key in self.review.fingerprints:
            config, separator, split = key.partition("::")
            if not separator or not config or split not in self.splits:
                raise ValueError(f"invalid fingerprint key: {key!r}")
            if self.config_template and config not in self.allowed_configs:
                raise ValueError(f"fingerprint key uses unknown config: {key!r}")
            if not self.config_template and config != "_default":
                raise ValueError(f"no-config benchmark fingerprint must use _default: {key!r}")
        return self

    @staticmethod
    def fingerprint_key(config: Optional[str], split: str) -> str:
        return f"{config or '_default'}::{split}"

    def identity(
        self,
        benchmark_id: str,
        *,
        resolved_config: Optional[str],
        split: str,
        data_fingerprint: str,
        publishable: bool = False,
    ) -> DatasetIdentity:
        if not self.enabled:
            raise ValueError(f"benchmark {benchmark_id!r} is not enabled for runtime use")
        if split not in self.splits:
            raise ValueError(f"benchmark {benchmark_id!r} does not define split {split!r}")
        if self.config_template and not resolved_config:
            raise ValueError(f"benchmark {benchmark_id!r} requires a resolved config")
        if self.config_template and resolved_config not in self.allowed_configs:
            raise ValueError(f"benchmark {benchmark_id!r} does not define config {resolved_config!r}")
        if not self.config_template and resolved_config:
            raise ValueError(f"benchmark {benchmark_id!r} does not accept a config")
        data_fingerprint = _require_sha256(data_fingerprint)
        if publishable:
            expected = self.review.fingerprints.get(self.fingerprint_key(resolved_config, split))
            if not self.review.approved or expected != data_fingerprint:
                raise ValueError(f"benchmark {benchmark_id!r} is not approved for this exact data fingerprint")
        return DatasetIdentity(
            benchmark=benchmark_id,
            source_identifier=self.source.identifier,
            source_revision=self.source.revision,
            config=resolved_config,
            split=split,
            data_fingerprint=data_fingerprint,
            selection=self.selection,
            preprocessing=self.preprocessing,
            text_column=self.text_column,
            audio_column=self.audio_column,
        )


class BenchmarkCatalog(_StrictModel):
    schema_version: Literal[1]
    benchmarks: dict[str, BenchmarkSpec] = Field(min_length=1)

    def get(self, benchmark_id: str) -> BenchmarkSpec:
        try:
            return self.benchmarks[benchmark_id]
        except KeyError as exc:
            raise KeyError(f"unknown benchmark {benchmark_id!r}; known: {sorted(self.benchmarks)}") from exc

    def identity(self, benchmark_id: str, **kwargs: Any) -> DatasetIdentity:
        return self.get(benchmark_id).identity(benchmark_id, **kwargs)

    def verify_publishable_identity(self, identity: DatasetIdentity) -> None:
        expected = self.identity(
            identity.benchmark,
            resolved_config=identity.config,
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
    try:
        return BenchmarkCatalog.model_validate(document)
    except ValidationError as exc:
        raise CatalogValidationError(str(exc)) from exc


def load_catalog(path: Optional[str | Path] = None) -> BenchmarkCatalog:
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
    """Hash an ordered sequence of JSON records."""
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_json(record).encode("utf-8") + b"\n")
    return f"sha256:{digest.hexdigest()}"
