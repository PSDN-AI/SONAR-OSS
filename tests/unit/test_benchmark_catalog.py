"""Offline validation and immutable identity tests for the benchmark catalog."""

from __future__ import annotations

from copy import deepcopy

import pytest

from psdn_sonar.data import (
    CatalogValidationError,
    fingerprint_records,
    load_catalog,
    validate_catalog_document,
)

DATA_FINGERPRINT = "sha256:" + "1" * 64


def test_bundled_catalog_and_generated_schema_validate_offline(monkeypatch):
    def network_forbidden(*args, **kwargs):
        raise AssertionError("catalog validation must not use the network")

    monkeypatch.setattr("socket.create_connection", network_forbidden)

    catalog = load_catalog()

    assert catalog.schema_version == 1
    assert catalog.catalog_version >= 1
    assert catalog.get("common_voice").runtime == "disabled"
    assert catalog.get("multilingual_librispeech").text_column == "transcript"
    assert len(catalog.get("openslr53").source.artifacts) == 17


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        "schema_version: 1\nschema_version: 1\ncatalog_version: 1\nbenchmarks: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(CatalogValidationError, match="duplicate catalog key: 'schema_version'"):
        load_catalog(path)


def test_floating_huggingface_revision_is_rejected():
    document = load_catalog().model_dump(mode="json")
    document["benchmarks"]["fleurs"]["source"]["revision"] = "main"

    with pytest.raises(CatalogValidationError, match="floating source revision"):
        validate_catalog_document(document)


def test_identity_rejects_an_unapproved_config():
    catalog = load_catalog()

    with pytest.raises(ValueError, match="does not define config"):
        catalog.identity(
            "fleurs",
            resolved_config="not_a_real_config",
            split="test",
            data_fingerprint=DATA_FINGERPRINT,
        )


def test_each_identity_dimension_changes_the_cache_key():
    catalog = load_catalog()
    base = catalog.identity(
        "fleurs",
        resolved_config="en_us",
        split="test",
        data_fingerprint=DATA_FINGERPRINT,
    )

    revised_document = deepcopy(catalog.model_dump(mode="json"))
    revised_document["benchmarks"]["fleurs"]["source"]["revision"] = "2" * 40
    revised_catalog = validate_catalog_document(revised_document)

    variants = [
        revised_catalog.identity(
            "fleurs",
            resolved_config="en_us",
            split="test",
            data_fingerprint=DATA_FINGERPRINT,
        ),
        catalog.identity(
            "fleurs",
            resolved_config="en_us",
            split="train",
            data_fingerprint=DATA_FINGERPRINT,
        ),
        catalog.identity(
            "fleurs",
            resolved_config="bn_in",
            split="test",
            data_fingerprint=DATA_FINGERPRINT,
        ),
    ]

    keys = {base.fingerprint, *(identity.fingerprint for identity in variants)}
    assert len(keys) == 1 + len(variants)


def test_pending_rights_cannot_produce_a_publishable_identity():
    catalog = load_catalog()

    with pytest.raises(ValueError, match="not an approved public default"):
        catalog.identity(
            "fleurs",
            resolved_config="en_us",
            split="test",
            data_fingerprint=DATA_FINGERPRINT,
            publishable=True,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("approved_by", "   ", "non-empty approver"),
        ("approved_at", "not-a-date", "ISO date"),
        ("evidence_url", "https://", "with a host"),
    ],
)
def test_human_approval_requires_reviewable_evidence(field, value, message):
    document = load_catalog().model_dump(mode="json")
    approval = {
        "status": "approved",
        "approved_by": "reviewer",
        "approved_at": "2026-08-10",
        "evidence_url": "https://example.com/approval",
    }
    approval[field] = value
    document["benchmarks"]["fleurs"]["rights_approval"] = approval

    with pytest.raises(CatalogValidationError, match=message):
        validate_catalog_document(document)


def test_record_fingerprint_is_key_order_stable_but_record_order_sensitive():
    records = [{"id": 1, "text": "alpha"}, {"id": 2, "text": "beta"}]
    same_records_different_key_order = [{"text": "alpha", "id": 1}, {"text": "beta", "id": 2}]

    digest = fingerprint_records(records)
    assert digest == fingerprint_records(records)
    assert digest == fingerprint_records(same_records_different_key_order)
    assert digest != fingerprint_records(list(reversed(records)))
