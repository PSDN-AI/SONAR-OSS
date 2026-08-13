"""Offline validation and immutable identity tests for the benchmark catalog."""

from __future__ import annotations

import pytest

from psdn_sonar.data.catalog import CatalogValidationError, fingerprint_records, load_catalog, validate_catalog_document

DATA_FINGERPRINT = "sha256:" + "1" * 64


def test_bundled_catalog_validates_offline():
    catalog = load_catalog()
    assert catalog.schema_version == 1
    assert catalog.get("common_voice").enabled is False
    assert catalog.get("multilingual_librispeech").text_column == "transcript"
    assert len(catalog.get("fleurs").review.fingerprints) == 12
    assert len(catalog.get("zeroth").review.fingerprints) == 2


def test_duplicate_yaml_keys_are_rejected(tmp_path):
    path = tmp_path / "duplicate.yaml"
    path.write_text("schema_version: 1\nschema_version: 1\nbenchmarks: {}\n", encoding="utf-8")
    with pytest.raises(CatalogValidationError, match="duplicate catalog key: 'schema_version'"):
        load_catalog(path)


def test_floating_huggingface_revision_is_rejected():
    document = load_catalog().model_dump(mode="json")
    document["benchmarks"]["fleurs"]["source"]["revision"] = "main"
    with pytest.raises(CatalogValidationError, match="full 40-character commit SHA"):
        validate_catalog_document(document)


def test_identity_changes_for_each_material_dimension():
    catalog = load_catalog()
    base = catalog.identity("fleurs", resolved_config="en_us", split="test", data_fingerprint=DATA_FINGERPRINT)
    variants = [
        base.model_copy(update={"source_revision": "2" * 40}),
        base.model_copy(update={"selection": "different_selection@1"}),
        base.model_copy(update={"preprocessing": "different_preprocessing@1"}),
        base.model_copy(update={"split": "train"}),
        base.model_copy(update={"config": "bn_in"}),
        base.model_copy(update={"data_fingerprint": "sha256:" + "2" * 64}),
    ]
    assert len({base.fingerprint, *(identity.fingerprint for identity in variants)}) == 1 + len(variants)


def test_only_approved_exact_data_is_publishable():
    catalog = load_catalog()
    recorded = catalog.get("fleurs").review.fingerprints["af_za::test"]
    kwargs = dict(resolved_config="af_za", split="test", data_fingerprint=recorded)
    with pytest.raises(ValueError, match="not approved"):
        catalog.identity("fleurs", publishable=True, **kwargs)

    document = catalog.model_dump(mode="json")
    document["benchmarks"]["fleurs"]["review"] = {
        "decision": "reference_only",
        "rationale": "Approved test fixture.",
        "approved_by": "reviewer",
        "approved_at": "2026-08-11",
        "evidence_url": "https://example.com/review",
        "fingerprints": {"af_za::test": recorded},
    }
    approved = validate_catalog_document(document)
    identity = approved.identity("fleurs", publishable=True, **kwargs)
    approved.verify_publishable_identity(identity)

    with pytest.raises(ValueError, match="not approved"):
        approved.identity("fleurs", publishable=True, **{**kwargs, "data_fingerprint": "sha256:" + "2" * 64})
    with pytest.raises(ValueError, match="approved catalog identity"):
        approved.verify_publishable_identity(identity.model_copy(update={"selection": "changed@1"}))


def test_approved_review_requires_named_evidence():
    document = load_catalog().model_dump(mode="json")
    review = document["benchmarks"]["fleurs"]["review"]
    review.update(decision="reference_only", fingerprints={"en_us::test": DATA_FINGERPRINT})
    with pytest.raises(CatalogValidationError, match="named evidence"):
        validate_catalog_document(document)


def test_record_fingerprint_is_deterministic_and_ordered():
    records = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}]
    assert fingerprint_records(records) == fingerprint_records([{"text": "a", "id": 1}, {"text": "b", "id": 2}])
    assert fingerprint_records(records) != fingerprint_records(list(reversed(records)))
