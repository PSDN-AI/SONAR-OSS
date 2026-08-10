"""Dataset discovery and preparation utilities."""

from .catalog import (
    BenchmarkCatalog,
    BenchmarkSpec,
    CatalogValidationError,
    DatasetIdentity,
    fingerprint_bytes,
    fingerprint_file,
    fingerprint_records,
    generated_catalog_schema,
    load_catalog,
    load_catalog_schema,
    validate_catalog_document,
    validate_catalog_schema_sync,
    validate_huggingface_revision,
)
from .discovery import DatasetDiscovery
from .preparer import DatasetPreparer, prepare_dataset
from .registry import (
    DATASET_REGISTRY,
    AvailableDataset,
    DatasetSpec,
)

__all__ = [
    "DatasetDiscovery",
    "DatasetPreparer",
    "BenchmarkCatalog",
    "BenchmarkSpec",
    "CatalogValidationError",
    "DATASET_REGISTRY",
    "AvailableDataset",
    "DatasetIdentity",
    "DatasetSpec",
    "fingerprint_bytes",
    "fingerprint_file",
    "fingerprint_records",
    "generated_catalog_schema",
    "load_catalog",
    "load_catalog_schema",
    "prepare_dataset",
    "validate_catalog_document",
    "validate_catalog_schema_sync",
    "validate_huggingface_revision",
]
