"""Dataset discovery and preparation utilities."""

from .catalog import (
    BenchmarkCatalog,
    BenchmarkSpec,
    CatalogValidationError,
    DatasetIdentity,
    fingerprint_records,
    load_catalog,
    validate_catalog_document,
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
    "fingerprint_records",
    "load_catalog",
    "prepare_dataset",
    "validate_catalog_document",
    "validate_huggingface_revision",
]
