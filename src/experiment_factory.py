"""Compatibility wrapper for config-driven dataset construction."""

from src.dataset_catalog import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle

__all__ = ["DatasetBundle", "SUPPORTED_DATASETS", "build_dataset_bundle"]
