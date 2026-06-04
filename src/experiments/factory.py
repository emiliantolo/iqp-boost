"""Compatibility wrapper for config-driven dataset construction."""

from src.datasets import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle

__all__ = ["DatasetBundle", "SUPPORTED_DATASETS", "build_dataset_bundle"]
