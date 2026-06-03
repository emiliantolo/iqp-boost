"""Dataset integrations."""

__all__ = ["DatasetBundle", "SUPPORTED_DATASETS", "build_dataset_bundle"]


def __getattr__(name):
    if name in __all__:
        from src.datasets.catalog import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle

        exports = {
            "DatasetBundle": DatasetBundle,
            "SUPPORTED_DATASETS": SUPPORTED_DATASETS,
            "build_dataset_bundle": build_dataset_bundle,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
