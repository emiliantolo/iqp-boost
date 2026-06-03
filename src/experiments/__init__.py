"""Experiment entrypoints."""

__all__ = [
    "DEFAULT_RUN_CONFIG",
    "DatasetBundle",
    "SUPPORTED_DATASETS",
    "build_dataset_bundle",
    "run_hpo",
    "run_suite",
]


def __getattr__(name):
    if name in {"DatasetBundle", "SUPPORTED_DATASETS", "build_dataset_bundle"}:
        from src.experiments.factory import DatasetBundle, SUPPORTED_DATASETS, build_dataset_bundle

        exports = {
            "DatasetBundle": DatasetBundle,
            "SUPPORTED_DATASETS": SUPPORTED_DATASETS,
            "build_dataset_bundle": build_dataset_bundle,
        }
        return exports[name]
    if name in {"DEFAULT_RUN_CONFIG", "run_suite"}:
        from src.experiments.suite import DEFAULT_RUN_CONFIG, run_suite

        return {"DEFAULT_RUN_CONFIG": DEFAULT_RUN_CONFIG, "run_suite": run_suite}[name]
    if name == "run_hpo":
        from src.experiments.hpo import run_hpo

        return run_hpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
