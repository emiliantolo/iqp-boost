# Experiments

This branch keeps Hopfield and Hamming Balls as the runnable dataset families.
`src/dataset_catalog.py` is the source of truth for supported config keys and
dataset-specific defaults.

## Supported Datasets

- `hopfield`: available now through `configs/datasets/hopfield_16q_grid.json` and the Hopfield HPO configs in `configs/hpo/`.
- `hamming_balls`: available now through `configs/datasets/benchmark_suite_hamming_balls/`.

Historical benchmark and grid configs for other removed datasets were deleted so stale experiments are not advertised as runnable.

See `docs/dataset_catalog.md` for the developer checklist for adding a dataset,
including single-instance configs, HPO configs, tests, and docs.
