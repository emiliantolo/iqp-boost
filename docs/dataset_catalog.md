# Dataset Catalog

`src/datasets/catalog.py` is the source of truth for config-driven datasets.
It owns supported dataset keys, dataset construction, plot defaults, and the
typed `DatasetBundle` Interface consumed by experiment runners.

`DatasetBundle` is the runner-facing dataset integration Module. It carries the
generated train/test samples, exact probabilities when available, dataset metric
hooks, and custom visualization hook behind one Interface. Runners should pass a
bundle directly instead of unpacking dataset capabilities into separate keyword
arguments.

## Supported Dataset Keys

- `hopfield`: Hopfield patterns with exact probabilities when feasible.
- `hamming_balls`: Hamming Balls patterns with exact probabilities when feasible.
- `mnist`: MNIST digit images resized to `10x10`, binarized with a default
  threshold of `0.4`, and flattened to 100-bit samples.

The singular key `hamming_ball` is intentionally unsupported; use
`hamming_balls` in configs.

## Optional Train/Test Splits

All catalog datasets support optional train/test split params inside
`dataset.params`.

- `test_samples`: when positive, enables a test split and controls the desired
  number of test samples.
- `train_split_ratio`: optional explicit split ratio. When omitted, the catalog
  uses `train_samples / (train_samples + test_samples)` so the returned train
  and test arrays match the requested counts.
- `balanced_per_class`: MNIST-only opt-in. When true, `train_samples` and
  `test_samples` must be divisible by the selected class count and the catalog
  samples the same number of examples from each class in each split.

When `test_samples` is omitted or zero, bundles include only `x_train` and
`x_test` remains `None`.

## Config Locations

Single-instance experiment suites live under `configs/datasets/`.

- `configs/datasets/hamming_balls/`
- `configs/datasets/mnist/`

HPO experiment configs live under per-dataset folders in `configs/hpo/`.

- `configs/hpo/hamming_balls/`
- `configs/hpo/hopfield/`
- `configs/hpo/mnist/`

HPO and single-instance experiments both resolve datasets through the same
catalog Module.

## Plot Kinds

- `none`: disable dataset-specific custom visualization.
- `boltzmann_summary`: Hopfield Boltzmann summary visualizations.
- `hamming_balls_mode_evolution`: accepted for Hamming Balls; currently a no-op
  custom visualization hook.

If a run omits `plot.kind`, the catalog chooses the dataset default:
`boltzmann_summary` for Hopfield and `hamming_balls_mode_evolution` for Hamming
Balls. MNIST defaults to `none`.

## Adding A Dataset

To add a dataset, implement the dataset class under `src/datasets/`, then add a
catalog entry in `src/datasets/catalog.py` that builds a `DatasetBundle`.

The entry should define:

- the config key;
- default plot kind;
- supported params and defaults;
- generated `x_train` and optional `x_test`;
- exact probabilities and generation evaluation hooks when available.
- any dataset metric or custom visualization hooks exposed through the bundle.

Also add at least one single-instance config, an HPO config if the dataset is
expected to support HPO, catalog tests for construction and unsupported keys,
and README/experiment documentation updates.
