# iqp-boost

Config-driven experiments for IQP ensemble boosting on Hopfield and Hamming Balls binary datasets.

## Run Experiments

Use a single CLI entrypoint and pass a JSON/TOML experiment file containing a list of runs.

```bash
uv run main.py --config configs/datasets/hopfield_16q_grid.json
```

Optional controls:

```bash
# list run names found in the config
uv run main.py --config configs/datasets/hopfield_16q_grid.json --list-runs

# validate config and selected runs without training
uv run main.py --config configs/datasets/hopfield_16q_grid.json --dry-run

# run only selected named runs from the config
uv run main.py --config configs/datasets/hopfield_16q_grid.json --only 0

# run selection also accepts 0-based indices from --list-runs order
uv run main.py --config configs/datasets/hopfield_16q_grid.json --only 0

# override output base directory
uv run main.py --config configs/datasets/hopfield_16q_grid.json --output-dir out_custom

# override config values for all selected runs
uv run main.py --config configs/datasets/hopfield_16q_grid.json --set n_models=16 --set learning_rate=0.03

# example: force analytical mode for speed
uv run main.py --config configs/datasets/hopfield_16q_grid.json --set skip_sampling=true
```

## Dataset Configs

Dataset-specific config files are available in:

- `configs/datasets/hopfield_16q_grid.json`
- `configs/datasets/benchmark_suite_hamming_balls/hamming_balls_16q.json`
- `configs/datasets/benchmark_suite_hamming_balls/hamming_balls_20q.json`
- `configs/datasets/benchmark_suite_hamming_balls/hamming_balls_50q.json`
- `configs/datasets/benchmark_suite_hamming_balls/hamming_balls_100q.json`

Use `hamming_balls` as the dataset key for Hamming Balls configs.

Examples:

```bash
uv run main.py --config configs/datasets/hopfield_16q_grid.json
uv run main.py --config configs/datasets/hopfield_16q_grid.json --set skip_sampling=true
```

## Config Schema

- `output`: suite output settings (`base_dir`, `suite_name`)
- `defaults`: default training/circuit config merged into each run
- `runs`: list of run objects

Each run supports:

- `name`: subfolder name for the run output
- `dataset`: dataset selection (`name`: `hopfield|hamming_balls`) and optional params
- `config`: per-run overrides merged on top of `defaults`
- `plot`: optional plotting mode and params (`none|boltzmann_summary|hamming_balls_mode_evolution`)
- `metric_configs`: optional metric progression overrides
- `baseline_epochs`: optional standalone baseline epochs override

Example config: `configs/datasets/hopfield_16q_grid.json`

The dataset catalog in `src/dataset_catalog.py` is the source of truth for
supported dataset keys, construction defaults, and dataset-specific plot modes.
See `docs/dataset_catalog.md` when adding or changing dataset integrations.

All supported datasets accept optional split params under `dataset.params`:
`test_samples` enables an `x_test` split, and `train_split_ratio` can override
the inferred train/test ratio.

## Outputs

Each invocation creates one suite folder, then one subfolder per run:

- `out/<suite_name>_<timestamp>/<run_name>/log.txt`
- `out/<suite_name>_<timestamp>/<run_name>/results.csv`
- `out/<suite_name>_<timestamp>/<run_name>/config.json`
- plus plots and optional custom visualization files
