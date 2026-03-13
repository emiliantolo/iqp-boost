# iqp-boost

Config-driven experiments for IQP ensemble boosting on binary datasets (BAS, parity, Gaussian mixture, blobs, and shapes).

## Run Experiments

Use a single CLI entrypoint and pass a JSON/TOML experiment file containing a list of runs.

```bash
uv run main.py --config configs/experiments.example.json
```

Optional controls:

```bash
# list run names found in the config
uv run main.py --config configs/experiments.example.json --list-runs

# validate config and selected runs without training
uv run main.py --config configs/experiments.example.json --dry-run

# run only selected named runs from the config
uv run main.py --config configs/experiments.example.json --only blobs_default parity_scan

# run selection also accepts 0-based indices from --list-runs order
uv run main.py --config configs/experiments.example.json --only 0 2

# override output base directory
uv run main.py --config configs/experiments.example.json --output-dir out_custom

# override config values for all selected runs
uv run main.py --config configs/experiments.example.json --set n_models=16 --set learning_rate=0.03

# example: force analytical mode for speed
uv run main.py --config configs/experiments.example.json --set skip_sampling=true
```

## Dataset Configs

Dataset-specific config files are available in:

- `configs/datasets/bas_4x4.json`
- `configs/datasets/bas_3x3.json`
- `configs/datasets/blobs.json`
- `configs/datasets/gaussian.json`
- `configs/datasets/parity.json`
- `configs/datasets/shapes.json`

Examples:

```bash
uv run main.py --config configs/datasets/blobs.json
uv run main.py --config configs/datasets/shapes.json --set skip_sampling=true
```

## Config Schema

- `output`: suite output settings (`base_dir`, `suite_name`)
- `defaults`: default training/circuit config merged into each run
- `runs`: list of run objects

Each run supports:

- `name`: subfolder name for the run output
- `dataset`: dataset selection (`name`: `bas|blobs|gaussian|parity|shapes`) and optional params
- `config`: per-run overrides merged on top of `defaults`
- `plot`: standardized plotting mode and params (`none|histogram|sample_grid|gaussian_summary`)
- `metric_configs`: optional metric progression overrides
- `baseline_epochs`: optional standalone baseline epochs override

Example config: `configs/experiments.example.json`

## Outputs

Each invocation creates one suite folder, then one subfolder per run:

- `out/<suite_name>_<timestamp>/<run_name>/log.txt`
- `out/<suite_name>_<timestamp>/<run_name>/results.csv`
- `out/<suite_name>_<timestamp>/<run_name>/config.json`
- plus plots and optional custom visualization files
