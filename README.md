# iqp-boost

Config-driven experiments for IQP ensemble boosting on Hopfield and Hamming Balls binary datasets.

## Run Experiments

Use a single CLI entrypoint and pass a JSON/TOML experiment file containing a list of runs.

```bash
uv run main.py --config configs/datasets/hopfield_16q_grid.json
```

`main.py` is a thin wrapper around `src.experiments.suite`. The equivalent
direct module command is:

```bash
python3 -m src.experiments.suite --config configs/datasets/hopfield_16q_grid.json
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

## Run HPO

Optuna HPO configs live under `configs/hpo/` and use the HPO entrypoint:

```bash
python3 -m src.hpo --config configs/hpo/hopfield_20q_p1_b15.json
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

The dataset catalog in `src/datasets/catalog.py` is the source of truth for
supported dataset keys, construction defaults, and dataset-specific plot modes.
See `docs/dataset_catalog.md` when adding or changing dataset integrations.

All supported datasets accept optional split params under `dataset.params`:
`test_samples` enables an `x_test` split, and `train_split_ratio` can override
the inferred train/test ratio.

## Experiment Architecture

The source tree is organized into layer packages:

- `src/run/`: experiment execution flow (`runner`, baselines, boosting steps,
  and final evaluation).
- `src/core/`: circuit setup, ensembles, losses, metrics, weighting, sigma
  selection, and evaluation policy.
- `src/datasets/`: dataset catalog, dataset classes, exact probability helpers,
  and dataset-specific visualization/evaluation adapters.
- `src/io/`: reporting, plotting, logging, output paths, and CSV/JSON writing.
- `src/experiments/`: runnable suite and HPO entrypoints.

The main public imports are curated at the package level:

- `src.run` exports `run_boosting_experiment` and run phase context/result types.
- `src.core` exports `BoostedEnsemble`, `EvaluationPolicy`, circuit setup, and
  weight strategy interfaces.
- `src.datasets` exports `DatasetBundle`, `SUPPORTED_DATASETS`, and
  `build_dataset_bundle`.
- `src.experiments` exports suite/HPO entrypoints and dataset factory helpers.

Experiment runs are orchestrated by `src/run/runner.py`, with two focused
modules owning the main config-driven seams:

- `src/datasets/catalog.py` builds a typed dataset bundle from each run's
  `dataset` spec.
- `src/core/evaluation.py` owns sampling and metric evaluation through
  `EvaluationPolicy`.

`EvaluationPolicy` centralizes the run's training data, kernel sigma, shot
count, RNG seed, optional dataset metric callbacks, exact probabilities, and
the `skip_sampling` / `final_eval_sampling` flags. The runner uses it for
baseline, ensemble, FCFW, final, and held-out test evaluation so sampling
rules and metric shapes stay consistent across those paths.

When `skip_sampling=true`, intermediate ensemble reporting uses analytical
training MMD without drawing samples. When `final_eval_sampling=true`, final
sample-based metrics are still computed even if intermediate sampling was
skipped.

## Outputs

Each invocation creates one suite folder, then one subfolder per run:

- `out/<suite_name>_<timestamp>/<run_name>/log.txt`
- `out/<suite_name>_<timestamp>/<run_name>/results.csv`
- `out/<suite_name>_<timestamp>/<run_name>/config.json`
- plus plots and optional custom visualization files


## Backend inference & metrics

Scripts
- `scripts/run_backend_inference.py` — replay a saved circuit artifact to produce shot files (.npy).
- `scripts/compute_backend_metrics.py` — compute metrics from shot files and the original artifact.

### 1) Run inference (replay saved models):
#### run_backend_inference.py

```bash
python scripts/run_backend_inference.py /path/to/circuit_artifact.json [--backend BACKEND] [--shots N] [--subset N] [--mode MODE] [--dry-run]
```

- `artifact_path`: required path to the `circuit_artifact.json` produced by training.
- `--backend`: backend to use for inference (default: `simulator`).
- `--shots`: shots per model (default 1024).
- `--subset`: run only first N ensemble models (for tests).
- `--mode`: `ensemble`, `standalone`, or `both` (default: `both`).
- `--dry-run`: validate without executing sampling.

#### Output

Outputs a timestamped per-run folder under `inference_results/` containing `inference_metadata.json` and `.npy` shot files.

#### Backends:

| Backend | Usage | Requirements |
|---------|-------|--------------|
| `simulator` (default) | Local testing | PennyLane only |
| `qiskit-simulator` | Qiskit Aer (classical) | `pip install pennylane-qiskit` |
| `qiskit-ibm` | Real IBM Quantum hardware | `.env` with `IBM_TOKEN` |


### 2) Compute metrics from a run:

#### compute_backend_metrics.py

```bash
# Single mode (original behavior)
python scripts/compute_backend_metrics.py /path/to/circuit_artifact.json [--inference-dir PATH] [--shots N]

# Batch mode: process all artifacts in a directory tree
python scripts/compute_backend_metrics.py /path/to/experiments/ [--shots N]
```

- `artifact_path`: path to a `circuit_artifact.json` file (single mode) **or** a parent directory whose subdirectories are searched recursively for artifacts (batch mode).
- `--inference-dir`: optional path to a specific run folder under `inference_results/`. If omitted, **all** inference runs under `inference_results/` are processed.
- `--shots`: number of generated samples to use for metrics (default: 1024).

Creates `metrics/backend_metrics.json` and `metrics/common_metrics_*.png` plots inside the selected inference run folder; includes per-model metrics and ensemble aggregates (standard and FCFW-weighted when available), plus baselines.

#### Metrics Definitions:
- **MMD**: Maximum Mean Discrepancy (lower is better)
- **TVD**: Total Variation Distance (lower is better)
- **KL**: Kullback-Leibler divergence vs. training distribution (lower is better)
- **Coverage**: % of training set states observed in samples (higher is better)
- **Validity**: % of samples that are valid bitstrings (always 100% for binary data)

