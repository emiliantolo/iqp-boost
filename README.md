# Backend inference & metrics

## Prerequisites
- Python and project dependencies installed (see `pyproject.toml`).

Scripts
- `scripts/run_backend_inference.py` — replay a saved circuit artifact to produce shot files (.npy).
- `scripts/compute_backend_metrics.py` — compute metrics from shot files and the original artifact.

## Usage

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
python scripts/compute_backend_metrics.py /path/to/circuit_artifact.json [--inference-dir PATH] [--shots N]
```
- `artifact_path`: required path to the `circuit_artifact.json` produced by training.
- `--inference-dir`: optional path to a specific run folder under `inference_results/`. If omitted the script picks the latest run.
- `--shots`: sample budget, total number of generated samples for metrics (default 1024).

#### Output

Creates `inference_results/backend_metrics_*.json` saved into the selected run folder; includes per-model metrics and ensemble aggregates (standard and FCFW-weighted when available), plus baselines.

#### Metrics Definitions:
- **MMD**: Maximum Mean Discrepancy (lower is better)
- **TVD**: Total Variation Distance (lower is better)
- **KL**: Kullback-Leibler divergence vs. training distribution (lower is better)
- **Coverage**: % of training set states observed in samples (higher is better)
- **Validity**: % of samples that are valid bitstrings (always 100% for binary data)


## Example (BAS on Simulator)

### 1. Execute on local simulator

```bash
uv run scripts/run_backend_inference.py out_benchmark_suite_aachen/benchmark_suite_bas_16q_20260528_184359/circuit_artifact.json --backend simulator --shots 1024
```

#### Expected output:
```
out_benchmark_suite_aachen/benchmark_suite_bas_16q_20260528_184359/inference_results/model_*.npy
out_benchmark_suite_aachen/benchmark_suite_bas_16q_20260528_184359/inference_results/inference_metadata.json
```

### 2. Compute metrics
```bash
uv run scripts/compute_backend_metrics.py out_benchmark_suite_aachen/benchmark_suite_bas_16q_20260528_184359/circuit_artifact.json --shots 1024
```

#### Expected output:
```
out_benchmark_suite_aachen/benchmark_suite_bas_16q_20260528_184359/inference_results/backend_metrics_*.json
```
