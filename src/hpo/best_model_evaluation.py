"""Dataset-specific best-model evaluation for HPO finalization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.benchmark_metrics import compute_hamming_balls_metrics
from src.core import setup_iqp_circuit
from src.datasets.hamming_balls import HammingBallsDataset
from src.core.ensemble import BoostedEnsemble


def evaluate_best_model(
    hpo_spec: dict,
    best_config_path: Path | None,
    best_model_path: Path,
    fcfw_weights_list: list | None,
) -> tuple[str, dict] | None:
    """Return ``(summary_key, metrics)`` for datasets with HPO postprocessing."""
    dataset_spec = hpo_spec.get("dataset", {})
    if dataset_spec.get("name") != "hamming_balls":
        return None
    metrics = evaluate_best_model_hamming_balls(
        hpo_spec=hpo_spec,
        best_config_path=best_config_path,
        best_model_path=best_model_path,
        fcfw_weights_list=fcfw_weights_list,
    )
    if metrics is None:
        return None
    return "hamming_balls_metrics", metrics


def evaluate_best_model_hamming_balls(
    hpo_spec: dict,
    best_config_path: Path | None,
    best_model_path: Path,
    fcfw_weights_list: list | None,
) -> dict | None:
    """Reconstruct the best HammingBalls ensemble and evaluate sampled metrics."""
    if best_config_path is None or not best_config_path.exists():
        print("[Postprocess] best_config.json missing, skipping HammingBalls metrics")
        return None
    if not best_model_path.exists():
        print("[Postprocess] best_model.json missing, skipping HammingBalls metrics")
        return None

    best_cfg = json.loads(best_config_path.read_text())
    dataset_spec = hpo_spec.get("dataset", {})
    params = dict(dataset_spec.get("params", {}))
    dataset = HammingBallsDataset(
        n_qubits=int(params.get("n_qubits", 16)),
        K=int(params.get("K", 8)),
        p=float(params.get("p", 0.1)),
        pattern_seed=int(params.get("pattern_seed", 0)),
        batch_size=int(params.get("batch_size", 2**20)),
        max_exact_states=int(params.get("max_exact_states", 2**20)),
    )

    circuit_cfg = dict(best_cfg.get("circuit_config", {}))
    circuit, _, _, _ = setup_iqp_circuit(dataset.n_qubits, **circuit_cfg)
    ensemble = BoostedEnsemble.load(
        str(best_model_path),
        iqp_circuit=circuit,
        n_samples=int(best_cfg.get("n_samples", 512)),
    )

    shots = int(best_cfg.get("shots", 10000))
    rng_seed = int(best_cfg.get("rng_seed", 42))
    radius_fraction = float(params.get("radius_fraction", 0.15))
    rng = np.random.default_rng(rng_seed)

    normal_samples = ensemble.sample(shots, rng)
    metrics = {
        "ensemble": compute_hamming_balls_metrics(
            dataset.centers,
            normal_samples,
            dataset.n_qubits,
            radius_fraction=radius_fraction,
        )
    }

    if fcfw_weights_list is not None and len(fcfw_weights_list) > 0:
        fcfw_weights = np.asarray(fcfw_weights_list, dtype=np.float64)
        fcfw_samples = ensemble.sample(shots, rng, weights_override=fcfw_weights)
        metrics["ensemble_fcfw"] = compute_hamming_balls_metrics(
            dataset.centers,
            fcfw_samples,
            dataset.n_qubits,
            radius_fraction=radius_fraction,
        )

    return metrics
