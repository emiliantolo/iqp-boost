from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False

from src.core import setup_iqp_circuit


ARTIFACT_SCHEMA_VERSION = "1.0"


def _params_to_list(params) -> list[float]:
    return np.asarray(params, dtype=np.float64).tolist()


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _models_to_payload(models) -> list[dict]:
    payload = []
    for idx, params in enumerate(models or []):
        payload.append({
            "index": int(idx),
            "params": _params_to_list(params),
        })
    return payload


def _serialize_gates(gates: list) -> list[list]:
    """Serialize gate list to human-readable format: [[q0, q1, ...], ...].
    
    Each gate is a list of qubit indices [q0] for single-qubit or [q0, q1, ...] for multi-qubit.
    Converts all NumPy int64 values to Python int for JSON serialization.
    """
    gate_list = []
    for gate in gates:
        if hasattr(gate, '__iter__') and not isinstance(gate, str):
            # Gate is already iterable (list or tuple of qubit indices)
            # Convert each qubit index to JSON-serializable Python int
            gate_list.append([_to_jsonable(q) for q in gate])
        else:
            # Fallback: single qubit
            gate_list.append([_to_jsonable(gate)])
    return gate_list


def save_circuit_artifact(
    path: str | Path,
    *,
    dataset_name: str,
    run_name: str | None,
    config: dict,
    dataset_spec: dict | None,
    x_train,
    sigma,
    circuit,
    circuit_config: dict,
    n_visible_qubits: int,
    wires,
    ensemble,
    ensemble_metrics_history: dict,
    ensemble_fcfw_weights=None,
    standalone_params=None,
    data_only_ensemble=None,
    data_only_history: dict | None = None,
    data_only_fcfw_weights=None,
) -> Path:
    """Save trained circuit parameters and weights for backend inference."""
    path = Path(path)

    dataset_samples_filename = "dataset_train_samples.npz"
    dataset_samples_path = path.parent / dataset_samples_filename
    dataset_train = np.asarray(x_train, dtype=np.int8)
    np.savez_compressed(dataset_samples_path, x_train=dataset_train)

    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": {
            "dataset_name": str(dataset_name),
            "run_name": str(run_name) if run_name is not None else None,
        },
        "dataset": {
            "spec": _to_jsonable(dataset_spec) if dataset_spec is not None else None,
            "run_config": _to_jsonable(config),
            "train_samples_file": dataset_samples_filename,
            "train_samples_shape": list(dataset_train.shape),
        },
        "metrics_context": {
            "sigma": _to_jsonable(sigma),
        },
        "circuit": {
            "n_visible_qubits": int(n_visible_qubits),
            "n_total_qubits": int(circuit.n_qubits),
            "wires": None if wires is None else [int(w) for w in wires],
            "config": dict(circuit_config),
            "gate_count": int(len(circuit.gates)),
            "gates": _serialize_gates(circuit.gates),
            "topology": str(circuit_config.get("topology", "unknown")),
        },
        "weights": {
            "strategy": str(config.get("weight_strategy", "frank_wolfe")),
            # Per-step accepted alpha values (Frank-Wolfe style coefficients).
            "alpha_history": [float(a) for a in ensemble_metrics_history.get("alpha", [])],
            # Final mixture weights after any corrective reweighting.
            "final_mixture_weights": [float(w) for w in ensemble.weights],
            # Optional: Fully Corrective Frank-Wolfe reweighted final mixture
            **({
                "fcfw": {
                    "final_mixture_weights": [float(w) for w in ensemble_fcfw_weights]
                }
            } if ensemble_fcfw_weights is not None else {}),
        },
        "ensemble": {
            "model_count": int(len(ensemble.models)),
            "models": _models_to_payload(ensemble.models),
            "training_loss_history": [float(v) for v in ensemble_metrics_history.get("training_loss", [])],
            "step_history": [int(v) for v in ensemble_metrics_history.get("step", [])],
        },
        "baselines": {
            "standalone": None,
            "data_only": None,
        },
        "execution_context": {
            "framework": "pennylane",
            "backend_hints": {
                "preferred_backend": "qiskit",
                "min_qubits_supported": int(circuit.n_qubits),
                "supports_classical_shadows": False,
            }
        },
    }

    if standalone_params is not None:
        artifact["baselines"]["standalone"] = {
            "params": _params_to_list(standalone_params),
        }

    if data_only_ensemble is not None:
        artifact["baselines"]["data_only"] = {
            "model_count": int(len(data_only_ensemble.models)),
            "models": _models_to_payload(data_only_ensemble.models),
            "final_mixture_weights": [float(w) for w in data_only_ensemble.weights],
            "alpha_history": [float(a) for a in (data_only_history or {}).get("alpha", [])],
            "training_loss_history": [float(v) for v in (data_only_history or {}).get("training_loss", [])],
            "step_history": [int(v) for v in (data_only_history or {}).get("step", [])],
            # Optional FCFW weights for data-only baseline
            **({
                "fcfw": {
                    "final_mixture_weights": [float(w) for w in data_only_fcfw_weights]
                }
            } if data_only_fcfw_weights is not None else {}),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
    return path


def restore_circuit_artifact(path: str | Path) -> dict:
    """Restore runnable circuit objects and parameter arrays from an artifact file."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    circuit_info = artifact.get("circuit", {})
    n_visible_qubits = int(circuit_info.get("n_visible_qubits"))
    circuit_config = dict(circuit_info.get("config", {}))

    dataset_train_samples = None
    dataset_entry = artifact.get("dataset", {})
    samples_file = dataset_entry.get("train_samples_file")
    if samples_file:
        samples_path = path.parent / str(samples_file)
        if samples_path.exists():
            with np.load(samples_path) as payload:
                dataset_train_samples = np.asarray(payload["x_train"], dtype=np.int8)

    circuit_kwargs = dict(circuit_config)
    if dataset_train_samples is not None:
        circuit_kwargs.setdefault("data", dataset_train_samples)

    circuit, _, _, wires = setup_iqp_circuit(n_visible_qubits, **circuit_kwargs)

    ensemble_models = [
        np.asarray(model["params"], dtype=np.float64)
        for model in artifact.get("ensemble", {}).get("models", [])
    ]
    ensemble_weights = np.asarray(
        artifact.get("weights", {}).get("final_mixture_weights", []),
        dtype=np.float64,
    )
    ensemble_fcfw_weights = np.asarray(
        artifact.get("weights", {}).get("fcfw", {}).get("final_mixture_weights", []),
        dtype=np.float64,
    )

    standalone_entry = artifact.get("baselines", {}).get("standalone")
    standalone_params = None
    if standalone_entry and standalone_entry.get("params") is not None:
        standalone_params = np.asarray(standalone_entry["params"], dtype=np.float64)

    data_only_entry = artifact.get("baselines", {}).get("data_only")
    data_only = None
    if data_only_entry:
        data_only = {
            "models": [
                np.asarray(model["params"], dtype=np.float64)
                for model in data_only_entry.get("models", [])
            ],
            "weights": np.asarray(data_only_entry.get("final_mixture_weights", []), dtype=np.float64),
            "alpha_history": np.asarray(data_only_entry.get("alpha_history", []), dtype=np.float64),
            "fcfw_weights": np.asarray(data_only_entry.get("fcfw", {}).get("final_mixture_weights", []), dtype=np.float64),
        }

    return {
        "artifact": artifact,
        "circuit": circuit,
        "wires": wires,
        "dataset_train_samples": dataset_train_samples,
        "sigma": _to_jsonable(artifact.get("metrics_context", {}).get("sigma")),
        "ensemble": {
            "models": ensemble_models,
            "weights": ensemble_weights,
            "alpha_history": np.asarray(artifact.get("weights", {}).get("alpha_history", []), dtype=np.float64),
            "fcfw_weights": ensemble_fcfw_weights,
        },
        "standalone": standalone_params,
        "data_only": data_only,
    }


def execute_circuit_native(
    circuit,
    params: np.ndarray,
    shots: int = 1024,
    wires: Optional[list] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Execute circuit using native IqpSimulator.sample() method.
    
    Args:
        circuit: IqpSimulator circuit object.
        params: Circuit parameters.
        shots: Number of samples to draw.
        wires: Optional list of visible qubit indices (for read-out).
        seed: Random seed for reproducibility.
    
    Returns:
        Array of samples with shape (shots, n_visible_qubits).
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = circuit.sample(params, shots=shots, wires=wires)
    return np.asarray(samples, dtype=np.int8)


def execute_circuit_ensemble_mixture(
    circuit,
    ensemble_models: list[np.ndarray],
    ensemble_weights: np.ndarray,
    shots: int = 1024,
    wires: Optional[list] = None,
    seed: Optional[int] = None,
) -> dict:
    """Execute ensemble as weighted mixture of models.
    
    Args:
        circuit: IqpSimulator circuit object.
        ensemble_models: List of parameter arrays for each model.
        ensemble_weights: Mixture weights (should sum to ~1.0).
        shots: Total number of shots to draw.
        wires: Optional list of visible qubit indices.
        seed: Random seed.
    
    Returns:
        Dict with keys:
            - 'samples': Aggregated samples from mixture (shape: shots x n_visible)
            - 'model_shots': List of shot counts per model
            - 'metadata': Info about shot allocation
    """
    if seed is not None:
        np.random.seed(seed)
    
    n_models = len(ensemble_models)
    weights = np.asarray(ensemble_weights, dtype=np.float64)
    weights = weights / weights.sum()  # Normalize
    
    # Allocate shots proportionally to weights
    model_shots = (weights * shots).astype(int)
    remainder = shots - model_shots.sum()
    if remainder > 0:
        # Assign remaining shots to model with highest weight
        model_shots[weights.argmax()] += remainder
    
    all_samples = []
    for idx, (model_params, n_shots) in enumerate(zip(ensemble_models, model_shots)):
        if n_shots > 0:
            model_samples = execute_circuit_native(circuit, model_params, shots=n_shots, wires=wires)
            all_samples.append(model_samples)
    
    aggregated = np.vstack(all_samples) if all_samples else np.empty((0, circuit.n_qubits), dtype=np.int8)
    
    return {
        "samples": aggregated,
        "model_shots": model_shots.tolist(),
        "weights": weights.tolist(),
        "total_shots": int(shots),
    }


def save_execution_results(
    path: str | Path,
    execution_metadata: dict,
    samples: np.ndarray,
) -> Path:
    """Save circuit execution results (shots and metadata).
    
    Args:
        path: Output file path (will save as .npz with accompanying .json).
        execution_metadata: Dict with execution info (model_shots, weights, etc.).
        samples: Sample array from circuit execution.
    
    Returns:
        Path to saved .npz file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save samples as compressed npz
    samples_array = np.asarray(samples, dtype=np.int8)
    np.savez_compressed(path, samples=samples_array)
    
    # Save metadata as JSON alongside
    metadata_path = path.with_suffix(".json")
    execution_metadata["samples_file"] = path.name
    execution_metadata["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(execution_metadata, f, indent=2)
    
    return path


def load_execution_results(path: str | Path) -> dict:
    """Load circuit execution results (shots and metadata).
    
    Args:
        path: Path to .npz file (or .json, will find paired file).
    
    Returns:
        Dict with 'samples' (ndarray), 'metadata' (dict from JSON).
    """
    path = Path(path)
    
    # Find the paired files
    npz_path = path.with_suffix(".npz") if path.suffix != ".npz" else path
    json_path = path.with_suffix(".json") if path.suffix != ".json" else path
    
    # Load samples
    samples = None
    if npz_path.exists():
        with np.load(npz_path) as f:
            samples = np.asarray(f["samples"], dtype=np.int8)
    
    # Load metadata
    metadata = {}
    if json_path.exists():
        with open(json_path, "r") as f:
            metadata = json.load(f)
    
    return {
        "samples": samples,
        "metadata": metadata,
    }

