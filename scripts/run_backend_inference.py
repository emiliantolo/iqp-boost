"""Load circuit artifacts and execute on quantum backends (simulator or real hardware)."""

import sys
from pathlib import Path

# Add parent directory to path so we can import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import re
from datetime import datetime

import numpy as np

# Set up logging before any imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.circuit_artifacts import execute_circuit_native, restore_circuit_artifact


def sample_circuit(circuit_obj, params: np.ndarray, n_qubits: int, shots: int = None, wires=None, dry_run: bool = False):
    """Sample from the circuit with given parameters."""
    if dry_run:
        logger.info("[DRY RUN] Would execute circuit with %d qubits, %d shots", n_qubits, shots or 1024)
        return np.random.randint(0, 2, size=(shots or 1024, n_qubits), dtype=np.int8)

    return execute_circuit_native(
        circuit_obj,
        params,
        shots=shots or 1024,
        wires=wires,
    )


def execute_and_save_samples(
    circuit,
    params: np.ndarray,
    n_qubits: int,
    output_dir: Path,
    sample_name: str,
    shots: int,
    wires=None,
    dry_run: bool = False,
) -> dict:
    """Execute one parameter set and persist the shot matrix."""
    samples = sample_circuit(
        circuit,
        params,
        n_qubits,
        shots=shots,
        wires=wires,
        dry_run=dry_run,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shot_file = output_dir / f"{sample_name}_shots_{timestamp}.npy"
    np.save(shot_file, samples)

    logger.info("  Saved %d shots to: %s", len(samples), shot_file)
    return {
        "sample_name": sample_name,
        "shots_file": str(shot_file.relative_to(output_dir.parent)),
        "num_shots": int(len(samples)),
        "timestamp": timestamp,
    }


def make_inference_run_dir(artifact_path: Path, backend: str, mode: str) -> Path:
    """Create a dedicated subfolder for a single inference run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_backend = re.sub(r"[^A-Za-z0-9_.-]+", "-", backend)
    safe_mode = re.sub(r"[^A-Za-z0-9_.-]+", "-", mode)
    run_name = f"run_{timestamp}_{safe_backend}_{safe_mode}"
    run_dir = artifact_path.parent / "inference_results" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main(args: argparse.Namespace):
    """Load artifact and run inference on specified backend."""
    
    artifact_path = Path(args.artifact_path)
    if not artifact_path.exists():
        logger.error("Artifact file not found: %s", artifact_path)
        return
    
    logger.info("Loading circuit artifact from: %s", artifact_path)
    
    try:
        restored = restore_circuit_artifact(artifact_path)
    except Exception as e:
        logger.error("Failed to restore artifact: %s", e)
        return
    
    artifact = restored["artifact"]
    circuit = restored["circuit"]
    ensemble_models = restored["ensemble"]["models"]
    standalone_params = restored.get("standalone")
    wires = restored.get("wires")
    n_qubits = artifact["circuit"]["n_visible_qubits"]
    
    logger.info("Circuit restored: %d qubits, %d ensemble models", n_qubits, len(ensemble_models))
    
    # Set up a dedicated output directory for this run
    output_dir = make_inference_run_dir(artifact_path, args.backend, args.mode)
    
    if args.backend != "simulator":
        logger.warning(
            "Backend '%s' is currently ignored by this artifact replay path; using native circuit sampling.",
            args.backend,
        )
    
    shot_results = []

    if args.mode in ("ensemble", "both"):
        subset_end = len(ensemble_models)
        if args.subset is not None:
            subset_end = min(args.subset, len(ensemble_models))

        logger.info("Running ensemble inference on %d models (out of %d), %d shots per model",
                    subset_end, len(ensemble_models), args.shots)

        for model_idx in range(subset_end):
            params = ensemble_models[model_idx]
            model_name = f"ensemble_model_{model_idx}"

            logger.info("[%d/%d] Executing %s with %d shots...",
                        model_idx + 1, subset_end, model_name, args.shots)

            try:
                shot_results.append(
                    execute_and_save_samples(
                        circuit,
                        params,
                        n_qubits,
                        output_dir,
                        model_name,
                        args.shots,
                        wires=wires,
                        dry_run=args.dry_run,
                    ) | {"model_index": int(model_idx), "kind": "ensemble"}
                )
            except Exception as e:
                logger.error("  Failed to execute %s: %s", model_name, e)
                if not args.dry_run:
                    raise

    if args.mode in ("standalone", "both"):
        if standalone_params is None:
            logger.warning("Standalone execution requested, but the artifact has no standalone baseline.")
        else:
            logger.info("Running standalone inference with %d shots", args.shots)
            try:
                shot_results.append(
                    execute_and_save_samples(
                        circuit,
                        standalone_params,
                        n_qubits,
                        output_dir,
                        "standalone",
                        args.shots,
                        wires=wires,
                        dry_run=args.dry_run,
                    ) | {"kind": "standalone"}
                )
            except Exception as e:
                logger.error("  Failed to execute standalone baseline: %s", e)
                if not args.dry_run:
                    raise
    
    # Save inference metadata
    metadata = {
        "artifact_path": str(artifact_path),
        "backend": args.backend,
        "mode": args.mode,
        "shots": args.shots,
        "n_qubits": n_qubits,
        "ensemble_models_run": min(args.subset, len(ensemble_models)) if args.subset is not None else len(ensemble_models),
        "standalone_run": bool(args.mode in ("standalone", "both") and standalone_params is not None),
        "run_dir": str(output_dir.relative_to(artifact_path.parent)),
        "timestamp": datetime.now().isoformat(),
        "shot_results": shot_results,
    }
    
    metadata_file = output_dir / "inference_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("")
    logger.info("Inference complete.")
    logger.info("Metadata: %s", metadata_file)
    logger.info("Shots saved to: %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load a circuit artifact and execute on a quantum backend."
    )
    parser.add_argument(
        "artifact_path",
        type=str,
        help="Path to the circuit_artifact.json file",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="simulator",
        choices=["simulator", "qiskit-simulator", "qiskit-ibm"],
        help="Backend to use. Default: simulator (local)",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of shots per model. All models get the same number of shots (can be subsampled later). Default: 1024",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Run only the first N models (for testing). Default: all",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["ensemble", "standalone", "both"],
        help="Which circuit(s) to execute. Default: both (baseline + ensemble)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without executing the circuit.",
    )
    
    args = parser.parse_args()
    main(args)
