"""Compute metrics from backend inference shots and artifact context."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.circuit_artifacts import restore_circuit_artifact, execute_circuit_native
from src.utils import compute_kl_divergence, compute_mmd, compute_tvd


def compute_sample_metrics(ground_truth: np.ndarray, samples: np.ndarray, sigma: float) -> dict:
    """Compute standard metrics: TVD, MMD, KL divergence."""
    metrics = {}
    
    try:
        metrics['mmd'] = float(compute_mmd(ground_truth, samples, sigma))
    except Exception as e:
        logger.warning("Failed to compute MMD: %s", e)
        metrics['mmd'] = float('nan')
    
    try:
        metrics['tvd'] = float(compute_tvd(ground_truth, samples))
    except Exception as e:
        logger.warning("Failed to compute TVD: %s", e)
        metrics['tvd'] = float('nan')
    
    try:
        metrics['kl'] = float(compute_kl_divergence(ground_truth, samples))
    except Exception as e:
        logger.warning("Failed to compute KL: %s", e)
        metrics['kl'] = float('nan')
    
    return metrics


def compute_coverage_validity(samples: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Compute coverage and validity metrics."""
    metrics = {}
    
    # Coverage: fraction of unique ground truth samples observed in generated samples
    unique_gt = set(map(tuple, ground_truth))
    observed = set(map(tuple, samples))
    coverage_count = len(observed & unique_gt)
    metrics['coverage'] = float(coverage_count) / float(len(unique_gt)) * 100.0 if len(unique_gt) > 0 else 0.0
    
    # Validity: always 100% for binary samples (all bitstrings are valid)
    metrics['validity'] = 100.0
    
    return metrics


def subsample_shots(shots: np.ndarray, n_samples: int) -> np.ndarray:
    """Randomly subsample shots to n_samples if needed."""
    if len(shots) <= n_samples:
        return shots
    indices = np.random.choice(len(shots), size=n_samples, replace=False)
    return shots[indices]


def combine_shots_by_weights(shot_pools: list[np.ndarray], weights: np.ndarray, total_n: int) -> np.ndarray:
    """Combine per-model shot pools into a single sample set according to mixture weights.

    Args:
        shot_pools: list of arrays (n_shots_i, n_qubits)
        weights: array-like weights for each pool
        total_n: total number of samples to draw for the ensemble

    Returns:
        Combined samples array (total_n, n_qubits)
    """
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(shot_pools), dtype=float)
    weights = weights / weights.sum()

    # Allocate counts
    counts = np.floor(weights * total_n).astype(int)
    remainder = total_n - counts.sum()
    if remainder > 0:
        # assign remainder to largest weights
        idxs = np.argsort(-weights)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1

    selected = []
    for pool, c in zip(shot_pools, counts):
        if c <= 0:
            continue
        if len(pool) == 0:
            continue
        if c <= len(pool):
            idxs = np.random.choice(len(pool), size=c, replace=False)
        else:
            idxs = np.random.choice(len(pool), size=c, replace=True)
        selected.append(pool[idxs])

    if not selected:
        return np.empty((0, 0), dtype=np.int8)
    return np.vstack(selected)


def sample_models_direct(models: list[np.ndarray], weights: np.ndarray, circuit, wires, total_n: int, seed: int | None = None) -> np.ndarray:
    """Sample directly from a list of model parameter arrays using the provided circuit.

    Allocates samples proportional to `weights` and draws from each model via `execute_circuit_native`.
    """
    if seed is not None:
        np.random.seed(seed)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(models), dtype=float)
    weights = weights / weights.sum()

    counts = np.floor(weights * total_n).astype(int)
    remainder = total_n - counts.sum()
    if remainder > 0:
        idxs = np.argsort(-weights)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1

    pools = []
    for params, c in zip(models, counts):
        if c <= 0:
            continue
        pool = execute_circuit_native(circuit, params, shots=int(c), wires=wires)
        pools.append(pool)

    if not pools:
        return np.empty((0, 0), dtype=np.int8)
    return np.vstack(pools)


def load_inference_shot_entries(inference_dir: Path) -> tuple[list[dict], dict | None]:
    """Load shot metadata entries from inference_metadata.json, with filename fallback."""
    metadata_path = inference_dir / "inference_metadata.json"
    metadata = None
    entries: list[dict] = []

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            entries = list(metadata.get("shot_results", []) or [])
        except Exception as e:
            logger.warning("Failed to read inference metadata at %s: %s", metadata_path, e)

    if entries:
        return entries, metadata

    # Backward-compatible fallback for older runs without metadata or with partial metadata.
    fallback_files = sorted(
        list(inference_dir.glob("ensemble_model_*_shots_*.npy"))
        + list(inference_dir.glob("model_*_shots_*.npy"))
        + list(inference_dir.glob("standalone_shots_*.npy"))
    )
    for shot_file in fallback_files:
        stem = shot_file.stem
        kind = "standalone" if stem.startswith("standalone_") else "ensemble"
        entries.append({
            "sample_name": stem.rsplit("_shots_", 1)[0],
            "shots_file": shot_file.name,
            "kind": kind,
        })

    return entries, metadata


def resolve_shot_path(inference_dir: Path, shot_ref: str) -> Path:
    """Resolve a shot file path from metadata or a bare filename."""
    shot_path = Path(shot_ref)
    if shot_path.is_absolute():
        return shot_path
    if shot_path.parts and shot_path.parts[0] == inference_dir.name:
        return inference_dir.parent / shot_path
    if len(shot_path.parts) >= 2 and shot_path.parts[0] == inference_dir.parent.name and shot_path.parts[1] == inference_dir.name:
        return inference_dir.parent / shot_path
    return inference_dir / shot_path


def resolve_inference_dir(artifact_path: Path, inference_dir_arg: str | None) -> Path:
    """Resolve the inference run directory to analyze.

    If a folder is passed explicitly, use it. Otherwise, pick the latest run folder
    under inference_results/ and fall back to the legacy flat directory.
    """
    if inference_dir_arg:
        candidate = Path(inference_dir_arg)
        if candidate.exists():
            return candidate
        return candidate if candidate.is_absolute() else artifact_path.parent / candidate

    root = artifact_path.parent / "inference_results"
    if not root.exists():
        return root

    metadata_runs = sorted(
        [p.parent for p in root.glob("*/inference_metadata.json")],
        key=lambda p: (p.stat().st_mtime, p.name),
    )
    if metadata_runs:
        return metadata_runs[-1]

    subdirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: (p.stat().st_mtime, p.name))
    if subdirs:
        return subdirs[-1]

    return root


def main(args: argparse.Namespace):
    """Compute metrics from backend inference shots."""
    
    artifact_path = Path(args.artifact_path)
    if not artifact_path.exists():
        logger.error("Artifact file not found: %s", artifact_path)
        return
    
    logger.info("Loading circuit artifact: %s", artifact_path)
    
    try:
        restored = restore_circuit_artifact(artifact_path)
    except Exception as e:
        logger.error("Failed to restore artifact: %s", e)
        return
    
    artifact = restored["artifact"]
    dataset_train_samples = restored["dataset_train_samples"]
    sigma = restored["sigma"]
    ensemble_models = restored["ensemble"]["models"]
    ensemble_weights = restored["ensemble"]["weights"]
    n_qubits = artifact["circuit"]["n_visible_qubits"]
    
    if dataset_train_samples is None:
        logger.error("Training dataset samples not available in artifact")
        return
    
    logger.info("Dataset: %d training samples, %d qubits", len(dataset_train_samples), n_qubits)
    logger.info("Sigma: %s", sigma)
    logger.info("Ensemble: %d models, weights sum=%.4f", len(ensemble_models), ensemble_weights.sum())
    
    # Determine inference results directory
    inference_dir = resolve_inference_dir(artifact_path, args.inference_dir)
    if not inference_dir.exists():
        logger.error("Inference results directory not found: %s", inference_dir)
        logger.info("Run: python scripts/run_backend_inference.py %s", artifact_path)
        return
    
    # Load shot entries from metadata first, with filename fallback.
    shot_entries, inference_metadata = load_inference_shot_entries(inference_dir)
    if not shot_entries:
        logger.error("No shot files found in: %s", inference_dir)
        return
    
    logger.info("Found %d shot entries", len(shot_entries))
    
    # Determine sampling configuration
    shots_budget = args.shots
    if shots_budget is None:
        shots_budget = len(dataset_train_samples)
    logger.info("Using shots=%d (metric sample budget)", shots_budget)
    
    # Compute per-model metrics
    model_metrics = []
    ensemble_shots_all = []
    
    standalone_shots = None

    for entry_idx, entry in enumerate(shot_entries):
        shot_file = resolve_shot_path(inference_dir, entry.get("shots_file", ""))
        sample_name = entry.get("sample_name", shot_file.name)
        kind = entry.get("kind", "ensemble")

        if not shot_file.exists():
            logger.warning("Skipping missing shot file: %s", shot_file)
            continue

        logger.info("[%d/%d] Computing metrics for: %s", 
                   entry_idx + 1, len(shot_entries), shot_file.name)
        
        try:
            shots = np.load(shot_file)
            logger.info("  Loaded %d shots", len(shots))
        except Exception as e:
            logger.error("  Failed to load shots: %s", e)
            continue
        
        # Subsample if needed
        if shots_budget and len(shots) > shots_budget:
            shots = subsample_shots(shots, shots_budget)
            logger.info("  Subsampled to %d shots", len(shots))
        
        # Compute metrics
        sample_mets = compute_sample_metrics(dataset_train_samples, shots, sigma)
        coverage_mets = compute_coverage_validity(shots, dataset_train_samples)
        
        if kind == "standalone":
            standalone_shots = shots
        else:
            model_metrics.append({
                "sample_name": sample_name,
                "n_shots_used": int(len(shots)),
                "metrics": {**sample_mets, **coverage_mets}
            })
        
        logger.info("  MMD: %.6f, TVD: %.4f, KL: %.4f, Coverage: %.2f%%",
                   sample_mets.get('mmd', float('nan')),
                   sample_mets.get('tvd', float('nan')),
                   sample_mets.get('kl', float('nan')),
                   coverage_mets.get('coverage', float('nan')))
        
        # Store for ensemble computation
        if kind != "standalone":
            ensemble_shots_all.append(shots)
    
    # Compute ensemble metrics (standard weights)
    ensemble_sample_mets = None
    ensemble_coverage_mets = None
    ensemble_fcfw_sample_mets = None
    ensemble_fcfw_coverage_mets = None

    if ensemble_shots_all and len(ensemble_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble metrics (standard weights)...")
        total_n = shots_budget
        ensemble_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_weights, total_n)
        if ensemble_shots.size:
            ensemble_sample_mets = compute_sample_metrics(dataset_train_samples, ensemble_shots, sigma)
            ensemble_coverage_mets = compute_coverage_validity(ensemble_shots, dataset_train_samples)
            logger.info("Ensemble MMD: %.6f", ensemble_sample_mets.get('mmd', float('nan')))
            logger.info("Ensemble TVD: %.4f", ensemble_sample_mets.get('tvd', float('nan')))
            logger.info("Ensemble KL: %.4f", ensemble_sample_mets.get('kl', float('nan')))
            logger.info("Ensemble Coverage: %.2f%%", ensemble_coverage_mets.get('coverage', float('nan')))
    else:
        logger.warning("Ensemble metrics skipped: incomplete shot data for standard weights")

    # If FCFW weights are present in artifact, compute ensemble metrics using them as well
    ensemble_fcfw_weights = restored.get('ensemble', {}).get('fcfw_weights', None)
    if ensemble_shots_all and ensemble_fcfw_weights is not None and len(ensemble_fcfw_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble metrics (FCFW weights)...")
        total_n = shots_budget
        ensemble_fcfw_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_fcfw_weights, total_n)
        if ensemble_fcfw_shots.size:
            ensemble_fcfw_sample_mets = compute_sample_metrics(dataset_train_samples, ensemble_fcfw_shots, sigma)
            ensemble_fcfw_coverage_mets = compute_coverage_validity(ensemble_fcfw_shots, dataset_train_samples)
            logger.info("Ensemble(FCFW) MMD: %.6f", ensemble_fcfw_sample_mets.get('mmd', float('nan')))
            logger.info("Ensemble(FCFW) TVD: %.4f", ensemble_fcfw_sample_mets.get('tvd', float('nan')))
            logger.info("Ensemble(FCFW) KL: %.4f", ensemble_fcfw_sample_mets.get('kl', float('nan')))
            logger.info("Ensemble(FCFW) Coverage: %.2f%%", ensemble_fcfw_coverage_mets.get('coverage', float('nan')))
    else:
        if ensemble_fcfw_weights is not None:
            logger.warning("Ensemble FCFW metrics skipped: inconsistent shot count or missing shots")
    
    # Optional standalone baseline metrics from inference output.
    standalone_sample_mets = None
    standalone_coverage_mets = None
    if standalone_shots is not None:
        logger.info("")
        logger.info("Computing standalone baseline metrics...")
        standalone_sample_mets = compute_sample_metrics(dataset_train_samples, standalone_shots, sigma)
        standalone_coverage_mets = compute_coverage_validity(standalone_shots, dataset_train_samples)

    # Save results
    results = {
        "artifact_path": str(artifact_path),
        "inference_dir": str(inference_dir),
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_qubits": n_qubits,
            "n_training_samples": int(len(dataset_train_samples)),
            "shots_for_metrics": int(shots_budget),
            "sigma": float(sigma) if isinstance(sigma, (int, float)) else str(sigma),
        },
        "model_results": model_metrics,
        "ensemble": {
            "n_models": len(ensemble_models),
            "weights": [float(w) for w in ensemble_weights],
            "weights_fcfw": [float(w) for w in (
                restored.get('ensemble', {}).get('fcfw_weights')
                if restored.get('ensemble', {}).get('fcfw_weights') is not None
                else []
            )],
        },
    }

    if standalone_sample_mets is not None:
        results.setdefault("baselines", {})
        results["baselines"]["standalone"] = {**standalone_sample_mets, **standalone_coverage_mets}
    
    if ensemble_sample_mets is not None:
        results["ensemble"]["metrics_standard"] = {**ensemble_sample_mets, **ensemble_coverage_mets}
    if ensemble_fcfw_sample_mets is not None:
        results["ensemble"]["metrics_fcfw"] = {**ensemble_fcfw_sample_mets, **ensemble_fcfw_coverage_mets}

    # Baseline: data-only ensemble metrics (sample directly from saved models if present)
    data_only_entry = restored.get('data_only', None)
    if data_only_entry is not None:
        results.setdefault('baselines', {})
        data_only_models = data_only_entry.get('models', [])
        data_only_weights = data_only_entry.get('weights', None)
        data_only_fcfw_weights = data_only_entry.get('fcfw_weights', None)

        if data_only_models and data_only_weights is not None:
            logger.info("")
            logger.info("Computing data-only baseline metrics (standard weights) by sampling from saved models...")
            sampled = sample_models_direct(data_only_models, data_only_weights, restored['circuit'], restored['wires'], shots_budget)
            if sampled.size:
                mets = compute_sample_metrics(dataset_train_samples, sampled, sigma)
                cov = compute_coverage_validity(sampled, dataset_train_samples)
                results['baselines']['data_only_metrics_standard'] = {**mets, **cov}

        if data_only_models and data_only_fcfw_weights is not None:
            logger.info("")
            logger.info("Computing data-only baseline metrics (FCFW weights) by sampling from saved models...")
            sampled_fcfw = sample_models_direct(data_only_models, data_only_fcfw_weights, restored['circuit'], restored['wires'], shots_budget)
            if sampled_fcfw.size:
                mets_f = compute_sample_metrics(dataset_train_samples, sampled_fcfw, sigma)
                cov_f = compute_coverage_validity(sampled_fcfw, dataset_train_samples)
                results['baselines']['data_only_metrics_fcfw'] = {**mets_f, **cov_f}
    
    output_file = inference_dir / f"backend_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("")
    logger.info("Metrics saved to: %s", output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics from backend inference shots."
    )
    parser.add_argument(
        "artifact_path",
        type=str,
        help="Path to the circuit_artifact.json file",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of generated samples to use for metrics (ensemble/sample budget). Default: 1024",
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        default=None,
        help="Specific inference run folder to analyze. Default: latest run under inference_results.",
    )
    
    args = parser.parse_args()
    main(args)
