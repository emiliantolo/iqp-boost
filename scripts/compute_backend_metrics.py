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
from src.utils import compute_kl_divergence, compute_mmd, compute_tvd, compute_jsd, compute_precision_recall_f1

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def compute_sample_metrics(ground_truth: np.ndarray, samples: np.ndarray, sigma: float) -> dict:
    """Compute standard metrics: MMD, TVD, KL, JSD, Precision/Recall/F1, plus sample statistics."""
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

    try:
        metrics['jsd'] = float(compute_jsd(ground_truth, samples))
    except Exception as e:
        logger.warning("Failed to compute JSD: %s", e)
        metrics['jsd'] = float('nan')

    try:
        sigma_f = sigma[0] if isinstance(sigma, (list, tuple, np.ndarray)) else sigma
        pr = compute_precision_recall_f1(ground_truth, samples, sigma_f)
        metrics.update(pr)
    except Exception as e:
        logger.warning("Failed to compute Precision/Recall/F1: %s", e)
        metrics['precision'] = float('nan')
        metrics['recall'] = float('nan')
        metrics['f_score'] = float('nan')
        metrics['support_match'] = float('nan')

    try:
        n_unique = len(np.unique(samples, axis=0))
        metrics['unique_fraction'] = float(n_unique / len(samples)) if len(samples) > 0 else 0.0
    except Exception as e:
        logger.warning("Failed to compute unique fraction: %s", e)
        metrics['unique_fraction'] = float('nan')

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


def plot_common_metrics(ground_truth: np.ndarray, methods: list[tuple[str, np.ndarray]], save_dir: Path) -> None:
    """Dataset-agnostic sample-based plots: probability spectrum + cumulative mass, all methods overlaid."""
    if plt is None or not methods:
        return
    n_qubits = ground_truth.shape[1]
    n_states = 2 ** n_qubits

    def _samples_to_sorted_probs(s):
        if s is None or len(s) == 0:
            return np.zeros(n_states, dtype=np.float64), np.zeros(n_states, dtype=np.float64)
        indices = np.sum(s.astype(int) * (2 ** np.arange(n_qubits)), axis=1)
        counts = np.bincount(indices, minlength=n_states)
        p = counts.astype(np.float64) / counts.sum()
        sorted_p = np.sort(p)[::-1]
        return sorted_p, np.cumsum(sorted_p)

    gt_sorted, gt_cum = _samples_to_sorted_probs(ground_truth)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)

    # Left: sorted probability spectrum (first 100 modes)
    max_modes = min(100, n_states)
    x = np.arange(1, max_modes + 1)
    axes[0].plot(x, gt_sorted[:max_modes], 'o-', markersize=3, color='gray', label='Ground Truth', alpha=0.5)
    axes[0].set_yscale('log')

    # Right: cumulative probability mass with log x-scale
    x_pct = np.linspace(1e-10, 1, n_states)
    axes[1].plot(x_pct, gt_cum, color='gray', label='Ground Truth', alpha=0.5)
    axes[1].plot([1e-10, 1], [1e-10, 1], 'k--', linewidth=0.5, alpha=0.4)
    axes[1].set_xscale('log')

    for i, (label, samples) in enumerate(methods):
        model_sorted, model_cum = _samples_to_sorted_probs(samples)
        axes[0].plot(x, model_sorted[:max_modes], 's-', markersize=3, color=colors[i], label=label, alpha=0.7)
        axes[1].plot(x_pct, model_cum, color=colors[i], label=label, alpha=0.7)

    axes[0].set_xlabel('Mode rank')
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Probability spectrum (top 100 modes)')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Fraction of modes (log)')
    axes[1].set_ylabel('Cumulative probability')
    axes[1].set_title('Cumulative probability mass')
    axes[1].legend(fontsize=8)

    path = save_dir / 'common_metrics.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info('  Saved common metrics plot to: %s', path)
    plt.close(fig)


def plot_correlation_heatmaps(ground_truth: np.ndarray, methods: list[tuple[str, np.ndarray]], save_dir: Path) -> None:
    """Pairwise bit correlation heatmaps: one row per method + ground truth, shared color scale."""
    if plt is None or not methods:
        return

    def _corr(samples):
        return np.corrcoef(samples.astype(np.float64).T)

    gt_corr = _corr(ground_truth)
    n_methods = len(methods)
    n_rows = n_methods + 1  # +1 for ground truth

    fig, axes = plt.subplots(1, n_rows, figsize=(4 * n_rows, 3.8), constrained_layout=True)

    vmin = min(gt_corr.min(), 0)
    vmax = max(gt_corr.max(), 1)
    for _, s in methods:
        c = _corr(s)
        vmin = min(vmin, c.min())
        vmax = max(vmax, c.max())

    im = axes[0].imshow(gt_corr, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='equal')
    axes[0].set_title('Ground Truth')
    axes[0].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, samples) in enumerate(methods):
        ax = axes[i + 1]
        c = _corr(samples)
        ax.imshow(c, cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='equal')
        ax.set_title(label)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
    cbar.set_label('Pearson correlation')

    path = save_dir / 'common_correlations.png'
    fig.savefig(path, dpi=160, bbox_inches='tight')
    fig.savefig(str(path).replace('.png', '.pdf'), bbox_inches='tight')
    logger.info('  Saved correlation heatmaps to: %s', path)
    plt.close(fig)


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


def find_inference_dirs(artifact_path: Path) -> list[Path]:
    """Return all inference run subdirectories for an artifact, newest first."""
    root = artifact_path.parent / "inference_results"
    if not root.exists():
        return []

    metadata_runs = sorted(
        [p.parent for p in root.glob("*/inference_metadata.json")],
        key=lambda p: p.name,
        reverse=True,
    )
    if metadata_runs:
        return metadata_runs

    subdirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    return subdirs


def _process_inference_run(restored: dict, inference_dir: Path, shots_budget: int) -> None:
    """Compute and save metrics for a single inference run directory."""
    artifact = restored["artifact"]
    dataset_train_samples = restored["dataset_train_samples"]
    sigma = restored["sigma"]
    ensemble_models = restored["ensemble"]["models"]
    ensemble_weights = restored["ensemble"]["weights"]
    n_qubits = artifact["circuit"]["n_visible_qubits"]

    if not inference_dir.exists():
        logger.error("Inference results directory not found: %s", inference_dir)
        return

    # Load shot entries
    shot_entries, inference_metadata = load_inference_shot_entries(inference_dir)
    if not shot_entries:
        logger.error("No shot files found in: %s", inference_dir)
        return

    logger.info("Found %d shot entries", len(shot_entries))

    if shots_budget is None:
        shots_budget = len(dataset_train_samples)
    logger.info("Using shots=%d (metric sample budget)", shots_budget)

    metrics_dir = inference_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Per-model metrics
    model_metrics = []
    ensemble_shots_all = []
    standalone_shots = None
    methods_to_plot = []

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
        
        if shots_budget and len(shots) > shots_budget:
            shots = subsample_shots(shots, shots_budget)
            logger.info("  Subsampled to %d shots", len(shots))
        
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
        
        logger.info("  MMD: %.6f, TVD: %.4f, KL: %.4f, JSD: %.4f, Coverage: %.2f%%",
                   sample_mets.get('mmd', float('nan')),
                   sample_mets.get('tvd', float('nan')),
                   sample_mets.get('kl', float('nan')),
                   sample_mets.get('jsd', float('nan')),
                   coverage_mets.get('coverage', float('nan')))
        
        if kind != "standalone":
            ensemble_shots_all.append(shots)

    # Ensemble metrics (standard weights)
    ensemble_sample_mets = ensemble_coverage_mets = None
    ensemble_fcfw_sample_mets = ensemble_fcfw_coverage_mets = None

    if ensemble_shots_all and len(ensemble_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble metrics (standard weights)...")
        ensemble_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_weights, shots_budget)
        if ensemble_shots.size:
            ensemble_sample_mets = compute_sample_metrics(dataset_train_samples, ensemble_shots, sigma)
            ensemble_coverage_mets = compute_coverage_validity(ensemble_shots, dataset_train_samples)
            log_ensemble_metrics("Ensemble", ensemble_sample_mets, ensemble_coverage_mets)
            methods_to_plot.append(("Ensemble", ensemble_shots))
    else:
        logger.warning("Ensemble metrics skipped: incomplete shot data for standard weights")

    # FCFW ensemble metrics
    ensemble_fcfw_weights = restored.get('ensemble', {}).get('fcfw_weights', None)
    if ensemble_shots_all and ensemble_fcfw_weights is not None and len(ensemble_fcfw_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble metrics (FCFW weights)...")
        ensemble_fcfw_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_fcfw_weights, shots_budget)
        if ensemble_fcfw_shots.size:
            ensemble_fcfw_sample_mets = compute_sample_metrics(dataset_train_samples, ensemble_fcfw_shots, sigma)
            ensemble_fcfw_coverage_mets = compute_coverage_validity(ensemble_fcfw_shots, dataset_train_samples)
            log_ensemble_metrics("Ensemble(FCFW)", ensemble_fcfw_sample_mets, ensemble_fcfw_coverage_mets)
            methods_to_plot.append(("Ensemble_FCFW", ensemble_fcfw_shots))
    else:
        if ensemble_fcfw_weights is not None:
            logger.warning("Ensemble FCFW metrics skipped: inconsistent shot count or missing shots")

    # Standalone baseline
    standalone_sample_mets = standalone_coverage_mets = None
    if standalone_shots is not None:
        logger.info("")
        logger.info("Computing standalone baseline metrics...")
        standalone_sample_mets = compute_sample_metrics(dataset_train_samples, standalone_shots, sigma)
        standalone_coverage_mets = compute_coverage_validity(standalone_shots, dataset_train_samples)
        methods_to_plot.append(("Standalone", standalone_shots))

    # Build results
    results = {
        "artifact_path": str(restored.get('_artifact_path', '')),
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

    # Data-only baselines
    data_only_entry = restored.get('data_only', None)
    if data_only_entry is not None:
        results.setdefault('baselines', {})
        do_models = data_only_entry.get('models', [])
        do_weights = data_only_entry.get('weights', None)
        do_fcfw = data_only_entry.get('fcfw_weights', None)

        if do_models and do_weights is not None:
            logger.info("")
            logger.info("Computing data-only metrics (standard weights)...")
            sampled = sample_models_direct(do_models, do_weights, restored['circuit'], restored['wires'], shots_budget)
            if sampled.size:
                mets = compute_sample_metrics(dataset_train_samples, sampled, sigma)
                cov = compute_coverage_validity(sampled, dataset_train_samples)
                results['baselines']['data_only_metrics_standard'] = {**mets, **cov}
                methods_to_plot.append(("DataOnly", sampled))

        if do_models and do_fcfw is not None:
            logger.info("")
            logger.info("Computing data-only metrics (FCFW weights)...")
            sampled = sample_models_direct(do_models, do_fcfw, restored['circuit'], restored['wires'], shots_budget)
            if sampled.size:
                mets = compute_sample_metrics(dataset_train_samples, sampled, sigma)
                cov = compute_coverage_validity(sampled, dataset_train_samples)
                results['baselines']['data_only_metrics_fcfw'] = {**mets, **cov}
                methods_to_plot.append(("DataOnly_FCFW", sampled))

    # Plot all methods in canonical order (skip missing)
    _order = ["Standalone", "DataOnly", "DataOnly_FCFW", "Ensemble", "Ensemble_FCFW"]
    _method_map = dict(methods_to_plot)
    methods_ordered = [(k, _method_map[k]) for k in _order if k in _method_map]
    plot_common_metrics(dataset_train_samples, methods_ordered, metrics_dir)
    plot_correlation_heatmaps(dataset_train_samples, methods_ordered, metrics_dir)
    methods_to_plot.clear()

    output_file = metrics_dir / "backend_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Metrics saved to: %s", output_file)


def log_ensemble_metrics(label: str, sample_mets: dict, coverage_mets: dict) -> None:
    """Log ensemble-level metrics."""
    logger.info("%s MMD: %.6f", label, sample_mets.get('mmd', float('nan')))
    logger.info("%s TVD: %.4f", label, sample_mets.get('tvd', float('nan')))
    logger.info("%s KL: %.4f", label, sample_mets.get('kl', float('nan')))
    logger.info("%s JSD: %.4f", label, sample_mets.get('jsd', float('nan')))
    logger.info("%s Precision: %.4f, Recall: %.4f, F1: %.4f",
                label, sample_mets.get('precision', float('nan')),
                sample_mets.get('recall', float('nan')),
                sample_mets.get('f_score', float('nan')))
    logger.info("%s Coverage: %.2f%%", label, coverage_mets.get('coverage', float('nan')))


def process_artifact(artifact_path: Path, shots_budget: int, inference_dir_override: str | None = None) -> None:
    """Compute metrics for a single circuit artifact.

    When *inference_dir_override* is given, only that run is processed.
    Otherwise, all inference runs under ``inference_results/`` are processed.
    """
    if not artifact_path.exists():
        logger.error("Artifact file not found: %s", artifact_path)
        return

    logger.info("Loading circuit artifact: %s", artifact_path)
    try:
        restored = restore_circuit_artifact(artifact_path)
    except Exception as e:
        logger.error("Failed to restore artifact: %s", e)
        return

    restored.setdefault('_artifact_path', str(artifact_path))

    dataset_train_samples = restored["dataset_train_samples"]
    if dataset_train_samples is None:
        logger.error("Training dataset samples not available in artifact")
        return

    ensemble_models = restored["ensemble"]["models"]
    ensemble_weights = restored["ensemble"]["weights"]
    sigma = restored["sigma"]
    n_qubits = restored["artifact"]["circuit"]["n_visible_qubits"]

    logger.info("Dataset: %d training samples, %d qubits", len(dataset_train_samples), n_qubits)
    logger.info("Sigma: %s", sigma)
    logger.info("Ensemble: %d models, weights sum=%.4f", len(ensemble_models), ensemble_weights.sum())

    if inference_dir_override is not None:
        dirs = [resolve_inference_dir(artifact_path, inference_dir_override)]
    else:
        dirs = find_inference_dirs(artifact_path)
        if not dirs:
            logger.error("No inference runs found under %s", artifact_path.parent / "inference_results")
            logger.info("Run: python scripts/run_backend_inference.py %s", artifact_path)
            return
        logger.info("Found %d inference run(s)", len(dirs))

    for inf_dir in dirs:
        logger.info("")
        logger.info("--- Inference run: %s ---", inf_dir.name)
        _process_inference_run(restored, inf_dir, shots_budget)


def build_recap_table(root: Path) -> str:
    """Aggregate metrics from all artifacts under *root* into a Markdown table."""
    headers = ['Dataset', 'Method', 'MMD', 'TVD', 'KL', 'JSD', 'Precision', 'Recall', 'F1', 'Coverage (%)']
    rows = []
    artifacts = sorted(root.rglob('circuit_artifact.json'))
    for ap in artifacts:
        dataset_name = ap.parent.name
        # Pick the latest inference run
        inf_dirs = find_inference_dirs(ap)
        if not inf_dirs:
            continue
        metrics_file = inf_dirs[0] / "metrics" / "backend_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            data = json.load(f)
        dataset_rows = []
        for method_key, method_label in [('standalone', 'Standalone'),
                                          ('data_only_metrics_standard', 'DataOnly'),
                                          ('data_only_metrics_fcfw', 'DataOnly_FCFW'),
                                          ('metrics_standard', 'Ensemble'),
                                          ('metrics_fcfw', 'Ensemble_FCFW')]:
            baselines = data.get('baselines', {})
            ens = data.get('ensemble', {})
            m = baselines.get(method_key, {}) if method_key in baselines else ens.get(method_key, {})
            if not m:
                continue
            dataset_rows.append([dataset_name, method_label,
                                 f"{m.get('mmd', float('nan')):.4f}",
                                 f"{m.get('tvd', float('nan')):.4f}",
                                 f"{m.get('kl', float('nan')):.4f}",
                                 f"{m.get('jsd', float('nan')):.4f}",
                                 f"{m.get('precision', float('nan')):.3f}",
                                 f"{m.get('recall', float('nan')):.3f}",
                                 f"{m.get('f_score', float('nan')):.3f}",
                                 f"{m.get('coverage', float('nan')):.1f}"])
        if dataset_rows:
            if rows:
                rows.append(None)  # separator
            rows.extend(dataset_rows)

    if not rows:
        return ""

    valid_rows = [r for r in rows if r is not None]
    col_widths = [max([len(str(r[i])) for r in valid_rows] + [len(headers[i])]) for i in range(len(headers))]
    sep = '| ' + ' | '.join(h.ljust(w) for h, w in zip(headers, col_widths)) + ' |'
    line = '|-' + '-|-'.join('-' * w for w in col_widths) + '-|'
    sep_line = '| ' + ' | '.join(' ' * w for w in col_widths) + ' |'

    parts = [sep, line]
    for r in rows:
        if r is None:
            parts.append(sep_line)
        else:
            parts.append('| ' + ' | '.join(str(r[i]).ljust(w) for i, w in enumerate(col_widths)) + ' |')
    return '\n'.join(parts)


def main(args: argparse.Namespace):
    """Compute metrics from backend inference shots.

    Single-artifact mode: pass a path to a ``circuit_artifact.json`` file.
    Batch mode: pass a parent directory; all ``circuit_artifact.json`` files
    found recursively under it are processed.
    """
    root = Path(args.artifact_path)

    if root.is_file():
        # Single mode (original behavior)
        process_artifact(root, args.shots, inference_dir_override=args.inference_dir)
    else:
        # Batch mode
        artifacts = sorted(root.rglob('circuit_artifact.json'))
        if not artifacts:
            logger.error("No circuit_artifact.json found under %s", root)
            return
        logger.info("Batch mode: found %d artifact(s) under %s", len(artifacts), root)
        for ap in artifacts:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Processing: %s", ap)
            logger.info("=" * 60)
            process_artifact(ap, args.shots, inference_dir_override=None)

        # Recap table across all datasets
        table = build_recap_table(root)
        if table:
            table_path = root / 'metrics_recap.md'
            with open(table_path, 'w') as f:
                f.write('# Metrics Recap\n\n')
                f.write(table)
                f.write('\n')
            logger.info("")
            logger.info("Recap table saved to: %s", table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute metrics from backend inference shots."
    )
    parser.add_argument(
        "artifact_path",
        type=str,
        help="Path to a circuit_artifact.json file (single mode), or a parent directory"
             " whose subdirectories are searched recursively for circuit_artifact.json files"
             " (batch mode).",
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
