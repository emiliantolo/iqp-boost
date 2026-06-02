"""Compute dataset-specific metrics from backend inference shots.

Supported datasets: bas, hamming_balls, hopfield.
Reads saved dataset metadata (dataset.json + dataset.npz) alongside the circuit artifact.
Produces step-wise metrics over boosting models and final ensemble metrics.
"""

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
from src.utils import compute_hamming_matrix
from src.benchmark_metrics import compute_hamming_balls_recall_distance

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---------------------------------------------------------------------------
# Dataset class map (for reconstruction from saved params)
# ---------------------------------------------------------------------------
DATASET_CLASSES: dict[str, type] = {}


def _register_datasets():
    try:
        from src.datasets.bas import BarsAndStripesDataset
        DATASET_CLASSES['bas'] = BarsAndStripesDataset
        DATASET_CLASSES['noisy_bas'] = BarsAndStripesDataset
    except ImportError:
        pass
    try:
        from src.datasets.hopfield import HopfieldDataset
        DATASET_CLASSES['hopfield'] = HopfieldDataset
    except ImportError:
        pass
    try:
        from src.datasets.hamming_balls import HammingBallsDataset
        DATASET_CLASSES['hamming_balls'] = HammingBallsDataset
    except ImportError:
        pass
    try:
        from src.datasets.blobs import BlobsDataset
        DATASET_CLASSES['blobs'] = BlobsDataset
    except ImportError:
        pass
    try:
        from src.datasets.fashion_mnist import FashionMNISTDownscaledDataset
        DATASET_CLASSES['fashion_mnist'] = FashionMNISTDownscaledDataset
    except ImportError:
        pass
    try:
        from src.datasets.mps import MPSDataset
        DATASET_CLASSES['mps'] = MPSDataset
    except ImportError:
        pass


_register_datasets()

# ---------------------------------------------------------------------------
# Shared helper functions (mirrored from compute_backend_metrics.py)
# ---------------------------------------------------------------------------


def subsample_shots(shots: np.ndarray, n_samples: int) -> np.ndarray:
    if len(shots) <= n_samples:
        return shots
    indices = np.random.choice(len(shots), size=n_samples, replace=False)
    return shots[indices]


def combine_shots_by_weights(shot_pools: list[np.ndarray], weights: np.ndarray, total_n: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones(len(shot_pools), dtype=float)
    weights = weights / weights.sum()
    counts = np.floor(weights * total_n).astype(int)
    remainder = total_n - counts.sum()
    if remainder > 0:
        idxs = np.argsort(-weights)
        for i in range(remainder):
            counts[idxs[i % len(idxs)]] += 1
    selected = []
    for pool, c in zip(shot_pools, counts):
        if c <= 0 or len(pool) == 0:
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
    shot_path = Path(shot_ref)
    if shot_path.is_absolute():
        return shot_path
    if shot_path.parts and shot_path.parts[0] == inference_dir.name:
        return inference_dir.parent / shot_path
    if len(shot_path.parts) >= 2 and shot_path.parts[0] == inference_dir.parent.name and shot_path.parts[1] == inference_dir.name:
        return inference_dir.parent / shot_path
    return inference_dir / shot_path


def resolve_inference_dir(artifact_path: Path, inference_dir_arg: str | None) -> Path:
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


# ---------------------------------------------------------------------------
# Dataset sidecar loading
# ---------------------------------------------------------------------------


def load_dataset_sidecar(artifact_dir: Path) -> tuple[str, dict, dict]:
    """Load dataset metadata and arrays from the artifact's parent directory.

    Returns:
        (dataset_key, dataset_params, arrays_dict)
    """
    meta_file = artifact_dir / "dataset.json"
    if not meta_file.exists():
        logger.error("dataset.json not found in %s", artifact_dir)
        return "", {}, {}

    with open(meta_file) as f:
        meta = json.load(f)

    arrays: dict = {}
    npz_file = artifact_dir / "dataset.npz"
    if npz_file.exists():
        try:
            with np.load(npz_file, allow_pickle=False) as d:
                for key in d:
                    arrays[key] = d[key].copy()
        except Exception as e:
            logger.warning("Failed to load dataset.npz: %s", e)

    return meta.get("dataset_key", ""), meta.get("dataset_params", {}), arrays


def _filter_constructor_kwargs(cls: type, kwargs: dict) -> dict:
    """Keep only kwargs that match the class's __init__ signature."""
    import inspect
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {'self'}
    return {k: v for k, v in kwargs.items() if k in valid}


def reconstruct_dataset(dataset_key: str, dataset_params: dict):
    """Reconstruct a dataset object from its saved params."""
    cls = DATASET_CLASSES.get(dataset_key)
    if cls is None:
        return None
    params = dict(dataset_params)
    # Normalize BAS params: `dims` -> `height`/`width`
    if dataset_key in ("bas", "noisy_bas") and "dims" in params:
        dims = params.pop("dims")
        params.setdefault("height", int(dims[0]))
        params.setdefault("width", int(dims[1]))
    params = _filter_constructor_kwargs(cls, params)
    try:
        return cls(**params)
    except Exception as e:
        logger.warning("Failed to reconstruct dataset %s: %s", dataset_key, e)
        return None


# ---------------------------------------------------------------------------
# Dataset-specific metric functions
# ---------------------------------------------------------------------------


def compute_bas_metrics(ds, samples: np.ndarray) -> dict:
    """BAS validity and coverage (structural, not sample-overlap)."""
    if len(samples) == 0:
        return {"validity": 0.0, "coverage": 0.0}
    return {
        "validity": float(ds.validity_rate(samples) * 100.0),
        "coverage": float(ds.coverage_rate(None, samples) * 100.0),
    }


def hopfield_energy(samples: np.ndarray, J: np.ndarray) -> np.ndarray:
    """E = -0.5 * s^T J s, s = 1 - 2*x maps {0,1} -> {+1,-1}."""
    spins = 1.0 - 2.0 * samples.astype(np.float64)
    return -0.5 * np.einsum("ni,ij,nj->n", spins, J, spins)


def compute_hopfield_metrics(samples: np.ndarray, J: np.ndarray, patterns: np.ndarray,
                             radius: int = 2) -> dict:
    """Hopfield energy stats and pattern proximity/coverage."""
    if len(samples) == 0:
        return {"energy_mean": float("nan"), "energy_std": float("nan"),
                "pattern_proximity": float("nan"), "pattern_coverage": 0.0}

    energies = hopfield_energy(samples, J)
    # Pattern proximity: Hamming distance to nearest pattern
    # patterns are in {-1,+1}, convert to {0,1}
    p_bits = ((1.0 - patterns) / 2.0).astype(np.int8)
    H = compute_hamming_matrix(samples, p_bits)
    min_dists = H.min(axis=1)
    covered = np.any(H <= radius, axis=0)

    return {
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "pattern_proximity": float(min_dists.mean()),
        "pattern_coverage": float(covered.mean()),
    }


def _hamming_balls_analytical_recall(ds) -> float | None:
    """Expected recall distance under the true Hamming Balls mixture distribution.

    Uses the exact probabilities (*ds.probs*) when available (n ≤ 20-25),
    otherwise falls back to *None*.
    """
    probs = getattr(ds, "probs", None)
    centers = getattr(ds, "centers", None)
    nq = getattr(ds, "n_qubits", 0)
    if probs is None or centers is None or nq == 0:
        return None
    if 2 ** nq > 2 ** 22:
        return None
    indices = np.arange(2 ** nq, dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(nq, dtype=np.int64)) & 1).astype(np.int8)
    H = (bits[:, None, :] != centers[None, :, :]).sum(axis=2)
    min_dists = H.min(axis=1)
    return float((probs * min_dists).sum())


def compute_hamming_balls_metrics(centers: np.ndarray, samples: np.ndarray,
                                  n_qubits: int = 0, radius_fraction: float = 0.15) -> dict:
    """Recall distance: mean min Hamming distance to nearest center (threshold-free)."""
    if len(samples) == 0 or len(centers) == 0:
        return {"recall_distance": float("nan")}
    recall_dist = compute_hamming_balls_recall_distance(centers, samples)
    return {"recall_distance": float(recall_dist)}


# ---------------------------------------------------------------------------
# Dataset-specific plot functions
# ---------------------------------------------------------------------------


def plot_dataset_metrics(ground_truth: np.ndarray | None,
                         step_metrics: list[dict] | None,
                         methods: list[tuple[str, np.ndarray]],
                         dataset_key: str,
                         ds: object,
                         save_dir: Path) -> None:
    """Generate dataset-specific plots: step-wise progression + distribution panels."""
    if plt is None:
        return

    _figures = _build_dataset_figures(ground_truth, step_metrics, methods, dataset_key, ds)

    for name, fig in _figures:
        stem = f"dataset_{dataset_key}_{name}"
        path = save_dir / f"{stem}.png"
        fig.savefig(path, dpi=160)
        fig.savefig(save_dir / f"{stem}.pdf")
        logger.info("  Saved %s plot to: %s", name, path)
        plt.close(fig)


def _build_dataset_figures(ground_truth, step_metrics, methods, dataset_key, ds):
    """Return list of (name, figure) tuples for the given dataset."""
    figs = []

    if dataset_key in ("bas", "noisy_bas"):
        fig = _plot_bas_stepwise(step_metrics)
        if fig:
            figs.append(("stepwise", fig))
        return figs

    if dataset_key == "hopfield":
        fig = _plot_hopfield_stepwise(step_metrics)
        if fig:
            figs.append(("stepwise", fig))
        # Energy histogram (distribution per method)
        fig = _plot_hopfield_energy_hist(ground_truth, methods, ds)
        if fig:
            figs.append(("energy_hist", fig))
        return figs

    if dataset_key == "hamming_balls":
        centers = getattr(ds, "centers", None)
        nq = getattr(ds, "n_qubits", centers.shape[1] if centers is not None else 0)
        analytical = _hamming_balls_analytical_recall(ds)
        fig = _plot_hamming_balls_stepwise(step_metrics, analytical=analytical)
        if fig:
            figs.append(("stepwise", fig))
        fig = _plot_hamming_balls_distance_dist(ground_truth, methods, centers,
                                                analytical=analytical)
        if fig:
            figs.append(("distance_dist", fig))
        return figs

    return figs


def _plot_bas_stepwise(step_metrics: list[dict] | None) -> object | None:
    if not step_metrics or plt is None:
        return None
    steps = list(range(len(step_metrics)))
    validities = [m.get("validity", float("nan")) for m in step_metrics]
    coverages = [m.get("coverage", float("nan")) for m in step_metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax1.plot(steps, validities, "s-", color="tab:blue", label="Validity (%)")
    ax1.set_xlabel("Boosting step")
    ax1.set_ylabel("Validity (%)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(steps, coverages, "o-", color="tab:orange", label="Coverage (%)")
    ax2.set_ylabel("Coverage (%)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.set_title("BAS — incremental ensemble validity & coverage")
    return fig


def _plot_hopfield_stepwise(step_metrics: list[dict] | None) -> object | None:
    if not step_metrics or plt is None:
        return None
    steps = list(range(len(step_metrics)))
    prox = [m.get("pattern_proximity", float("nan")) for m in step_metrics]
    cov = [m.get("pattern_coverage", float("nan")) for m in step_metrics]

    fig, ax1 = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax1.plot(steps, prox, "s-", color="tab:blue", label="Pattern proximity (Hamming)")
    ax1.set_xlabel("Boosting step")
    ax1.set_ylabel("Mean Hamming distance", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(steps, cov, "o-", color="tab:orange", label="Pattern coverage")
    ax2.set_ylabel("Coverage", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax1.set_title("Hopfield — incremental ensemble proximity & coverage")
    return fig


def _plot_hopfield_energy_hist(ground_truth, methods, ds) -> object | None:
    if plt is None or not methods:
        return None
    J = getattr(ds, "J", None)
    if J is None:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    # Ground truth energy distribution
    if ground_truth is not None and len(ground_truth) > 0:
        gt_e = hopfield_energy(ground_truth, J)
        ax.hist(gt_e, bins=50, density=True, alpha=0.5, color="gray", label="Ground Truth")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, samples) in enumerate(methods):
        if len(samples) == 0:
            continue
        e = hopfield_energy(samples, J)
        ax.hist(e, bins=50, density=True, alpha=0.6, color=colors[i], label=label)

    ax.set_xlabel("Energy E = -½ sᵀJ s")
    ax.set_ylabel("Density")
    ax.set_title("Hopfield energy distribution")
    ax.legend(fontsize=8)
    return fig


def _plot_hamming_balls_stepwise(step_metrics: list[dict] | None,
                                 analytical: float | None = None) -> object | None:
    if not step_metrics or plt is None:
        return None
    steps = list(range(len(step_metrics)))
    distances = [m.get("recall_distance", float("nan")) for m in step_metrics]

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.plot(steps, distances, "o-", color="tab:green", label="Incremental ensemble")
    if analytical is not None:
        ax.axhline(analytical, color="black", ls=":", lw=1.2, label=f"Analytical = {analytical:.3f}")
    ax.set_xlabel("Boosting step")
    ax.set_ylabel("Mean Hamming distance to nearest center")
    ax.set_title("Hamming Balls — incremental ensemble recall distance")
    ax.legend(fontsize=8)
    return fig


def _plot_hamming_balls_distance_dist(ground_truth, methods, centers,
                                      analytical: float | None = None) -> object | None:
    """Side-by-side panels: one subplot per method, histogram + line + analytical value."""
    if plt is None or len(centers) == 0:
        return None
    if not methods and ground_truth is None:
        return None

    panels = []
    if ground_truth is not None and len(ground_truth) > 0:
        gt_dists = compute_hamming_matrix(ground_truth, centers).min(axis=1)
        panels.append(("Ground Truth", gt_dists, "gray"))
    for i, (label, samples) in enumerate(methods):
        if len(samples) == 0:
            continue
        dists = compute_hamming_matrix(samples, centers).min(axis=1)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        panels.append((label, dists, colors[i % len(colors)]))

    n_panels = len(panels)
    n_cols = 2
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.5 * n_rows),
                             constrained_layout=True, squeeze=False, sharex=True)

    for idx, (label, dists, color) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        counts, bins, _ = ax.hist(dists, bins=min(30, len(set(dists))),
                                  density=True, alpha=0.5, color=color, edgecolor=color)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax.plot(bin_centers, counts, color=color, lw=1.2, marker=".", markersize=3)

        method_mean = float(dists.mean())
        if label == "Ground Truth":
            ax.axvline(method_mean, color="gray", ls="--", lw=1.2, label=f"Mean = {method_mean:.3f}")
        else:
            ax.axvline(method_mean, color=color, ls="--", lw=1.0, alpha=0.6, label=f"Mean = {method_mean:.3f}")
        if analytical is not None:
            ax.axvline(analytical, color="black", ls=":", lw=1.2, label=f"Analytical = {analytical:.3f}")

        ax.set_xlabel("Hamming distance")
        ax.set_ylabel("Density")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7)

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    return fig


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------


def _compute_dataset_metrics_fn(dataset_key: str, ds, samples: np.ndarray) -> dict:
    """Dispatch to the correct dataset-specific metric function."""
    if dataset_key in ("bas", "noisy_bas"):
        return compute_bas_metrics(ds, samples)
    if dataset_key == "hopfield":
        J = getattr(ds, "J", None)
        patterns = getattr(ds, "patterns", None)
        if J is None or patterns is None:
            return {}
        return compute_hopfield_metrics(samples, J, patterns)
    if dataset_key == "hamming_balls":
        centers = getattr(ds, "centers", None)
        if centers is None:
            return {}
        nq = getattr(ds, "n_qubits", centers.shape[1])
        return compute_hamming_balls_metrics(centers, samples, nq)
    return {}


def log_dataset_metrics(label: str, m: dict) -> None:
    items = " | ".join(f"{k}: {v:.4f}" for k, v in m.items() if isinstance(v, float))
    if items:
        logger.info("%s: %s", label, items)


def _process_inference_run(restored: dict, inference_dir: Path, shots_budget: int,
                           dataset_key: str, ds) -> None:
    """Compute and save dataset-specific metrics for a single inference run."""
    ensemble_weights = restored["ensemble"]["weights"]
    n_qubits = restored["artifact"]["circuit"]["n_visible_qubits"]
    dataset_train_samples = restored.get("dataset_train_samples")

    if not inference_dir.exists():
        logger.error("Inference results directory not found: %s", inference_dir)
        return

    shot_entries, _ = load_inference_shot_entries(inference_dir)
    if not shot_entries:
        logger.error("No shot files found in: %s", inference_dir)
        return

    logger.info("Found %d shot entries", len(shot_entries))
    if shots_budget is None and dataset_train_samples is not None:
        shots_budget = len(dataset_train_samples)
    logger.info("Using shots=%d (metric sample budget)", shots_budget)

    metrics_dir = inference_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Per-model loop — load shots, defer metric computation to incremental ensemble
    ensemble_shots_all: list[np.ndarray] = []
    standalone_shots: np.ndarray | None = None
    methods_to_plot: list[tuple[str, np.ndarray]] = []

    ground_truth = dataset_train_samples

    for entry_idx, entry in enumerate(shot_entries):
        shot_file = resolve_shot_path(inference_dir, entry.get("shots_file", ""))
        kind = entry.get("kind", "ensemble")

        if not shot_file.exists():
            logger.warning("Skipping missing shot file: %s", shot_file)
            continue

        logger.info("[%d/%d] Loading: %s", entry_idx + 1, len(shot_entries), shot_file.name)
        try:
            shots = np.load(shot_file)
        except Exception as e:
            logger.error("Failed to load shots: %s", e)
            continue

        if shots_budget and len(shots) > shots_budget:
            shots = subsample_shots(shots, shots_budget)

        if kind == "standalone":
            standalone_shots = shots
        else:
            ensemble_shots_all.append(shots)

    # Incremental ensemble metrics: at each step k, combine models 0..k
    step_metrics: list[dict] = []
    n_ensemble = len(ensemble_shots_all)
    if n_ensemble > 0 and len(ensemble_weights) >= n_ensemble:
        logger.info("")
        logger.info("Computing incremental ensemble metrics over %d model(s)...", n_ensemble)
        for k in range(n_ensemble):
            prefix_weights = np.asarray(ensemble_weights[:k + 1], dtype=float)
            prefix_weights = prefix_weights / prefix_weights.sum()
            combined = combine_shots_by_weights(ensemble_shots_all[:k + 1], prefix_weights, shots_budget)
            m = _compute_dataset_metrics_fn(dataset_key, ds, combined)
            if m:
                step_metrics.append(m)
                log_dataset_metrics(f"  Step {k + 1}/{n_ensemble}", m)

    # Ensemble metrics (standard weights)
    ensemble_sample_mets: dict | None = None
    ensemble_fcfw_sample_mets: dict | None = None
    ensemble_sample_mets_fcfw = ensemble_sample_mets_std = None

    if ensemble_shots_all and len(ensemble_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble dataset metrics (standard weights)...")
        ensemble_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_weights, shots_budget)
        if ensemble_shots.size:
            ensemble_sample_mets_std = _compute_dataset_metrics_fn(dataset_key, ds, ensemble_shots)
            log_dataset_metrics("Ensemble", ensemble_sample_mets_std)
            methods_to_plot.append(("Ensemble", ensemble_shots))
    else:
        logger.warning("Ensemble dataset metrics skipped: incomplete shot data")

    # FCFW ensemble metrics
    ensemble_fcfw_weights = restored.get("ensemble", {}).get("fcfw_weights", None)
    if ensemble_shots_all and ensemble_fcfw_weights is not None and len(ensemble_fcfw_weights) == len(ensemble_shots_all):
        logger.info("")
        logger.info("Computing ensemble dataset metrics (FCFW weights)...")
        ensemble_shots = combine_shots_by_weights(ensemble_shots_all, ensemble_fcfw_weights, shots_budget)
        if ensemble_shots.size:
            ensemble_sample_mets_fcfw = _compute_dataset_metrics_fn(dataset_key, ds, ensemble_shots)
            log_dataset_metrics("Ensemble(FCFW)", ensemble_sample_mets_fcfw)
            methods_to_plot.append(("Ensemble_FCFW", ensemble_shots))
    else:
        if ensemble_fcfw_weights is not None:
            logger.warning("Ensemble FCFW dataset metrics skipped: inconsistent data")

    # Standalone baseline
    standalone_sample_mets: dict | None = None
    if standalone_shots is not None:
        logger.info("")
        logger.info("Computing standalone dataset metrics...")
        standalone_sample_mets = _compute_dataset_metrics_fn(dataset_key, ds, standalone_shots)
        log_dataset_metrics("Standalone", standalone_sample_mets)
        methods_to_plot.append(("Standalone", standalone_shots))

    # Data-only baselines
    data_only_entry = restored.get("data_only", None)
    if data_only_entry is not None:
        do_models = data_only_entry.get("models", [])
        do_weights = data_only_entry.get("weights", None)
        do_fcfw = data_only_entry.get("fcfw_weights", None)

        if do_models and do_weights is not None:
            logger.info("Computing data-only dataset metrics (standard weights)...")
            sampled = sample_models_direct(do_models, do_weights, restored["circuit"], restored["wires"], shots_budget)
            if sampled.size:
                m = _compute_dataset_metrics_fn(dataset_key, ds, sampled)
                log_dataset_metrics("DataOnly", m)
                methods_to_plot.append(("DataOnly", sampled))

        if do_models and do_fcfw is not None:
            logger.info("Computing data-only dataset metrics (FCFW weights)...")
            sampled = sample_models_direct(do_models, do_fcfw, restored["circuit"], restored["wires"], shots_budget)
            if sampled.size:
                m = _compute_dataset_metrics_fn(dataset_key, ds, sampled)
                log_dataset_metrics("DataOnly(FCFW)", m)
                methods_to_plot.append(("DataOnly_FCFW", sampled))

    # Build results
    results: dict = {
        "artifact_path": str(restored.get("_artifact_path", "")),
        "inference_dir": str(inference_dir),
        "dataset_key": dataset_key,
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_qubits": n_qubits,
            "shots_for_metrics": int(shots_budget) if shots_budget else 0,
        },
        "step_metrics": step_metrics,
    }

    # Reorder methods for display
    _order = ["Standalone", "DataOnly", "DataOnly_FCFW", "Ensemble", "Ensemble_FCFW"]
    _method_map = dict(methods_to_plot)
    methods_ordered = [(k, _method_map[k]) for k in _order if k in _method_map]

    # Ensemble metrics in results
    ensemble_block: dict = {}
    if ensemble_sample_mets_std:
        ensemble_block["metrics_standard"] = ensemble_sample_mets_std
    if ensemble_sample_mets_fcfw:
        ensemble_block["metrics_fcfw"] = ensemble_sample_mets_fcfw
    if ensemble_block:
        results["ensemble"] = ensemble_block

    baselines_block: dict = {}
    if standalone_sample_mets:
        baselines_block["standalone"] = standalone_sample_mets
    if data_only_entry:
        for key in ("data_only_metrics_standard", "data_only_metrics_fcfw"):
            if key in results.get("baselines", {}):
                baselines_block[key] = results["baselines"][key]
    if baselines_block:
        results["baselines"] = baselines_block

    # Plots
    plot_dataset_metrics(ground_truth, step_metrics, methods_ordered, dataset_key, ds, metrics_dir)

    # Save JSON
    output_file = metrics_dir / "dataset_metrics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Dataset metrics saved to: %s", output_file)


def process_artifact(artifact_path: Path, shots_budget: int,
                     inference_dir_override: str | None = None) -> None:
    """Compute dataset-specific metrics for a single circuit artifact."""
    if not artifact_path.exists():
        logger.error("Artifact file not found: %s", artifact_path)
        return

    # Load dataset sidecar
    artifact_dir = artifact_path.parent
    dataset_key, dataset_params, arrays = load_dataset_sidecar(artifact_dir)

    if not dataset_key:
        logger.error("No dataset.json found alongside artifact; cannot compute dataset metrics.")
        return

    ds = reconstruct_dataset(dataset_key, dataset_params)
    if ds is None:
        logger.error("Cannot reconstruct dataset %s from params %s", dataset_key, dataset_params)
        return

    # Restore arrays from NPZ onto ds if present
    if hasattr(ds, "centers") and "centers" in arrays:
        ds.centers = arrays["centers"]
    if hasattr(ds, "J") and "J" in arrays:
        ds.J = arrays["J"]
    if hasattr(ds, "patterns") and "patterns" in arrays:
        ds.patterns = arrays["patterns"]
    if "probs" in arrays:
        ds.probs = arrays["probs"]

    logger.info("Dataset: %s, %d qubits", dataset_key,
                dataset_params.get("n_qubits", dataset_params.get("rows", 0)))

    # Load circuit artifact
    logger.info("Loading circuit artifact: %s", artifact_path)
    try:
        restored = restore_circuit_artifact(artifact_path)
    except Exception as e:
        logger.error("Failed to restore artifact: %s", e)
        return

    restored.setdefault("_artifact_path", str(artifact_path))

    if restored.get("dataset_train_samples") is None:
        logger.error("Training dataset samples not available in artifact")
        return

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
        _process_inference_run(restored, inf_dir, shots_budget, dataset_key, ds)


# ---------------------------------------------------------------------------
# Recap table
# ---------------------------------------------------------------------------


def build_recap_table(root: Path) -> str:
    """Aggregate dataset metrics from all artifacts under *root* into a Markdown table."""
    dataset_key_map = {}
    for ap in sorted(root.rglob("circuit_artifact.json")):
        dk, _, _ = load_dataset_sidecar(ap.parent)
        if dk:
            dataset_key_map[ap.parent] = dk

    # Determine metric columns from the first available dataset_key
    metric_cols_by_key = {
        "bas": ["validity", "coverage"],
        "noisy_bas": ["validity", "coverage"],
        "hopfield": ["energy_mean", "pattern_proximity", "pattern_coverage"],
        "hamming_balls": ["recall_distance"],
    }

    headers = ["Dataset", "Method"]
    # We'll figure out columns dynamically — use a union of all possible metrics
    all_metric_cols = sorted({c for cols in metric_cols_by_key.values() for c in cols})
    headers.extend(all_metric_cols)

    rows = []
    artifacts = sorted(root.rglob("circuit_artifact.json"))
    for ap in artifacts:
        dataset_name = ap.parent.name
        dk = dataset_key_map.get(ap.parent, "")
        cols = metric_cols_by_key.get(dk, [])

        inf_dirs = find_inference_dirs(ap)
        if not inf_dirs:
            continue
        metrics_file = inf_dirs[0] / "metrics" / "dataset_metrics.json"
        if not metrics_file.exists():
            continue
        with open(metrics_file) as f:
            data = json.load(f)

        dataset_rows = []
        for method_key, method_label in [("standalone", "Standalone"),
                                          ("data_only_metrics_standard", "DataOnly"),
                                          ("data_only_metrics_fcfw", "DataOnly_FCFW"),
                                          ("metrics_standard", "Ensemble"),
                                          ("metrics_fcfw", "Ensemble_FCFW")]:
            baselines = data.get("baselines", {})
            ens = data.get("ensemble", {})
            m = baselines.get(method_key, {}) if method_key in baselines else ens.get(method_key, {})
            if not m:
                continue
            vals = [dataset_name, method_label]
            for c in all_metric_cols:
                v = m.get(c, float("nan"))
                if isinstance(v, float):
                    vals.append(f"{v:.4f}")
                else:
                    vals.append(str(v))
            dataset_rows.append(vals)
        if dataset_rows:
            if rows:
                rows.append(None)  # separator
            rows.extend(dataset_rows)

    if not rows:
        return ""

    valid_rows = [r for r in rows if r is not None]
    col_widths = [max([len(str(r[i])) for r in valid_rows] + [len(headers[i])]) for i in range(len(headers))]
    sep = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |"
    line = "|-" + "-|-".join("-" * w for w in col_widths) + "-|"
    sep_row = "| " + " | ".join(" " * w for w in col_widths) + " |"

    parts = [sep, line]
    for r in rows:
        if r is None:
            parts.append(sep_row)
        else:
            parts.append("| " + " | ".join(str(r[i]).ljust(w) for i, w in enumerate(col_widths)) + " |")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace):
    root = Path(args.artifact_path)

    if root.is_file():
        process_artifact(root, args.shots, inference_dir_override=args.inference_dir)
    else:
        artifacts = sorted(root.rglob("circuit_artifact.json"))
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

        table = build_recap_table(root)
        if table:
            table_path = root / "dataset_metrics_recap.md"
            with open(table_path, "w") as f:
                f.write("# Dataset-Specific Metrics Recap\n\n")
                f.write(table)
                f.write("\n")
            logger.info("")
            logger.info("Dataset metrics recap saved to: %s", table_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute dataset-specific metrics from backend inference shots."
    )
    parser.add_argument(
        "artifact_path",
        type=str,
        help="Path to a circuit_artifact.json file (single mode), or a parent directory"
             " whose subdirectories are searched recursively (batch mode).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=1024,
        help="Number of generated samples to use for metrics. Default: 1024",
    )
    parser.add_argument(
        "--inference-dir",
        type=str,
        default=None,
        help="Specific inference run folder. Default: latest (single mode) or all (batch mode).",
    )

    args = parser.parse_args()
    main(args)
