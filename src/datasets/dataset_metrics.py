"""Local dataset-specific diagnostics for sampled experiment outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.metrics import compute_hamming_matrix

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - exercised by environments without plotting deps
    plt = None


def compute_and_save_dataset_metrics(
    *,
    dataset_bundle,
    output,
    final_samples: np.ndarray | None,
    per_model_samples: list[np.ndarray] | None = None,
    baseline_samples: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Compute dataset diagnostics from local final-evaluation samples."""
    if final_samples is None or len(final_samples) == 0:
        print("Skipping dataset metrics because final samples are unavailable.")
        return None

    dataset_obj = getattr(dataset_bundle, "dataset_obj", None)
    dataset_key = _dataset_key(dataset_obj)
    if dataset_key is None:
        print(f"Skipping dataset metrics for unsupported dataset: {dataset_bundle.dataset_name}")
        return None

    metrics = _compute_metrics(dataset_key, dataset_obj, np.asarray(final_samples, dtype=np.int8))
    if not metrics:
        print(f"Skipping dataset metrics for unsupported dataset: {dataset_bundle.dataset_name}")
        return None

    payload = {
        "dataset": dataset_key,
        "final_ensemble": metrics,
    }
    output.save_metrics(payload, filename="dataset_metrics.json")

    if plt is not None:
        methods = _methods(final_samples, per_model_samples, baseline_samples)
        _plot_dataset_metrics(
            dataset_key=dataset_key,
            dataset_obj=dataset_obj,
            ground_truth=dataset_bundle.x_train,
            methods=methods,
            output_dir=Path(output.run_dir),
        )

    return payload


def compute_hopfield_metrics(samples: np.ndarray, J: np.ndarray, patterns: np.ndarray, radius: int = 2) -> dict:
    """Return Hopfield energy and pattern-recall diagnostics."""
    samples = np.asarray(samples, dtype=np.int8)
    if len(samples) == 0:
        return {
            "energy_mean": float("nan"),
            "energy_std": float("nan"),
            "pattern_proximity": float("nan"),
            "pattern_coverage": 0.0,
        }

    energies = hopfield_energy(samples, J)
    pattern_bits = ((1.0 - np.asarray(patterns)) / 2.0).astype(np.int8)
    distances = compute_hamming_matrix(samples, pattern_bits)
    min_distances = distances.min(axis=1)
    covered = np.any(distances <= radius, axis=0)

    return {
        "energy_mean": float(energies.mean()),
        "energy_std": float(energies.std()),
        "pattern_proximity": float(min_distances.mean()),
        "pattern_coverage": float(covered.mean()),
    }


def hopfield_energy(samples: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Hopfield energy E = -0.5 * s.T J s for binary samples."""
    spins = 1.0 - 2.0 * np.asarray(samples, dtype=np.float64)
    return -0.5 * np.einsum("ni,ij,nj->n", spins, np.asarray(J, dtype=np.float64), spins)


def compute_hamming_balls_metrics(
    centers: np.ndarray,
    samples: np.ndarray,
    analytical_threshold: float | None = None,
) -> dict:
    """Return Hamming Balls recall distance and optional threshold coverage."""
    centers = np.asarray(centers, dtype=np.int8)
    samples = np.asarray(samples, dtype=np.int8)
    if len(samples) == 0 or len(centers) == 0:
        return {"recall_distance": float("nan"), "coverage": float("nan")}

    distances = compute_hamming_matrix(samples, centers).min(axis=1)
    metrics = {"recall_distance": float(distances.mean())}
    if analytical_threshold is not None:
        metrics["coverage"] = float((distances < analytical_threshold).mean())
    return metrics


def hamming_balls_analytical_recall(dataset_obj) -> tuple[float | None, float | None]:
    """Return expected nearest-center distances under true and uniform distributions."""
    probs = getattr(dataset_obj, "probs", None)
    centers = getattr(dataset_obj, "centers", None)
    n_qubits = int(getattr(dataset_obj, "n_qubits", 0))
    if probs is None or centers is None or n_qubits <= 0 or 2**n_qubits > 2**22:
        return None, None

    indices = np.arange(2**n_qubits, dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n_qubits, dtype=np.int64)) & 1).astype(np.int8)
    distances = compute_hamming_matrix(bits, centers).min(axis=1)
    return float(np.asarray(probs) @ distances), float(distances.mean())


def _dataset_key(dataset_obj) -> str | None:
    if dataset_obj is None:
        return None
    if hasattr(dataset_obj, "J") and hasattr(dataset_obj, "patterns"):
        return "hopfield"
    if hasattr(dataset_obj, "centers") and hasattr(dataset_obj, "p"):
        return "hamming_balls"
    return None


def _compute_metrics(dataset_key: str, dataset_obj, samples: np.ndarray) -> dict:
    if dataset_key == "hopfield":
        return compute_hopfield_metrics(samples, dataset_obj.J, dataset_obj.patterns)
    if dataset_key == "hamming_balls":
        analytical, _ = hamming_balls_analytical_recall(dataset_obj)
        return compute_hamming_balls_metrics(dataset_obj.centers, samples, analytical_threshold=analytical)
    return {}


def _methods(final_samples, per_model_samples, baseline_samples) -> list[tuple[str, np.ndarray]]:
    methods = []
    if baseline_samples is not None and len(baseline_samples) > 0:
        methods.append(("Standalone", np.asarray(baseline_samples, dtype=np.int8)))
    methods.append(("Final ensemble", np.asarray(final_samples, dtype=np.int8)))
    for idx, samples in enumerate(per_model_samples or []):
        if samples is not None and len(samples) > 0:
            methods.append((f"Model {idx}", np.asarray(samples, dtype=np.int8)))
    return methods


def _plot_dataset_metrics(dataset_key: str, dataset_obj, ground_truth, methods, output_dir: Path) -> None:
    if dataset_key == "hopfield":
        fig = _plot_hopfield_energy_hist(ground_truth, methods, dataset_obj)
        _save_figure(fig, output_dir, "dataset_hopfield_energy_hist")
    elif dataset_key == "hamming_balls":
        analytical, random_analytical = hamming_balls_analytical_recall(dataset_obj)
        fig = _plot_hamming_balls_distance_dist(
            ground_truth,
            methods,
            dataset_obj.centers,
            analytical=analytical,
            random_analytical=random_analytical,
        )
        _save_figure(fig, output_dir, "dataset_hamming_balls_distance_dist")


def _plot_hopfield_energy_hist(ground_truth, methods, dataset_obj):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if ground_truth is not None and len(ground_truth) > 0:
        ax.hist(
            hopfield_energy(ground_truth, dataset_obj.J),
            bins=40,
            density=True,
            alpha=0.45,
            color="gray",
            label="Train data",
        )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, samples) in enumerate(methods):
        ax.hist(
            hopfield_energy(samples, dataset_obj.J),
            bins=40,
            density=True,
            alpha=0.45,
            color=colors[idx % len(colors)],
            label=label,
        )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Density")
    ax.set_title("Hopfield energy distribution")
    ax.legend(fontsize=8)
    return fig


def _plot_hamming_balls_distance_dist(
    ground_truth,
    methods,
    centers,
    analytical: float | None = None,
    random_analytical: float | None = None,
):
    if plt is None or centers is None or len(centers) == 0:
        return None
    panels = []
    if ground_truth is not None and len(ground_truth) > 0:
        panels.append(("Train data", compute_hamming_matrix(ground_truth, centers).min(axis=1), "gray"))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (label, samples) in enumerate(methods):
        panels.append((label, compute_hamming_matrix(samples, centers).min(axis=1), colors[idx % len(colors)]))
    if not panels:
        return None

    n_cols = 2
    n_rows = (len(panels) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.6 * n_rows), constrained_layout=True, squeeze=False)
    for idx, (label, distances, color) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        bins = np.arange(0, int(np.max(distances)) + 2) - 0.5
        ax.hist(distances, bins=bins, density=True, alpha=0.5, color=color, edgecolor=color)
        ax.axvline(float(np.mean(distances)), color=color, linestyle="--", linewidth=1.0, label="Mean")
        if analytical is not None:
            ax.axvline(analytical, color="black", linestyle=":", linewidth=1.2, label="True expected")
        if random_analytical is not None:
            ax.axvline(random_analytical, color="gray", linestyle="-.", linewidth=1.0, label="Uniform expected")
        ax.set_title(label)
        ax.set_xlabel("Distance to nearest center")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)
    for idx in range(len(panels), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    return fig


def _save_figure(fig, output_dir: Path, stem: str) -> None:
    if fig is None:
        return
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=160, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved dataset metrics plot to: {png_path}")
