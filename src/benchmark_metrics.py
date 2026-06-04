"""Reusable dataset-specific benchmark metrics."""

from __future__ import annotations

import math

import numpy as np


def compute_hamming_balls_metrics(
    centers: np.ndarray,
    samples: np.ndarray,
    n_qubits: int,
    radius_fraction: float = 0.15,
) -> dict:
    """Compute generation metrics for the Hamming Balls dataset."""
    centers = np.asarray(centers, dtype=np.int8)
    samples = np.asarray(samples, dtype=np.int8)
    n_qubits = int(n_qubits)

    if centers.ndim != 2:
        raise ValueError("centers must be a 2D array")
    if centers.shape[1] != n_qubits:
        raise ValueError("centers width must match n_qubits")
    if samples.size == 0:
        return {
            "recall_distance": float("nan"),
            "coverage": 0.0,
            "per_center_coverage": [False for _ in range(len(centers))],
            "per_center_hit_counts": [0 for _ in range(len(centers))],
            "mean_distance_to_nearest_center": float("nan"),
            "median_distance_to_nearest_center": float("nan"),
        }
    if samples.ndim != 2 or samples.shape[1] != n_qubits:
        raise ValueError("samples must be a 2D array with width n_qubits")

    distances = np.sum(samples[:, None, :] != centers[None, :, :], axis=2)
    min_distances = np.min(distances, axis=1)
    radius = int(math.ceil(float(radius_fraction) * n_qubits))
    hits = distances <= radius
    covered = np.any(hits, axis=0)
    hit_counts = np.sum(hits, axis=0)

    return {
        "recall_distance": float(np.mean(min_distances)),
        "coverage": float(np.mean(covered)),
        "per_center_coverage": [bool(v) for v in covered.tolist()],
        "per_center_hit_counts": [int(v) for v in hit_counts.tolist()],
        "mean_distance_to_nearest_center": float(np.mean(min_distances)),
        "median_distance_to_nearest_center": float(np.median(min_distances)),
    }


def compute_mps_nll(dataset, samples: np.ndarray) -> float:
    """Compute MPS negative log-likelihood when the dataset exposes a scorer."""
    if hasattr(dataset, "negative_log_likelihood"):
        return float(dataset.negative_log_likelihood(samples))
    if hasattr(dataset, "nll"):
        return float(dataset.nll(samples))
    raise AttributeError("MPS dataset does not expose negative_log_likelihood or nll")
