"""Standalone evaluation utilities for Hopfield dataset integrations.

These functions are intentionally not wired into the experiment runner or HPO
loop. They can be imported after training to evaluate saved/final samples.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

from src.datasets.boltzmann_utils import sample_from_probs


def _as_binary_sample_matrix(samples: np.ndarray, n_qubits: int) -> np.ndarray:
    """Validate and coerce samples to a 2D binary matrix."""
    arr = np.asarray(samples)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("samples must be a 1D or 2D array")
    if arr.shape[1] != n_qubits:
        raise ValueError(f"expected samples with {n_qubits} columns, got {arr.shape[1]}")

    binary = (arr.astype(float) >= 0.5).astype(np.int8)
    return binary


def compute_hopfield_energies(dataset, samples: np.ndarray) -> np.ndarray:
    """Compute Hopfield energies for a batch of binary samples.

    Args:
        dataset: HopfieldDataset-like object with ``n_qubits`` and ``J``.
        samples: Array with shape ``(batch, n_qubits)`` containing ``{0, 1}``
            values. A single 1D sample is also accepted.

    Returns:
        1D array of energies using ``E(s) = -0.5 * s^T J s`` after converting
        bits ``{0, 1}`` to spins ``{+1, -1}``.
    """
    binary = _as_binary_sample_matrix(samples, int(dataset.n_qubits))
    spins = 1.0 - 2.0 * binary
    J_matrix = getattr(dataset, 'J', getattr(dataset, 'J_dense', None))
    if J_matrix is None:
        raise AttributeError("dataset has neither 'J' nor 'J_dense' coupling matrix attribute")
    J = np.asarray(J_matrix, dtype=np.float64)
    return -0.5 * np.einsum('bi,ij,bj->b', spins, J, spins)


def _sample_hopfield_reference(dataset, n_samples: int, seed: int = 0) -> np.ndarray:
    """Draw reference samples without mutating ``dataset.data``."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    probs = getattr(dataset, 'probs', None)
    if probs is not None:
        return sample_from_probs(probs, int(dataset.n_qubits), int(n_samples), seed=seed)

    if not hasattr(dataset, '_generate_mcmc'):
        raise ValueError("dataset has no exact probs and no _generate_mcmc method")
    return dataset._generate_mcmc(int(n_samples), seed=seed)


def evaluate_energy_wasserstein(
    dataset,
    qcbm_samples: np.ndarray,
    n_baseline: int = 10000,
    seed: int = 0,
) -> float:
    """Compare target and generated Hopfield energy distributions.

    Computes the 1-Wasserstein distance between scalar energy samples from the
    Hopfield target distribution and the supplied model/QCBM samples. Lower is
    better.

    The reference samples are drawn without calling ``dataset.generate()`` so
    this function does not overwrite ``dataset.data``.
    """
    qcbm_samples = _as_binary_sample_matrix(qcbm_samples, int(dataset.n_qubits))
    true_samples = _sample_hopfield_reference(dataset, int(n_baseline), seed=seed)

    true_energies = compute_hopfield_energies(dataset, true_samples)
    qcbm_energies = compute_hopfield_energies(dataset, qcbm_samples)
    return float(wasserstein_distance(true_energies, qcbm_energies))


def evaluate_memory_recall(dataset, qcbm_samples: np.ndarray) -> dict[str, float | np.ndarray]:
    """Evaluate distance from generated samples to stored Hopfield memories.

    For each generated bitstring, computes the minimum Hamming distance to any
    stored pattern or its inverse. Lower mean/quantile distance is better;
    higher exact recall is better.
    """
    samples = _as_binary_sample_matrix(qcbm_samples, int(dataset.n_qubits))
    patterns = np.asarray(dataset.patterns, dtype=np.float64)
    if patterns.ndim != 2 or patterns.shape[1] != samples.shape[1]:
        raise ValueError("dataset.patterns must have shape (n_patterns, n_qubits)")

    patterns_bin = ((1.0 - patterns) / 2.0).astype(np.int8)
    patterns_inv_bin = 1 - patterns_bin

    samples_expanded = samples[:, None, :]
    dists_to_patterns = (samples_expanded != patterns_bin[None, :, :]).sum(axis=2)
    dists_to_inverses = (samples_expanded != patterns_inv_bin[None, :, :]).sum(axis=2)
    min_dists = np.minimum(dists_to_patterns.min(axis=1), dists_to_inverses.min(axis=1))

    return {
        "memory_mean_distance": float(min_dists.mean()),
        "memory_exact_recall_rate": float((min_dists == 0).mean()),
        "memory_distance_p50": float(np.percentile(min_dists, 50)),
        "memory_distance_p90": float(np.percentile(min_dists, 90)),
        "memory_distance_p99": float(np.percentile(min_dists, 99)),
        "distances_array": min_dists,
    }
