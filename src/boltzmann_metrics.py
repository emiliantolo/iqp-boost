from __future__ import annotations

import numpy as np


def _ensure_2d_samples(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.int8)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError("samples must be a 2D array of binary strings")
    return (values > 0).astype(np.int8)


def samples_to_hamming_weights(samples: np.ndarray) -> np.ndarray:
    """Return Hamming weights for a batch of binary samples."""
    values = _ensure_2d_samples(samples)
    return np.sum(values, axis=1).astype(np.int32)


def samples_to_histogram(samples: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
    """Convert binary samples to an empirical distribution over bitstrings."""
    values = _ensure_2d_samples(samples)
    if values.size == 0:
        if n_qubits is None:
            return np.array([], dtype=np.float64)
        return np.zeros(2 ** int(n_qubits), dtype=np.float64)

    n_qubits = int(n_qubits if n_qubits is not None else values.shape[1])
    indices = np.sum(values.astype(np.int64) * (2 ** np.arange(n_qubits)), axis=1)
    counts = np.bincount(indices, minlength=2 ** n_qubits).astype(np.float64)
    total = counts.sum()
    return counts / total if total > 0 else counts


def spin_covariance_from_samples(samples: np.ndarray) -> np.ndarray:
    """Estimate the spin covariance matrix from binary samples in {0, 1}."""
    values = _ensure_2d_samples(samples)
    if len(values) == 0:
        return np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)

    spins = 1.0 - 2.0 * values.astype(np.float64)
    mean = spins.mean(axis=0, keepdims=True)
    centered = spins - mean
    return (centered.T @ centered) / len(spins)


def spin_covariance_from_probs(exact_probs: np.ndarray) -> np.ndarray:
    """Compute the exact spin covariance matrix from a full probability tensor."""
    p = np.asarray(exact_probs, dtype=np.float64)
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("exact_probs must be a non-empty 1D probability vector")

    n_qubits = int(round(np.log2(len(p))))
    if 2 ** n_qubits != len(p):
        raise ValueError("exact_probs length must be a power of 2")

    indices = np.arange(len(p), dtype=np.int64)
    bits = ((indices[:, None] >> np.arange(n_qubits)) & 1).astype(np.float64)
    spins = 1.0 - 2.0 * bits
    mean = p @ spins
    second_moment = (spins.T * p) @ spins
    return second_moment - np.outer(mean, mean)


def covariance_matrices(
    data_samples: np.ndarray | None,
    model_samples: np.ndarray,
    exact_probs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return data/model spin covariance matrices for plotting and scoring."""
    sigma_model = spin_covariance_from_samples(model_samples)
    if exact_probs is not None:
        sigma_data = spin_covariance_from_probs(exact_probs)
    elif data_samples is not None:
        sigma_data = spin_covariance_from_samples(data_samples)
    else:
        raise ValueError("either data_samples or exact_probs must be provided")
    return sigma_data, sigma_model


def pairwise_correlation_frobenius_error(
    data_samples: np.ndarray | None,
    model_samples: np.ndarray,
    exact_probs: np.ndarray | None = None,
) -> float:
    """Return ||Sigma_data - Sigma_model||_F using spins in {-1, +1}."""
    sigma_data, sigma_model = covariance_matrices(data_samples, model_samples, exact_probs=exact_probs)
    return float(np.linalg.norm(sigma_data - sigma_model, ord='fro'))
