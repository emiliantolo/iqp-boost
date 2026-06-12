"""Evaluation utilities for Ising generative models."""

import numpy as np
from src.hopfield_evaluation import evaluate_energy_wasserstein

def compute_pairwise_correlation_matrix(samples: np.ndarray) -> np.ndarray:
    """Compute spin-spin correlation matrix C_ij = <s_i s_j> where s in {+1, -1}."""
    arr = np.asarray(samples)
    binary = (arr >= 0.5).astype(np.int8)
    spins = 1.0 - 2.0 * binary
    n_samples, n_qubits = spins.shape
    return (spins.T @ spins) / n_samples

def evaluate_pairwise_correlation_error(
    dataset,
    qcbm_samples: np.ndarray,
    n_baseline: int = 10000,
    seed: int = 0,
) -> float:
    """Compute Frobenius norm of the difference between true and generated pairwise correlation matrices."""
    from src.hopfield_evaluation import _sample_hopfield_reference
    
    true_samples = _sample_hopfield_reference(dataset, int(n_baseline), seed=seed)
    
    c_true = compute_pairwise_correlation_matrix(true_samples)
    c_qcbm = compute_pairwise_correlation_matrix(qcbm_samples)
    
    return float(np.linalg.norm(c_true - c_qcbm, 'fro'))

def evaluate_magnetization_absolute_error(
    dataset,
    qcbm_samples: np.ndarray,
    n_baseline: int = 10000,
    seed: int = 0,
) -> float:
    """Compute absolute difference in average absolute magnetization <|M|>."""
    from src.hopfield_evaluation import _sample_hopfield_reference
    
    true_samples = _sample_hopfield_reference(dataset, int(n_baseline), seed=seed)
    
    def abs_mag(s):
        spins = 1.0 - 2.0 * (s >= 0.5)
        return np.abs(spins.mean(axis=1))
        
    m_true = abs_mag(true_samples).mean()
    m_qcbm = abs_mag(qcbm_samples).mean()
    
    return float(np.abs(m_true - m_qcbm))
