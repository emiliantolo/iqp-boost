from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp


def dense_boltzmann_probs(J: np.ndarray, beta: float, batch_size: int = 2**20) -> np.ndarray:
    """Compute exact Boltzmann probabilities for a dense Ising coupling matrix.

    The Hamiltonian is assumed to be

        E(s) = -1/2 * s^T J s

    with spins s in {-1, +1} and zero diagonal in J.
    """
    J = np.asarray(J, dtype=np.float64)
    n_qubits = J.shape[0]
    total_states = 2 ** n_qubits
    bs = min(int(batch_size), total_states)
    J_jnp = jnp.array(J)

    @jax.jit
    def batch_energies(start_idx):
        idx = start_idx + jnp.arange(bs, dtype=jnp.int32)
        bits = jnp.arange(n_qubits)
        x = jnp.bitwise_and(jnp.right_shift(idx[:, None], bits), 1)
        spins = 1 - 2 * x
        return -0.5 * jnp.einsum('bi,ij,bj->b', spins, J_jnp, spins)

    energies = []
    for start in range(0, total_states, bs):
        batch_e = batch_energies(start)
        remaining = total_states - start
        if remaining < bs:
            batch_e = batch_e[:remaining]
        energies.append(batch_e)

    energies = jnp.concatenate(energies)
    log_z = logsumexp(-beta * energies)
    probs = jnp.exp(-beta * energies - log_z)
    return np.asarray(probs)


def sample_from_probs(probs: np.ndarray, n_qubits: int, n_samples: int, seed: int = 0) -> np.ndarray:
    """Sample binary bitstrings from a probability vector over all 2**n states."""
    rng = np.random.default_rng(seed)
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    indices = rng.choice(len(p), size=int(n_samples), p=p)
    bit_positions = np.arange(n_qubits)
    return ((indices[:, None] >> bit_positions) & 1).astype(np.int8)
