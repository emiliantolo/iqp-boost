"""Frustrated Ising model dataset.

Generates binary samples from the exact Boltzmann distribution of a
frustrated 2D Ising model on a square grid with random +/-1 couplings.
"""

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from .base import BinaryDataset


class FrustratedIsingDataset(BinaryDataset):
    """
    Samples from the exact Boltzmann distribution of a frustrated Ising model
    on a 2D square grid with nearest-neighbour random +/-1 couplings.

    The Hamiltonian is  H = -sum_{<i,j>} J_ij s_i s_j  where s in {+1, -1}
    and J_ij in {+1, -1} are drawn randomly (quenched disorder).

    Since the distribution has full support on all 2^n states, validity,
    coverage, and F1 metrics are not meaningful.  The dataset returns
    trivial ``validity_rate`` (always 1.0) and ``coverage_rate``
    (always 1.0).
    """

    def __init__(self, rows: int = 4, cols: int = 4,
                 beta: float = 1.0, j_seed: int = 0,
                 batch_size: int = 2**20):
        """
        Initialise the frustrated Ising dataset.

        Args:
            rows: Number of rows in the square lattice.
            cols: Number of columns in the square lattice.
            beta: Inverse temperature. Higher beta -> more peaked distribution.
            j_seed: Seed for generating the quenched random +-1 couplings.
            batch_size: States processed at once during the exact
                Boltzmann computation (tune to avoid OOM).
        """
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols
        self.beta = beta
        self.j_seed = j_seed
        self.batch_size = batch_size

        # Build frustrated grid
        self.edges, self.J_weights = self._build_frustrated_grid()

        # Compute exact Boltzmann probabilities
        self.probs = self._compute_boltzmann()

        # Validity / coverage are not meaningful for this dataset.
        self._valid_patterns = None

    def _build_frustrated_grid(self):
        """Build nearest-neighbour edges on a 2D grid with random +/-1 couplings."""
        edges = []
        for r in range(self.rows):
            for c in range(self.cols):
                idx = r * self.cols + c
                # Horizontal neighbour
                if c + 1 < self.cols:
                    edges.append([idx, idx + 1])
                # Vertical neighbour
                if r + 1 < self.rows:
                    edges.append([idx, idx + self.cols])

        edges = jnp.array(edges)

        rng = np.random.default_rng(self.j_seed)
        J_weights = jnp.array(
            rng.choice([-1.0, 1.0], size=len(edges))
        )
        return edges, J_weights

    def _compute_boltzmann(self):
        """Compute exact Boltzmann probabilities via batched JAX enumeration."""
        total_states = 2 ** self.n_qubits
        edges = self.edges
        J_weights = self.J_weights
        beta = self.beta
        bs = min(self.batch_size, total_states)

        @jax.jit
        def get_batch_energies(start_idx):
            idx = start_idx + jnp.arange(bs, dtype=jnp.int32)
            u = edges[:, 0]
            v = edges[:, 1]
            bit_u = jnp.bitwise_and(jnp.right_shift(idx[:, None], u), 1)
            bit_v = jnp.bitwise_and(jnp.right_shift(idx[:, None], v), 1)
            spin_u = 1 - 2 * bit_u
            spin_v = 1 - 2 * bit_v
            return -jnp.sum(J_weights * spin_u * spin_v, axis=1)

        energies = []
        for start in range(0, total_states, bs):
            batch_e = get_batch_energies(start)
            remaining = total_states - start
            if remaining < bs:
                batch_e = batch_e[:remaining]  # truncate padded tail
            energies.append(batch_e)
        energies = jnp.concatenate(energies)

        log_Z = logsumexp(-beta * energies)
        probs = jnp.exp(-beta * energies - log_Z)
        return np.asarray(probs)

    # ------------------------------------------------------------------
    # BinaryDataset interface
    # ------------------------------------------------------------------

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Sample bitstrings from the exact Boltzmann distribution.

        Args:
            n_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_samples, n_qubits)`` with binary values.
        """
        rng = np.random.default_rng(seed)

        # Use float64 probabilities for accurate sampling
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()  # re-normalise to avoid fp rounding issues
        indices = rng.choice(len(p), size=n_samples, p=p)

        # Vectorised integer-to-bitstring conversion
        bit_positions = np.arange(self.n_qubits)          # (n_qubits,)
        samples = ((indices[:, None] >> bit_positions) & 1).astype(np.int8)

        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        """All bitstrings are valid Ising configurations -> always 1.0."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Not meaningful for Ising (full support) -> always 1.0."""
        return 1.0

    def top_k_tvd(self, k: int) -> float:
        """TVD of the oracle top-k baseline vs the exact Boltzmann distribution.

        The oracle baseline places mass on the top-k modes with weights
        proportional to their exact Boltzmann probabilities:

            q_i = p_i / sum(top-k p) for the k highest-probability states,
            q_i = 0 otherwise.

        Returns 0.5 * sum|p - q|, which is the best TVD achievable by ANY
        k-mode delta-function mixture.  A boosted ensemble of k models
        should be compared against this lower bound.
        """
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        top_k_idx = np.argsort(p)[::-1][:k]

        q = np.zeros_like(p)
        q[top_k_idx] = p[top_k_idx]
        q = q / q.sum()  # re-normalise to the top-k subspace

        return float(0.5 * np.abs(p - q).sum())

    def visualize(self, sample: np.ndarray, ax=None):
        """Visualise a sample as a 2D spin grid (+1 / -1)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(max(3, self.cols * 0.7),
                                            max(3, self.rows * 0.7)))

        spins = 1 - 2 * sample.reshape(self.rows, self.cols).astype(float)
        ax.imshow(spins, cmap='RdBu', vmin=-1, vmax=1,
                  interpolation='nearest')

        for r in range(self.rows):
            for c in range(self.cols):
                ax.text(c, r, '+' if spins[r, c] > 0 else '-',
                        ha='center', va='center', fontsize=10,
                        color='white' if spins[r, c] > 0 else 'black',
                        fontweight='bold')

        # Grid lines
        for i in range(self.rows + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.cols + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Ising {self.rows}x{self.cols}  beta={self.beta}')
        return ax