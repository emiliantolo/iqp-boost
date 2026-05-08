import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from .base import BinaryDataset

class HopfieldDataset(BinaryDataset):
    """Dataset based on the Boltzmann distribution of a Hopfield network."""
    
    def __init__(self, n_qubits: int = 16, n_patterns: int = 5,
                 beta: float = 2.0, pattern_seed: int = 0,
                 batch_size: int = 2**20):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_patterns = n_patterns
        self.beta = beta
        self.pattern_seed = pattern_seed
        self.batch_size = batch_size

        # Generate random patterns in {-1, +1}
        rng = np.random.default_rng(pattern_seed)
        self.patterns = rng.choice([-1.0, 1.0],
                                    size=(n_patterns, n_qubits))

        # Hopfield coupling matrix (Hebbian rule)
        self.J = sum(np.outer(p, p) for p in self.patterns) / n_qubits
        np.fill_diagonal(self.J, 0)

        # Boltzmann distribution has full support on all 2^n bitstrings.
        # Stored memories are high-probability modes, not the only valid states.
        self._valid_patterns = None

        self.probs = self._compute_boltzmann()

    def _compute_boltzmann(self):
        total_states = 2 ** self.n_qubits
        bs = min(self.batch_size, total_states)
        J = jnp.array(self.J)

        @jax.jit
        def batch_energies(start_idx):
            idx = start_idx + jnp.arange(bs, dtype=jnp.int32)
            bits = jnp.arange(self.n_qubits)
            x = jnp.bitwise_and(jnp.right_shift(idx[:, None], bits), 1)
            s = 1 - 2 * x  # {0,1} -> {+1,-1}
            return -0.5 * jnp.einsum('bi,ij,bj->b', s, J, s)

        energies = []
        for start in range(0, total_states, bs):
            batch_e = batch_energies(start)
            remaining = total_states - start
            if remaining < bs:
                batch_e = batch_e[:remaining]
            energies.append(batch_e)

        energies = jnp.concatenate(energies)
        log_Z = logsumexp(-self.beta * energies)
        probs = jnp.exp(-self.beta * energies - log_Z)
        return np.asarray(probs)

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        indices = rng.choice(len(p), size=n_samples, p=p)
        bit_positions = np.arange(self.n_qubits)
        samples = ((indices[:, None] >> bit_positions) & 1).astype(np.int8)
        self.data = samples
        return self.data

    def validity_rate(self, samples: np.ndarray) -> float:
        """All bitstrings are valid Hopfield configurations -> always 1.0."""
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """Not meaningful for full-support distributions -> always 1.0."""
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a Hopfield sample.
        Tries to reshapes to a square grid if possible, otherwise displays as 1D.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        
        # Try to find a square layout
        side = int(np.sqrt(self.n_qubits))
        if side * side == self.n_qubits:
            display_data = sample.reshape(side, side)
        else:
            display_data = sample.reshape(1, -1)
            
        im = ax.imshow(display_data, cmap='binary', interpolation='nearest')
        
        # Add labels if it's small enough
        if self.n_qubits <= 64:
            if side * side == self.n_qubits:
                for i in range(side):
                    for j in range(side):
                        ax.text(j, i, str(int(display_data[i, j])),
                                ha='center', va='center',
                                color='red' if display_data[i, j] == 0 else 'gray')
            else:
                for i, bit in enumerate(sample):
                    ax.text(i, 0, str(int(bit)), ha='center', va='center',
                            color='red' if bit == 0 else 'gray')

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Hopfield Sample ({self.n_qubits} qubits)")
        return ax

    def top_k_tvd(self, k: int) -> float:
        p = np.asarray(self.probs, dtype=np.float64)
        p = p / p.sum()
        top_k_idx = np.argsort(p)[::-1][:k]
        q = np.zeros_like(p)
        q[top_k_idx] = p[top_k_idx]
        q = q / q.sum()
        return float(0.5 * np.abs(p - q).sum())
