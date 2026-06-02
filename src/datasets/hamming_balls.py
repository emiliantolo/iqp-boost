import numpy as np
import matplotlib.pyplot as plt

from .base import BinaryDataset
from .boltzmann_utils import sample_from_probs


class HammingBallsDataset(BinaryDataset):
    """Mixture of Hamming Balls: K planted binary centers with bit-flip noise."""

    def __init__(
        self,
        n_qubits: int = 16,
        K: int = 8,
        p: float = 0.1,
        pattern_seed: int = 0,
        batch_size: int = 2**20,
        max_exact_states: int = 2**20,
        train_split_ratio: float | None = None,
    ):
        super().__init__(train_split_ratio=train_split_ratio)
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if K <= 0:
            raise ValueError("K must be positive")
        if not 0 < p < 1:
            raise ValueError("p must be in (0, 1)")

        self.n_qubits = int(n_qubits)
        self.K = int(K)
        self.p = float(p)
        self.pattern_seed = int(pattern_seed)
        self.batch_size = int(batch_size)
        self.max_exact_states = int(max_exact_states)
        self._valid_patterns = None

        rng = np.random.default_rng(self.pattern_seed)
        self.centers = rng.integers(0, 2, size=(self.K, self.n_qubits), dtype=np.int8)

        self._exact_mode = 2**self.n_qubits <= self.max_exact_states
        self.probs = self._compute_exact_probs() if self._exact_mode else None

    def _compute_exact_probs(self) -> np.ndarray:
        total = 2**self.n_qubits
        indices = np.arange(total, dtype=np.int64)[:, None]
        bit_shifts = np.arange(self.n_qubits, dtype=np.int64)
        all_bits = ((indices >> bit_shifts) & 1).astype(np.int8)
        d = np.sum(all_bits[:, None, :] != self.centers[None, :, :], axis=2)
        probs_per_center = (self.p ** d) * ((1.0 - self.p) ** (self.n_qubits - d))
        probs = probs_per_center.mean(axis=1)
        return probs / probs.sum()

    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if self.probs is not None:
            return sample_from_probs(self.probs, self.n_qubits, n_samples, seed=seed)
        rng = np.random.default_rng(seed)
        k_indices = rng.integers(0, self.K, size=n_samples)
        samples = self.centers[k_indices].copy()
        flip_mask = rng.random((n_samples, self.n_qubits)) < self.p
        samples ^= flip_mask.astype(np.int8)
        return samples

    def probability(self, x: np.ndarray) -> float:
        """Scalable evaluation: Exact analytical probability of a single bitstring string."""
        x = np.asarray(x, dtype=np.int8)
        d = np.sum(x[None, :] != self.centers, axis=1)
        probs_per_center = (self.p ** d) * ((1.0 - self.p) ** (self.n_qubits - d))
        return float(probs_per_center.mean())

    def validity_rate(self, samples: np.ndarray) -> float:
        return 1.0

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        return 1.0

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(max(3, self.n_qubits * 0.15), 2))
        ax.imshow(sample.reshape(1, -1), cmap='binary', interpolation='nearest', aspect='auto')
        ax.set_yticks([])
        ax.set_xticks(range(self.n_qubits))
        ax.set_xticklabels([])
        ax.set_title(f'Hamming Balls (n={self.n_qubits}, K={self.K})')
        return ax
