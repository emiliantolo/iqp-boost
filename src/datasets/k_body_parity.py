"""Hidden k-body parity dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .base import BinaryDataset


class KBodyParityDataset(BinaryDataset):
    """Generate bitstrings satisfying a hidden parity constraint."""

    def __init__(
        self,
        n_qubits: int = 20,
        k: int = 10,
        parity: int = 1,
        subset_seed: int = 0,
        hidden_indices: list[int] | tuple[int, ...] | None = None,
        max_exact_states: int = 2**24,
    ):
        super().__init__()
        if n_qubits <= 0:
            raise ValueError("n_qubits must be positive")
        if k < 0:
            raise ValueError("k must be non-negative")
        if k > n_qubits:
            raise ValueError("k must be less than or equal to n_qubits")
        if parity not in {0, 1}:
            raise ValueError("parity must be 0 or 1")
        if k == 0 and parity != 0:
            raise ValueError("empty parity can only equal 0")
        if max_exact_states < 0:
            raise ValueError("max_exact_states must be non-negative")

        self.n_qubits = int(n_qubits)
        self.k = int(k)
        self.parity = int(parity)
        self.subset_seed = int(subset_seed)
        self.max_exact_states = int(max_exact_states)

        if hidden_indices is None:
            rng = np.random.default_rng(self.subset_seed)
            self.hidden_indices = tuple(sorted(int(index) for index in rng.choice(self.n_qubits, size=self.k, replace=False)))
        else:
            self.hidden_indices = tuple(sorted(int(index) for index in hidden_indices))
            if len(self.hidden_indices) != self.k:
                raise ValueError("hidden_indices length must equal k")
            if len(set(self.hidden_indices)) != self.k:
                raise ValueError("hidden_indices must not contain duplicates")
            if any(index < 0 or index >= self.n_qubits for index in self.hidden_indices):
                raise ValueError("hidden_indices must be valid bit positions")

        self.support_size = 2 ** (self.n_qubits - 1) if self.k > 0 else 2**self.n_qubits
        self.probs = self._build_probs() if 2**self.n_qubits <= self.max_exact_states else None

    def _build_probs(self) -> np.ndarray:
        probabilities = np.zeros(2**self.n_qubits, dtype=np.float64)
        if self.n_qubits >= 64:
            return probabilities

        mask = sum(1 << index for index in self.hidden_indices)
        for index in range(len(probabilities)):
            if (index & mask).bit_count() % 2 == self.parity:
                probabilities[index] = 1.0 / self.support_size
        return probabilities

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(seed)
        samples = rng.integers(0, 2, size=(int(n_samples), self.n_qubits), dtype=np.int8)
        if self.k > 0:
            hidden = np.asarray(self.hidden_indices, dtype=int)
            current = np.bitwise_xor.reduce(samples[:, hidden], axis=1)
            mismatched = current != self.parity
            samples[mismatched, hidden[0]] = 1 - samples[mismatched, hidden[0]]
        self.data = samples.astype(np.int8)
        return self.data

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        values = np.asarray(samples, dtype=np.int8)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != self.n_qubits:
            raise ValueError(f"expected samples with shape (n, {self.n_qubits})")
        return (values > 0).astype(np.int8)

    def valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        if len(values) == 0:
            return 0
        if self.k == 0:
            return int(len(values))
        hidden = np.asarray(self.hidden_indices, dtype=int)
        parity_values = np.bitwise_xor.reduce(values[:, hidden], axis=1)
        return int(np.count_nonzero(parity_values == self.parity))

    def validity_rate(self, samples: np.ndarray) -> float:
        values = self._validate_samples(samples)
        if len(values) == 0:
            return 0.0
        return float(self.valid_count(values) / len(values))

    def unique_valid_count(self, samples: np.ndarray) -> int:
        values = self._validate_samples(samples)
        if len(values) == 0:
            return 0
        if self.k == 0:
            valid = values
        else:
            hidden = np.asarray(self.hidden_indices, dtype=int)
            parity_values = np.bitwise_xor.reduce(values[:, hidden], axis=1)
            valid = values[parity_values == self.parity]
        return int(len({tuple(int(bit) for bit in sample) for sample in valid}))

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        del ground_truth
        if self.support_size <= 0:
            return 0.0
        return float(self.unique_valid_count(samples) / self.support_size)

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(5, 1.5))

        sample = np.asarray(sample).reshape(1, -1)
        ax.imshow(sample, cmap="binary", interpolation="nearest", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Hidden k-parity (n={self.n_qubits}, k={self.k}, parity={self.parity})")
        return ax
