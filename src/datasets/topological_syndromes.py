"""Topological error syndrome dataset.

Toric code on an L×L lattice. Independent bit-flip errors at rate p,
compute the X-stabilizer syndrome grid. Each sample is the syndrome
grid (which stabilizers are violated), binarized and flattened.
"""

from __future__ import annotations

import numpy as np
from src.datasets.base import BinaryDataset


class TopologicalSyndromeDataset(BinaryDataset):
    """Toric code X-stabilizer syndromes under bit-flip noise."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 4,
        error_rate: float = 0.1,
        train_split_ratio: float | None = None,
    ):
        super().__init__(data=None, train_split_ratio=train_split_ratio)
        self.rows = int(rows)
        self.cols = int(cols)
        self.error_rate = float(error_rate)
        self.n_qubits = self.rows * self.cols

    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        H, W = self.rows, self.cols
        samples = []
        for _ in range(n_samples):
            data_qubits = rng.binomial(1, self.error_rate, size=(H, W))
            syndrome = np.zeros((H, W), dtype=np.int8)
            for i in range(H):
                for j in range(W):
                    s = 0
                    s ^= data_qubits[i, j]
                    s ^= data_qubits[(i + 1) % H, j]
                    s ^= data_qubits[i, (j + 1) % W]
                    s ^= data_qubits[(i + 1) % H, (j + 1) % W]
                    syndrome[i, j] = s
            samples.append(syndrome.ravel())
        return np.stack(samples).astype(np.int8)

    def visualize(self, sample: np.ndarray, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        image = np.asarray(sample, dtype=np.int8).reshape(self.rows, self.cols)
        ax.imshow(image, cmap="viridis", interpolation="nearest", vmin=0, vmax=1)
        ax.axis("off")
        return ax
