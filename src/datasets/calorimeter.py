"""Synthetic binarized calorimeter shower dataset.

Generates N Gaussian blobs with random (cx, cy, σ) on an H×W grid,
binarized at a threshold. A momentum-conservation constraint biases
the blob centroid vector sum toward zero.
"""

from __future__ import annotations

import numpy as np
from src.datasets.base import BinaryDataset


class CalorimeterDataset(BinaryDataset):
    """Binarized calorimeter shower dataset with synthetic Gaussian blobs."""

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        n_blobs: int = 3,
        blob_std_range: tuple[float, float] = (0.5, 2.0),
        threshold: float = 0.3,
        momentum_strength: float = 2.0,
        train_split_ratio: float | None = None,
    ):
        super().__init__(data=None, train_split_ratio=train_split_ratio)
        self.rows = int(rows)
        self.cols = int(cols)
        self.n_blobs = int(n_blobs)
        self.blob_std_range = blob_std_range
        self.threshold = float(threshold)
        self.momentum_strength = float(momentum_strength)
        self.n_qubits = self.rows * self.cols

    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        H, W = self.rows, self.cols
        ys, xs = np.mgrid[0:H, 0:W]
        grid = np.stack([xs, ys], axis=-1).reshape(-1, 2)
        samples = []
        for _ in range(n_samples):
            for attempt in range(100):
                cx = rng.uniform(0, W, size=self.n_blobs)
                cy = rng.uniform(0, H, size=self.n_blobs)
                sigmas = rng.uniform(*self.blob_std_range, size=self.n_blobs)
                if self.momentum_strength > 0:
                    momenta = np.stack([cx - W / 2, cy - H / 2], axis=1)
                    total_momentum = np.linalg.norm(momenta.sum(axis=0))
                    accept_prob = np.exp(-self.momentum_strength * total_momentum / (self.n_blobs * max(sigmas)))
                    if rng.uniform() > accept_prob:
                        continue
                density = np.zeros(H * W, dtype=np.float64)
                for b in range(self.n_blobs):
                    dist2 = ((grid[:, 0] - cx[b]) ** 2 + (grid[:, 1] - cy[b]) ** 2) / (2 * sigmas[b] ** 2)
                    density += np.exp(-dist2)
                binary = (density > self.threshold).astype(np.int8)
                samples.append(binary)
                break
            else:
                cx = rng.uniform(0, W, size=self.n_blobs)
                cy = rng.uniform(0, H, size=self.n_blobs)
                density = np.zeros(H * W, dtype=np.float64)
                for b in range(self.n_blobs):
                    dist2 = ((grid[:, 0] - cx[b]) ** 2 + (grid[:, 1] - cy[b]) ** 2)
                    density += np.exp(-dist2 / 2)
                binary = (density > self.threshold).astype(np.int8)
                samples.append(binary)
        return np.stack(samples)

    def visualize(self, sample: np.ndarray, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        image = np.asarray(sample, dtype=np.int8).reshape(self.rows, self.cols)
        ax.imshow(image, cmap="hot_r", interpolation="nearest")
        ax.axis("off")
        return ax
