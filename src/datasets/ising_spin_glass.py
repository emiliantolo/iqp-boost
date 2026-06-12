"""2D Frustrated Ising Spin Glass dataset.

Edwards-Anderson model on an L×L grid with quenched random Jij ∈ {±1}
on nearest-neighbor edges. Samples drawn via Metropolis–Hastings MCMC
at configurable temperature.
"""

from __future__ import annotations

import numpy as np
from src.datasets.base import BinaryDataset


class IsingSpinGlassDataset(BinaryDataset):
    """2D Edwards-Anderson spin glass with Metropolis sampling."""

    def __init__(
        self,
        rows: int = 5,
        cols: int = 4,
        beta: float = 2.0,
        coupling_seed: int = 0,
        burn_in: int = 1000,
        thinning: int = 100,
        sweeps_per_sample: int = 1,
        train_split_ratio: float | None = None,
    ):
        super().__init__(data=None, train_split_ratio=train_split_ratio)
        self.rows = int(rows)
        self.cols = int(cols)
        self.beta = float(beta)
        self.coupling_seed = int(coupling_seed)
        self.burn_in = int(burn_in)
        self.thinning = int(thinning)
        self.sweeps_per_sample = int(sweeps_per_sample)
        self.n_qubits = self.rows * self.cols
        self._build_couplings()

    def _build_couplings(self):
        rng = np.random.default_rng(self.coupling_seed)
        H = self.rows
        W = self.cols
        n_sites = H * W
        self.J_rows = np.zeros((H - 1, W), dtype=np.int8)
        self.J_cols = np.zeros((H, W - 1), dtype=np.int8)
        for i in range(H - 1):
            for j in range(W):
                self.J_rows[i, j] = 1 if rng.uniform() < 0.5 else -1
        for i in range(H):
            for j in range(W - 1):
                self.J_cols[i, j] = 1 if rng.uniform() < 0.5 else -1

    def _energy(self, spins: np.ndarray) -> float:
        H, W = self.rows, self.cols
        spins_2d = spins.reshape(H, W)
        energy = 0.0
        for i in range(H - 1):
            for j in range(W):
                energy -= self.J_rows[i, j] * spins_2d[i, j] * spins_2d[i + 1, j]
        for i in range(H):
            for j in range(W - 1):
                energy -= self.J_cols[i, j] * spins_2d[i, j] * spins_2d[i, j + 1]
        return energy

    def _metropolis_step(self, spins: np.ndarray, beta: float, rng: np.random.Generator):
        H, W = self.rows, self.cols
        i = rng.integers(H)
        j = rng.integers(W)
        idx = i * W + j

        dE = 0
        if i > 0:
            dE += 2 * self.J_rows[i - 1, j] * spins[idx] * spins[(i - 1) * W + j]
        if i < H - 1:
            dE += 2 * self.J_rows[i, j] * spins[idx] * spins[(i + 1) * W + j]
        if j > 0:
            dE += 2 * self.J_cols[i, j - 1] * spins[idx] * spins[i * W + (j - 1)]
        if j < W - 1:
            dE += 2 * self.J_cols[i, j] * spins[idx] * spins[i * W + (j + 1)]

        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            spins[idx] = -spins[idx]

    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n = self.n_qubits
        spins = np.ones(n, dtype=np.int8)
        for _ in range(self.burn_in * n):
            self._metropolis_step(spins, self.beta, rng)
        samples = []
        for _ in range(n_samples):
            for _ in range(self.thinning * n):
                self._metropolis_step(spins, self.beta, rng)
            for _ in range(self.sweeps_per_sample * n):
                self._metropolis_step(spins, self.beta, rng)
            samples.append(((spins + 1) // 2).astype(np.int8))
        return np.stack(samples)

    def visualize(self, sample: np.ndarray, ax=None):
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots()
        image = np.asarray(sample, dtype=np.int8).reshape(self.rows, self.cols)
        ax.imshow(image, cmap="RdBu_r", interpolation="nearest", vmin=0, vmax=1)
        ax.axis("off")
        return ax
