"""D-Wave quantum hardware samples dataset loaded from PennyLane."""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .base import BinaryDataset
from .reduction import pca_reduce, umap_reduce, variance_mi_reduce


class DWaveDataset(BinaryDataset):
    """
    D-Wave quantum annealer samples for generative quantum ML.

    Loads the ``d-wave`` dataset from PennyLane (10 000 training samples of
    484 spins from a 3-nearest-neighbour Ising model on a 22x22 grid,
    measured on a D-Wave quantum processor at 100 us anneal time).  Spin
    values are converted from +/-1 to {0, 1} and then reduced to
    ``rows x cols`` qubits using PCA, UMAP, or variance+MI feature selection.

    Source: `XanaduAI/gqml_datasets <https://github.com/XanaduAI/gqml_datasets>`_
    Original data: `Zenodo 7250436 <https://zenodo.org/records/7250436>`_

    Args:
        rows: Number of rows for the output grid (default 4).
        cols: Number of columns for the output grid (default 4).
              Total qubits = rows x cols.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        reduction: str = "pca",
        variance_filter_dims: int = 100,
    ):
        super().__init__()
        if reduction not in ("pca", "umap", "variance_mi"):
            raise ValueError(f"reduction must be 'pca', 'umap', or 'variance_mi', got {reduction!r}")
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols
        self.reduction = reduction
        self.variance_filter_dims = variance_filter_dims

        # Load from PennyLane
        [ds] = qml.data.load(
            "other", name="d-wave",
            progress_bar=False, folder_path="./data",
        )
        raw = np.array(ds.train, dtype=np.float64)

        # The original data uses +/-1 encoding; convert to {0, 1}
        if raw.min() < 0:
            raw = ((raw + 1) / 2).astype(np.int8)
        else:
            raw = raw.astype(np.int8)

        if reduction == "pca":
            self._reduced = pca_reduce(raw, n_components=self.n_qubits)
        elif reduction == "umap":
            self._reduced = umap_reduce(raw, n_components=self.n_qubits)
        else:
            self._reduced = variance_mi_reduce(
                raw,
                n_components=self.n_qubits,
                variance_filter_dims=self.variance_filter_dims,
            )

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Sample *n_samples* PCA-reduced D-Wave vectors.

        Args:
            n_samples: Number of samples to draw (with replacement).
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_samples, n_qubits)`` with binary values.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._reduced), size=n_samples, replace=True)
        self.data = self._reduced[indices]
        return self.data

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a single D-Wave sample as a 2-D binary grid.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        sample_2d = sample.reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Blues", aspect="equal", interpolation="nearest")
        ax.set_title("D-Wave sample")

        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()
