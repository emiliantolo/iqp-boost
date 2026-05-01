"""Binarized MNIST dataset loaded from PennyLane."""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .base import BinaryDataset
from .reduction import pca_reduce, spatial_downsample, umap_reduce, variance_mi_reduce


class BinarizedMNISTDataset(BinaryDataset):
    """
    Binarized MNIST dataset for generative quantum machine learning.

    Loads the ``binarized-mnist`` dataset from PennyLane (50 000 training
    images of shape 784 = 28x28, already thresholded to {0, 1}).  Images
    are then reduced to ``rows x cols`` qubits using either spatial
    average-pooling or PCA.

    Source: `XanaduAI/gqml_datasets <https://github.com/XanaduAI/gqml_datasets>`_

    Args:
        rows: Height of the output image grid (default 4).
        cols: Width of the output image grid (default 4).
              Total qubits = rows x cols.
        digit: If not None, restrict to a single MNIST digit class (0-9).
        reduction: ``'spatial'`` for average-pool downscaling (preserves 2-D
            structure), ``'pca'`` for PCA-based reduction, ``'umap'`` for
            UMAP-based nonlinear reduction, or ``'variance_mi'`` for
            variance filtering followed by pairwise MI ranking.
    """

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        digit: int | None = None,
        reduction: str = "spatial",
        variance_filter_dims: int = 100,
    ):
        super().__init__()
        if reduction not in ("spatial", "pca", "umap", "variance_mi"):
            raise ValueError(
                f"reduction must be 'spatial', 'pca', 'umap', or 'variance_mi', got {reduction!r}"
            )

        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols
        self.digit = digit
        self.reduction = reduction
        self.variance_filter_dims = variance_filter_dims

        # Load from PennyLane
        [ds] = qml.data.load(
            "other", name="binarized-mnist",
            progress_bar=False, folder_path="./data",
        )
        all_inputs = np.array(ds.train["inputs"], dtype=np.int8)
        all_labels = np.array(ds.train["labels"], dtype=int)

        # Optionally filter by digit
        if digit is not None:
            mask = all_labels == int(digit)
            all_inputs = all_inputs[mask]
            all_labels = all_labels[mask]

        self._raw_inputs = all_inputs       # (N, 784)
        self._raw_labels = all_labels       # (N,)

        # Reduce dimensionality
        if reduction == "spatial":
            self._reduced = spatial_downsample(
                all_inputs, src_size=28,
                dst_height=rows, dst_width=cols,
            )
        elif reduction == "pca":
            self._reduced = pca_reduce(
                all_inputs, n_components=self.n_qubits,
            )
        elif reduction == "umap":
            self._reduced = umap_reduce(
                all_inputs, n_components=self.n_qubits,
            )
        else:
            self._reduced = variance_mi_reduce(
                all_inputs,
                n_components=self.n_qubits,
                variance_filter_dims=self.variance_filter_dims,
            )

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Sample *n_samples* reduced MNIST images.

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
        Visualize a single sample as a 2-D binary image.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        sample_2d = sample.reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Greys", aspect="equal", interpolation="nearest")
        title = "MNIST"
        if self.digit is not None:
            title += f" (digit {self.digit})"
        ax.set_title(title)

        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()
