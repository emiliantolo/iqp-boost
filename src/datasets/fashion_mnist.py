"""Downscaled binarized Fashion-MNIST dataset."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .base import BinaryDataset


class FashionMNISTDownscaledDataset(BinaryDataset):
    """Load Fashion-MNIST, area-resize it, and threshold to binary samples."""

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        threshold: float = 0.5,
        data_dir: str | Path = "./data",
    ):
        super().__init__()
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")

        self.rows = int(rows)
        self.cols = int(cols)
        self.threshold = float(threshold)
        self.data_dir = Path(data_dir)
        self.n_qubits = self.rows * self.cols
        self._reduced = self._load_reduced()

    def _load_reduced(self) -> np.ndarray:
        try:
            import torch
            import torch.nn.functional as functional
            from torchvision.datasets import FashionMNIST
        except ImportError as exc:
            raise ImportError(
                "FashionMNISTDownscaledDataset requires torch and torchvision. "
                "Install project dependencies with `uv sync`."
            ) from exc

        dataset = FashionMNIST(root=str(self.data_dir), train=True, download=True)
        images = dataset.data.to(dtype=torch.float32).unsqueeze(1) / 255.0
        resized = functional.interpolate(images, size=(self.rows, self.cols), mode="area")
        binary = (resized.squeeze(1) >= self.threshold).to(dtype=torch.int8)
        return binary.reshape(images.shape[0], self.n_qubits).numpy().astype(np.int8)

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self._reduced), size=int(n_samples), replace=True)
        self.data = self._reduced[indices].astype(np.int8)
        return self.data

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(2, 2))

        sample_2d = np.asarray(sample).reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Greys", aspect="equal", interpolation="nearest")
        ax.set_title("Fashion-MNIST")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
