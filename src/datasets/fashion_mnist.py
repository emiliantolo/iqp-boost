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
        classes: list[int] | None = None,
        train_split_ratio: float | None = None,
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
        self.classes = classes
        self.train_split_ratio = train_split_ratio

        if train_split_ratio is not None:
            self._train_data, self._test_data = self._load_split()
        else:
            self._all_data = self._load_single()

    def _preprocess(self, dataset) -> np.ndarray:
        """Area-resize and threshold a torchvision FashionMNIST dataset."""
        import torch
        import torch.nn.functional as functional

        images = dataset.data.to(dtype=torch.float32).unsqueeze(1) / 255.0
        resized = functional.interpolate(images, size=(self.rows, self.cols), mode="area")
        binary = (resized.squeeze(1) >= self.threshold).to(dtype=torch.int8)
        return binary.reshape(images.shape[0], self.n_qubits).numpy().astype(np.int8)

    def _filter_by_class(self, data: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.classes is None:
            return data
        mask = np.isin(labels, self.classes)
        return data[mask]

    def _load_single(self) -> np.ndarray:
        try:
            from torchvision.datasets import FashionMNIST
        except ImportError as exc:
            raise ImportError(
                "FashionMNISTDownscaledDataset requires torchvision. "
                "Install project dependencies with `uv sync`."
            ) from exc

        dataset = FashionMNIST(root=str(self.data_dir), train=True, download=True)
        data = self._preprocess(dataset)
        labels = np.array(dataset.targets, dtype=int)
        return self._filter_by_class(data, labels)

    def _load_split(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            from torchvision.datasets import FashionMNIST
        except ImportError as exc:
            raise ImportError(
                "FashionMNISTDownscaledDataset requires torchvision. "
                "Install project dependencies with `uv sync`."
            ) from exc

        train_dataset = FashionMNIST(root=str(self.data_dir), train=True, download=True)
        test_dataset = FashionMNIST(root=str(self.data_dir), train=False, download=True)

        train_data = self._preprocess(train_dataset)
        test_data = self._preprocess(test_dataset)

        train_labels = np.array(train_dataset.targets, dtype=int)
        test_labels = np.array(test_dataset.targets, dtype=int)

        train_data = self._filter_by_class(train_data, train_labels)
        test_data = self._filter_by_class(test_data, test_labels)

        return train_data, test_data

    def generate(self, n_samples: int | None = None, seed: int = 0, split: str = "train") -> np.ndarray:
        if self.train_split_ratio is not None:
            if split == "train":
                pool = self._train_data
            elif split == "test":
                pool = self._test_data
            elif split == "all":
                pool = np.concatenate([self._train_data, self._test_data], axis=0)
            else:
                raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")
        else:
            pool = self._all_data

        if n_samples is None:
            n_samples = len(pool)
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        rng = np.random.default_rng(seed)
        indices = rng.choice(len(pool), size=int(n_samples), replace=True)
        self.data = pool[indices]
        return self.data

    def set_split(self, split: str = "train") -> "FashionMNISTDownscaledDataset":
        if split in {"train", "test", "all"}:
            self.active_split = split
        else:
            raise ValueError(f"split must be 'train', 'test', or 'all', got {split!r}")
        return self

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(2, 2))

        sample_2d = np.asarray(sample).reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Greys", aspect="equal", interpolation="nearest")
        ax.set_title("Fashion-MNIST")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax
