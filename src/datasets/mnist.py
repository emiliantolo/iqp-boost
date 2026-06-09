"""Binarized MNIST dataset integration."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.datasets.base import BinaryDataset


class MNISTDataset(BinaryDataset):
    """MNIST images resized to a binary grid and flattened to bitstrings."""

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        threshold: float = 0.4,
        classes: list[int] | None = None,
        data_dir: str | Path = "./data",
    ):
        super().__init__(data=None, train_split_ratio=None)
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if classes is not None:
            classes = [int(label) for label in classes]
            invalid = [label for label in classes if label < 0 or label > 9]
            if invalid:
                raise ValueError(f"MNIST classes must be digit labels 0-9, got {invalid}")
            if not classes:
                raise ValueError("classes must be non-empty when provided")

        self.rows = int(rows)
        self.cols = int(cols)
        self.threshold = float(threshold)
        self.classes = classes
        self.data_dir = Path(data_dir)
        self.n_qubits = self.rows * self.cols
        self._split_cache: dict[str, np.ndarray] = {}
        self._split_label_cache: dict[str, np.ndarray] = {}

    def _generate_samples(self, n_samples: int, seed: int = 0) -> np.ndarray:
        return self._sample_split("train", n_samples=n_samples, seed=seed)

    def generate(self, n_samples: int | None = None, seed: int = 0, split: str = "train") -> np.ndarray:
        self._validate_split(split)
        if split == "all":
            raise ValueError("MNISTDataset supports split='train' or split='test'")
        if n_samples is None:
            raise ValueError("n_samples must be provided")
        self.data = self._sample_split(split, n_samples=n_samples, seed=seed)
        self.active_split = split
        return self.data

    def generate_balanced(self, n_per_class: int, seed: int = 0, split: str = "train") -> np.ndarray:
        self._validate_split(split)
        if split == "all":
            raise ValueError("MNISTDataset supports split='train' or split='test'")
        if n_per_class <= 0:
            raise ValueError("n_per_class must be positive")

        data, labels = self._load_split_with_labels(split)
        classes = self.classes if self.classes is not None else list(range(10))
        rng = np.random.default_rng(seed)
        selected_indices = []
        for label in classes:
            class_indices = np.flatnonzero(labels == label)
            if len(class_indices) == 0:
                raise ValueError(f"No MNIST samples found for class {label}")
            replace = n_per_class > len(class_indices)
            selected_indices.extend(rng.choice(class_indices, size=n_per_class, replace=replace).tolist())

        selected_indices = np.asarray(selected_indices, dtype=np.int64)
        selected_indices = selected_indices[rng.permutation(len(selected_indices))]
        self.data = np.asarray(data[selected_indices], dtype=np.int8)
        self.active_split = split
        return self.data

    def _sample_split(self, split: str, n_samples: int, seed: int) -> np.ndarray:
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        data = self._load_split(split)
        rng = np.random.default_rng(seed)
        replace = n_samples > len(data)
        indices = rng.choice(len(data), size=n_samples, replace=replace)
        return np.asarray(data[indices], dtype=np.int8)

    def _load_split(self, split: str) -> np.ndarray:
        data, _ = self._load_split_with_labels(split)
        return data

    def _load_split_with_labels(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._split_cache.get(split)
        cached_labels = self._split_label_cache.get(split)
        if cached is not None and cached_labels is not None:
            return cached, cached_labels

        from torchvision.datasets import MNIST

        dataset = MNIST(root=str(self.data_dir), train=(split == "train"), download=True)
        images = dataset.data
        targets = dataset.targets

        if self.classes is not None:
            class_tensor = torch.as_tensor(self.classes, dtype=targets.dtype, device=targets.device)
            mask = torch.isin(targets, class_tensor)
            images = images[mask]
            targets = targets[mask]
            if len(images) == 0:
                raise ValueError(f"No MNIST samples found for classes {self.classes}")

        images = images.to(dtype=torch.float32).unsqueeze(1) / 255.0
        resized = F.interpolate(
            images,
            size=(self.rows, self.cols),
            mode="bilinear",
            align_corners=False,
        )
        binary = (resized.squeeze(1) > self.threshold).to(torch.int8)
        flattened = binary.reshape(binary.shape[0], self.n_qubits).cpu().numpy()
        self._split_cache[split] = np.asarray(flattened, dtype=np.int8)
        self._split_label_cache[split] = np.asarray(targets.cpu().numpy(), dtype=np.int64)
        return self._split_cache[split], self._split_label_cache[split]

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        image = np.asarray(sample, dtype=np.int8).reshape(self.rows, self.cols)
        ax.imshow(image, cmap="gray_r", interpolation="nearest")
        ax.axis("off")
        return ax
