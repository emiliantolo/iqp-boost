"""PennyLane Hidden Manifold dataset loaded from XanaduAI/qml-benchmarks."""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .base import BinaryDataset

class PennylaneHMDataset(BinaryDataset):
    """
    PennyLane Hidden Manifold (HM) benchmark binarized for generative quantum ML.

    Loads the ``hidden-manifold`` dataset from PennyLane 'other' category.
    Available native dims represent total qubits: '2' up to '20'.
    """

    def __init__(self, rows: int = 4, cols: int = 4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols

        dimension_key = str(self.n_qubits)

        [ds] = qml.data.load(
            "other", name="hidden-manifold",
            progress_bar=False, folder_path="./data",
        )
        
        if dimension_key not in ds.train:
            raise ValueError(f"Dimension {dimension_key} (from {rows}x{cols}) not available in PennyLane HM dataset. Available: {list(ds.train.keys())}")
            
        raw_inputs = np.array(ds.train[dimension_key]['inputs'])
        raw = (raw_inputs > 0).astype(np.int8)

        self.data = raw

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.data), size=n_samples, replace=True)
        return self.data[indices]

    def visualize(self, sample: np.ndarray, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        sample_2d = sample.reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Greys", aspect="equal", interpolation="nearest")
        ax.set_title("PennyLane HM")

        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()
