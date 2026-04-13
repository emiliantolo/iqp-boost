"""PennyLane Ising samples dataset loaded from XanaduAI/gqml_datasets."""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml

from .base import BinaryDataset

class PennylaneIsingDataset(BinaryDataset):
    """
    PennyLane Ising model samples for generative quantum ML.

    Loads the ``ising`` dataset from PennyLane 'other' category. 
    It contains binary vector samples representing configurations 
    from the Ising model at specific grid dimensions.
    
    Args:
        rows: Number of rows for the output grid (default 4).
        cols: Number of columns for the output grid (default 4).
              Total qubits = rows x cols.
    """

    def __init__(self, rows: int = 4, cols: int = 4):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.n_qubits = rows * cols

        dimension_key = f"({rows},{cols})"

        # Load from PennyLane
        [ds] = qml.data.load(
            "other", name="ising",
            progress_bar=False, folder_path="./data",
        )
        
        if dimension_key not in ds.train:
            raise ValueError(f"Dimension {dimension_key} not available in PennyLane Ising dataset. Available: {list(ds.train.keys())}")
            
        raw = np.array(ds.train[dimension_key], dtype=np.float64)

        # The original data uses {0, 1} encoding, but just to be safe:
        if raw.min() < 0:
            raw = ((raw + 1) / 2).astype(np.int8)
        else:
            raw = raw.astype(np.int8)

        self.data = raw

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Sample *n_samples* vectors from the dataset.

        Args:
            n_samples: Number of samples to draw (with replacement).
            seed: Random seed for reproducibility.

        Returns:
            Array of shape ``(n_samples, n_qubits)`` with binary values.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.data), size=n_samples, replace=True)
        return self.data[indices]

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a single Ising sample as a 2-D binary grid.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))

        sample_2d = sample.reshape((self.rows, self.cols))
        ax.imshow(sample_2d, cmap="Blues", aspect="equal", interpolation="nearest")
        ax.set_title("PennyLane Ising")

        ax.set_xticks(np.arange(-0.5, self.cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()
