"""Blobs dataset."""

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from .base import BinaryDataset


class BlobsDataset(BinaryDataset):
    """
    Generates binary blobs patterns from the PennyLane binary-blobs dataset.
    Valid patterns are exactly those that appear in the PennyLane dataset.
    """

    def __init__(self):
        """
        Initialize the Blobs dataset generator.
        Loads the full PennyLane dataset to define the valid pattern space.
        """
        super().__init__()
        self.n_qubits = 4 * 4  # 4x4 grid
        
        # Load and store all unique patterns from PennyLane dataset
        [ds] = qml.data.load("other", name="binary-blobs", progress_bar=False, folder_path="./data")
        all_patterns = np.array(ds.train['inputs'], dtype=np.int8)
        # Store as set of tuples for O(1) membership checking
        self._valid_patterns = set(map(tuple, all_patterns))
        self.all_patterns_array = all_patterns  # Keep for reference

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Generate binary blobs samples by sampling from the PennyLane dataset.
               
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, n_qubits) with binary values
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self.all_patterns_array), size=n_samples, replace=True)
        self.data = self.all_patterns_array[indices]
        return self.data

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualizes a single shapes sample as a 2D image.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(2, 2))
            
        sample_2d = sample.reshape((4, 4))
        ax.imshow(sample_2d, cmap='Blues', aspect='equal')
        ax.set_title("Binary Blobs Sample")
        
        # Add a grid to distinguish qubits
        ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if ax.figure is not None and ax is ax.figure.axes[0]:
            plt.tight_layout()

    def is_valid(self, sample: np.ndarray) -> bool:
        """
        Check if a sample is a valid blob pattern (exists in the PennyLane dataset).
        
        Args:
            sample: 1D binary array of length n_qubits
            
        Returns:
            True if sample is in the valid pattern space
        """
        return super().is_valid(sample)
