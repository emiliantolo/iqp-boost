"""Bars and Stripes (BAS) dataset."""

import numpy as np
import matplotlib.pyplot as plt
from .base import BinaryDataset


class BarsAndStripesDataset(BinaryDataset):
    """
    Generates Bars and Stripes (BAS) patterns.
    
    Each sample is a binary grid that is either:
    - A bar: All rows are identical
    - A stripe: All columns are identical
    """

    def __init__(self, height: int = 3, width: int = 3):
        """
        Initialize the BAS dataset generator.
        
        Args:
            height: Height of the grid (number of rows)
            width: Width of the grid (number of columns)
        """
        super().__init__()
        self.height = height
        self.width = width
        self.n_qubits = height * width

    def generate(self, n_samples: int, seed: int = 0) -> np.ndarray:
        """
        Generate BAS samples.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Array of shape (n_samples, height*width) with binary values
        """
        rng = np.random.default_rng(seed)
        samples = []
        
        for _ in range(n_samples):
            if rng.random() < 0.5:
                # Vertical bars: All rows identical
                bars = rng.integers(0, 2, size=(self.width,))
                img = np.tile(bars, (self.height, 1))
            else:
                # Horizontal stripes: All columns identical
                stripes = rng.integers(0, 2, size=(self.height,))
                img = np.tile(stripes[:, None], (1, self.width))
            
            samples.append(img.reshape(-1))
        
        self.data = np.array(samples, dtype=np.int8)
        return self.data

    def is_valid(self, sample: np.ndarray) -> bool:
        """Check if a sample is a valid BAS pattern."""
        img = sample.reshape(self.height, self.width)
        
        # Check if it's a bar (all rows identical)
        is_bar = all(np.array_equal(img[0], row) for row in img)
        if is_bar:
            return True
        
        # Check if it's a stripe (all columns identical)
        is_stripe = all(np.array_equal(img[:, 0], col) for col in img.T)
        if is_stripe:
            return True
        
        return False

    def validity_rate(self, samples: np.ndarray) -> float:
        """Compute fraction of samples that are valid BAS patterns."""
        valid = sum(self.is_valid(s) for s in samples)
        return valid / len(samples)

    def coverage_rate(self, ground_truth: np.ndarray, samples: np.ndarray) -> float:
        """
        Compute fraction of ground truth patterns that appear in samples.
        
        Args:
            ground_truth: Ground truth BAS samples
            samples: Model-generated samples
            
        Returns:
            Coverage rate (fraction of unique ground truth patterns in samples)
        """
        gt_unique = set(tuple(s.astype(int)) for s in ground_truth)
        model_unique = set(tuple(s.astype(int)) for s in samples if self.is_valid(s))
        
        if len(gt_unique) == 0:
            return 0.0
        
        return len(gt_unique & model_unique) / len(gt_unique)

    def visualize(self, sample: np.ndarray, ax=None):
        """
        Visualize a BAS sample as a 2D grid.
        
        Args:
            sample: 1D binary array of length height*width
            ax: Matplotlib axis to plot on (creates new if None)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        
        img = sample.reshape(self.height, self.width)
        ax.imshow(img, cmap='binary', interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add grid lines
        for i in range(self.height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(self.width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)
        
        # Label as bar or stripe
        is_bar = self.is_valid(sample)
        img_2d = sample.reshape(self.height, self.width)
        is_vertical = all(np.array_equal(img_2d[0], row) for row in img_2d)
        
        if is_bar:
            label = "Bar (vertical)" if is_vertical else "Stripe (horizontal)"
            ax.set_title(label)
        else:
            ax.set_title("Invalid BAS pattern")
        
        return ax
