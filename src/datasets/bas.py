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
    
    Valid patterns are exactly those that satisfy one of these constraints.
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
        
        # Pre-compute all unique valid bar and stripe patterns
        self._valid_patterns = self._compute_valid_patterns()

    def _compute_valid_patterns(self) -> set:
        """
        Enumerate all unique valid BAS patterns (bars and stripes).
        
        Returns:
            Set of tuples representing all valid patterns
        """
        patterns = set()
        
        # All possible bars (all rows identical)
        for row_pattern in range(2**self.width):
            row = np.array([(row_pattern >> i) & 1 for i in range(self.width)], dtype=np.int8)
            grid = np.tile(row, (self.height, 1))
            patterns.add(tuple(grid.flatten().astype(int)))
        
        # All possible stripes (all columns identical)
        for col_pattern in range(2**self.height):
            col = np.array([(col_pattern >> i) & 1 for i in range(self.height)], dtype=np.int8)
            grid = np.tile(col[:, None], (1, self.width))
            patterns.add(tuple(grid.flatten().astype(int)))
        
        return patterns

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
